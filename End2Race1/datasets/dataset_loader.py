import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial


DEFAULT_TEACHER_DATA_ROOT = Path("/home/jovyan/kdkd/dataset/dataset")


def load_teacher_lidar_bin(bin_path: Path) -> torch.Tensor:
    """
    라이다 .bin을 (N, 8) float32로 로드.
    필드 추정: [x, y, z, pad?, intensity, ring, ts0, ts1]
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 8)
    return torch.as_tensor(points, dtype=torch.float32)


def pointcloud_to_bev(
    points: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: float = 0.2,
) -> torch.Tensor:
    """
    간단한 BEV rasterizer: max height, mean intensity, density 3채널.
    points: [N, 8] (x,y,z,?,intensity,ring,ts0,ts1)
    """
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 4]
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    bev_w = int((x_max - x_min) / resolution)
    bev_h = int((y_max - y_min) / resolution)

    # grid index
    ix = ((x - x_min) / resolution).long().clamp(0, bev_w - 1)
    iy = ((y - y_min) / resolution).long().clamp(0, bev_h - 1)

    max_height = torch.full((bev_h, bev_w), z_min, device=points.device, dtype=torch.float32)
    sum_intensity = torch.zeros((bev_h, bev_w), device=points.device, dtype=torch.float32)
    count = torch.zeros((bev_h, bev_w), device=points.device, dtype=torch.float32)

    for xi, yi, zi, inten in zip(ix, iy, z, intensity):
        if zi > max_height[yi, xi]:
            max_height[yi, xi] = zi
        sum_intensity[yi, xi] += inten
        count[yi, xi] += 1.0

    mean_intensity = torch.where(count > 0, sum_intensity / count, torch.zeros_like(sum_intensity))
    density = torch.clamp(count / count.max().clamp(min=1.0), 0.0, 1.0)

    bev = torch.stack([max_height, mean_intensity, density], dim=0)  # [3, H, W]
    return bev


class TeacherDataset(Dataset):
    """
    고정 데이터 경로(/media/sws/X9 Pro/dataset/)를 사용하는 Teacher 전용 Dataset.
    manifest(json)에는 lidar/gt_controls(/meta) 경로 정보를 포함한다고 가정한다.
    """

    def __init__(
        self,
        data_root: Path = DEFAULT_TEACHER_DATA_ROOT,
        split: str = "train",
        manifest: Optional[Path] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.teacher_data_root = Path(data_root)
        self.teacher_split = split
        self.teacher_manifest = Path(manifest) if manifest else self.teacher_data_root / f"{split}.json"
        self.teacher_transform = transform

        self.teacher_samples = self._load_manifest(self.teacher_manifest)
        self.teacher_bev_builder = partial(
            pointcloud_to_bev,
            x_range=(-50.0, 50.0),
            y_range=(-50.0, 50.0),
            z_range=(-5.0, 5.0),
            resolution=0.2,
        )

    def _resolve_path(self, maybe_path: str) -> Path:
        path = Path(maybe_path)
        if not path.is_absolute():
            path = self.teacher_data_root / path
        return path

    def _load_manifest(self, manifest_path: Path) -> List[Dict]:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Teacher manifest not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            entries = json.load(f)
        teacher_samples: List[Dict] = []
        for entry in entries:
            lidar_path = self._resolve_path(entry["lidar"])
            teacher_control = entry["control"]
            teacher_meta = entry.get("meta")  # optional meta features
            teacher_speed = entry.get("speed")  # optional speed value/path
            teacher_samples.append(
                {
                    "lidar": lidar_path,
                    "control": teacher_control,
                    "meta": teacher_meta,
                    "speed": teacher_speed,
                }
            )
        return teacher_samples

    def __len__(self) -> int:
        return len(self.teacher_samples)

    def _load_tensor(self, source) -> torch.Tensor:
        if isinstance(source, str):
            source = self._resolve_path(source)
        if isinstance(source, Path):
            if source.suffix in {".npy", ".npz"}:
                data = np.load(source)
                if isinstance(data, np.lib.npyio.NpzFile):
                    data = data["arr_0"]
                return torch.as_tensor(data, dtype=torch.float32)
            if source.suffix == ".bin":
                return load_teacher_lidar_bin(source)
            with open(source, "r") as f:
                data = json.load(f)
            return torch.as_tensor(data, dtype=torch.float32)
        return torch.as_tensor(source, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.teacher_samples[idx]
        teacher_lidar = self._load_tensor(sample["lidar"])

        # 라이다 .bin 포인트를 BEV 맵으로 변환 (기본: 3채널 max height / mean intensity / density)
        if teacher_lidar.dim() == 2 and teacher_lidar.shape[1] == 8:
            teacher_lidar = self.teacher_bev_builder(teacher_lidar)
        elif teacher_lidar.dim() == 3 and teacher_lidar.shape[-1] == 8:
            # [N, 8] 형태를 명시적으로 맞춰줌
            teacher_lidar = self.teacher_bev_builder(teacher_lidar)
        else:
            # 라이다 형태 정규화: BEV 맵은 [C, H, W], 벡터는 [F] 또는 [T, F]
            if teacher_lidar.dim() == 2:
                teacher_lidar = teacher_lidar.unsqueeze(0)  # [1, H, W]
            elif teacher_lidar.dim() == 3:
                # HWC -> CHW로 변환 (채널 축이 마지막이고 작을 때)
                if teacher_lidar.shape[0] not in {1, 2, 3, 4} and teacher_lidar.shape[-1] in {1, 2, 3, 4}:
                    teacher_lidar = teacher_lidar.permute(2, 0, 1)
                # 그 외 [C, H, W] 또는 [T, F, ?] 형태는 그대로 유지

        if self.teacher_transform is not None:
            teacher_lidar = self.teacher_transform(teacher_lidar)

        teacher_meta = sample.get("meta")
        teacher_meta_tensor = self._load_tensor(teacher_meta) if teacher_meta is not None else None
        teacher_speed = sample.get("speed")
        teacher_speed_tensor = self._load_tensor(teacher_speed) if teacher_speed is not None else None
        teacher_controls = torch.as_tensor(sample["control"], dtype=torch.float32)

        batch: Dict[str, torch.Tensor] = {
            "lidar": teacher_lidar,
            "gt_controls": teacher_controls,
        }
        # meta/speed가 없으면 None을 collate하지 않도록 키를 생략한다
        if teacher_meta_tensor is not None:
            batch["meta"] = teacher_meta_tensor
        if teacher_speed_tensor is not None:
            batch["speed"] = teacher_speed_tensor

        return batch


def build_teacher_dataloader(
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    data_root: Path = DEFAULT_TEACHER_DATA_ROOT,
    split: str = "train",
    manifest: Optional[Path] = None,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> DataLoader:
    dataset = TeacherDataset(data_root=data_root, split=split, manifest=manifest, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader
