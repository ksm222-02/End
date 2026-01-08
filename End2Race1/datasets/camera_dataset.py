from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CameraSample:
    images: torch.Tensor  # (T, 3, H, W)
    speed: torch.Tensor  # (T, 1)
    steering_input: torch.Tensor  # (T, 1)
    gt_actions: torch.Tensor  # (T, A)
    key: str


class CameraVehicleDataset(Dataset):
    """
    Camera sequence dataset skeleton.

    Expected directory structure (same as LiDAR VehicleDataset):
      {root_dir}/{split}/segment_x/meta/data.csv

    Expected CSV columns (minimum):
      - velocity
      - wheel_angle
      - image_filepath  (relative to segment_path)

    Notes:
      - steering_input follows teacher convention: input at t is wheel_angle at t-1.
      - gt_actions default: [wheel_angle, dummy_action] to match teacher scripts.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        sequence_length: int = 5,
        image_column: str = "image_filepath",
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.root_dir = str(root_dir)
        self.split = str(split)
        self.sequence_length = int(sequence_length)
        self.image_column = str(image_column)
        self.image_size = tuple(image_size)

        split_path = os.path.join(self.root_dir, self.split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Split directory not found: {split_path}")

        self.segment_dfs = {}
        self.all_sequences_info = []

        segment_dirs = [
            d
            for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d)) and d.startswith("segment_")
        ]
        for segment_name in sorted(segment_dirs):
            segment_path = os.path.join(split_path, segment_name)
            meta_path = os.path.join(segment_path, "meta", "data.csv")
            if not os.path.exists(meta_path):
                continue

            df = pd.read_csv(meta_path)
            missing = [c for c in ("velocity", "wheel_angle", self.image_column) if c not in df.columns]
            if missing:
                raise KeyError(f"Missing columns in {meta_path}: {missing}")

            self.segment_dfs[segment_path] = df
            for start in range(len(df) - self.sequence_length + 1):
                self.all_sequences_info.append((segment_path, start))

        if not self.all_sequences_info:
            raise RuntimeError(f"No valid sequences found for split '{self.split}' in '{self.root_dir}'.")

    def __len__(self) -> int:
        return len(self.all_sequences_info)

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            from PIL import Image
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Pillow is required to load camera images. Install it (e.g., `pip install pillow`) or "
                "replace image loading with your preferred backend."
            ) from e

        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)

    def __getitem__(self, idx: int) -> CameraSample:
        segment_path, start_index = self.all_sequences_info[idx]
        df = self.segment_dfs[segment_path]
        seq = df.iloc[start_index : start_index + self.sequence_length]

        steering_input_seq = []
        if start_index == 0:
            steering_input_seq.append(0.0)
        else:
            steering_input_seq.append(float(df.iloc[start_index - 1]["wheel_angle"]))
        steering_input_seq.extend(seq["wheel_angle"].iloc[:-1].astype(float).tolist())

        images = []
        speed = []
        actions = []
        for _, row in seq.iterrows():
            image_rel = row[self.image_column]
            image_path = os.path.join(segment_path, str(image_rel))
            images.append(self._load_image(image_path))
            speed.append(float(row["velocity"]))
            actions.append([float(row["wheel_angle"]), 0.0])

        images_t = torch.stack(images, dim=0)  # (T,3,H,W)
        speed_t = torch.tensor(speed, dtype=torch.float32).view(-1, 1)
        steering_in_t = torch.tensor(steering_input_seq, dtype=torch.float32).view(-1, 1)
        actions_t = torch.tensor(actions, dtype=torch.float32)

        segment_name = os.path.basename(segment_path.rstrip(os.sep))
        key = f"{self.split}/{segment_name}:{start_index}"
        return CameraSample(
            images=images_t,
            speed=speed_t,
            steering_input=steering_in_t,
            gt_actions=actions_t,
            key=key,
        )


def collate_camera_samples(batch: Sequence[CameraSample]):
    images = torch.stack([b.images for b in batch], dim=0)  # (B,T,3,H,W)
    speed = torch.stack([b.speed for b in batch], dim=0)  # (B,T,1)
    steering_input = torch.stack([b.steering_input for b in batch], dim=0)  # (B,T,1)
    gt_actions = torch.stack([b.gt_actions for b in batch], dim=0)  # (B,T,A)
    keys = [b.key for b in batch]
    return images, speed, steering_input, gt_actions, keys
