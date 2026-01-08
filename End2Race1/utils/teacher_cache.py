from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch


def _key_to_filename(key: str) -> str:
    # filesystem-safe, stable mapping
    return (
        key.replace("/", "__")
        .replace(":", "_")
        .replace("\\", "__")
        .replace("..", "_")
    ) + ".npz"


@dataclass(frozen=True)
class TeacherCacheSample:
    actions: torch.Tensor  # (T, A)
    gru_out: Optional[torch.Tensor] = None  # (T, H)
    bev_vec: Optional[torch.Tensor] = None  # (T, D)
    bev_feat: Optional[torch.Tensor] = None  # (T, C, H, W)


class TeacherCache:
    """
    Offline cache for teacher outputs/features, keyed by a stable sequence key.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def path_for_key(self, key: str) -> Path:
        return self.cache_dir / _key_to_filename(key)

    def has(self, key: str) -> bool:
        return self.path_for_key(key).exists()

    def load(self, key: str, device: Optional[torch.device] = None) -> TeacherCacheSample:
        path = self.path_for_key(key)
        payload = np.load(path, allow_pickle=False)
        to_tensor = lambda x: torch.from_numpy(x).to(device) if device is not None else torch.from_numpy(x)

        actions = to_tensor(payload["actions"]).float()
        gru_out = to_tensor(payload["gru_out"]).float() if "gru_out" in payload else None
        bev_vec = to_tensor(payload["bev_vec"]).float() if "bev_vec" in payload else None
        bev_feat = to_tensor(payload["bev_feat"]).float() if "bev_feat" in payload else None
        return TeacherCacheSample(actions=actions, gru_out=gru_out, bev_vec=bev_vec, bev_feat=bev_feat)


class TeacherCacheWriter:
    """
    Writes per-sequence teacher cache files.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        key: str,
        *,
        actions: torch.Tensor,
        gru_out: Optional[torch.Tensor] = None,
        bev_vec: Optional[torch.Tensor] = None,
        bev_feat: Optional[torch.Tensor] = None,
    ) -> Path:
        path = self.cache_dir / _key_to_filename(key)

        def to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().float().numpy()

        payload: Dict[str, np.ndarray] = {"actions": to_np(actions)}
        if gru_out is not None:
            payload["gru_out"] = to_np(gru_out)
        if bev_vec is not None:
            payload["bev_vec"] = to_np(bev_vec)
        if bev_feat is not None:
            payload["bev_feat"] = to_np(bev_feat)
        np.savez_compressed(path, **payload)
        return path

    def write_many(self, keys: Iterable[str], samples: Iterable[TeacherCacheSample]) -> None:
        for key, sample in zip(keys, samples):
            self.write(
                key,
                actions=sample.actions,
                gru_out=sample.gru_out,
                bev_vec=sample.bev_vec,
                bev_feat=sample.bev_feat,
            )

