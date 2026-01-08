from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_teacher_checkpoint(
    teacher_checkpoint_path: Path,
    teacher_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
) -> None:
    teacher_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "teacher_state_dict": teacher_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, teacher_checkpoint_path)


def load_teacher_checkpoint(
    teacher_checkpoint_path: Path,
    teacher_model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
) -> int:
    checkpoint = torch.load(teacher_checkpoint_path, map_location=map_location)
    teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", 0))

