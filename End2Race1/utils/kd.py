from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _maybe_normalize(x: torch.Tensor, normalize: bool, dim: int) -> torch.Tensor:
    return F.normalize(x, dim=dim) if normalize else x


def distill_action_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    loss_type: str = "huber",
    delta: float = 1.0,
) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(student_actions, teacher_actions)
    if loss_type == "mse":
        return F.mse_loss(student_actions, teacher_actions)
    if loss_type == "huber":
        return F.smooth_l1_loss(student_actions, teacher_actions, beta=delta)
    raise ValueError(f"Unsupported distill_action_loss type: {loss_type}")


def distill_feature_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    *,
    normalize: bool = False,
    loss_type: str = "mse",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Supports vector features (B,T,D) or BEV maps (B,T,C,H,W).
    If mask is provided, it must be broadcastable to feature shape.
    """
    if normalize:
        if student_features.dim() in (3, 2):
            student_features = _maybe_normalize(student_features, True, dim=-1)
            teacher_features = _maybe_normalize(teacher_features, True, dim=-1)
        elif student_features.dim() == 5:
            student_features = _maybe_normalize(student_features, True, dim=2)
            teacher_features = _maybe_normalize(teacher_features, True, dim=2)

    if mask is not None:
        student_features = student_features * mask
        teacher_features = teacher_features * mask

    if loss_type == "mse":
        return F.mse_loss(student_features, teacher_features)
    if loss_type == "l1":
        return F.l1_loss(student_features, teacher_features)
    if loss_type == "huber":
        return F.smooth_l1_loss(student_features, teacher_features)
    raise ValueError(f"Unsupported distill_feature_loss type: {loss_type}")

