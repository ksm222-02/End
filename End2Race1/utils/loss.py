from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def teacher_output_loss(
    teacher_outputs: torch.Tensor,
    teacher_targets: torch.Tensor,
    loss_type: str = "huber",
    delta: float = 1.0,
) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(teacher_outputs, teacher_targets)
    if loss_type == "mse":
        return F.mse_loss(teacher_outputs, teacher_targets)
    if loss_type == "huber":
        return F.smooth_l1_loss(teacher_outputs, teacher_targets, beta=delta)
    raise ValueError(f"Unsupported teacher_output_loss type: {loss_type}")


def teacher_feature_loss(
    teacher_features: torch.Tensor,
    teacher_feature_targets: Optional[torch.Tensor],
    normalize: bool = False,
) -> torch.Tensor:
    if teacher_feature_targets is None:
        return torch.zeros((), device=teacher_features.device, dtype=teacher_features.dtype)
    if normalize:
        teacher_features = F.normalize(teacher_features, dim=-1)
        teacher_feature_targets = F.normalize(teacher_feature_targets, dim=-1)
    return F.mse_loss(teacher_features, teacher_feature_targets)


def compute_teacher_losses(
    teacher_outputs: torch.Tensor,
    teacher_targets: torch.Tensor,
    teacher_features: Optional[torch.Tensor] = None,
    teacher_feature_targets: Optional[torch.Tensor] = None,
    output_weight: float = 1.0,
    feature_weight: float = 0.0,
    output_loss_type: str = "huber",
    delta: float = 1.0,
    normalize_features: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    output_loss = teacher_output_loss(
        teacher_outputs=teacher_outputs,
        teacher_targets=teacher_targets,
        loss_type=output_loss_type,
        delta=delta,
    )
    feature_loss = torch.zeros_like(output_loss)
    if teacher_features is not None and feature_weight > 0:
        feature_loss = teacher_feature_loss(
            teacher_features=teacher_features,
            teacher_feature_targets=teacher_feature_targets,
            normalize=normalize_features,
        )

    teacher_total_loss = output_weight * output_loss + feature_weight * feature_loss
    teacher_loss_dict = {
        "teacher_total_loss": teacher_total_loss.detach(),
        "teacher_output_loss": output_loss.detach(),
        "teacher_feature_loss": feature_loss.detach(),
    }
    return teacher_total_loss, teacher_loss_dict

