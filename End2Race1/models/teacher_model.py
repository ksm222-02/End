from typing import Dict, Optional

import torch
import torch.nn as nn

from models.teacher_encoder import TeacherEncoder
from models.teacher_mlp import TeacherMLP
from models.teacher_temporal import TeacherTemporal


class TeacherModel(nn.Module):
    """
    TeacherEncoder + TeacherMLP를 결합하여 제어 출력과 KD 중간 feature를 동시에 반환한다.
    """

    def __init__(
        self,
        teacher_encoder: TeacherEncoder,
        teacher_temporal: TeacherTemporal,
        teacher_mlp: TeacherMLP,
    ):
        super().__init__()
        self.teacher_encoder = teacher_encoder
        self.teacher_temporal = teacher_temporal
        self.teacher_mlp = teacher_mlp

    def forward(self, teacher_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            teacher_batch: dataloader가 제공하는 입력 딕셔너리.
                - lidar: [B, C, H, W] 또는 [B, F] 라이다 입력
                - meta (옵션): [B, meta_dim] 추가 메타 피쳐
                - speed (옵션): [B] 또는 [B, T, 1] 속도 입력
        Returns:
            dict: {"teacher_outputs": tensor, "teacher_features": tensor, "teacher_temporal_features": tensor}
        """
        teacher_lidar: torch.Tensor = teacher_batch.get("lidar")
        teacher_meta: Optional[torch.Tensor] = teacher_batch.get("meta")
        teacher_speed: Optional[torch.Tensor] = teacher_batch.get("speed")

        teacher_features = self.teacher_encoder(teacher_lidar, teacher_meta)

        # 시퀀스 차원 확보
        if teacher_features.dim() == 2:
            teacher_features_seq = teacher_features.unsqueeze(1)  # [B, 1, F]
        else:
            teacher_features_seq = teacher_features

        if teacher_speed is not None and teacher_speed.dim() == 1:
            teacher_speed = teacher_speed.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        elif teacher_speed is not None and teacher_speed.dim() == 2:
            teacher_speed = teacher_speed.unsqueeze(2)  # [B, T, 1]

        teacher_gru_out, _ = self.teacher_temporal(teacher_features_seq, teacher_speed)
        teacher_temporal_features = teacher_gru_out[:, -1, :]  # 마지막 타임스텝 사용
        teacher_outputs = self.teacher_mlp(teacher_temporal_features)

        return {
            "teacher_outputs": teacher_outputs,
            "teacher_features": teacher_features,
            "teacher_temporal_features": teacher_temporal_features,
        }

    def freeze_teacher_encoder(self) -> None:
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
