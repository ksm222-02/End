import torch
import torch.nn as nn
from typing import Optional, Tuple


class TeacherTemporal(nn.Module):
    """
    End2Race GRU 스타일을 참고한 Teacher 전용 temporal 모듈.
    BEV feature 시퀀스와 speed 입력을 결합해 시계열 정보를 인코딩한다.
    """

    def __init__(
        self,
        feature_dim: int,
        speed_dim: int = 1,
        hidden_scale: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.teacher_speed_mlp = nn.Sequential(
            nn.Linear(speed_dim, feature_dim // 6),
            nn.ReLU(),
        )
        processed_features = feature_dim + feature_dim // 6

        self.teacher_gru = nn.GRU(
            input_size=processed_features,
            hidden_size=processed_features * hidden_scale,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        teacher_features_seq: torch.Tensor,
        teacher_speed_seq: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            teacher_features_seq: [B, T, F] encoder output sequence (T=1 가능).
            teacher_speed_seq: [B, T, speed_dim] or None.
        Returns:
            teacher_gru_out: [B, T, H] GRU 출력.
            teacher_last_hidden: [num_layers, B, H] 마지막 히든.
        """
        if teacher_speed_seq is None:
            teacher_speed_seq = torch.zeros(
                teacher_features_seq.size(0),
                teacher_features_seq.size(1),
                1,
                device=teacher_features_seq.device,
                dtype=teacher_features_seq.dtype,
            )

        teacher_speed_emb = self.teacher_speed_mlp(teacher_speed_seq)
        teacher_fused = torch.cat([teacher_features_seq, teacher_speed_emb], dim=-1)
        teacher_gru_out, teacher_last_hidden = self.teacher_gru(teacher_fused, teacher_hidden)
        return teacher_gru_out, teacher_last_hidden

