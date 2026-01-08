import torch
import torch.nn as nn
from typing import Optional


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation for TeacherEncoder: {name}")


class TeacherEncoder(nn.Module):
    """
    라이다 입력(BEV 맵 또는 벡터)을 받아 KD용 중간 feature를 생성하는 Teacher 전용 Encoder.
    """

    def __init__(
        self,
        lidar_channels: int = 1,
        lidar_feature_dim: int = 360,
        base_channels: int = 64,
        feature_dim: int = 256,
        projection_dim: Optional[int] = None,
        meta_dim: int = 0,
        activation: str = "relu",
        input_type: str = "bev",  # "bev" => [B, C, H, W], "vector" => [B, F] or [B, T, F]
    ):
        super().__init__()
        teacher_activation = _get_activation(activation)
        self.teacher_input_type = input_type

        if input_type == "bev":
            # 간단한 BEV conv 인코더
            self.teacher_stem = nn.Sequential(
                nn.Conv2d(lidar_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(base_channels),
                teacher_activation,
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                teacher_activation,
            )

            self.teacher_encoder_blocks = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                teacher_activation,
                nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                teacher_activation,
            )
            self.teacher_pool = nn.AdaptiveAvgPool2d((1, 1))
            backbone_out_dim = base_channels * 4
        elif input_type == "vector":
            # 라이다 벡터 입력 (예: 360-빔 스캔 또는 BEV flatten)
            self.teacher_vector_encoder = nn.Sequential(
                nn.Linear(lidar_feature_dim, feature_dim * 2),
                teacher_activation,
                nn.Linear(feature_dim * 2, feature_dim),
                teacher_activation,
            )
            backbone_out_dim = feature_dim
        else:
            raise ValueError(f"Unsupported teacher input_type: {input_type}")

        fused_dim = backbone_out_dim
        self.teacher_meta_adapter: Optional[nn.Module] = None
        if meta_dim > 0:
            self.teacher_meta_adapter = nn.Sequential(
                nn.Linear(meta_dim, feature_dim),
                teacher_activation,
            )
            fused_dim = fused_dim + feature_dim

        self.teacher_feature_head = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            teacher_activation,
            nn.LayerNorm(feature_dim),
        )

        self.teacher_projection: Optional[nn.Module] = None
        if projection_dim is not None:
            self.teacher_projection = nn.Sequential(
                nn.Linear(feature_dim, projection_dim),
                teacher_activation,
                nn.LayerNorm(projection_dim),
            )

    def forward(
        self,
        teacher_lidar: torch.Tensor,
        teacher_meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            teacher_lidar: 라이다 입력. BEV일 때 [B, C, H, W], 벡터일 때 [B, F] 또는 [B, T, F].
            teacher_meta: 추가 메타 피쳐(옵션) [B, meta_dim].
        Returns:
            teacher_features: KD 증류 대상이 되는 중간 특징.
        """
        if self.teacher_input_type == "bev":
            x = self.teacher_stem(teacher_lidar)
            x = self.teacher_encoder_blocks(x)
            x = self.teacher_pool(x).flatten(1)
        else:
            # 벡터 또는 시퀀스 벡터 처리
            if teacher_lidar.dim() == 3:  # [B, T, F] -> 평균 pooling으로 단일 벡터
                teacher_lidar = teacher_lidar.mean(dim=1)
            x = self.teacher_vector_encoder(teacher_lidar)

        if teacher_meta is not None and self.teacher_meta_adapter is not None:
            meta_features = self.teacher_meta_adapter(teacher_meta)
            x = torch.cat([x, meta_features], dim=1)

        teacher_features = self.teacher_feature_head(x)

        if self.teacher_projection is not None:
            teacher_features = self.teacher_projection(teacher_features)

        return teacher_features
