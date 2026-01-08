from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityViewTransform(nn.Module):
    """
    Placeholder for Lift-Splat / view transform.
    For now, it just resizes a 2D feature map into a fixed BEV grid.
    """

    def __init__(self, in_channels: int, bev_channels: int = 64, bev_hw: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.bev_hw = tuple(bev_hw)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, bev_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        bev = self.proj(feat_2d)
        bev = F.interpolate(bev, size=self.bev_hw, mode="bilinear", align_corners=False)
        return bev


class CameraBEVStudent(nn.Module):
    """
    Student model (camera + ego) producing control, with BEV-aligned intermediate features for KD.

    Inputs:
      images: Tensor (B, T, 3, H, W)
      speed_input: Tensor (B, T, 1)
      steering_input: Tensor (B, T, 1)  (previous wheel angle, same convention as teacher dataset)

    Outputs:
      actions: Tensor (B, T, num_actions)
      last_hidden: Tensor (1, B, hidden_size)
      aux (optional): dict with intermediate tensors for KD
    """

    def __init__(
        self,
        num_actions: int = 2,
        hidden_scale: int = 2,
        img_feat_channels: int = 64,
        bev_feat_channels: int = 64,
        bev_hw: Tuple[int, int] = (128, 128),
        speed_embed_dim: int = 32,
        steering_embed_dim: int = 32,
        bev_vec_dim: int = 256,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.hidden_scale = int(hidden_scale)

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, img_feat_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(img_feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(img_feat_channels, img_feat_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(img_feat_channels),
            nn.ReLU(inplace=True),
        )

        self.view_transform = IdentityViewTransform(
            in_channels=img_feat_channels,
            bev_channels=bev_feat_channels,
            bev_hw=bev_hw,
        )

        self.bev_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(bev_feat_channels, bev_vec_dim),
            nn.ReLU(inplace=True),
        )

        self.speed_mlp = nn.Sequential(
            nn.Linear(1, speed_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(speed_embed_dim, speed_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.steering_mlp = nn.Sequential(
            nn.Linear(1, steering_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(steering_embed_dim, steering_embed_dim),
            nn.ReLU(inplace=True),
        )

        gru_in_dim = bev_vec_dim + speed_embed_dim + steering_embed_dim
        gru_hidden_dim = gru_in_dim * self.hidden_scale
        self.gru = nn.GRU(
            input_size=gru_in_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gru_in_dim, self.num_actions),
        )

    def forward(
        self,
        images: torch.Tensor,
        speed_input: torch.Tensor,
        steering_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        if images.dim() != 5:
            raise ValueError("images must be shaped (B, T, 3, H, W).")
        if speed_input.dim() != 3 or speed_input.size(-1) != 1:
            raise ValueError("speed_input must be shaped (B, T, 1).")
        if steering_input.dim() != 3 or steering_input.size(-1) != 1:
            raise ValueError("steering_input must be shaped (B, T, 1).")

        B, T, C, H, W = images.shape
        images_flat = images.reshape(B * T, C, H, W)

        feat_2d = self.backbone(images_flat)  # (B*T, C2, h, w)
        bev_feat = self.view_transform(feat_2d)  # (B*T, C_bev, H_bev, W_bev)
        bev_vec = self.bev_pool(bev_feat)  # (B*T, D)

        bev_feat_seq = bev_feat.reshape(B, T, bev_feat.shape[1], bev_feat.shape[2], bev_feat.shape[3])
        bev_vec_seq = bev_vec.reshape(B, T, bev_vec.shape[1])

        speed_emb = self.speed_mlp(speed_input)
        steering_emb = self.steering_mlp(steering_input)
        gru_in = torch.cat([bev_vec_seq, speed_emb, steering_emb], dim=-1)

        gru_out, last_hidden = self.gru(gru_in, hidden)
        actions = self.output_layer(gru_out)

        if not return_aux:
            return actions, last_hidden, None

        aux = {
            "bev_feat_seq": bev_feat_seq,
            "bev_vec_seq": bev_vec_seq,
            "gru_in": gru_in,
            "gru_out": gru_out,
        }
        return actions, last_hidden, aux

