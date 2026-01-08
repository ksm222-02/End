import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .pointpillars_teacher import PointPillarsBEV


class LiDARBEVTeacher(nn.Module):
    """
    Teacher model:
        LiDAR(PointCloud) + Speed + Steering -> BEV -> CNN -> BEV feature(vec) -> GRU -> Control

    Expected LiDAR input format (sequence):
        batched_pts_seq: List[List[Tensor]]
            Outer list length = B (batch)
            Inner list length = T (sequence length)
            Each Tensor: (N_i, 4)  (x,y,z,intensity) in LiDAR frame, consistent with your voxelizer settings.

    Scalar inputs:
        speed_input: Tensor (B, T, 1)
        steering_input: Tensor (B, T, 1)

    Output:
        actions: Tensor (B, T, num_actions)
        last_hidden: Tensor (1, B, hidden_size)  (GRU hidden state)
        aux: dict with intermediate features (optional, useful for KD/MemDistill)
    """
    def __init__(
        self,
        num_actions: int = 2,
        mask_prob: float = 0.0,
        hidden_scale: int = 2,
        bev_vec_dim: int = 256,
        speed_embed_dim: int = 32,
        steering_embed_dim: int = 32,
        voxel_size=(0.16, 0.16, 4.0),
        point_cloud_range=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0), #!!!!
        max_num_points: int = 32,
        max_voxels=(16000, 40000),
    ):
        super().__init__()

        self.mask_prob = float(mask_prob)
        self.hidden_scale = int(hidden_scale)
        self.num_actions = int(num_actions)

        # LiDAR -> BEV feature extractor (no detection)
        self.bev_encoder = PointPillarsBEV(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
            vec_dim=bev_vec_dim,
        )

        # Speed embedding
        self.speed_mlp = nn.Sequential(
            nn.Linear(1, speed_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(speed_embed_dim, speed_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dummy_embedding = nn.Parameter(torch.randn(1, speed_embed_dim))

        # Steering embedding
        self.steering_mlp = nn.Sequential(
            nn.Linear(1, steering_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(steering_embed_dim, steering_embed_dim),
            nn.ReLU(inplace=True),
        )

        # Temporal model
        gru_in_dim = bev_vec_dim + speed_embed_dim + steering_embed_dim
        gru_hidden_dim = gru_in_dim * self.hidden_scale

        self.gru = nn.GRU(
            input_size=gru_in_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Control head
        self.output_layer = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gru_in_dim, self.num_actions),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
        nn.init.xavier_normal_(self.dummy_embedding)

    def _encode_bev_sequence(
        self,
        batched_pts_seq: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Encodes LiDAR point clouds into a BEV vector sequence.

        Returns:
            bev_vec_seq: Tensor (B, T, D)
            aux: dict of optional intermediate tensors (can be large)
        """
        B = len(batched_pts_seq)
        if B == 0:
            raise ValueError("batched_pts_seq is empty.")
        T = len(batched_pts_seq[0])
        if any(len(seq) != T for seq in batched_pts_seq):
            raise ValueError("All batch elements must have the same sequence length T.")

        bev_vec_list = []
        # Optional (large) feature retention:
        bev_pseudo_list = []
        bev_feat_list = []

        for t in range(T):
            batched_pts_t = [batched_pts_seq[b][t] for b in range(B)]
            bev_pseudo, bev_feat, bev_vec = self.bev_encoder(batched_pts_t)
            bev_vec_list.append(bev_vec)          # (B, D)
            bev_pseudo_list.append(bev_pseudo)    # (B, 64, y, x)
            bev_feat_list.append(bev_feat)        # (B, C, H, W)

        bev_vec_seq = torch.stack(bev_vec_list, dim=1)  # (B, T, D)

        aux = {
            "bev_vec_seq": bev_vec_seq,
            # The next two are big; keep them only if you need feature-level KD:
            "bev_pseudo_seq": torch.stack(bev_pseudo_list, dim=1),  # (B, T, 64, y, x)
            "bev_feat_seq": torch.stack(bev_feat_list, dim=1),      # (B, T, C, H, W)
        }
        return bev_vec_seq, aux

    def forward(
        self,
        batched_pts_seq: List[List[torch.Tensor]],
        speed_input: torch.Tensor,
        steering_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Args:
            batched_pts_seq: List[List[Tensor]]  (B, T, ...)
            speed_input: Tensor (B, T, 1)
            steering_input: Tensor (B, T, 1)
            hidden: optional GRU hidden (1, B, H)
            return_aux: if True, return intermediate tensors for KD/debug

        Returns:
            actions: (B, T, num_actions)
            last_hidden: (1, B, hidden_size)
            aux (optional): dict
        """
        if speed_input.dim() != 3 or speed_input.size(-1) != 1:
            raise ValueError("speed_input must be shaped (B, T, 1).")
        if steering_input.dim() != 3 or steering_input.size(-1) != 1:
            raise ValueError("steering_input must be shaped (B, T, 1).")

        bev_vec_seq, aux = self._encode_bev_sequence(batched_pts_seq)  # (B,T,D)
        B, T, D = bev_vec_seq.shape

        # Scalar embeddings
        speed_emb = self.speed_mlp(speed_input)  # (B, T, E_s)
        steering_emb = self.steering_mlp(steering_input) # (B, T, E_steer)

        # Speed masking (optional, training-only)
        # Note: could extend this to steering or other inputs
        if self.training and self.mask_prob > 0.0:
            mask = torch.rand(B, T, 1, device=speed_input.device) < self.mask_prob
            dummy = self.dummy_embedding.expand(B, T, -1)
            speed_emb = torch.where(mask, dummy, speed_emb)

        # GRU input
        gru_in = torch.cat([bev_vec_seq, speed_emb, steering_emb], dim=-1)  # (B, T, D+E_s+E_steer)

        gru_out, last_hidden = self.gru(gru_in, hidden)       # (B, T, H)
        actions = self.output_layer(gru_out)                  # (B, T, A)

        if return_aux:
            aux = {
                **aux,
                "speed_warning": "bev_pseudo_seq/bev_feat_seq are large; disable return_aux for training speed.",
                "gru_in": gru_in,
                "gru_out": gru_out,
            }
            return actions, last_hidden, aux

        return actions, last_hidden, None
