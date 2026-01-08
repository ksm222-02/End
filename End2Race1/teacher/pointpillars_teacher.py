import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep the same voxelization dependency as your original implementation.
# This file intentionally strips all detection-specific components (anchors/head/NMS).
from pointpillars.ops import Voxelization


class PillarLayer(nn.Module):
    """
    Voxelizes a list of point clouds (one tensor per batch element) into pillars/voxels.

    Input:
        batched_pts: list[Tensor], length B
            each Tensor shape: (N, C_in) with C_in >= 3 (typically 4: x,y,z,intensity)

    Output:
        pillars: Tensor (P_total, max_points, C_in)
        coors_batch: Tensor (P_total, 1 + 3)  [batch_idx, x_idx, y_idx, z_idx] (order defined by Voxelization)
        npoints_per_pillar: Tensor (P_total,)
    """
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

    @torch.no_grad()
    def forward(self, batched_pts):
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            # The voxelization op requires contiguous input.
            pts = pts.contiguous()
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)

        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))  # prepend batch index
        coors_batch = torch.cat(coors_batch, dim=0)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    """
    Encodes raw pillar points into a BEV pseudo-image.

    Output:
        bev: Tensor (B, out_channel, y_l, x_l)
    """
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel

        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        device = pillars.device

        # 1) Offset to points center (within each pillar)
        offset_pt_center = pillars[:, :, :3] - (
            torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]
        )

        # 2) Offset to pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)

        # 3) Feature concat (PointPillars standard: 4 + 3 + 1 + 1 = 9 dims)
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)

        # Keep the mmdet3d-consistent overwrite (as in your original file)
        features[:, :, 0:1] = x_offset_pi_center
        features[:, :, 1:2] = y_offset_pi_center

        # 4) Mask padded points
        voxel_ids = torch.arange(0, pillars.size(1), device=device)  # (max_points,)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]      # (max_points, P_total)
        mask = mask.permute(1, 0).contiguous()                       # (P_total, max_points)
        features *= mask[:, :, None]

        # 5) Embed + max pool
        features = features.permute(0, 2, 1).contiguous()            # (P_total, 9, max_points)
        features = F.relu(self.bn(self.conv(features)))              # (P_total, out_channel, max_points)
        pooling_features = torch.max(features, dim=-1)[0]            # (P_total, out_channel)

        # 6) Scatter to BEV canvas
        bs = int(coors_batch[:, 0].max().item()) + 1
        batched_canvas = []
        for i in range(bs):
            cur_mask = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_mask, :]
            cur_feat = pooling_features[cur_mask]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)

            # NOTE:
            # The coors layout depends on Voxelization. This matches your original implementation.
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_feat
            canvas = canvas.permute(2, 1, 0).contiguous()  # (C, y, x)

            batched_canvas.append(canvas)

        bev = torch.stack(batched_canvas, dim=0)  # (B, C, y_l, x_l)
        return bev


class Backbone(nn.Module):
    """
    Simple 2D CNN backbone (multi-scale). Kept because it is commonly useful as the BEV CNN.
    """
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=(2, 2, 2)):
        super().__init__()
        assert len(out_channels) == len(layer_nums) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = [
                nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_nums[i]):
                blocks += [
                    nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1),
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ]
            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        outs = []
        for blk in self.multi_blocks:
            x = blk(x)
            outs.append(x)
        return outs  # list of feature maps


class Neck(nn.Module):
    """
    Lightweight FPN-style neck: upsample multi-scale features then concat.
    """
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for ic, s, oc in zip(in_channels, upsample_strides, out_channels):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ic, oc, s, stride=s, bias=False),
                    nn.BatchNorm2d(oc, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                )
            )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, xs):
        ups = [blk(x) for blk, x in zip(self.decoder_blocks, xs)]
        return torch.cat(ups, dim=1)


class PointPillarsBEV(nn.Module):
    """
    Minimal LiDAR->BEV feature extractor (no detection).
    Returns:
        bev_pseudo: (B, 64, y_l, x_l)
        bev_feat:   (B, C_out, H_out, W_out)  (after backbone+neck)
        bev_vec:    (B, D)                   (global pooled vector)
    """
    def __init__(
        self,
        voxel_size=(0.16, 0.16, 4.0),
        point_cloud_range=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
        max_num_points=32,
        max_voxels=(16000, 40000),
        backbone_out_channels=(64, 128, 256),
        backbone_layer_nums=(3, 5, 5),
        neck_in_channels=(64, 128, 256),
        neck_upsample_strides=(1, 2, 4),
        neck_out_channels=(128, 128, 128),
        vec_dim=256,
    ):
        super().__init__()
        self.pillar_layer = PillarLayer(
            voxel_size=list(voxel_size),
            point_cloud_range=list(point_cloud_range),
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )
        self.pillar_encoder = PillarEncoder(
            voxel_size=list(voxel_size),
            point_cloud_range=list(point_cloud_range),
            in_channel=9,
            out_channel=64,
        )
        self.backbone = Backbone(
            in_channel=64,
            out_channels=list(backbone_out_channels),
            layer_nums=list(backbone_layer_nums),
        )
        self.neck = Neck(
            in_channels=list(neck_in_channels),
            upsample_strides=list(neck_upsample_strides),
            out_channels=list(neck_out_channels),
        )

        # Optional projection to a fixed vector dim for temporal modeling / memory bank key/value.
        self.vec_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Flatten(1),            # (B, C)
            nn.Linear(sum(neck_out_channels), vec_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, batched_pts):
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        bev_pseudo = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)  # (B,64,y,x)
        xs = self.backbone(bev_pseudo)
        bev_feat = self.neck(xs)  # (B, sum(neck_out_channels), H, W)
        bev_vec = self.vec_proj(bev_feat)
        return bev_pseudo, bev_feat, bev_vec
