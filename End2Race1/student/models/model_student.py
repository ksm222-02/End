import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Dict


class BEVConfig:
    POINT_CLOUD_RANGE = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
    VOXEL_SIZE = [0.16, 0.16, 4.0]
    
    BEV_WIDTH = int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0])
    BEV_HEIGHT = int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1])

    TEACHER_BEV_CHANNELS = 384
    TEACHER_GRU_HIDDEN = 640

class LSSViewTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depth_channels = 64 
        self.out_channels = out_channels
        
        self.depth_net = nn.Conv2d(in_channels, self.depth_channels + out_channels, kernel_size=1)
        
    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        x = self.depth_net(feat_2d)
        
        depth = x[:, :self.depth_channels].softmax(dim=1)
        context = x[:, self.depth_channels:]
        
        bev = context.unsqueeze(2) * depth.unsqueeze(1) 
        bev = bev.mean(dim=2)
        
        bev = F.interpolate(bev, size=(BEVConfig.BEV_WIDTH, BEVConfig.BEV_HEIGHT), 
                            mode='bilinear', align_corners=False)
        return bev

class CameraBEVStudent(nn.Module):
    def __init__(self, num_actions: int = 2, bev_channels: int = 128):
        super().__init__()
        
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.view_transform = LSSViewTransform(in_channels=512, out_channels=bev_channels)
        
        self.bev_adapter = nn.Conv2d(bev_channels, BEVConfig.TEACHER_BEV_CHANNELS, kernel_size=1)
        self.hidden_adapter = nn.Linear(256, BEVConfig.TEACHER_GRU_HIDDEN)
        
        self.bev_pool = nn.AdaptiveAvgPool2d(1)
        self.speed_mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.steering_mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        
        self.gru = nn.GRU(input_size=bev_channels + 32 + 32, hidden_size=256, batch_first=True)
        
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, images, speed_input, steering_input, hidden=None, return_aux=False):
        B, T, C, H, W = images.shape
        images_flat = images.reshape(B * T, C, H, W)
        
        feat_2d = self.backbone(images_flat)
        bev_feat = self.view_transform(feat_2d)
        
        kd_bev_feat = self.bev_adapter(bev_feat).reshape(B, T, BEVConfig.TEACHER_BEV_CHANNELS, 
                                                        BEVConfig.BEV_WIDTH, BEVConfig.BEV_HEIGHT)
        
        bev_vec = self.bev_pool(bev_feat).flatten(1)
        speed_emb = self.speed_mlp(speed_input.reshape(B * T, 1))
        steer_emb = self.steering_mlp(steering_input.reshape(B * T, 1))
        
        gru_in = torch.cat([bev_vec, speed_emb, steer_emb], dim=-1).reshape(B, T, -1)
        gru_out, last_hidden = self.gru(gru_in, hidden)
        
        kd_gru_out = self.hidden_adapter(gru_out)
        
        actions = self.output_layer(gru_out)
        
        if return_aux:
            return actions, last_hidden, {
                "bev_feat_seq": kd_bev_feat,
                "gru_out": kd_gru_out
            }
        return actions, last_hidden