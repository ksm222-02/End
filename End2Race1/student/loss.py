import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3):
        super().__init__()
        self.alpha = alpha  # BEV Feature KD 가중치
        self.beta = beta    # Temporal Logic KD 가중치
        self.mse = nn.MSELoss()

    def forward(self, student_output, teacher_output, gt_actions):
        # student_output: (actions, bev_feat, gru_out)
        # teacher_output: (actions, bev_feat, gru_out)
        s_act, s_bev, s_gru = student_output
        t_act, t_bev, t_gru = teacher_output

        # 1. Task Loss: 실제 운전 데이터(GT)와의 오차
        loss_task = self.mse(s_act, gt_actions)

        # 2. LabelDistill (BEV Feature KD): 
        # Teacher가 LiDAR로 파악한 공간 정보를 카메라가 따라하도록 함
        loss_bev = self.mse(s_bev, t_bev)

        # 3. MemDistill (Temporal KD):
        # Teacher의 과거 데이터 축적 방식(GRU 로직)을 모방
        loss_mem = self.mse(s_gru, t_gru)

        total_loss = loss_task + (self.alpha * loss_bev) + (self.beta * loss_mem)
        
        return total_loss, {
            "task": loss_task.item(),
            "bev": loss_bev.item(),
            "mem": loss_mem.item()
        }