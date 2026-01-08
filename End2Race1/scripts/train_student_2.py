import torch
from student.models.model_student_v2 import CameraBEVStudent
from teacher.model_teacher import LiDARBEVTeacher
from student.loss.kd_loss import KDLoss

def train_one_epoch(student, teacher, dataloader, optimizer, criterion, device):
    student.train()
    teacher.eval() # Teacher는 고정 (중요!)
    
    for batch in dataloader:
        # 데이터 로드 (이미지, LiDAR, 속도, GT)
        imgs = batch['images'].to(device)       # (B, T, 3, H, W)
        lidar = batch['lidar']                  # Teacher용 LiDAR
        speed = batch['speed'].to(device)
        steer = batch['steering'].to(device)
        gt_actions = batch['actions'].to(device)

        # 1. Teacher 추론 (Gradient 계산 안함)
        with torch.no_grad():
            t_actions, _, t_aux = teacher(lidar, speed, steer, return_aux=True)
            # t_aux 내부의 'bev_feat_seq', 'gru_out'을 사용

        # 2. Student 추론
        s_actions, _, s_aux = student(imgs, speed, steer, return_aux=True)

        # 3. Loss 계산 (Distillation)
        loss, loss_dict = criterion(
            (s_actions, s_aux['bev_feat_seq'], s_aux['gru_out']),
            (t_actions, t_aux['bev_feat_seq'], t_aux['gru_out']),
            gt_actions
        )

        # 4. 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()