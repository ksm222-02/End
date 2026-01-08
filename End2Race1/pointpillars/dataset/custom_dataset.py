import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Union

class VehicleDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        sequence_length: int = 5,
        return_key: bool = False,
        lidar_num_features: Union[int, str] = "auto",
    ):
        """
        Args:
            root_dir (str): 데이터셋의 최상위 경로 (예: '/media/sws/X9 Pro/dataset/').
            split (str): 'train', 'valid', 'test' 중 하나.
            sequence_length (int): 모델에 입력으로 사용할 시퀀스의 길이.
            return_key (bool): 시퀀스 고유 키(split/segment:start_index)를 함께 반환할지 여부.
            lidar_num_features (int | "auto"): LiDAR .bin 포맷의 feature 수.
              - 4: (x,y,z,intensity)
              - 8: (x,y,z,?,intensity,ring,ts0,ts1) -> (x,y,z,intensity)로 변환
              - "auto": 4/8 중 heuristics로 추정
        """
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.return_key = bool(return_key)
        self.lidar_num_features = lidar_num_features
        self.all_sequences_info = []
        self.segment_dfs = {} # 데이터프레임 캐싱을 위한 딕셔너리

        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Split directory not found: {split_path}")

        segment_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d)) and d.startswith('segment_')]
        if not segment_dirs:
            print(f"No segment directories found in {split_path}. Please check data organization.")

        for segment_name in sorted(segment_dirs):
            segment_path = os.path.join(split_path, segment_name)
            meta_path = os.path.join(segment_path, 'meta', 'data.csv')

            if not os.path.exists(meta_path):
                print(f"Warning: data.csv not found for segment {segment_name}. Skipping.")
                continue

            # 성능 향상을 위해 데이터프레임을 읽어 캐싱합니다.
            segment_df = pd.read_csv(meta_path)
            self.segment_dfs[segment_path] = segment_df
            
            # 유효한 시퀀스의 시작 인덱스를 계산합니다.
            # 시작 인덱스는 0부터 (전체 길이 - 시퀀스 길이) 까지 가능합니다.
            for i in range(len(segment_df) - self.sequence_length + 1):
                self.all_sequences_info.append((segment_path, i))

        if not self.all_sequences_info:
            raise RuntimeError(f"No valid sequences found for split '{split}' in '{root_dir}'.")

    def __len__(self) -> int:
        return len(self.all_sequences_info)

    def _load_lidar_points(self, lidar_filepath: str) -> torch.Tensor:
        raw = np.fromfile(lidar_filepath, dtype=np.float32)
        if raw.size == 0:
            raise ValueError(f"Empty lidar bin: {lidar_filepath}")
        if raw.size % 4 != 0:
            raise ValueError(f"Unexpected lidar bin float count: {raw.size} ({lidar_filepath})")

        fmt = self.lidar_num_features
        if isinstance(fmt, str) and fmt != "auto":
            raise ValueError("lidar_num_features must be 4, 8, or 'auto'")

        if fmt == 4:
            pts4 = raw.reshape(-1, 4)
            return torch.from_numpy(pts4)

        if fmt == 8:
            if raw.size % 8 != 0:
                raise ValueError(f"lidar_num_features=8 but float count not divisible by 8: {raw.size} ({lidar_filepath})")
            pts8 = raw.reshape(-1, 8)
            pts4 = pts8[:, [0, 1, 2, 4]]
            return torch.from_numpy(pts4)

        # auto-detect (4 vs 8)
        if raw.size % 8 == 0:
            pts8 = raw.reshape(-1, 8)
            pad = pts8[:, 3]
            intensity = pts8[:, 4]
            xyz = pts8[:, :3]

            # Heuristic: in many (N,8) dumps, column 3 is a near-constant pad (often ~0),
            # and intensity is finite and in a reasonable range.
            pad_std = float(np.std(pad))
            pad_mean_abs = float(np.mean(np.abs(pad)))
            intensity_finite = np.isfinite(intensity).mean()
            xyz_finite = np.isfinite(xyz).mean()
            intensity_in_range = float(np.mean((intensity >= 0.0) & (intensity <= 255.0)))

            if (
                xyz_finite > 0.999
                and intensity_finite > 0.999
                and pad_std < 1e-3
                and pad_mean_abs < 1e-2
                and intensity_in_range > 0.80
            ):
                pts4 = pts8[:, [0, 1, 2, 4]]
                return torch.from_numpy(pts4)

        # fallback to (N,4)
        pts4 = raw.reshape(-1, 4)
        return torch.from_numpy(pts4)

    def __getitem__(self, idx: int):
        """
        하나의 데이터 시퀀스를 불러옵니다.
        
        Returns:
            Tuple:
                return_key=False:
                  (LiDAR 시퀀스, 속도 시퀀스, 조향각 입력 시퀀스, 행동(정답) 시퀀스)
                return_key=True:
                  (LiDAR 시퀀스, 속도 시퀀스, 조향각 입력 시퀀스, 행동(정답) 시퀀스, key)
        """
        segment_path, start_index = self.all_sequences_info[idx]
        
        segment_df = self.segment_dfs[segment_path]

        # LiDAR, 속도, 정답 액션을 위한 메인 시퀀스
        sequence_df = segment_df.iloc[start_index : start_index + self.sequence_length]

        # --- steering_input 시퀀스 생성 (t 시점의 입력은 t-1 시점의 조향각) ---
        steering_input_seq = []
        if start_index == 0:
            # 세그먼트의 가장 첫 프레임인 경우, 이전 조향각이 없으므로 0을 사용합니다.
            steering_input_seq.append(0.0)
        else:
            # 이전 시간 스텝의 wheel_angle 값을 가져옵니다.
            prev_wheel_angle = segment_df.iloc[start_index - 1]['wheel_angle']
            steering_input_seq.append(prev_wheel_angle)
        
        # 나머지 steering_input 값들을 채웁니다.
        # (메인 시퀀스의 마지막 값을 제외한 모든 wheel_angle 값)
        steering_input_seq.extend(sequence_df['wheel_angle'].iloc[:-1].tolist())

        # --- 메인 시퀀스 데이터 처리 ---
        lidar_seq = []
        speed_seq = []
        action_seq = []
        
        for _, row in sequence_df.iterrows():
            lidar_filepath = os.path.join(segment_path, row['lidar_filepath'])
            lidar_seq.append(self._load_lidar_points(lidar_filepath))
            
            speed_seq.append(row['velocity'])
            
            # 정답 액션: [wheel_angle(deg), target_speed(m/s)]
            action_seq.append([float(row['wheel_angle']), float(row['velocity'])]) 

        # 텐서로 변환
        speed_tensor = torch.tensor(speed_seq, dtype=torch.float32).view(-1, 1)
        steering_input_tensor = torch.tensor(steering_input_seq, dtype=torch.float32).view(-1, 1)
        action_tensor = torch.tensor(action_seq, dtype=torch.float32)

        if not self.return_key:
            return lidar_seq, speed_tensor, steering_input_tensor, action_tensor

        segment_name = os.path.basename(segment_path.rstrip(os.sep))
        key = f"{self.split}/{segment_name}:{start_index}"
        return lidar_seq, speed_tensor, steering_input_tensor, action_tensor, key
