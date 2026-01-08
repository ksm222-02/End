import os
import torch
from camera_dataset import CameraVehicleDataset 

class DistillDataset(CameraVehicleDataset):
    def __init__(self, cache_dir, *args, **kwargs):
        """
        cache_dir: export_teacher_cache.py 임시 경로
        """
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir

    def __getitem__(self, index):
       
        sample = super().__getitem__(index)
        
        cache_filename = sample.key.replace("/", "_").replace(":", "_") + ".pt"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        if os.path.exists(cache_path):
            teacher_data = torch.load(cache_path, map_location="cpu")
        else:
            # Dummy (학습 초기 테스트용)
            teacher_data = {
                "bev_feat_seq": torch.zeros((self.sequence_length, 384, 108, 124)),
                "gru_out": torch.zeros((self.sequence_length, 640)),
                "actions": torch.zeros((self.sequence_length, 2))
            }
            
        return {
            "student_input": {
                "images": sample.images,         
                "speed": sample.speed,           
                "steer_in": sample.steering_input 
            },
            "teacher_labels": teacher_data,      
            "gt_actions": sample.gt_actions,     
            "key": sample.key
        }