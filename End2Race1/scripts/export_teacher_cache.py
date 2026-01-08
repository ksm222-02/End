import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from teacher.model_teacher import LiDARBEVTeacher
from pointpillars.dataset.custom_dataset import VehicleDataset
from utils.teacher_cache import TeacherCacheWriter


def collate_with_key(batch):
    lidar_seqs_batch = [item[0] for item in batch]
    speed_tensors_batch = torch.stack([item[1] for item in batch])
    steering_input_tensors_batch = torch.stack([item[2] for item in batch])
    action_tensors_batch = torch.stack([item[3] for item in batch])
    keys = [item[4] for item in batch]
    return lidar_seqs_batch, speed_tensors_batch, steering_input_tensors_batch, action_tensors_batch, keys


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Export teacher aux/features to an offline cache for student KD.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_bev_feat", action="store_true", help="Also save full BEV feature map (large).")
    parser.add_argument(
        "--lidar_num_features",
        type=str,
        default="auto",
        help="LiDAR .bin feature count: 4, 8, or 'auto' (auto-detect).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    writer = TeacherCacheWriter(Path(args.cache_dir))

    lidar_num_features = args.lidar_num_features
    if isinstance(lidar_num_features, str) and lidar_num_features.isdigit():
        lidar_num_features = int(lidar_num_features)

    dataset = VehicleDataset(
        root_dir=args.data_root,
        split=args.split,
        sequence_length=args.seq_len,
        return_key=True,
        lidar_num_features=lidar_num_features,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_key,
        pin_memory=True,
    )

    model = LiDARBEVTeacher(num_actions=2).to(device)
    state = torch.load(args.teacher_ckpt, map_location="cpu")
    if isinstance(state, dict) and "teacher_state_dict" in state:
        state = state["teacher_state_dict"]
    model.load_state_dict(state)
    model.eval()

    for batched_pts_seq, speed_input, steering_input, _, keys in loader:
        speed_input = speed_input.to(device)
        steering_input = steering_input.to(device)
        batched_pts_seq_device = [[pts.to(device) for pts in seq] for seq in batched_pts_seq]

        actions, _, aux = model(
            batched_pts_seq_device,
            speed_input,
            steering_input,
            hidden=None,
            return_aux=True,
        )

        # per-sequence save (assumes batch_size small; keys are per sample)
        for i, key in enumerate(keys):
            actions_i = actions[i].detach().cpu()
            gru_out_i = aux["gru_out"][i].detach().cpu()
            bev_vec_i = aux["bev_vec_seq"][i].detach().cpu()
            bev_feat_i = aux["bev_feat_seq"][i].detach().cpu() if args.save_bev_feat else None
            writer.write(key, actions=actions_i, gru_out=gru_out_i, bev_vec=bev_vec_i, bev_feat=bev_feat_i)


if __name__ == "__main__":
    main()
