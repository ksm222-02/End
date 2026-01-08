import argparse
import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pointpillars.dataset.custom_dataset import VehicleDataset
from teacher.model_teacher import LiDARBEVTeacher
from utils.metrics import OnlineRegressionMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Offline evaluation for LiDAR teacher.")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root containing train/valid/test")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to teacher .pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action_names", type=str, default="steer,speed", help="Comma-separated action names")
    parser.add_argument(
        "--overall_weights",
        type=str,
        default="",
        help="Optional comma-separated weights to compute overall_weighted_mse (e.g. '1.0,0.05')",
    )
    parser.add_argument("--csv_path", type=str, default="", help="Optional path to write metrics as a 1-row CSV.")
    parser.add_argument(
        "--lidar_num_features",
        type=str,
        default="auto",
        help="LiDAR .bin feature count: 4, 8, or 'auto' (auto-detect).",
    )
    return parser.parse_args()


def collate_fn(batch):
    lidar_seqs_batch = [item[0] for item in batch]
    speed_tensors_batch = torch.stack([item[1] for item in batch])
    steering_input_tensors_batch = torch.stack([item[2] for item in batch])
    action_tensors_batch = torch.stack([item[3] for item in batch])
    return lidar_seqs_batch, speed_tensors_batch, steering_input_tensors_batch, action_tensors_batch


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    lidar_num_features = args.lidar_num_features
    if isinstance(lidar_num_features, str) and lidar_num_features.isdigit():
        lidar_num_features = int(lidar_num_features)

    dataset = VehicleDataset(
        root_dir=args.data_root,
        split=args.split,
        sequence_length=args.seq_len,
        lidar_num_features=lidar_num_features,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    model = LiDARBEVTeacher(num_actions=2).to(device)
    state = torch.load(args.teacher_ckpt, map_location="cpu")
    if isinstance(state, dict) and "teacher_state_dict" in state:
        state = state["teacher_state_dict"]
    model.load_state_dict(state)
    model.eval()

    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0

    action_names = [s.strip() for s in args.action_names.split(",") if s.strip()]
    overall_weights = [float(s) for s in args.overall_weights.split(",") if s.strip()] if args.overall_weights else None
    metrics = OnlineRegressionMetrics(num_dims=2)

    for batched_pts_seq, speed_input, steering_input, targets in tqdm(loader, desc=f"Eval({args.split})"):
        speed_input = speed_input.to(device)
        steering_input = steering_input.to(device)
        targets = targets.to(device)
        batched_pts_seq_device = [[pts.to(device) for pts in seq] for seq in batched_pts_seq]

        preds, _, _ = model(batched_pts_seq_device, speed_input, steering_input, hidden=None, return_aux=False)
        loss = criterion(preds, targets)
        total_loss += float(loss.item())

        metrics.update(preds, targets)

    avg_loss = total_loss / max(1, len(loader))
    out = metrics.finalize(
        names=action_names if len(action_names) == 2 else None,
        overall_weights=overall_weights,
    )

    print(f"\nTeacher offline evaluation ({args.split})")
    print(f"- avg_mse_loss: {avg_loss:.6f}")
    for k in sorted(out.keys()):
        print(f"- {k}: {out[k]}")

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "split": args.split,
            "teacher_ckpt": str(args.teacher_ckpt),
            "avg_mse_loss": avg_loss,
            **out,
        }
        fieldnames = list(row.keys())
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nSaved CSV metrics to {csv_path}")


if __name__ == "__main__":
    main()
