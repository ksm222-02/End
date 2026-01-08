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

from datasets.camera_dataset import CameraVehicleDataset, collate_camera_samples
from student.model_student import CameraBEVStudent
from utils.metrics import OnlineRegressionMetrics
from utils.teacher_cache import TeacherCache


def parse_args():
    parser = argparse.ArgumentParser(description="Offline evaluation for Camera student.")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root containing train/valid/test")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--student_ckpt", type=str, required=True, help="Path to student .pth (state_dict)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action_names", type=str, default="steer,speed", help="Comma-separated action names")
    parser.add_argument("--teacher_cache_dir", type=str, default=None, help="Optional teacher cache dir for KD metrics")
    parser.add_argument(
        "--overall_weights",
        type=str,
        default="",
        help="Optional comma-separated weights to compute overall_weighted_mse (e.g. '1.0,0.05')",
    )
    parser.add_argument("--csv_path", type=str, default="", help="Optional path to write metrics as a 1-row CSV.")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    teacher_cache = TeacherCache(args.teacher_cache_dir) if args.teacher_cache_dir else None

    dataset = CameraVehicleDataset(root_dir=args.data_root, split=args.split, sequence_length=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_camera_samples,
        pin_memory=True,
        drop_last=False,
    )

    model = CameraBEVStudent(num_actions=2).to(device)
    state = torch.load(args.student_ckpt, map_location="cpu")
    if isinstance(state, dict) and "student_state_dict" in state:
        state = state["student_state_dict"]
    model.load_state_dict(state)
    model.eval()

    action_names = [s.strip() for s in args.action_names.split(",") if s.strip()]
    overall_weights = [float(s) for s in args.overall_weights.split(",") if s.strip()] if args.overall_weights else None
    metrics_bc = OnlineRegressionMetrics(num_dims=2)
    metrics_kd = OnlineRegressionMetrics(num_dims=2) if teacher_cache is not None else None
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0

    for images, speed, steering_input, gt_actions, keys in tqdm(loader, desc=f"Eval({args.split})"):
        images = images.to(device)
        speed = speed.to(device)
        steering_input = steering_input.to(device)
        gt_actions = gt_actions.to(device)

        pred_actions, _, _ = model(images, speed, steering_input, hidden=None, return_aux=False)
        loss = criterion(pred_actions, gt_actions)
        total_loss += float(loss.item())
        metrics_bc.update(pred_actions, gt_actions)

        if metrics_kd is not None:
            teacher_actions = []
            keep = []
            for i, key in enumerate(keys):
                if teacher_cache is None or not teacher_cache.has(key):
                    continue
                t = teacher_cache.load(key, device=device)
                teacher_actions.append(t.actions)
                keep.append(i)
            if keep:
                teacher_actions_t = torch.stack(teacher_actions, dim=0)
                metrics_kd.update(pred_actions[keep], teacher_actions_t)

    avg_loss = total_loss / max(1, len(loader))
    out_bc = metrics_bc.finalize(
        names=action_names if len(action_names) == 2 else None,
        prefix="bc_",
        overall_weights=overall_weights,
    )

    print(f"\nStudent offline evaluation ({args.split})")
    print(f"- avg_mse_loss: {avg_loss:.6f}")
    for k in sorted(out_bc.keys()):
        print(f"- {k}: {out_bc[k]}")

    out_kd = None
    if metrics_kd is not None:
        out_kd = metrics_kd.finalize(
            names=action_names if len(action_names) == 2 else None,
            prefix="kd_",
            overall_weights=overall_weights,
        )
        print("\nKD metrics (student vs teacher cache)")
        for k in sorted(out_kd.keys()):
            print(f"- {k}: {out_kd[k]}")

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "split": args.split,
            "student_ckpt": str(args.student_ckpt),
            "avg_mse_loss": avg_loss,
            **out_bc,
        }
        if out_kd is not None:
            row.update(out_kd)

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
