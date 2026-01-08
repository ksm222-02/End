import argparse
import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.camera_dataset import CameraVehicleDataset, collate_camera_samples
from student.model_student import CameraBEVStudent
from utils.kd import distill_action_loss, distill_feature_loss
from utils.metrics import OnlineRegressionMetrics
from utils.teacher_cache import TeacherCache


def _save_checkpoint(path: str, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_score: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "student_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def _load_checkpoint(path: str, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, map_location: str = "cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "student_state_dict" in payload:
        model.load_state_dict(payload["student_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload
    if isinstance(payload, dict):
        model.load_state_dict(payload)
        return {"epoch": 0, "best_score": float("inf")}
    raise ValueError(f"Unsupported checkpoint format: {type(payload)}")


def _load_student_state_dict(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "student_state_dict" in payload:
        return payload["student_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported weights format: {type(payload)}")


def _init_from_weights(path: str, *, model: torch.nn.Module, strict: bool) -> None:
    state = _load_student_state_dict(path)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"Init-from-weights: missing={len(missing)} unexpected={len(unexpected)}")


@torch.no_grad()
def evaluate_bc(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_names: list[str],
    overall_weights: list[float] | None = None,
):
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    metrics = OnlineRegressionMetrics(num_dims=2)
    total_loss = 0.0
    for images, speed, steering_input, gt_actions, _ in loader:
        images = images.to(device)
        speed = speed.to(device)
        steering_input = steering_input.to(device)
        gt_actions = gt_actions.to(device)

        pred_actions, _, _ = model(images, speed, steering_input, hidden=None, return_aux=False)
        loss = criterion(pred_actions, gt_actions)
        total_loss += float(loss.item())
        metrics.update(pred_actions, gt_actions)

    avg_loss = total_loss / max(1, len(loader))
    metric_dict = metrics.finalize(
        prefix="val_",
        names=action_names if len(action_names) == 2 else None,
        overall_weights=overall_weights,
    )
    return avg_loss, metric_dict


def main():
    parser = argparse.ArgumentParser(description="Train Camera Student with optional offline teacher KD.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--val_split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="checkpoints/student_model.pth")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/student_last.pt", help="Path to save last checkpoint for resume.")
    parser.add_argument("--resume_path", type=str, default="", help="Path to resume checkpoint (student_last.pt).")
    parser.add_argument("--init_path", type=str, default="", help="Path to weights for transfer learning init (state_dict or checkpoint).")
    parser.add_argument("--init_strict", action="store_true", help="Use strict=True when loading init weights.")
    parser.add_argument("--csv_log_path", type=str, default="", help="Optional CSV path to append per-epoch metrics.")
    parser.add_argument("--teacher_cache_dir", type=str, default=None)
    parser.add_argument("--kd_action_w", type=float, default=0.0)
    parser.add_argument("--kd_bevvec_w", type=float, default=0.0)
    parser.add_argument("--kd_temporal_w", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action_names", type=str, default="steer,speed", help="Comma-separated action names")
    parser.add_argument(
        "--overall_weights",
        type=str,
        default="",
        help="Optional comma-separated weights for val_overall_weighted_mse (e.g. '1.0,0.05')",
    )
    parser.add_argument("--no_validate", action="store_true", help="Disable per-epoch validation on val_split")
    args = parser.parse_args()

    device = torch.device(args.device)
    teacher_cache = TeacherCache(args.teacher_cache_dir) if args.teacher_cache_dir else None

    action_names = [s.strip() for s in args.action_names.split(",") if s.strip()]
    overall_weights = [float(s) for s in args.overall_weights.split(",") if s.strip()] if args.overall_weights else None

    train_dataset = CameraVehicleDataset(root_dir=args.data_root, split=args.train_split, sequence_length=args.seq_len)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_camera_samples,
        pin_memory=True,
    )

    val_loader = None
    if not args.no_validate:
        try:
            val_dataset = CameraVehicleDataset(root_dir=args.data_root, split=args.val_split, sequence_length=args.seq_len)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_camera_samples,
                pin_memory=True,
                drop_last=False,
            )
        except Exception as e:
            print(f"Warning: validation disabled because val dataset failed to load: {e}")
            val_loader = None

    model = CameraBEVStudent(num_actions=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bc_criterion = nn.MSELoss()
    best_score = float("inf")
    start_epoch = 0

    if args.resume_path:
        resume_path = Path(args.resume_path)
        if resume_path.exists():
            ckpt = _load_checkpoint(str(resume_path), model=model, optimizer=optimizer, map_location="cpu")
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_score = float(ckpt.get("best_score", best_score))
            print(f"✓ Resumed from {resume_path} (start_epoch={start_epoch}, best_score={best_score:.6f})")
        else:
            print(f"Warning: resume_path not found: {resume_path} (starting from scratch)")
    elif args.init_path:
        init_path = Path(args.init_path)
        if init_path.exists():
            _init_from_weights(str(init_path), model=model, strict=args.init_strict)
            print(f"✓ Initialized from weights: {init_path}")
        else:
            print(f"Warning: init_path not found: {init_path} (starting from scratch)")

    csv_path = Path(args.csv_log_path) if args.csv_log_path else None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    expected_val_keys = []
    if len(action_names) == 2:
        for name in action_names:
            expected_val_keys.extend(
                [
                    f"val_{name}_mae",
                    f"val_{name}_mse",
                    f"val_{name}_rmse",
                    f"val_{name}_max_error",
                    f"val_{name}_r2",
                ]
            )
    expected_val_keys.extend(["val_mse_mean", "val_mae_mean"])
    if overall_weights is not None:
        expected_val_keys.append("val_overall_weighted_mse")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        total = 0.0
        for images, speed, steering_input, gt_actions, keys in loader:
            images = images.to(device)
            speed = speed.to(device)
            steering_input = steering_input.to(device)
            gt_actions = gt_actions.to(device)

            pred_actions, _, aux_s = model(images, speed, steering_input, hidden=None, return_aux=True)
            loss_bc = bc_criterion(pred_actions, gt_actions)

            loss_kd = torch.zeros_like(loss_bc)
            if teacher_cache is not None and (args.kd_action_w > 0 or args.kd_bevvec_w > 0 or args.kd_temporal_w > 0):
                # load teacher targets per sample; skip missing entries
                teacher_actions = []
                teacher_bev_vec = []
                teacher_gru_out = []
                keep = []
                for i, key in enumerate(keys):
                    if not teacher_cache.has(key):
                        continue
                    t = teacher_cache.load(key, device=device)
                    teacher_actions.append(t.actions)
                    teacher_bev_vec.append(t.bev_vec if t.bev_vec is not None else None)
                    teacher_gru_out.append(t.gru_out if t.gru_out is not None else None)
                    keep.append(i)

                if keep:
                    pred_keep = pred_actions[keep]
                    loss_kd_actions = torch.zeros_like(loss_bc)
                    loss_kd_bevvec = torch.zeros_like(loss_bc)
                    loss_kd_temporal = torch.zeros_like(loss_bc)

                    if args.kd_action_w > 0:
                        teacher_actions_t = torch.stack(teacher_actions, dim=0)  # (Bkeep,T,A)
                        loss_kd_actions = distill_action_loss(pred_keep, teacher_actions_t)

                    if args.kd_bevvec_w > 0 and all(v is not None for v in teacher_bev_vec):
                        teacher_bevvec_t = torch.stack(teacher_bev_vec, dim=0)  # (Bkeep,T,D)
                        loss_kd_bevvec = distill_feature_loss(aux_s["bev_vec_seq"][keep], teacher_bevvec_t, normalize=True)

                    if args.kd_temporal_w > 0 and all(v is not None for v in teacher_gru_out):
                        teacher_gru_t = torch.stack(teacher_gru_out, dim=0)  # (Bkeep,T,H)
                        loss_kd_temporal = distill_feature_loss(aux_s["gru_out"][keep], teacher_gru_t, normalize=False)

                    loss_kd = (
                        args.kd_action_w * loss_kd_actions
                        + args.kd_bevvec_w * loss_kd_bevvec
                        + args.kd_temporal_w * loss_kd_temporal
                    )

            loss = loss_bc + loss_kd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())

        train_avg = total / max(1, len(loader))
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss={train_avg:.6f}")

        val_avg = None
        val_metrics = {}
        if val_loader is not None:
            val_avg, val_metrics = evaluate_bc(
                model,
                val_loader,
                device=device,
                action_names=action_names,
                overall_weights=overall_weights,
            )
            print(f"  val_loss={val_avg:.6f}")
            # keep output compact but informative
            for k in sorted(val_metrics.keys()):
                if k.endswith("_mae") or k.endswith("_rmse") or k.endswith("_r2"):
                    print(f"  {k}={val_metrics[k]}")
            if "val_overall_weighted_mse" in val_metrics:
                print(f"  val_overall_weighted_mse={val_metrics['val_overall_weighted_mse']}")

        score = (
            val_metrics.get("val_overall_weighted_mse")
            if val_loader is not None and overall_weights is not None and "val_overall_weighted_mse" in val_metrics
            else (val_avg if val_avg is not None else train_avg)
        )
        if score < best_score:
            best_score = score
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✓ Saved best student checkpoint to {args.save_path} (score={best_score:.6f})")

        if args.checkpoint_path:
            _save_checkpoint(
                args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
            )

        if csv_path is not None:
            row = {
                "epoch": epoch + 1,
                "train_split": args.train_split,
                "val_split": args.val_split if val_loader is not None else "",
                "train_loss": train_avg,
                "val_loss": val_avg if val_avg is not None else "",
                "best_score": best_score,
            }
            for k in expected_val_keys:
                row[k] = val_metrics.get(k, "")

            fieldnames = list(row.keys())
            write_header = not csv_path.exists()
            with csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)


if __name__ == "__main__":
    main()
