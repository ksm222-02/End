import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

# 프로젝트 루트 디렉토리를 Python 경로에 추가하여 모듈을 찾을 수 있도록 함
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from teacher.model_teacher import LiDARBEVTeacher
from pointpillars.dataset.custom_dataset import VehicleDataset


def _save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "teacher_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def _load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    # Support loading either a full checkpoint dict or a raw state_dict.
    if isinstance(payload, dict) and "teacher_state_dict" in payload:
        model.load_state_dict(payload["teacher_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload
    if isinstance(payload, dict):
        model.load_state_dict(payload)
        return {"epoch": 0, "best_val_loss": float("inf")}
    raise ValueError(f"Unsupported checkpoint format: {type(payload)}")


def _load_teacher_state_dict(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "teacher_state_dict" in payload:
        return payload["teacher_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported weights format: {type(payload)}")


def _init_from_weights(
    path: str,
    *,
    model: torch.nn.Module,
    strict: bool,
    skip_output_layer: bool,
) -> None:
    state = _load_teacher_state_dict(path)
    if skip_output_layer:
        state = {k: v for k, v in state.items() if not k.startswith("output_layer.")}
        strict = False
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"Init-from-weights: missing={len(missing)} unexpected={len(unexpected)}")

def ddp_setup(rank: int, world_size: int):
    """
    DDP를 위한 프로세스 그룹을 설정합니다.
    Args:
        rank: 현재 프로세스의 순위(ID).
        world_size: 총 프로세스(GPU)의 수.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # 사용할 포트
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def ddp_cleanup():
    """DDP 프로세스 그룹을 정리합니다."""
    dist.destroy_process_group()

def collate_fn(batch):
    """
    데이터 로더가 생성하는 배치를 수동으로 구성합니다.
    """
    lidar_seqs_batch = [item[0] for item in batch]
    speed_tensors_batch = torch.stack([item[1] for item in batch])
    steering_input_tensors_batch = torch.stack([item[2] for item in batch])
    action_tensors_batch = torch.stack([item[3] for item in batch])
    return lidar_seqs_batch, speed_tensors_batch, steering_input_tensors_batch, action_tensors_batch

def _smoothness_loss(pred_actions: torch.Tensor, target_actions: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    if pred_actions.size(1) < 2:
        return torch.zeros((), device=pred_actions.device, dtype=pred_actions.dtype)
    pred_diff = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
    target_diff = target_actions[:, 1:, :] - target_actions[:, :-1, :]
    return criterion(pred_diff, target_diff)


def train_one_epoch(rank, model, criterion, optimizer, train_loader, epoch, writer, args):
    model.train()
    total_loss = 0.0
    # DistributedSampler 사용 시, 에폭마다 수동으로 설정해야 함
    train_loader.sampler.set_epoch(epoch)
    
    num_train_steps = len(train_loader)
    for step, (batched_pts_seq, speed_input, steering_input, target_actions) in enumerate(train_loader):
        # 데이터를 해당 GPU(rank)로 이동
        speed_input = speed_input.to(rank)
        steering_input = steering_input.to(rank)
        target_actions = target_actions.to(rank)
        
        batched_pts_seq_device = [[pts.to(rank) for pts in seq] for seq in batched_pts_seq]
        
        predicted_actions, _, _ = model(batched_pts_seq_device, speed_input, steering_input, hidden=None)

        steer_loss = criterion(predicted_actions[..., 0], target_actions[..., 0])
        speed_loss = criterion(predicted_actions[..., 1], target_actions[..., 1])
        smooth_loss = _smoothness_loss(predicted_actions, target_actions, criterion)
        loss = args.steer_loss_weight * steer_loss + args.speed_loss_weight * speed_loss + args.smooth_weight * smooth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # 메인 프로세스(rank 0)에서만 로그 기록 및 출력
        if rank == 0:
            writer.add_scalar('Loss/train_step', loss.item(), epoch * num_train_steps + step)
            writer.add_scalar('Loss/train_step_steer', steer_loss.item(), epoch * num_train_steps + step)
            writer.add_scalar('Loss/train_step_speed', speed_loss.item(), epoch * num_train_steps + step)
            if args.smooth_weight > 0:
                writer.add_scalar('Loss/train_step_smooth', smooth_loss.item(), epoch * num_train_steps + step)
            if (step + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] Step [{step+1}/{num_train_steps}], Train Loss: {loss.item():.4f}")

    return total_loss / num_train_steps

def validate(rank, model, criterion, val_loader, epoch, writer, args):
    model.eval()
    total_loss = 0.0
    num_val_steps = len(val_loader)
    with torch.no_grad():
        for step, (batched_pts_seq, speed_input, steering_input, target_actions) in enumerate(val_loader):
            speed_input = speed_input.to(rank)
            steering_input = steering_input.to(rank)
            target_actions = target_actions.to(rank)
            
            batched_pts_seq_device = [[pts.to(rank) for pts in seq] for seq in batched_pts_seq]
            
            predicted_actions, _, _ = model(batched_pts_seq_device, speed_input, steering_input, hidden=None)

            steer_loss = criterion(predicted_actions[..., 0], target_actions[..., 0])
            speed_loss = criterion(predicted_actions[..., 1], target_actions[..., 1])
            smooth_loss = _smoothness_loss(predicted_actions, target_actions, criterion)
            loss = args.steer_loss_weight * steer_loss + args.speed_loss_weight * speed_loss + args.smooth_weight * smooth_loss
            total_loss += loss.item()

    avg_val_loss = total_loss / num_val_steps
    
    # 메인 프로세스(rank 0)에서만 로그 기록 및 출력
    if rank == 0:
        writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)
        print(f"\n[Epoch {epoch+1} Summary] Average Validation Loss: {avg_val_loss:.4f}\n")
        
    return avg_val_loss

def main_worker(rank: int, world_size: int, args: argparse.Namespace):

    """

    DDP의 각 프로세스에서 실행될 메인 워커 함수.

    """

    ddp_setup(rank, world_size)



    # --- TensorBoard 로거 설정 (메인 프로세스에서만) ---

    writer = SummaryWriter(args.log_dir) if rank == 0 else None

    if rank == 0:

        print(f"TensorBoard log will be saved in '{args.log_dir}'")



    # --- 검증 로그 파일 설정 (메인 프로세스에서만) ---

    val_log_file = None

    if rank == 0 and args.val_log_path:

        try:

            # Ensure directory exists for val_log_path

            Path(args.val_log_path).parent.mkdir(parents=True, exist_ok=True)

            val_log_file = open(args.val_log_path, 'a', buffering=1) # buffering=1 for line-buffering

            val_log_file.write("Epoch,TrainLoss,ValLoss,BestValLoss\n") # Write header

            val_log_file.flush() # Ensure header is written immediately

            print(f"Validation metrics will be logged to '{args.val_log_path}'")

        except IOError as e:

            print(f"Error opening validation log file {args.val_log_path}: {e}")

            val_log_file = None



    # --- 데이터셋 및 데이터로더 설정 ---

    lidar_num_features = args.lidar_num_features
    if isinstance(lidar_num_features, str) and lidar_num_features.isdigit():
        lidar_num_features = int(lidar_num_features)

    train_dataset = VehicleDataset(
        root_dir=args.data_root,
        split='train',
        sequence_length=args.seq_len,
        lidar_num_features=lidar_num_features,
    )

    val_dataset = VehicleDataset(
        root_dir=args.data_root,
        split='valid',
        sequence_length=args.seq_len,
        lidar_num_features=lidar_num_features,
    )

    

    if rank == 0:

        print(f"Train Dataset loaded: {len(train_dataset)} sequences.")

        print(f"Validation Dataset loaded: {len(val_dataset)} sequences.")



    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)



    train_loader = DataLoader(

        train_dataset, 

        batch_size=args.batch_size, 

        shuffle=False,  # Sampler가 셔플을 담당하므로 False로 설정

        collate_fn=collate_fn,

        num_workers=args.num_workers,

        sampler=train_sampler,

        pin_memory=True

    )

    val_loader = DataLoader(

        val_dataset, 

        batch_size=args.batch_size, 

        shuffle=False, 

        collate_fn=collate_fn,

        num_workers=args.num_workers,

        sampler=val_sampler,

        pin_memory=True

    )

    

    # --- 모델 초기화 및 DDP 래핑 ---

    model = LiDARBEVTeacher(num_actions=args.num_actions).to(rank)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    

    best_val_loss = float('inf')
    start_epoch = 0

    if args.resume_path:
        resume_path = Path(args.resume_path)
        if resume_path.exists():
            ckpt = _load_checkpoint(
                str(resume_path),
                model=model.module,
                optimizer=optimizer,
                map_location="cpu",
            )
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
            if rank == 0:
                print(f"✓ Resumed from {resume_path} (start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f})")
        else:
            if rank == 0:
                print(f"Warning: resume_path not found: {resume_path} (starting from scratch)")
    elif args.init_path:
        init_path = Path(args.init_path)
        if init_path.exists():
            _init_from_weights(
                str(init_path),
                model=model.module,
                strict=args.init_strict,
                skip_output_layer=args.init_skip_output_layer,
            )
            if rank == 0:
                print(f"✓ Initialized from weights: {init_path}")
        else:
            if rank == 0:
                print(f"Warning: init_path not found: {init_path} (starting from scratch)")



    if rank == 0:

        print("Starting DDP training with PointPillars BEV Teacher model...")

        

    # --- 학습 및 검증 루프 ---

    for epoch in range(start_epoch, args.epochs):

        # train_loader.sampler가 DistributedSampler 인스턴스인지 확인

        if hasattr(train_loader.sampler, 'set_epoch'):

            train_loader.sampler.set_epoch(epoch)

            

        avg_train_loss = train_one_epoch(rank, model, criterion, optimizer, train_loader, epoch, writer, args)

        avg_val_loss = validate(rank, model, criterion, val_loader, epoch, writer, args)

        

        # 메인 프로세스(rank 0)에서만 로그 출력 및 모델 저장

        if rank == 0:

            if writer:

                writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

                writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)

            

            print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")



            # 최고의 모델 저장 로직

            if avg_val_loss < best_val_loss:

                best_val_loss = avg_val_loss

                print(f"  ✓ New best validation loss: {best_val_loss:.6f}. Model saved to {args.save_path}")

                # DDP 모델은 .module.state_dict()를 저장

                torch.save(model.module.state_dict(), args.save_path)

            else:

                print(f"  Validation loss did not improve from {best_val_loss:.6f}.")

            

            # 검증 지표를 파일에 기록

            if val_log_file:

                val_log_file.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{best_val_loss:.6f}\n")

                val_log_file.flush() # Ensure data is written immediately

            if args.checkpoint_path:
                _save_checkpoint(
                    args.checkpoint_path,
                    model=model.module,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                )



    if rank == 0:

        print("Training finished.")

        print(f"Best validation loss: {best_val_loss:.6f}")

        print(f"Model with best validation performance saved to {args.save_path}")

        if writer:

            writer.close()

        if val_log_file:

            val_log_file.close() # Close the validation log file

            

    ddp_cleanup()
            
    ddp_cleanup()

def main():
    parser = argparse.ArgumentParser(description="Train Teacher model for KD using PointPillars BEV features and DDP.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size PER GPU.")
    parser.add_argument("--seq_len", type=int, default=5, help="Sequence length for temporal model.")
    parser.add_argument("--num_actions", type=int, default=2, help="Number of output actions (e.g., steer, throttle).")
    parser.add_argument("--steer_loss_weight", type=float, default=1.0, help="Weight for steering MSE.")
    parser.add_argument("--speed_loss_weight", type=float, default=0.05, help="Weight for speed MSE (E2R-style default).")
    parser.add_argument("--smooth_weight", type=float, default=0.0, help="Weight for temporal smoothness loss (diff matching).")
    parser.add_argument("--data_root", type=str, default='/home/jovyan/kdkd/dataset/dataset', help="Root directory of the dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading per GPU.")
    parser.add_argument(
        "--lidar_num_features",
        type=str,
        default="auto",
        help="LiDAR .bin feature count: 4, 8, or 'auto' (auto-detect).",
    )
    parser.add_argument("--log_dir", type=str, default='/home/jovyan/End2Race/runs/teacher_ddp_experiment', help="TensorBoard log directory.")
    parser.add_argument("--save_path", type=str, default='/home/jovyan/End2Race/teacher_model/teacher_model_ddp.pth', help="Path to save the trained model.")
    parser.add_argument("--val_log_path", type=str, default='/home/jovyan/End2Race/teacher_model/val_metrics.csv', help="Path to save validation metrics to a CSV file.")
    parser.add_argument("--checkpoint_path", type=str, default="/home/jovyan/End2Race/checkpoints/teacher_last.pt", help="Path to save last checkpoint for resume.")
    parser.add_argument("--resume_path", type=str, default="", help="Path to resume checkpoint (teacher_last.pt).")
    parser.add_argument("--init_path", type=str, default="", help="Path to weights for transfer learning init (state_dict or checkpoint).")
    parser.add_argument("--init_strict", action="store_true", help="Use strict=True when loading init weights.")
    parser.add_argument("--init_skip_output_layer", action="store_true", help="Drop output_layer.* when initializing weights.")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Spawning DDP processes...")
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size)

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your dataset root contains 'train/' and 'valid/' directories,")
        print("and each segment folder within them has a 'meta/data.csv' file.")
        print("Also, check if the C++/CUDA extensions were compiled successfully.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
