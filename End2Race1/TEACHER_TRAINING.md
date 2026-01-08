# LiDAR Teacher Training (End2Race)

이 문서는 `LiDARBEVTeacher`(PointPillars BEV + GRU + Control) 학습을 “처음부터” 다시 돌릴 때 필요한 최소 절차를 정리합니다.

## 1) 데이터셋 구조(필수)

`End2Race/scripts/train_teacher.py`는 `VehicleDataset`을 사용하며 아래 구조를 기대합니다.

```
<data_root>/
  train/
    segment_XXXX/
      meta/data.csv
      lidar/<timestamp>.bin
  valid/
    segment_YYYY/
      meta/data.csv
      lidar/<timestamp>.bin
```

`meta/data.csv` 최소 컬럼:
- `lidar_filepath` (예: `lidar/1766908460051370000.bin`)
- `velocity`
- `wheel_angle`

Teacher가 학습하는 제어 타깃(action):
- `action[0] = wheel_angle` (deg로 저장된 값 그대로 사용)
- `action[1] = velocity` (m/s, MORAI에서 target velocity로 사용 가능)

참고:
- LiDAR `.bin` 포맷은 `(N,4)` 또는 `(N,8)`을 지원합니다.
  - `(N,8)`이면 `[x,y,z,intensity]`로 자동 변환해 사용합니다. (`End2Race/pointpillars/dataset/custom_dataset.py`)

## 2) PointPillars CUDA/C++ extension 빌드(필수)

Teacher는 `pointpillars.ops` 확장을 사용합니다. 환경(CUDA/PyTorch)이 맞는 상태에서 아래를 실행합니다.

```
cd End2Race
python3 setup.py build_ext --inplace
```

## 3) Teacher 학습 실행

DDP 학습 스크립트:
`End2Race/scripts/train_teacher.py`

예시:
```
python3 End2Race/scripts/train_teacher.py \
  --data_root "<data_root>" \
  --epochs 50 \
  --batch_size 8 \
  --seq_len 5 \
  --lidar_num_features auto \
  --steer_loss_weight 1.0 \
  --speed_loss_weight 0.05 \
  --smooth_weight 0.0 \
  --checkpoint_path "End2Race/checkpoints/teacher_last.pt" \
  --save_path "End2Race/teacher_model/teacher_model_ddp.pth"
```

중간에 끊었다가 이어서 학습(resume):
```
python3 End2Race/scripts/train_teacher.py \
  --data_root "<data_root>" \
  --resume_path "End2Race/checkpoints/teacher_last.pt" \
  --checkpoint_path "End2Race/checkpoints/teacher_last.pt"
```

## 3.1) Teacher 전이학습(transfer learning) / finetune
resume와 다르게, 전이학습은 **가중치만 로드하고(epoch/optimizer는 새로 시작)** 합니다.

예시(같은 액션 차원/아키텍처):
```
python3 End2Race/scripts/train_teacher.py \
  --data_root "<new_data_root>" \
  --init_path "End2Race/teacher_model/teacher_model_ddp.pth"
```

액션 출력 차원이 달라져서 output head가 안 맞으면:
```
python3 End2Race/scripts/train_teacher.py \
  --data_root "<new_data_root>" \
  --init_path "<weights_or_ckpt>" \
  --init_skip_output_layer
```

## 4) Teacher 오프라인 평가

`End2Race/scripts/eval_teacher.py`:
```
python3 End2Race/scripts/eval_teacher.py \
  --data_root "<data_root>" \
  --split valid \
  --seq_len 5 \
  --teacher_ckpt "End2Race/teacher_model/teacher_model_ddp.pth"
```

E2R 스타일의 가중 overall MSE도 같이 보고 싶으면:
```
--overall_weights 1.0,0.05
```

CSV로 저장:
```
--csv_path End2Race/runs/eval_teacher.csv
```

## 5) KD용 memorybank(teacher cache) 생성(선택)

`End2Race/scripts/export_teacher_cache.py`:
```
python3 End2Race/scripts/export_teacher_cache.py \
  --data_root "<data_root>" \
  --split train \
  --seq_len 5 \
  --teacher_ckpt "End2Race/teacher_model/teacher_model_ddp.pth" \
  --cache_dir "End2Race/checkpoints/teacher_cache"
```
