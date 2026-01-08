# End2Race KD/MemDistill Handoff (LiDAR→Camera Control)

## Goal
- 목적: LiDAR 기반 teacher로 학습된 “제어(policy)”를 카메라 기반 student로 **지식증류(KD)** 해서 카메라 입력 제어 성능을 확보/가속.
- 현재 상태: LiDAR teacher 학습 완료(체크포인트 존재), 카메라 데이터셋은 아직 수집 전(구현 골격 먼저 진행).

## Model Designs (대화 기준)

### 1) Teacher (LiDAR)
- 입력: `LiDAR(PointCloud seq)` + `ego status (speed seq)` + `steering_input seq(이전 wheel_angle)`
- 출력: `Control (actions)` + (KD용) `BEV intermediate feature` + `temporal state`
- 구현 위치:
  - `End2Race/teacher/model_teacher.py`
  - `LiDARBEVTeacher(..., return_aux=True)`에서 KD에 쓸 중간값 제공:
    - `aux["bev_vec_seq"]` : (B,T,D)
    - `aux["bev_feat_seq"]` : (B,T,C,H,W) (용량 큼)
    - `aux["gru_out"]` : (B,T,H)
    - `actions` : (B,T,A)

### 2) Student (Camera)
- 입력: `Camera(image seq)` + `ego status (speed seq)` + `steering_input seq`
- 출력: `Control (actions)`
- 목표 KD 포인트:
  - student BEV-aligned feature ↔ teacher BEV feature/embedding
  - student temporal state(GRU out) ↔ teacher temporal state(GRU out)
  - student actions ↔ teacher actions(soft target)
- 현재 구현은 “골격 + placeholder view transform”:
  - `End2Race/student/model_student.py` (`CameraBEVStudent`)
  - `IdentityViewTransform`는 나중에 Lift-Splat / view transform으로 교체 예정

## MemDistill “Memorybank” 결정 요약

### 핵심 결론
- “teacher 모델(가중치)”를 memorybank에 넣는 게 아니라,
  - **teacher가 특정 시퀀스에서 뽑아낸 KD 타깃(출력/특징/temporal)** 을 저장하는 게 일반적인 memorybank.
- 구현/운영 효율 기준:
  - **(1) 같은 샘플(동일 시점) 페어 KD**가 가장 쉽고 안정적
  - **(2) retrieval 기반 memorybank(유사상황 top-k 검색)** 는 공수가 커서, 데이터 정렬이 깨지거나 페어가 약할 때 “2차 단계”로 추천

### memorybank에 저장하는 정보(권장)
- 필수(가볍고 효과 큼):
  - `teacher actions` (T,A)
  - `teacher bev_vec_seq` (T,D)  (feature map 대신 embedding)
  - `teacher gru_out` (T,H)      (temporal distill)
- 선택(무겁지만 feature-level KD용):
  - `teacher bev_feat_seq` (T,C,H,W)  (`--save_bev_feat` 옵션으로만 저장)

## Implemented: Offline Teacher Cache (Memorybank v1)

### 저장/로드 유틸
- `End2Race/utils/teacher_cache.py`
  - key 기반 `.npz` 캐시 저장/로드:
    - key 예: `train/segment_0001:123`
    - 파일명은 안전하게 변환되어 저장됨
  - 저장 항목: `actions`, `gru_out`, `bev_vec`, (옵션) `bev_feat`

### LiDAR dataset에서 key 생성
- `End2Race/pointpillars/dataset/custom_dataset.py`
  - `VehicleDataset(..., return_key=True)` 추가
  - 기존 teacher 학습에는 영향 없고, 캐시 생성용으로 key를 함께 반환 가능

### memorybank 생성 스크립트 (학습 아님)
- `End2Race/scripts/export_teacher_cache.py`
  - teacher 체크포인트를 로드해서 **추론(inference)** 으로 전체 시퀀스를 훑고 캐시 저장
  - `@torch.no_grad()` + `model.eval()` (optimizer/backprop 없음)

#### 실행 예시
```bash
python End2Race/scripts/export_teacher_cache.py \
  --data_root <lidar_dataset_root> \
  --split train \
  --seq_len 5 \
  --teacher_ckpt End2Race/teacher_model/teacher_model_ddp.pth \
  --cache_dir End2Race/checkpoints/teacher_cache
```

옵션:
- `--save_bev_feat` : (매우 큼) feature map까지 저장

### Student 학습 스켈레톤 (카메라 데이터 준비 전 단계)
- `End2Race/datasets/camera_dataset.py`
  - 카메라 시퀀스 데이터셋 골격
  - 예상 CSV 컬럼(최소): `velocity`, `wheel_angle`, `image_filepath`
  - 이미지 로드는 Pillow 필요(없으면 에러 메시지로 안내)
- `End2Race/scripts/train_student.py`
  - BC loss + (옵션) KD loss
  - `--teacher_cache_dir`가 주어지고 cache key가 존재하면 KD 적용
  - KD weight:
    - `--kd_action_w`
    - `--kd_bevvec_w`
    - `--kd_temporal_w`

## Important Notes / Assumptions
- memorybank(캐시) 생성에는 **LiDAR 데이터셋이 필요**:
  - teacher의 KD 타깃은 teacher에 입력을 넣어야 생성 가능하므로,
  - `export_teacher_cache.py`는 `VehicleDataset`을 통해 LiDAR 시퀀스를 읽는다.
- 카메라 데이터셋을 나중에 붙일 때:
  - 카메라 샘플이 LiDAR와 1:1로 매칭되도록 같은 기준의 key(`split/segment:start_index` 등)를 유지하는 것이 가장 단순함.
- `bev_feat_seq`는 저장 용량이 커서 기본은 비활성 권장(필요 시만 켜기).

## Next Steps (추천)
1) 카메라 데이터셋 포맷 확정:
   - 단일/멀티캠, 컬럼명, intrinsics/extrinsics 존재 여부
2) `IdentityViewTransform` → 실제 Lift-Splat/View Transform 구현으로 교체:
   - teacher BEV 좌표/해상도와 student BEV-aligned feature를 맞추고 KD 안정화
3) 필요 시 retrieval 기반 memorybank(MemDistill v2):
   - key/value 설계 + top-k 검색 + confidence weighting (페어가 약할 때 도입)

## Offline Evaluation (추가)
- Teacher 평가 스크립트: `End2Race/scripts/eval_teacher.py`
  - 실행 예:
    - `python End2Race/scripts/eval_teacher.py --data_root <lidar_dataset_root> --split valid --seq_len 5 --teacher_ckpt End2Race/teacher_model/teacher_model_ddp.pth`
  - E2R 스타일 가중 overall(MSE)도 출력하려면:
    - `--overall_weights 1.0,0.05`
  - CSV로 저장하려면:
    - `--csv_path End2Race/runs/eval_teacher.csv`
- Student 평가 스크립트: `End2Race/scripts/eval_student.py`
  - 카메라 데이터셋 준비 후 실행:
    - `python End2Race/scripts/eval_student.py --data_root <camera_dataset_root> --split valid --seq_len 5 --student_ckpt <student.pth>`
  - teacher cache가 있으면 KD 기준 오차도 같이 출력:
    - `--teacher_cache_dir End2Race/checkpoints/teacher_cache`
  - E2R 스타일 가중 overall(MSE)도 출력하려면:
    - `--overall_weights 1.0,0.05`
  - CSV로 저장하려면:
    - `--csv_path End2Race/runs/eval_student.csv`

## Resume Training (추가)
- Teacher 학습 재개:
  - `End2Race/scripts/train_teacher.py --resume_path End2Race/checkpoints/teacher_last.pt --checkpoint_path End2Race/checkpoints/teacher_last.pt`
- Student 학습 재개:
  - `End2Race/scripts/train_student.py --resume_path End2Race/checkpoints/student_last.pt --checkpoint_path End2Race/checkpoints/student_last.pt`

## Transfer Learning / Finetune (추가)
- Teacher 가중치로 초기화(optimizer/epoch 리셋):
  - `End2Race/scripts/train_teacher.py --init_path <teacher_weights_or_ckpt>`
  - head가 안 맞으면: `--init_skip_output_layer`
- Student 가중치로 초기화(optimizer/epoch 리셋):
  - `End2Race/scripts/train_student.py --init_path <student_weights_or_ckpt>`
