import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob

# --- 경로 설정 ---
# 데이터셋의 최상위 루트 경로
DATA_ROOT = Path("/media/sws/X9 Pro/dataset/")
# 학습 데이터가 포함된 경로
TRAIN_DATA_PATH = DATA_ROOT / "train"

# 생성될 train.json 및 val.json 파일의 전체 경로
TRAIN_MANIFEST_PATH = DATA_ROOT / "train.json"
VAL_MANIFEST_PATH = DATA_ROOT / "val.json"

# --- 모든 세그먼트 폴더 탐색 ---
segment_paths = sorted(glob.glob(str(TRAIN_DATA_PATH / "segment_*")))
print(f"총 {len(segment_paths)}개의 세그먼트 폴더를 찾았습니다.")
print("--------------------------------------------------")

# 모든 세그먼트의 데이터를 취합할 리스트
all_samples = []

for segment_path_str in segment_paths:
    segment_path = Path(segment_path_str)
    segment_name = segment_path.name  # 예: "segment_003"
    csv_path = segment_path / "meta" / "data.csv"
    
    if not csv_path.exists():
        print(f"경고: '{csv_path}' 파일이 없으므로 이 세그먼트는 건너뜁니다.")
        continue

    try:
        df = pd.read_csv(csv_path)
        print(f"'{csv_path}' 처리 중... {len(df)}개의 항목을 찾았습니다.")
        
        # --- 각 CSV 파일 내에서 샘플 데이터 준비 ---
        for index, row in df.iterrows():
            # DATA_ROOT 기준 상대 경로 생성
            # 예: "train/segment_003/lidar/타임스탬프.bin"
            relative_lidar_path = str(Path("train") / segment_name / row["lidar_filepath"])
            
            sample = {
                "lidar": relative_lidar_path,
                "control": [float(row["wheel_angle"]), float(row["velocity"])],
                "speed": float(row["velocity"])
            }
            all_samples.append(sample)
            
    except Exception as e:
        print(f"경고: '{csv_path}' 파일을 읽는 중 오류 발생: {e}. 이 세그먼트는 건너뜁니다.")
        continue

print("--------------------------------------------------")
print(f"모든 세그먼트에서 총 {len(all_samples)}개의 샘플을 취합했습니다.")

if not all_samples:
    print("처리할 샘플이 없습니다. 스크립트를 종료합니다.")
    exit()

# --- 학습/검증 데이터 분할 (80% 학습, 20% 검증) ---
train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42)

# --- 매니페스트 파일 저장 ---
try:
    with open(TRAIN_MANIFEST_PATH, "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"학습 샘플 {len(train_samples)}개를 '{TRAIN_MANIFEST_PATH}'에 저장했습니다.")

    with open(VAL_MANIFEST_PATH, "w") as f:
        json.dump(val_samples, f, indent=2)
    print(f"검증 샘플 {len(val_samples)}개를 '{VAL_MANIFEST_PATH}'에 저장했습니다.")
except IOError as e:
    print(f"오류: 매니페스트 파일을 저장할 수 없습니다. 쓰기 권한을 확인하거나 경로를 검토해 주세요. {e}")

print("\n모든 매니페스트 파일 생성이 완료되었습니다.")
