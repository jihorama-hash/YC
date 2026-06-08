import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = Path(r"C:\Home\YC")
INPUT_CSV = BASE_DIR / "runs" / "tracks_pixel.csv"
OUTPUT_CSV = BASE_DIR / "outputs" / "YC_pet_events.csv"

# =========================
# 2. 파라미터 설정
# =========================
CELL_SIZE = 40          # pixel 단위 grid 크기
PET_THRESHOLD = 5.0     # sec
MIN_TRACK_POINTS = 5    # 너무 짧은 track 제거

VRU_CLASSES = {"person", "bicycle"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# =========================
# 3. 데이터 읽기
# =========================
df = pd.read_csv(INPUT_CSV)

# 필요한 column 확인
required_cols = ["frame_id", "time_sec", "track_id", "class_name", "foot_x", "foot_y"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# =========================
# 4. 객체 그룹 부여
# =========================
def classify_group(class_name):
    if class_name in VRU_CLASSES:
        return "vru"
    elif class_name in VEHICLE_CLASSES:
        return "vehicle"
    else:
        return "other"

df["group"] = df["class_name"].apply(classify_group)
df = df[df["group"].isin(["vru", "vehicle"])].copy()

# =========================
# 5. 너무 짧은 track 제거
# =========================
track_counts = df.groupby("track_id").size()
valid_tracks = track_counts[track_counts >= MIN_TRACK_POINTS].index
df = df[df["track_id"].isin(valid_tracks)].copy()

# =========================
# 6. grid cell 부여
# =========================
df["cell_x"] = (df["foot_x"] // CELL_SIZE).astype(int)
df["cell_y"] = (df["foot_y"] // CELL_SIZE).astype(int)

# cell id
df["cell_id"] = df["cell_x"].astype(str) + "_" + df["cell_y"].astype(str)

# =========================
# 7. 같은 track이 같은 cell을 통과한 대표시각 만들기
#    한 객체가 같은 cell에 여러 프레임 머물 수 있으므로
#    cell 진입/점유를 하나의 event로 압축
# =========================
passages = (
    df.groupby(["cell_id", "cell_x", "cell_y", "track_id", "group", "class_name"])
      .agg(
          t_enter=("time_sec", "min"),
          t_exit=("time_sec", "max"),
          x_mean=("foot_x", "mean"),
          y_mean=("foot_y", "mean"),
          n_frames=("frame_id", "count")
      )
      .reset_index()
)

# =========================
# 8. cell별 VRU-vehicle PET 계산
# =========================
events = []

for cell_id, g in passages.groupby("cell_id"):
    vru_pass = g[g["group"] == "vru"]
    veh_pass = g[g["group"] == "vehicle"]

    if len(vru_pass) == 0 or len(veh_pass) == 0:
        continue

    for _, vru in vru_pass.iterrows():
        for _, veh in veh_pass.iterrows():

            # 같은 track끼리는 비교할 필요 없음
            if vru["track_id"] == veh["track_id"]:
                continue

            # 두 객체가 같은 cell을 점유한 시간 구간
            # interval A = [vru_enter, vru_exit]
            # interval B = [veh_enter, veh_exit]
            vru_enter = vru["t_enter"]
            vru_exit = vru["t_exit"]
            veh_enter = veh["t_enter"]
            veh_exit = veh["t_exit"]

            # PET 계산
            # 1) 시간이 겹치면 PET = 0
            # 2) VRU가 먼저 지나가고 차량이 나중에 오면 veh_enter - vru_exit
            # 3) 차량이 먼저 지나가고 VRU가 나중에 오면 vru_enter - veh_exit
            if (vru_enter <= veh_exit) and (veh_enter <= vru_exit):
                pet = 0.0
                order = "overlap"
            elif vru_exit < veh_enter:
                pet = veh_enter - vru_exit
                order = "vru_first"
            elif veh_exit < vru_enter:
                pet = vru_enter - veh_exit
                order = "vehicle_first"
            else:
                continue

            if pet <= PET_THRESHOLD:
                risk_score = (PET_THRESHOLD - pet) / PET_THRESHOLD

                events.append({
                    "cell_id": cell_id,
                    "cell_x": vru["cell_x"],
                    "cell_y": vru["cell_y"],
                    "event_x": (vru["x_mean"] + veh["x_mean"]) / 2,
                    "event_y": (vru["y_mean"] + veh["y_mean"]) / 2,
                    "vru_track_id": vru["track_id"],
                    "vru_class": vru["class_name"],
                    "vehicle_track_id": veh["track_id"],
                    "vehicle_class": veh["class_name"],
                    "vru_enter": vru_enter,
                    "vru_exit": vru_exit,
                    "vehicle_enter": veh_enter,
                    "vehicle_exit": veh_exit,
                    "pet": pet,
                    "order": order,
                    "risk_score": risk_score,
                })

events_df = pd.DataFrame(events)

# 중복이 많을 수 있으므로 동일 객체쌍/동일 cell 기준으로 가장 작은 PET만 남김
if len(events_df) > 0:
    events_df = (
        events_df.sort_values("pet")
                 .drop_duplicates(
                     subset=["cell_id", "vru_track_id", "vehicle_track_id"],
                     keep="first"
                 )
                 .reset_index(drop=True)
    )

events_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Saved: {OUTPUT_CSV}")
print(f"Total PET events: {len(events_df)}")

if len(events_df) > 0:
    print(events_df[["cell_id", "event_x", "event_y", "pet", "risk_score", "order"]].head(20))
else:
    print("No PET events found. Try larger CELL_SIZE or larger PET_THRESHOLD.")