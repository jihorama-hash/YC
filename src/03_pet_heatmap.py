import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = Path(r"C:\Home\YC")

VIDEO_PATH = BASE_DIR / "record" / "for yolo" / "20260528_CCTV001_yolo_5min01.mp4"
EVENT_CSV = BASE_DIR / "outputs" / "YC_pet_events.csv"
OUTPUT_IMG = BASE_DIR / "runs" / "YC_pet_heatmap.png"

# =========================
# 2. 파라미터 설정
# =========================
SIGMA = 30       # heatmap 부드러움. 클수록 넓게 퍼짐.
ALPHA = 0.55     # 원본 영상과 heatmap 합성 비율

# =========================
# 3. 첫 프레임 추출
# =========================
cap = cv2.VideoCapture(str(VIDEO_PATH))
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Cannot read video frame.")

# OpenCV는 BGR, matplotlib은 RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
height, width = frame_rgb.shape[:2]

# =========================
# 4. PET 이벤트 읽기
# =========================
events = pd.read_csv(EVENT_CSV)

if len(events) == 0:
    raise RuntimeError("No PET events found. Cannot create heatmap.")

# =========================
# 5. 빈 heatmap 생성
# =========================
heat = np.zeros((height, width), dtype=np.float32)

for _, row in events.iterrows():
    x = int(round(row["event_x"]))
    y = int(round(row["event_y"]))
    w = float(row["risk_score"])

    if 0 <= x < width and 0 <= y < height:
        heat[y, x] += w

# =========================
# 6. Gaussian smoothing
# =========================
heat_smooth = gaussian_filter(heat, sigma=SIGMA)

# 0~1 정규화
if heat_smooth.max() > 0:
    heat_norm = heat_smooth / heat_smooth.max()
else:
    heat_norm = heat_smooth

# =========================
# 7. 시각화
# =========================
plt.figure(figsize=(14, 8))

plt.imshow(frame_rgb)
plt.imshow(heat_norm, cmap="jet", alpha=ALPHA)

plt.title("Pixel-based PET Risk Heatmap")
plt.axis("off")

# 이벤트 점도 함께 표시
plt.scatter(
    events["event_x"],
    events["event_y"],
    s=10,
    c="white",
    alpha=0.6,
    label="PET events"
)

plt.legend(loc="lower right")

plt.savefig(OUTPUT_IMG, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved heatmap: {OUTPUT_IMG}")