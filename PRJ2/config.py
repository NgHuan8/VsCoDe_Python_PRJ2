# config.py
import os

# ── Đường dẫn dataset (dùng cho train_cnn.py) ────────────────────────────────
DATASET_PATH = "eye_dataset"
CLOSED_DIR = os.path.join(DATASET_PATH, "closed")
OPEN_DIR = os.path.join(DATASET_PATH, "open")

# ── Điểm mốc MediaPipe FaceMesh ──────────────────────────────────────────────
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES     = [78, 81, 13, 311, 308, 402, 14, 178]

# 6 điểm dùng cho solvePnP (nose tip, chin, left eye corner, right eye corner,
# left mouth corner, right mouth corner) — có tọa độ 3D đã biết trong tài liệu
HEAD_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FPS_ESTIMATE = 30   # dùng để tính PERCLOS window; webcam thực có thể khác

# ── Ngưỡng cơ bản (fallback khi adaptive chưa calibrate xong) ────────────────
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.60

# ── PERCLOS (Percentage of Eye Closure) ──────────────────────────────────────
# Tiêu chuẩn NHTSA: PERCLOS > 20% trong 60s → buồn ngủ
PERCLOS_WINDOW_SECONDS   = 60
PERCLOS_ALERT_THRESHOLD  = 0.20

# ── Adaptive EAR Threshold ───────────────────────────────────────────────────
CALIBRATION_FRAMES          = 900   # 30s × 30fps
EAR_CALIBRATION_PERCENTILE  = 15    # percentile thứ 15 của baseline làm ngưỡng
EAR_CALIBRATION_MIN         = 0.18  # loại EAR quá thấp khỏi calibration (chớp mắt)

# ── Head Pose ────────────────────────────────────────────────────────────────
HEAD_PITCH_ALERT_DEG = 20.0   # cúi đầu > 20° = dấu hiệu ngủ gật
HEAD_YAW_IGNORE_DEG  = 45.0   # ngoảnh đầu > 45° → EAR không đáng tin

# ── Fatigue Fusion Weights (tổng = 1.0) ──────────────────────────────────────
FUSION_W_PERCLOS = 0.40
FUSION_W_EAR     = 0.25
FUSION_W_MAR     = 0.15
FUSION_W_HEAD    = 0.20

ALERT_YELLOW_THRESHOLD = 30   # fatigue score 0–100
ALERT_RED_THRESHOLD    = 60

# # ── Đường dẫn Landmark Model (tùy chọn) ──────────────────────────────────────
# _HERE = os.path.dirname(os.path.abspath(__file__))
# LANDMARK_MODEL_PATH = os.path.join(_HERE, "..", "Train_Landmark", "weights",
#                                    "face_landmark_68_best.keras")
# LANDMARK_IMG_SIZE   = 112
# LANDMARK_PADDING    = 0.15