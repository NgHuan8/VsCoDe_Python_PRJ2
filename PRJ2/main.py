# main.py — Hệ thống giám sát tài xế tích hợp
#
# Pipeline:
#   Webcam → CLAHE → FaceMesh
#     ├─ [A] CNN eye classifier → PERCLOS
#     ├─ [B] EAR hình học (FaceMesh 6-pt) → Adaptive Threshold
#     ├─ [C] MAR ngáp
#     ├─ [D] Head Pose (solvePnP)
#     └─ [E] 68-pt Landmark model (tùy chọn, nếu tìm thấy .keras)
#                         ↓
#               FatigueFusion → fatigue_score → AlertLevel

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import csv
import json
import cv2
import mediapipe as mp
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import load_model

from config import *
from helpers import get_pixel_coords, calculate_mar, crop_eye, calculate_ear
from perclos import PerclosCalculator
from adaptive_threshold import AdaptiveEARThreshold
from head_pose import HeadPoseEstimator
from fatigue_fusion import (FatigueFusion, AlertLevel,
                             normalize_ear, normalize_mar)

# ── Thêm thư mục Train_Landmark vào sys.path để import tùy chọn ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_LANDMARK_ROOT = os.path.join(_HERE, "..", "Train_Landmark")
sys.path.insert(0, _LANDMARK_ROOT)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. KHỞI TẠO ÂM THANH
# ═══════════════════════════════════════════════════════════════════════════════
pygame.mixer.init()
try:
    _wav = os.path.join(_HERE, "warning.wav")
    alarm_sound = pygame.mixer.Sound(_wav)
except Exception as e:
    print(f"[Audio] Không tìm thấy warning.wav ({e}).")
    alarm_sound = None

alarm_playing = False

# ═══════════════════════════════════════════════════════════════════════════════
# 2. NẠP MÔ HÌNH CNN (bắt buộc)
# ═══════════════════════════════════════════════════════════════════════════════
CNN_MODEL_PATH = os.path.join(_HERE, "eye_state_classifier.keras")
LABELS_PATH    = os.path.join(_HERE, "class_indices.json")
CLOSED_CLASS_NAME = "sleepy"

print("Đang nạp mô hình CNN phân loại mắt...")
model = load_model(CNN_MODEL_PATH)

try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    closed_index = class_indices[CLOSED_CLASS_NAME]
    print(f"[CNN] Bản đồ nhãn: {class_indices} | '{CLOSED_CLASS_NAME}' → index {closed_index}")
except Exception as e:
    closed_index = 1
    print(f"[CNN] Không đọc được {LABELS_PATH} ({e}). Dùng closed_index=1.")

IMG_SIZE = 48  # phải khớp với lúc train

@tf.function(reduce_retracing=True)
def _cnn_infer(x):
    return model(x, training=False)

def preprocess_eye(eye_img: np.ndarray) -> np.ndarray:
    if eye_img.ndim == 3 and eye_img.shape[2] == 3:
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    if eye_img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        eye_img = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
    eye_img = eye_img.astype("float32")
    return np.expand_dims(eye_img, axis=-1)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. NẠP MÔ HÌNH LANDMARK 68-ĐIỂM (tùy chọn)
# ═══════════════════════════════════════════════════════════════════════════════
USE_LANDMARK_MODEL = False
landmark_model     = None

try:
    from webcam_demo import load_landmark_model, predict_68pts
    from cores.utils import denormalize_landmarks
    landmark_model     = load_landmark_model(LANDMARK_MODEL_PATH)
    USE_LANDMARK_MODEL = landmark_model is not None
except Exception as e:
    print(f"[Landmark] Import thất bại ({e}). Chạy không có landmark model.")

if USE_LANDMARK_MODEL:
    print("[Landmark] Pipeline 68-pt HOẠT ĐỘNG. EAR sẽ dùng cả hai nguồn.")
else:
    print("[Landmark] Chạy chế độ FaceMesh-only. EAR từ 6-pt MediaPipe.")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. KHỞI TẠO MEDIAPIPE FACE MESH
# ═══════════════════════════════════════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. KHỞI TẠO CÁC MODULE MỚI
# ═══════════════════════════════════════════════════════════════════════════════
perclos   = PerclosCalculator(
    window_seconds=PERCLOS_WINDOW_SECONDS,
    alert_threshold=PERCLOS_ALERT_THRESHOLD,
)

adaptive_ear = AdaptiveEARThreshold(
    fallback_threshold=EAR_THRESHOLD,
    calibration_frames=CALIBRATION_FRAMES,
    percentile_rank=EAR_CALIBRATION_PERCENTILE,
    min_valid_ear=EAR_CALIBRATION_MIN,
)

head_pose = HeadPoseEstimator(
    pitch_alert_deg=HEAD_PITCH_ALERT_DEG,
    yaw_ignore_deg=HEAD_YAW_IGNORE_DEG,
)

fusion = FatigueFusion(
    w_perclos=FUSION_W_PERCLOS,
    w_ear=FUSION_W_EAR,
    w_mar=FUSION_W_MAR,
    w_head=FUSION_W_HEAD,
    yellow_threshold=ALERT_YELLOW_THRESHOLD,
    red_threshold=ALERT_RED_THRESHOLD,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SESSION LOG (ghi dữ liệu để dùng cho ablation_fusion.py)
# ═══════════════════════════════════════════════════════════════════════════════
# Đặt ENABLE_SESSION_LOG = True để ghi session_log.csv khi chạy.
# Sau khi quay xong, mở CSV và điền cột "label" (0=tỉnh, 1=buồn ngủ) thủ công,
# rồi chạy: python ablation_fusion.py --data session_log.csv --grid_search
ENABLE_SESSION_LOG = False
_LOG_PATH = os.path.join(_HERE, "session_log.csv")
_log_file   = None
_log_writer = None

if ENABLE_SESSION_LOG:
    _log_file   = open(_LOG_PATH, "w", newline="", encoding="utf-8")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow(["perclos", "ear_score", "mar_score", "head_score", "label"])
    print(f"[Log] Ghi session log → {_LOG_PATH}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. BIẾN TRẠNG THÁI (giữ nguyên logic cũ song song với logic mới)
# ═══════════════════════════════════════════════════════════════════════════════
EYE_CLOSED_COUNTER    = 0   # vẫn giữ để so sánh / fallback
YAWN_COUNTER          = 0
SLEEP_FRAME_THRESHOLD = 30
YAWN_FRAME_THRESHOLD  = 90

# EMA cho EAR raw (giảm nhiễu giữa các frame)
_ear_ema        = 0.25
_EAR_EMA_ALPHA  = 0.30

# Landmark EMA smoother (giữ từ webcam_demo cũ)
_lm_previous    = None
_LM_EMA_ALPHA   = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# 7. HÀM VẼ HUD
# ═══════════════════════════════════════════════════════════════════════════════

_ALERT_COLORS = {
    AlertLevel.GREEN:  (0,   200,  0),
    AlertLevel.YELLOW: (0,   200, 255),
    AlertLevel.RED:    (0,     0, 255),
}

def _draw_hud(frame, avg_closed, ear_val, mar_val, pitch, yaw, roll,
              perclos_val, fatigue_score, alert_level,
              calib_progress, perclos_fill):
    """Vẽ toàn bộ thông tin lên frame."""
    h, w = frame.shape[:2]
    color = _ALERT_COLORS[alert_level]

    # ── Cột trái: tín hiệu thô ──────────────────────────────────────────────
    eye_status = "Closed" if avg_closed > 0.5 else "Open"
    eye_color  = (0, 0, 255) if avg_closed > 0.5 else (0, 255, 0)
    cv2.putText(frame, f"CNN Eye: {eye_status} ({avg_closed:.2f})",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
    cv2.putText(frame, f"EAR: {ear_val:.3f}  (thr={adaptive_ear.threshold:.3f})",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"MAR: {mar_val:.2f}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(frame, f"Head P:{pitch:+.1f} Y:{yaw:+.1f} R:{roll:+.1f}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2)
    cv2.putText(frame, f"PERCLOS: {perclos_val*100:.1f}% (fill={perclos_fill*100:.0f}%)",
                (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    # ── Fatigue score bar ────────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10, 155, 300, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), cv2.FILLED)
    fill = int(bar_w * fatigue_score / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                  color, cv2.FILLED)
    cv2.putText(frame, f"Fatigue: {fatigue_score:.0f}/100  [{alert_level.value}]",
                (bar_x, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── Calibration progress ─────────────────────────────────────────────────
    if not adaptive_ear.calibrated:
        cal_w = int(300 * calib_progress)
        cv2.rectangle(frame, (10, 190), (310, 204), (50, 50, 50), cv2.FILLED)
        cv2.rectangle(frame, (10, 190), (10 + cal_w, 204), (0, 165, 255), cv2.FILLED)
        cv2.putText(frame, f"Calibrating EAR... {calib_progress*100:.0f}%",
                    (10, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # ── Cảnh báo chính ───────────────────────────────────────────────────────
    if alert_level == AlertLevel.RED:
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 200), cv2.FILLED)
        cv2.putText(frame, "!!! CANH BAO: NGUOI LAI XE BUON NGU !!!",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    elif alert_level == AlertLevel.YELLOW:
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 150, 200), cv2.FILLED)
        cv2.putText(frame, "CANH BAO: Co dau hieu met moi",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # ── Badge landmark model ─────────────────────────────────────────────────
    badge_txt = "68-pt ON" if USE_LANDMARK_MODEL else "FaceMesh EAR"
    cv2.putText(frame, badge_txt, (w - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 200, 100) if USE_LANDMARK_MODEL else (180, 180, 180), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. VÒNG LẶP CHÍNH
# ═══════════════════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(CAMERA_INDEX)
print("Hệ thống giám sát tài xế đã SẴN SÀNG. Nhấn 'q' để thoát.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ── Tiền xử lý ảnh ──────────────────────────────────────────────────────
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch   = clahe.apply(l_ch)
    frame  = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    frame  = cv2.flip(frame, 1)
    h_img, w_img = frame.shape[:2]

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results    = face_mesh.process(rgb_frame)

    # Giá trị mặc định khi không thấy khuôn mặt
    avg_closed    = 0.0
    ear_val       = _ear_ema
    mar_val       = 0.0
    pitch = yaw = roll = 0.0
    fatigue_score = fusion.last_score
    alert_level   = fusion.classify(fatigue_score)

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark

        # ── [A] CNN eye classifier ────────────────────────────────────────────
        left_eye_pts  = get_pixel_coords(lms, LEFT_EYE_INDICES,  w_img, h_img)
        right_eye_pts = get_pixel_coords(lms, RIGHT_EYE_INDICES, w_img, h_img)
        mouth_pts     = get_pixel_coords(lms, MOUTH_INDICES,     w_img, h_img)

        left_eye_img  = crop_eye(gray_frame, left_eye_pts)
        right_eye_img = crop_eye(gray_frame, right_eye_pts)

        if left_eye_img is not None and right_eye_img is not None:
            # Debug windows (comment lại khi không cần)
            cv2.imshow("Debug Left Eye",  cv2.resize(left_eye_img,  (150, 150)))
            cv2.imshow("Debug Right Eye", cv2.resize(right_eye_img, (150, 150)))

            left_proc  = preprocess_eye(left_eye_img)
            right_proc = preprocess_eye(right_eye_img)
            batch      = np.stack([left_proc, right_proc], axis=0)
            preds      = _cnn_infer(tf.constant(batch, dtype=tf.float32)).numpy().ravel()

            closed_probs = preds if closed_index == 1 else (1.0 - preds)
            avg_closed   = float(np.mean(closed_probs))

            # Giữ nguyên bộ đếm cũ (backward compat / fallback)
            if avg_closed > 0.5:
                EYE_CLOSED_COUNTER += 1
            else:
                EYE_CLOSED_COUNTER = 0

            # PERCLOS update
            perclos.update(avg_closed > 0.5)

        # ── [B] EAR hình học từ FaceMesh ─────────────────────────────────────
        ear_left  = calculate_ear(left_eye_pts)
        ear_right = calculate_ear(right_eye_pts)
        ear_fm    = (ear_left + ear_right) / 2.0

        # ── [E] EAR từ 68-pt Landmark model (nếu có) ─────────────────────────
        if USE_LANDMARK_MODEL:
            # Xác định bbox khuôn mặt từ FaceMesh bounding box
            xs = [lm.x * w_img for lm in lms]
            ys = [lm.y * h_img for lm in lms]
            fm_x1 = max(0, int(min(xs) - 0.15 * (max(xs) - min(xs))))
            fm_y1 = max(0, int(min(ys) - 0.15 * (max(ys) - min(ys))))
            fm_x2 = min(w_img, int(max(xs) + 0.15 * (max(xs) - min(xs))))
            fm_y2 = min(h_img, int(max(ys) + 0.15 * (max(ys) - min(ys))))
            face_crop = frame[fm_y1:fm_y2, fm_x1:fm_x2]

            lm68 = predict_68pts(landmark_model, face_crop,
                                  fm_x1, fm_y1,
                                  fm_x2 - fm_x1, fm_y2 - fm_y1,
                                  LANDMARK_IMG_SIZE)
            if lm68 is not None:
                # EMA chống rung
                if _lm_previous is None:
                    _lm_previous = lm68
                else:
                    lm68         = _LM_EMA_ALPHA * lm68 + (1 - _LM_EMA_ALPHA) * _lm_previous
                    _lm_previous = lm68

                from cores.utils import calculate_ear as lm_ear
                ear_lm68 = (lm_ear(lm68[36:42]) + lm_ear(lm68[42:48])) / 2.0
                # Trọng số 60/40: ưu tiên landmark model nhưng vẫn tham khảo FaceMesh
                ear_fm   = 0.6 * ear_lm68 + 0.4 * ear_fm

        # EMA smoothing EAR cuối cùng
        _ear_ema = _EAR_EMA_ALPHA * ear_fm + (1 - _EAR_EMA_ALPHA) * _ear_ema
        ear_val  = _ear_ema

        # Adaptive threshold update
        adaptive_ear.update(ear_val, eyes_are_open=(avg_closed < 0.5))

        # ── [C] MAR ngáp ─────────────────────────────────────────────────────
        mar_val = calculate_mar(mouth_pts)
        if mar_val > MAR_THRESHOLD:
            YAWN_COUNTER += 1
        else:
            YAWN_COUNTER = 0

        # ── [D] Head Pose ─────────────────────────────────────────────────────
        pitch, yaw, roll = head_pose.estimate(lms, w_img, h_img,
                                               HEAD_POSE_LANDMARK_IDS)

        # ── Fusion ────────────────────────────────────────────────────────────
        ear_score  = normalize_ear(ear_val, adaptive_ear.threshold)
        mar_score  = normalize_mar(mar_val, MAR_THRESHOLD)
        head_score = head_pose.head_drowsiness_score(pitch, yaw)

        fatigue_score = fusion.compute(
            perclos_val=perclos.value,
            ear_score=ear_score,
            mar_score=mar_score,
            head_score=head_score,
        )
        alert_level = fusion.classify(fatigue_score)

    else:
        # Mất khuôn mặt → reset bộ đếm tránh cảnh báo sai
        EYE_CLOSED_COUNTER = 0
        YAWN_COUNTER       = 0
        perclos.reset()
        fusion.reset()
        _lm_previous = None

    # ── Âm thanh ─────────────────────────────────────────────────────────────
    # Kích hoạt cảnh báo âm thanh khi RED (thay vì chỉ khi frame_counter >= 30)
    should_alarm = (alert_level == AlertLevel.RED
                    or EYE_CLOSED_COUNTER >= SLEEP_FRAME_THRESHOLD  # backward compat
                    or YAWN_COUNTER >= YAWN_FRAME_THRESHOLD)
    if should_alarm:
        if alarm_sound and not alarm_playing:
            alarm_sound.play(loops=-1)
            alarm_playing = True
    else:
        if alarm_sound and alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

    # ── Session log ──────────────────────────────────────────────────────────
    if ENABLE_SESSION_LOG and _log_writer is not None:
        _log_writer.writerow([
            f"{perclos.value:.4f}",
            f"{normalize_ear(ear_val, adaptive_ear.threshold):.4f}",
            f"{normalize_mar(mar_val, MAR_THRESHOLD):.4f}",
            f"{head_pose.head_drowsiness_score(pitch, yaw):.4f}",
            "",   # label: điền thủ công sau khi quay xong
        ])

    # ── HUD ──────────────────────────────────────────────────────────────────
    _draw_hud(frame,
              avg_closed   = avg_closed,
              ear_val      = ear_val,
              mar_val      = mar_val,
              pitch        = pitch,
              yaw          = yaw,
              roll         = roll,
              perclos_val  = perclos.value,
              fatigue_score= fatigue_score,
              alert_level  = alert_level,
              calib_progress = adaptive_ear.progress,
              perclos_fill   = perclos.window_fill_ratio)

    cv2.imshow("Driver Drowsiness Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Dọn dẹp ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
if alarm_sound:
    alarm_sound.stop()
pygame.mixer.quit()
if _log_file is not None:
    _log_file.close()
    print(f"[Log] Đã lưu session log → {_LOG_PATH}")
