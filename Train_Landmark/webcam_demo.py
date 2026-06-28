"""
Train_Landmark/webcam_demo.py

Hai chế độ sử dụng:
  1. Chạy độc lập:  python webcam_demo.py
  2. Import từ PRJ2: from Train_Landmark.webcam_demo import load_landmark_model, predict_68pts

Lịch sử sửa đổi:
  v1: dùng MediaPipe FaceDetection → bbox bao cả đầu, y1 quá cao → landmarks lệch lên.
  v2: dùng FaceMesh tight bbox (min/max 468 điểm) → vẫn lệch vì 468 điểm có đường tóc.
  v3 (hiện tại): dùng FaceMesh nhưng chỉ lấy y_min từ ĐIỂM LÔNG MÀY,
     khớp với dataset_loader.py nơi y_min = min(68 GT landmarks) ≈ đỉnh lông mày.
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import config
from cores.utils import denormalize_landmarks, draw_landmarks, calculate_ear


# ── FaceMesh indices xấp xỉ vùng lông mày iBUG 68 (điểm 17-26) ─────────────
# Đây là các điểm cao nhất (y_min) trong tập iBUG 68 với khuôn mặt hướng thẳng.
# Dùng các chỉ số này để tính y_min của crop, thay vì dùng đường tóc.
_BROW_INDICES = [
    # Lông mày trái (xấp xỉ iBUG 17-21)
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    # Lông mày phải (xấp xỉ iBUG 22-26)
    296, 334, 293, 300, 276, 283, 282, 295, 285, 336,
]


# ─── Public API (dùng bởi PRJ2/main.py) ─────────────────────────────────────

def load_landmark_model(model_path: str | None = None):
    """
    Nạp mô hình 68-point landmark.
    Trả về model hoặc None nếu file không tồn tại.
    """
    if model_path is None:
        model_path = f"{config.WEIGHTS_DIR}/face_landmark_68_best.keras"
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"[Landmark] Nạp model thành công: {model_path}")
        return model
    except Exception as e:
        print(f"[Landmark] Không tìm thấy model ({e}). EAR sẽ dùng FaceMesh fallback.")
        return None


def predict_68pts(model, face_bgr: np.ndarray,
                  x1: int, y1: int, box_w: int, box_h: int,
                  img_size: int = 112) -> np.ndarray | None:
    """
    Dự đoán 68 landmarks từ vùng khuôn mặt đã crop.

    Args:
        model    : model Keras đã load
        face_bgr : ảnh vùng mặt (BGR, bất kỳ size)
        x1, y1   : góc trên-trái của bbox trong ảnh gốc (để denormalize)
        box_w, box_h: kích thước bbox
        img_size : kích thước input model (mặc định 112)

    Returns:
        np.ndarray shape (68, 2) tọa độ pixel trong ảnh gốc, hoặc None.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None
    face_resized    = cv2.resize(face_bgr, (img_size, img_size))
    face_rgb        = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = (face_rgb.astype(np.float32) / 127.5) - 1.0
    face_input      = np.expand_dims(face_normalized, axis=0)

    preds = model.predict(face_input, verbose=0)[0]
    return denormalize_landmarks(preds, x1, y1, box_w, box_h)


def get_tight_bbox_ibug_style(face_landmarks, img_w: int, img_h: int,
                               padding_ratio: float = 0.15):
    """
    Tính bounding box bắt chước cách dataset_loader.py tính từ iBUG 68 GT landmarks.

    Vấn đề cốt lõi:
      dataset_loader.py tính tight bbox từ min/max của 68 điểm iBUG.
      Điểm y_min của iBUG 68 = đỉnh lông mày (điểm 17-26), KHÔNG phải đường tóc.
      FaceMesh 468 điểm có đường tóc/trán → y_min quá cao → toàn bộ landmarks
      bị đặt sai vị trí lên trên khi denormalize.

    Giải pháp:
      - y_min : lấy từ _BROW_INDICES (lông mày) để khớp với iBUG điểm 17-26.
      - y_max : lấy từ toàn bộ 468 điểm (cằm vẫn là điểm thấp nhất).
      - x_min/x_max : lấy từ toàn bộ 468 điểm (má/hàm rộng nhất, trán không ảnh hưởng).
      - padding : 15% của landmark span, giống dataset_loader.py:63-64.

    Returns:
        (x1, y1, x2, y2, final_w, final_h) hoặc None nếu không hợp lệ.
    """
    xs = [lm.x * img_w for lm in face_landmarks]
    ys = [lm.y * img_h for lm in face_landmarks]

    # x_min/x_max: toàn bộ 468 điểm — má/hàm xác định cạnh trái phải
    x_min, x_max = min(xs), max(xs)

    # y_max: cằm là điểm thấp nhất trong cả 468 điểm
    y_max = max(ys)

    # y_min: CHỈ dùng điểm lông mày, không dùng đường tóc
    # → khớp với iBUG 68 nơi điểm 17-26 là các điểm cao nhất
    brow_ys = [face_landmarks[i].y * img_h for i in _BROW_INDICES]
    y_min   = min(brow_ys)

    # Padding tính trên landmark span — giống dataset_loader.py:63-64
    span_w = x_max - x_min
    span_h = y_max - y_min
    pad_w  = int(span_w * padding_ratio)
    pad_h  = int(span_h * padding_ratio)

    x1 = max(0,     int(x_min - pad_w))
    y1 = max(0,     int(y_min - pad_h))
    x2 = min(img_w, int(x_max + pad_w))
    y2 = min(img_h, int(y_max + pad_h))

    final_w = x2 - x1
    final_h = y2 - y1

    if final_w <= 0 or final_h <= 0:
        return None

    return x1, y1, x2, y2, final_w, final_h


# ─── Standalone demo ─────────────────────────────────────────────────────────

def _run_standalone_demo():
    print("Đang nạp mô hình Landmark...")
    model = load_landmark_model()
    if model is None:
        print("Không tìm thấy model. Dừng.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    EAR_THRESHOLD = 0.25
    ALARM_FRAMES  = 10
    sleep_counter = 0

    previous_landmarks = None
    alpha = 0.5

    print("Bật Camera...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        ih, iw, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            lms  = results.multi_face_landmarks[0].landmark
            bbox = get_tight_bbox_ibug_style(lms, iw, ih,
                                             padding_ratio=config.PADDING_RATIO)
            if bbox is None:
                previous_landmarks = None
                cv2.imshow("Driver Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            x1, y1, x2, y2, final_w, final_h = bbox
            face_crop = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            real_landmarks = predict_68pts(model, face_crop,
                                           x1, y1, final_w, final_h,
                                           config.IMG_SIZE)
            if real_landmarks is None:
                previous_landmarks = None
            else:
                # EMA chống rung
                if previous_landmarks is None:
                    previous_landmarks = real_landmarks
                else:
                    real_landmarks     = alpha * real_landmarks + (1 - alpha) * previous_landmarks
                    previous_landmarks = real_landmarks

                frame = draw_landmarks(frame, real_landmarks)

                left_eye  = real_landmarks[36:42]
                right_eye = real_landmarks[42:48]
                ear_avg   = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

                cv2.putText(frame, f"EAR: {ear_avg:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if ear_avg < EAR_THRESHOLD:
                    sleep_counter += 1
                    if sleep_counter >= ALARM_FRAMES:
                        cv2.putText(frame, "CANH BAO BUON NGU!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    sleep_counter = 0
        else:
            previous_landmarks = None

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_standalone_demo()
