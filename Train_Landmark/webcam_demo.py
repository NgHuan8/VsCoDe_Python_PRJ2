"""
Train_Landmark/webcam_demo.py

Hai chế độ sử dụng:
  1. Chạy độc lập:  python webcam_demo.py  (giữ nguyên hành vi cũ)
  2. Import từ PRJ2: from Train_Landmark.webcam_demo import load_landmark_model, predict_68pts
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import config
from cores.utils import denormalize_landmarks, draw_landmarks, calculate_ear


# ─── Public API (dùng bởi PRJ2/main.py) ────────────────────────────────────

def load_landmark_model(model_path: str | None = None):
    """
    Nạp mô hình 68-point landmark.
    Trả về model hoặc None nếu file không tồn tại (cho phép optional loading).
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
        box_w, box_h: kích thước bbox (sau padding)
        img_size : kích thước input model (mặc định 112)

    Returns:
        np.ndarray shape (68, 2) tọa độ pixel trong ảnh gốc,
        hoặc None nếu face_bgr rỗng.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None
    face_resized    = cv2.resize(face_bgr, (img_size, img_size))
    face_rgb        = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = (face_rgb.astype(np.float32) / 127.5) - 1.0
    face_input      = np.expand_dims(face_normalized, axis=0)

    preds = model.predict(face_input, verbose=0)[0]
    return denormalize_landmarks(preds, x1, y1, box_w, box_h)


# ─── Standalone demo (giữ nguyên hoạt động cũ) ──────────────────────────────

def _run_standalone_demo():
    print("Đang nạp mô hình Landmark...")
    model = load_landmark_model()
    if model is None:
        print("Không tìm thấy model. Dừng.")
        return

    mp_face_detection = mp.solutions.face_detection
    face_detection    = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC  = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                               int(bboxC.width * iw), int(bboxC.height * ih))

                pad_w = int(w * 0.15)
                pad_h = int(h * 0.15)
                x1 = max(0, x - pad_w);  y1 = max(0, y - pad_h)
                x2 = min(iw, x + w + pad_w); y2 = min(ih, y + h + pad_h)
                final_w = x2 - x1;  final_h = y2 - y1

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0 or final_w <= 0 or final_h <= 0:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                real_landmarks = predict_68pts(model, face_crop,
                                               x1, y1, final_w, final_h,
                                               config.IMG_SIZE)
                if real_landmarks is None:
                    continue

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
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_standalone_demo()
