# main.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ẩn các cảnh báo rác của TensorFlow

import json
import cv2
import mediapipe as mp
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import load_model   # dùng tf.keras cho khớp với train.py
from config import *
from helpers import get_pixel_coords, calculate_mar, crop_eye

# ==========================================
# CẤU HÌNH KHỚP VỚI train.py
# ==========================================
IMG_SIZE = 48                          # phải trùng với IMG_SIZE lúc train
MODEL_PATH = "eye_state_classifier.keras"
LABELS_PATH = "class_indices.json"     # file train.py mới đã xuất ra

# Đặt đúng TÊN THƯ MỤC lớp "nhắm mắt" mà bạn đã dùng khi train
# (ví dụ: "Closed", "closed", "Sleepy"...). Phải giống y hệt tên folder.
CLOSED_CLASS_NAME = "Closed"

# 1. KHỞI TẠO ÂM THANH
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound('c:\\Users\\Asus Zenbook14X OLED\\Documents\\VsCoDe_Python_PRJ2\\PRJ2\\warning.wav')
except Exception as e:
    print(f"Cảnh báo: Chưa tìm thấy file 'warning.wav' ({e}).")
    alarm_sound = None

alarm_playing = False

# 2. NẠP MÔ HÌNH CNN
print("Đang nạp mô hình AI phân loại mắt...")
model = load_model(MODEL_PATH)

# 2b. NẠP BẢN ĐỒ NHÃN ĐỂ XÁC ĐỊNH ĐÚNG CHIỀU DỰ ĐOÁN
# Model mới dùng sigmoid -> đầu ra = xác suất của lớp có index = 1.
# Tùy theo thứ tự alphabet của tên folder mà "nhắm mắt" có thể là index 0 hoặc 1,
# nên ta đọc class_indices.json để biết chắc, tránh bị ĐẢO NGƯỢC kết quả.
try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    closed_index = class_indices[CLOSED_CLASS_NAME]
    print(f"Bản đồ nhãn: {class_indices} | '{CLOSED_CLASS_NAME}' -> index {closed_index}")
except Exception as e:
    closed_index = 1  # fallback giữ nguyên hành vi cũ (sigmoid > 0.5 = nhắm mắt)
    print(f"Cảnh báo: không đọc được {LABELS_PATH} hoặc lớp '{CLOSED_CLASS_NAME}' ({e}). "
          f"Tạm dùng closed_index = 1.")

# Hàm dự đoán nhanh (model(x) nhanh hơn model.predict() rất nhiều cho từng frame).
# Bọc trong tf.function để có đường chạy đã biên dịch; input luôn cố định (2,48,48,1)
# nên không bị retrace lại mỗi frame.
@tf.function(reduce_retracing=True)
def infer(x):
    return model(x, training=False)

def preprocess_eye(eye_img):
    """Đưa ảnh mắt về đúng định dạng model cần: xám, 48x48, KHÔNG chia 255.
    Crop bây giờ đã là ảnh xám, nên nhánh BGR2GRAY chỉ còn là phòng hờ."""
    if eye_img.ndim == 3 and eye_img.shape[2] == 3:
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    if eye_img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        eye_img = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
    eye_img = eye_img.astype("float32")          # GIỮ NGUYÊN dải [0, 255]
    eye_img = np.expand_dims(eye_img, axis=-1)   # (48, 48) -> (48, 48, 1)
    return eye_img

# 3. KHỞI TẠO MEDIAPIPE FACE MESH
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# 4. CÁC BIẾN CẤU HÌNH VÀ ĐẾM THỜI GIAN
EYE_CLOSED_COUNTER = 0
YAWN_COUNTER = 0

SLEEP_FRAME_THRESHOLD = 30  # Khoảng 1 giây nhắm mắt
YAWN_FRAME_THRESHOLD = 90   # Khoảng 3 giây ngáp

cap = cv2.VideoCapture(CAMERA_INDEX)
print("Hệ thống giám sát trạng thái tài xế đã SẴN SÀNG.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # --- BỘ LỌC CLAHE TĂNG CƯỜNG SÁNG ---
    # 1. Đổi sang hệ màu LAB để bóc tách riêng kênh Ánh sáng (L)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # 2. Cân bằng histogram cục bộ (Tăng sáng vùng tối, giữ nguyên vùng sáng)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # 3. Gộp kênh và trả về ảnh BGR bình thường
    limg = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # ---------------------------------------------

    # Lật ảnh để dùng như gương soi
    frame = cv2.flip(frame, 1)
    h_img, w_img = frame.shape[:2]

    # Đổi sang ảnh xám MỘT LẦN duy nhất cho cả frame, dùng để cắt 2 mắt bên dưới
    # (model train bằng grayscale luma, nên dùng BGR2GRAY cho khớp -> không dùng kênh L).
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # MediaPipe cần ảnh RGB (vẫn dùng frame màu)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    display_alert = False
    alert_text = ""
    avg_closed = 0.0  # xác suất "nhắm mắt" trung bình của 2 mắt

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Lấy tọa độ pixel
        left_eye_pts = get_pixel_coords(landmarks, LEFT_EYE_INDICES, w_img, h_img)
        right_eye_pts = get_pixel_coords(landmarks, RIGHT_EYE_INDICES, w_img, h_img)
        mouth_pts = get_pixel_coords(landmarks, MOUTH_INDICES, w_img, h_img)

        # Cắt ảnh mắt TỪ FRAME XÁM (đã đổi 1 lần ở trên)
        # -> không cần đổi xám lại cho từng con mắt nữa
        left_eye_img = crop_eye(gray_frame, left_eye_pts)
        right_eye_img = crop_eye(gray_frame, right_eye_pts)

        if left_eye_img is not None and right_eye_img is not None:
            # Hiện Debug Windows (Bạn có thể comment lại 2 dòng này khi đã chạy tốt)
            cv2.imshow("Debug Left Eye", cv2.resize(left_eye_img, (150, 150)))
            cv2.imshow("Debug Right Eye", cv2.resize(right_eye_img, (150, 150)))

            # --- TIỀN XỬ LÝ + DỰ ĐOÁN GỘP 2 MẮT TRONG 1 LẦN GỌI ---
            # Lưu ý: model mới đã có sẵn lớp Rescaling(1/255) bên trong,
            # nên ở đây TUYỆT ĐỐI KHÔNG chia 255 nữa.
            left_proc = preprocess_eye(left_eye_img)
            right_proc = preprocess_eye(right_eye_img)

            batch = np.stack([left_proc, right_proc], axis=0)        # (2, 48, 48, 1)
            preds = infer(tf.constant(batch, dtype=tf.float32)).numpy().ravel()

            # preds[i] = xác suất của lớp index 1. Quy đổi về xác suất "nhắm mắt":
            if closed_index == 1:
                closed_probs = preds
            else:
                closed_probs = 1.0 - preds
            avg_closed = float(np.mean(closed_probs))

            # Kiểm tra trạng thái mắt (> 0.5 là Sleepy/Closed)
            if avg_closed > 0.5:
                EYE_CLOSED_COUNTER += 1
            else:
                EYE_CLOSED_COUNTER = 0

            # Kiểm tra ngáp
            mar = calculate_mar(mouth_pts)
            if mar > MAR_THRESHOLD:
                YAWN_COUNTER += 1
            else:
                YAWN_COUNTER = 0

            # Logic báo động
            if EYE_CLOSED_COUNTER >= SLEEP_FRAME_THRESHOLD:
                display_alert = True
                alert_text = "!!! CANH BAO: TAI XE NGU GAT !!!"
            elif YAWN_COUNTER >= YAWN_FRAME_THRESHOLD:
                display_alert = True
                alert_text = "!!! CANH BAO: TAI XE QUA MET MOI !!!"

            # Xử lý âm thanh
            if display_alert:
                if alarm_sound and not alarm_playing:
                    alarm_sound.play(loops=-1)
                    alarm_playing = True
            else:
                if alarm_sound and alarm_playing:
                    alarm_sound.stop()
                    alarm_playing = False

            # Vẽ thông tin lên màn hình
            eye_status_text = "Closed" if avg_closed > 0.5 else "Open"
            eye_color = (0, 0, 255) if avg_closed > 0.5 else (0, 255, 0)

            cv2.putText(frame, f"Eyes: {eye_status_text} ({avg_closed:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
            cv2.putText(frame, f"Mouth MAR: {mar:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if display_alert:
                cv2.rectangle(frame, (5, 100), (635, 150), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, alert_text, (20, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    else:
        # Mất khuôn mặt -> reset bộ đếm và tắt chuông để tránh báo động giả
        EYE_CLOSED_COUNTER = 0
        YAWN_COUNTER = 0
        if alarm_sound and alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

    cv2.imshow('Driver Drowsiness Detection System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
if alarm_sound: alarm_sound.stop()
pygame.mixer.quit()