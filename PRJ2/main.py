# main.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ẩn các cảnh báo rác của TensorFlow

import cv2
import mediapipe as mp
import numpy as np
import pygame
from keras.models import load_model
from config import *
from helpers import get_pixel_coords, calculate_mar, crop_eye

# 1. KHỞI TẠO ÂM THANH
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound('c:\\Users\\Asus Zenbook14X OLED\\Documents\\VsCoDe_Python_PRJ2\\PRJ2\\warning.wav') 
except:
    print("Cảnh báo: Chưa tìm thấy file 'warning.wav'.")
    alarm_sound = None

alarm_playing = False 

# 2. NẠP MÔ HÌNH CNN
print("Đang nạp mô hình AI phân loại mắt...")
model = load_model("eye_state_classifier.keras")

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
    
    # MediaPipe cần ảnh RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    display_alert = False
    alert_text = ""

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Lấy tọa độ pixel
        left_eye_pts = get_pixel_coords(landmarks, LEFT_EYE_INDICES, w_img, h_img)
        right_eye_pts = get_pixel_coords(landmarks, RIGHT_EYE_INDICES, w_img, h_img)
        mouth_pts = get_pixel_coords(landmarks, MOUTH_INDICES, w_img, h_img)
        
        # Cắt ảnh mắt (truyền vào frame BGR gốc)
        left_eye_img = crop_eye(frame, left_eye_pts)
        right_eye_img = crop_eye(frame, right_eye_pts)
        
        if left_eye_img is not None and right_eye_img is not None:
            # Hiện Debug Windows (Bạn có thể comment lại 2 dòng này khi đã chạy tốt)
            cv2.imshow("Debug Left Eye", cv2.resize(left_eye_img, (150, 150)))
            cv2.imshow("Debug Right Eye", cv2.resize(right_eye_img, (150, 150)))

            # --- TIỀN XỬ LÝ MA TRẬN CHUẨN XÁC CHO CNN ---
            # 1. Ép kiểu và chia 255.0
            left_norm = left_eye_img.astype("float32") / 255.0
            right_norm = right_eye_img.astype("float32") / 255.0
            
            # 2. Thêm chiều kênh màu (48, 48) -> (48, 48, 1)
            left_norm = np.expand_dims(left_norm, axis=-1)
            right_norm = np.expand_dims(right_norm, axis=-1)
            
            # 3. Thêm chiều batch size (48, 48, 1) -> (1, 48, 48, 1)
            left_input = np.expand_dims(left_norm, axis=0)
            right_input = np.expand_dims(right_norm, axis=0)
            
            # --- DỰ ĐOÁN (ĐẢO NGƯỢC TRÁI PHẢI DO FLIP) ---
            pred_right = model.predict(left_input, verbose=0)[0][0]
            pred_left = model.predict(right_input, verbose=0)[0][0]
            avg_prediction = (pred_left + pred_right) / 2.0
            
            # Kiểm tra trạng thái mắt (> 0.5 là Sleepy/Closed)
            if avg_prediction > 0.5:
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
            eye_status_text = "Closed" if avg_prediction > 0.5 else "Open"
            eye_color = (0, 0, 255) if avg_prediction > 0.5 else (0, 255, 0)
            
            cv2.putText(frame, f"Eyes: {eye_status_text} ({avg_prediction:.2f})", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
            cv2.putText(frame, f"Mouth MAR: {mar:.2f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if display_alert:
                cv2.rectangle(frame, (5, 100), (635, 150), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, alert_text, (20, 135), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    cv2.imshow('Driver Drowsiness Detection System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
if alarm_sound: alarm_sound.stop()
pygame.mixer.quit()