import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import config
from cores.utils import denormalize_landmarks, draw_landmarks, calculate_ear

def main():
    print("Đang nạp mô hình Landmark...")
    model_path = f"{config.WEIGHTS_DIR}/face_landmark_68_best.keras"
    model = tf.keras.models.load_model(model_path, compile=False) # compile=False vì lúc inference không cần hàm loss
    
    # Khởi tạo MediaPipe Face Detection thay cho Haar Cascade
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    
    # Tham số cảnh báo buồn ngủ
    EAR_THRESHOLD = 0.25
    ALARM_FRAMES = 10
    sleep_counter = 0

    # Tham số chống rung EMA (Exponential Moving Average)
    previous_landmarks = None
    alpha = 0.5 # Mức độ mượt (0.0 đến 1.0). Càng nhỏ càng mượt nhưng bám theo chuyển động chậm hơn.

    print("Bật Camera...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Nhận diện khuôn mặt bằng MediaPipe (Bắt góc nghiêng tốt)
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            for detection in results.detections:
                # Trích xuất tọa độ Bounding Box từ MediaPipe
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                # 1. Bù Padding 15% giống hệt tập Train để chống lệch điểm
                pad_w = int(w * 0.15)
                pad_h = int(h * 0.15)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(iw, x + w + pad_w)
                y2 = min(ih, y + h + pad_h)
                
                final_w = x2 - x1
                final_h = y2 - y1

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0 or final_w <= 0 or final_h <= 0: 
                    continue
                
                # Vẽ Box để test
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Tiền xử lý
                face_resized = cv2.resize(face_crop, (config.IMG_SIZE, config.IMG_SIZE))
                face_rgb_crop = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_normalized = (face_rgb_crop.astype(np.float32) / 127.5) - 1.0
                face_input = np.expand_dims(face_normalized, axis=0)
                
                # 2. Dự đoán 68 điểm
                preds = model.predict(face_input, verbose=0)[0]
                
                # 3. Phục hồi tọa độ dựa trên Box đã được Padding
                real_landmarks = denormalize_landmarks(preds, x1, y1, final_w, final_h)
                
                # 4. CHỐNG RUNG (EMA)
                if previous_landmarks is None:
                    previous_landmarks = real_landmarks
                else:
                    real_landmarks = alpha * real_landmarks + (1 - alpha) * previous_landmarks
                    previous_landmarks = real_landmarks
                
                # 5. Vẽ điểm
                frame = draw_landmarks(frame, real_landmarks)
                
                # 6. Tính toán EAR Cảnh báo buồn ngủ
                left_eye = real_landmarks[36:42]
                right_eye = real_landmarks[42:48]
                
                ear_left = calculate_ear(left_eye)
                ear_right = calculate_ear(right_eye)
                ear_avg = (ear_left + ear_right) / 2.0
                
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
            # Nếu không thấy mặt ai, reset lại biến lưu trạng thái trước đó
            previous_landmarks = None

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()