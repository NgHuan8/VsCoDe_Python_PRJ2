# core/utils.py
import cv2
import numpy as np

def denormalize_landmarks(norm_landmarks, x1, y1, box_w, box_h):
    """
    Hàm đảo ngược chuẩn hóa: 
    Biến đổi tọa độ từ dải [0, 1] do Model dự đoán trở lại tọa độ pixel thực tế trên ảnh gốc.
    
    Tham số:
    - norm_landmarks: Mảng 1D chứa 136 giá trị (hoặc 2D 68x2) từ output của model.
    - x1, y1: Tọa độ góc trên cùng bên trái của Bounding Box khuôn mặt.
    - box_w, box_h: Chiều rộng và chiều cao của Bounding Box.
    """
    # Đảm bảo mảng có shape (68, 2)
    landmarks = np.array(norm_landmarks).reshape(68, 2)
    
    real_landmarks = np.zeros_like(landmarks)
    for i in range(68):
        real_landmarks[i][0] = (landmarks[i][0] * box_w) + x1
        real_landmarks[i][1] = (landmarks[i][1] * box_h) + y1
        
    return real_landmarks

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2, thickness=-1):
    """
    Vẽ 68 điểm landmark lên ảnh.
    
    Tham số:
    - image: Ảnh gốc (numpy array từ cv2).
    - landmarks: Mảng tọa độ thực tế shape (68, 2).
    - color: Màu của điểm vẽ (Mặc định: Xanh lục BGR).
    """
    img_copy = image.copy()
    
    for (x, y) in landmarks:
        # Ép kiểu về số nguyên để vẽ pixel
        pt = (int(round(x)), int(round(y)))
        cv2.circle(img_copy, pt, radius, color, thickness)
        
    return img_copy

def draw_landmark_numbers(image, landmarks, color=(0, 0, 255), font_scale=0.3):
    """
    (Tùy chọn) Vẽ số thứ tự từ 0-67 lên từng điểm landmark để dễ debug.
    Rất hữu ích khi bạn muốn xác định điểm nào thuộc mắt, mũi hay miệng.
    """
    img_copy = image.copy()
    
    for i, (x, y) in enumerate(landmarks):
        pt = (int(round(x)), int(round(y)))
        cv2.putText(img_copy, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
    return img_copy

def calculate_ear(eye_landmarks):
    """
    Tính toán Eye Aspect Ratio (EAR) - Tỷ lệ khung hình mắt.
    Đây là hàm then chốt cho hệ thống nhận diện buồn ngủ sau này.
    
    Tham số:
    - eye_landmarks: Mảng chứa 6 tọa độ của 1 mắt (Ví dụ: từ điểm 36 đến 41 cho mắt trái).
    """
    # Tính khoảng cách chiều dọc (từ mí trên xuống mí dưới)
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Tính khoảng cách chiều ngang (từ khóe mắt tới đuôi mắt)
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # Công thức EAR
    ear = (A + B) / (2.0 * C)
    return ear