# helpers.py
import cv2
import numpy as np

def get_pixel_coords(landmarks, indices, img_w, img_h):
    """Chuyển đổi tọa độ chuẩn hóa sang pixel thực tế"""
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        coords.append(np.array([x, y]))
    return coords

def calculate_mar(mouth_pts):
    """Tính chỉ số MAR cho miệng để phát hiện ngáp"""
    v1 = np.linalg.norm(mouth_pts[1] - mouth_pts[7])
    v2 = np.linalg.norm(mouth_pts[2] - mouth_pts[6])
    v3 = np.linalg.norm(mouth_pts[3] - mouth_pts[5])
    h = np.linalg.norm(mouth_pts[0] - mouth_pts[4])
    return (v1 + v2 + v3) / (2.0 * h)

def calculate_ear(eye_pts):
    """Eye Aspect Ratio từ 6 điểm landmark (Soukupová & Čech, 2016).
    Thứ tự điểm: [ngoài, trên-trong, trên-ngoài, trong, dưới-trong, dưới-ngoài]
    theo chiều MediaPipe LEFT_EYE_INDICES / RIGHT_EYE_INDICES.
    Trả về float; gần 0 = mắt đóng, ~0.3 = mắt mở bình thường.
    """
    pts = [np.array(p) for p in eye_pts]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def crop_eye(frame, eye_pts, padding=10, target_size=(48, 48)):
    """Cắt vùng mắt và resize về target_size.

    Hỗ trợ cả ảnh màu (BGR) lẫn ảnh xám:
      - Nếu truyền vào frame MÀU  -> tự đổi sang xám (hành vi cũ).
      - Nếu truyền vào frame ĐÃ XÁM (main.py mới) -> giữ nguyên, không đổi lại
        (tránh lỗi 'invalid number of channels' khi gọi BGR2GRAY lên ảnh 1 kênh).
    Kết quả luôn là ảnh xám kích thước target_size.
    """
    h_img, w_img = frame.shape[:2]

    eye_pts = np.array(eye_pts)
    x_min, y_min = np.min(eye_pts, axis=0)
    x_max, y_max = np.max(eye_pts, axis=0)

    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(w_img, x_max + padding)
    y2 = min(h_img, y_max + padding)

    eye_crop = frame[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return None

    # Chỉ đổi sang xám khi đầu vào còn là ảnh màu (3 kênh).
    if eye_crop.ndim == 3:
        eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

    resized_eye = cv2.resize(eye_crop, target_size)
    return resized_eye