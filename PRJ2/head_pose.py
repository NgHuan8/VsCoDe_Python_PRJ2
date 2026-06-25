# head_pose.py
"""
Head Pose Estimation dùng cv2.solvePnP.

Nguyên lý:
  Có 6 điểm trên khuôn mặt với tọa độ 3D đã biết trong không gian metric
  (được đo từ mô hình khuôn mặt trung bình — Kazemi & Sullivan, 2014).
  MediaPipe cung cấp tọa độ 2D tương ứng trong ảnh.
  solvePnP giải bài toán Perspective-n-Point → ma trận xoay R, tịnh tiến T.
  Từ R ta tính góc Euler: pitch (cúi/ngẩng), yaw (quay trái/phải), roll (nghiêng).

Ứng dụng trong giám sát tài xế:
  - pitch > +20°: đầu cúi xuống → buồn ngủ
  - |yaw| > 45°: ngoảnh đầu nhiều → EAR không đáng tin (đầu quay → mắt bị méo)
  - roll > 30°: đầu nghiêng (ít ảnh hưởng hơn)

Hệ tọa độ (sau khi chuyển đổi):
  pitch dương = cúi đầu xuống (drowsy direction)
  yaw   dương = quay đầu sang phải
  roll  dương = nghiêng đầu sang phải
"""

import cv2
import numpy as np
import math


# Tọa độ 3D chuẩn của 6 điểm khuôn mặt (mm, gốc tại đầu mũi)
# Nguồn: OpenCV head pose estimation tutorial + Kazemi & Sullivan (2014)
# Thứ tự khớp với HEAD_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]
_MODEL_POINTS_3D = np.array([
    (  0.0,    0.0,    0.0),   # 1:   Nose tip
    (  0.0,  -63.6,  -12.5),   # 152: Chin
    (-43.3,   32.7,  -26.0),   # 33:  Left eye left corner
    ( 43.3,   32.7,  -26.0),   # 263: Right eye right corner
    (-28.9,  -28.9,  -24.1),   # 61:  Left mouth corner
    ( 28.9,  -28.9,  -24.1),   # 291: Right mouth corner
], dtype=np.float64)

# Giả sử không có méo lens (đúng với webcam thường)
_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


class HeadPoseEstimator:
    """Ước lượng góc đầu từ 6 landmarks MediaPipe FaceMesh."""

    def __init__(self, pitch_alert_deg: float = 20.0, yaw_ignore_deg: float = 45.0):
        self.pitch_alert_deg = pitch_alert_deg
        self.yaw_ignore_deg  = yaw_ignore_deg
        self._camera_matrix  = None
        self._last_img_shape = (0, 0)

    def _get_camera_matrix(self, img_w: int, img_h: int) -> np.ndarray:
        """
        Xây dựng camera intrinsic matrix từ kích thước ảnh.
        Giả sử focal length ≈ img_w (xấp xỉ hợp lý cho webcam thường).
        Cập nhật lại khi kích thước thay đổi (resized window, v.v.).
        """
        if (img_w, img_h) != self._last_img_shape:
            f = img_w  # focal length ước lượng
            cx, cy = img_w / 2.0, img_h / 2.0
            self._camera_matrix = np.array([
                [f,   0,  cx],
                [0,   f,  cy],
                [0,   0,   1],
            ], dtype=np.float64)
            self._last_img_shape = (img_w, img_h)
        return self._camera_matrix

    def estimate(self, landmarks, img_w: int, img_h: int,
                 landmark_ids: list) -> tuple:
        """
        Tính (pitch, yaw, roll) theo độ từ danh sách landmarks MediaPipe.

        landmarks    : results.multi_face_landmarks[0].landmark
        landmark_ids : thứ tự khớp với _MODEL_POINTS_3D, ví dụ [1,152,33,263,61,291]
        Trả về (pitch, yaw, roll) float degrees, hoặc (0, 0, 0) nếu solvePnP thất bại.
        """
        image_points = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in landmark_ids
        ], dtype=np.float64)

        cam_matrix = self._get_camera_matrix(img_w, img_h)
        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points, cam_matrix, _DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        # Chuyển rotation vector → rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Phân rã ma trận xoay → góc Euler (pitch, yaw, roll)
        # Dùng decomposeProjectionMatrix cho ổn định hơn atan2 thủ công
        proj_matrix = np.hstack([rmat, tvec])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler[0])   # x-axis: cúi/ngẩng
        yaw   = float(euler[1])   # y-axis: quay trái/phải
        roll  = float(euler[2])   # z-axis: nghiêng đầu

        # Hiệu chỉnh hướng: pitch dương = cúi xuống (buồn ngủ)
        # decomposeProjectionMatrix trả về góc âm khi cúi → đảo dấu
        pitch = -pitch

        return pitch, yaw, roll

    def is_nodding(self, pitch: float) -> bool:
        """True khi đầu cúi xuống quá ngưỡng cảnh báo."""
        return pitch > self.pitch_alert_deg

    def is_looking_away(self, yaw: float) -> bool:
        """True khi tài xế quay đầu nhiều → EAR không đáng tin."""
        return abs(yaw) > self.yaw_ignore_deg

    def head_drowsiness_score(self, pitch: float, yaw: float) -> float:
        """
        Điểm buồn ngủ từ tư thế đầu, trong [0.0, 1.0].
        Không tính nếu đang nhìn sang ngang (tránh false positive).
        """
        if self.is_looking_away(yaw):
            return 0.0
        # Normalize pitch: 0° → 0.0, pitch_alert_deg → 0.5, 2×pitch_alert → 1.0
        score = pitch / (2.0 * self.pitch_alert_deg)
        return float(np.clip(score, 0.0, 1.0))
