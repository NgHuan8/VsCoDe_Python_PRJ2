# adaptive_threshold.py
"""
Adaptive EAR Threshold — cá nhân hóa ngưỡng EAR theo từng người dùng.

Vấn đề với ngưỡng cứng (EAR = 0.22):
  - Người châu Á có EAR trung bình thấp hơn người châu Âu ~15–20%
  - Người đeo kính có EAR bị méo do phản xạ lens
  - Mỗi cá nhân có hình dạng mắt khác nhau

Giải pháp:
  Thu thập phân phối EAR của người dùng CỤ THỂ trong 30 giây đầu (khi tỉnh táo),
  rồi dùng percentile thứ 15 của phân phối đó làm ngưỡng.
  → Ngưỡng = "điểm EAR mà 85% thời gian tỉnh táo, mắt mở hơn mức này".
"""

import numpy as np


class AdaptiveEARThreshold:
    """
    Hiệu chỉnh ngưỡng EAR trong 30 giây đầu tiên người dùng sử dụng hệ thống.

    Trong giai đoạn calibration:
      - Chỉ thu thập EAR khi hệ thống chưa phát cảnh báo (mắt đang mở)
      - Lọc bỏ EAR < min_valid_ear (chớp mắt trong lúc calibrate)
      - Sau khi đủ mẫu → tính ngưỡng = percentile(baseline, percentile_rank)

    Sau calibration:
      - Dùng ngưỡng cá nhân hóa thay vì fallback_threshold
      - Hỗ trợ recalibration nếu gọi reset()
    """

    def __init__(self,
                 fallback_threshold: float = 0.22,
                 calibration_frames: int   = 900,
                 percentile_rank: int      = 15,
                 min_valid_ear: float      = 0.18):
        """
        fallback_threshold : ngưỡng dùng khi chưa calibrate xong
        calibration_frames : số frame cần thu thập (mặc định 900 = 30s×30fps)
        percentile_rank    : percentile của baseline làm ngưỡng (15 = bảo thủ)
        min_valid_ear      : lọc EAR quá thấp (chớp mắt, nhiễu) ra khỏi baseline
        """
        self.fallback          = fallback_threshold
        self.calibration_frames = calibration_frames
        self.percentile_rank   = percentile_rank
        self.min_valid_ear     = min_valid_ear

        self._baseline: list   = []
        self._threshold: float = fallback_threshold
        self.calibrated: bool  = False

    def update(self, ear: float, eyes_are_open: bool) -> None:
        """
        Thêm mẫu EAR vào baseline nếu đang trong giai đoạn calibration.

        eyes_are_open: True khi CNN dự đoán mắt MỞ (avg_closed < 0.5).
        Chỉ thu thập khi mắt mở để tránh baseline bị kéo thấp bởi buồn ngủ.
        """
        if self.calibrated:
            return
        if eyes_are_open and ear >= self.min_valid_ear:
            self._baseline.append(ear)
        if len(self._baseline) >= self.calibration_frames:
            self._finalize()

    def _finalize(self) -> None:
        arr = np.array(self._baseline)
        self._threshold = float(np.percentile(arr, self.percentile_rank))
        # Đảm bảo ngưỡng không thấp hơn min_valid_ear (phòng trường hợp dữ liệu bẩn)
        self._threshold = max(self._threshold, self.min_valid_ear)
        self.calibrated = True

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def progress(self) -> float:
        """Tiến trình calibration [0.0 → 1.0] để vẽ thanh tiến trình."""
        if self.calibrated:
            return 1.0
        return min(1.0, len(self._baseline) / self.calibration_frames)

    def is_drowsy(self, ear: float) -> bool:
        """True khi EAR dưới ngưỡng (mắt đóng theo tiêu chí cá nhân hóa)."""
        return ear < self._threshold

    def reset(self) -> None:
        """Bắt đầu lại calibration (dùng khi đổi người lái)."""
        self._baseline  = []
        self._threshold = self.fallback
        self.calibrated = False
