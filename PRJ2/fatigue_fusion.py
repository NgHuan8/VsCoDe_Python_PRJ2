# fatigue_fusion.py
"""
Fatigue Fusion — kết hợp nhiều tín hiệu thành một điểm mệt mỏi tổng hợp.

Tại sao cần fusion thay vì dùng 1 tín hiệu?
  - CNN bị lừa bởi ánh sáng yếu, kính râm
  - EAR bị lừa khi tài xế quay đầu (méo hình học)
  - MAR bị lừa khi tài xế nói chuyện, ho
  - Head pose bị lừa khi nhìn kính chiếu hậu
  → Kết hợp có trọng số: tín hiệu sai từ nguồn này được bù bởi nguồn khác

Trọng số (FUSION_W_*) có thể chỉnh trong config.py.
Kiến trúc này cho phép thay trọng số cố định bằng MLP nhỏ sau này.
"""

from enum import Enum
import numpy as np


class AlertLevel(Enum):
    GREEN  = "GREEN"    # bình thường
    YELLOW = "YELLOW"   # mệt mỏi nhẹ, cảnh báo sớm
    RED    = "RED"      # nguy hiểm, cảnh báo khẩn cấp


class FatigueFusion:
    """
    Kết hợp 4 tín hiệu thành fatigue_score trong [0, 100].

    Tín hiệu đầu vào (tất cả đã normalize về [0, 1] trước khi truyền vào):
      perclos_val   : PERCLOS (0=không bao giờ nhắm, 1=luôn nhắm)
      ear_score     : 1 - EAR_normalized (0=mắt mở hoàn toàn, 1=mắt đóng)
      mar_score     : MAR_normalized (0=miệng đóng, 1=ngáp to)
      head_score    : head_drowsiness_score (0=ngẩng đầu, 1=cúi sâu)
    """

    def __init__(self,
                 w_perclos: float = 0.40,
                 w_ear:     float = 0.25,
                 w_mar:     float = 0.15,
                 w_head:    float = 0.20,
                 yellow_threshold: int = 30,
                 red_threshold:    int = 60):
        assert abs(w_perclos + w_ear + w_mar + w_head - 1.0) < 1e-6, \
            "Tổng trọng số phải bằng 1.0"
        self.w_perclos = w_perclos
        self.w_ear     = w_ear
        self.w_mar     = w_mar
        self.w_head    = w_head
        self.yellow_th = yellow_threshold
        self.red_th    = red_threshold

        # EMA smoother để làm mịn score giữa các frame (tránh nhấp nháy)
        self._ema_score: float = 0.0
        self._ema_alpha: float = 0.15   # nhỏ = mượt hơn nhưng phản ứng chậm hơn

    def compute(self,
                perclos_val: float,
                ear_score:   float,
                mar_score:   float,
                head_score:  float) -> float:
        """
        Tính fatigue score thô [0, 100].
        Mỗi tín hiệu đã được normalize về [0, 1] bởi caller.
        """
        raw = (self.w_perclos * perclos_val
               + self.w_ear   * ear_score
               + self.w_mar   * mar_score
               + self.w_head  * head_score)

        raw_100 = float(np.clip(raw * 100, 0, 100))

        # EMA để tránh score nhảy đột ngột
        self._ema_score = (self._ema_alpha * raw_100
                           + (1 - self._ema_alpha) * self._ema_score)
        return self._ema_score

    def classify(self, score: float) -> AlertLevel:
        if score >= self.red_th:
            return AlertLevel.RED
        if score >= self.yellow_th:
            return AlertLevel.YELLOW
        return AlertLevel.GREEN

    @property
    def last_score(self) -> float:
        return self._ema_score

    def reset(self) -> None:
        self._ema_score = 0.0


def normalize_ear(ear: float, threshold: float) -> float:
    """
    Chuyển EAR thô thành ear_score [0, 1] dùng cho fusion.
    ear = threshold → score ≈ 0.5 (ranh giới)
    ear = 0         → score = 1.0 (mắt đóng hoàn toàn)
    ear = 2×threshold → score ≈ 0.0 (mắt mở rộng)
    """
    if threshold < 1e-6:
        return 0.0
    # Khoảng cách từ ngưỡng, chuẩn hóa bởi ngưỡng
    score = (threshold - ear) / threshold
    return float(np.clip(score, 0.0, 1.0))


def normalize_mar(mar: float, mar_threshold: float) -> float:
    """Chuyển MAR thô → [0, 1]. Vượt threshold → score > 0.5."""
    if mar_threshold < 1e-6:
        return 0.0
    return float(np.clip(mar / mar_threshold, 0.0, 1.0))
