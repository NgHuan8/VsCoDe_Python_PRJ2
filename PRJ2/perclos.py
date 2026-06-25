# perclos.py
"""
PERCLOS — Percentage of Eye Closure
Tiêu chuẩn đo buồn ngủ của NHTSA (National Highway Traffic Safety Administration).
Định nghĩa: tỷ lệ % thời gian mắt đóng > 80% trong cửa sổ trượt T giây.

Tài liệu gốc: Dinges et al. (1998) "Perclos: A Valid Psychophysiological
Measure of Alertness As Assessed by Psychomotor Vigilance."

Khác với đếm frame liên tiếp (reset khi mắt mở 1 frame),
PERCLOS tính TÍCH LŨY trong cửa sổ dài → không bị lừa bởi chớp mắt.
"""

from collections import deque
import time


class PerclosCalculator:
    """
    Cửa sổ trượt PERCLOS dựa trên thời gian thực (không phụ thuộc FPS ổn định).

    Dùng timestamp thay vì frame count để tránh sai số khi FPS dao động
    (máy yếu, nặng tải → FPS thực < 30, PERCLOS sẽ bị thổi phồng nếu dùng frame).
    """

    def __init__(self, window_seconds: float = 60.0, alert_threshold: float = 0.20):
        """
        window_seconds   : độ dài cửa sổ trượt (giây). NHTSA chuẩn = 60s.
        alert_threshold  : ngưỡng cảnh báo (0.0–1.0). NHTSA chuẩn = 0.20.
        """
        self.window_seconds  = window_seconds
        self.alert_threshold = alert_threshold

        # Mỗi phần tử: (timestamp_float, is_closed_bool)
        self._buffer: deque = deque()

        self._closed_count = 0   # số frame đóng trong buffer — duy trì O(1) update

    def update(self, is_closed: bool) -> None:
        """Gọi mỗi frame với trạng thái mắt hiện tại."""
        now = time.monotonic()
        self._buffer.append((now, is_closed))
        if is_closed:
            self._closed_count += 1

        # Loại bỏ các entry cũ hơn window_seconds
        cutoff = now - self.window_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            _, was_closed = self._buffer.popleft()
            if was_closed:
                self._closed_count -= 1

    @property
    def value(self) -> float:
        """PERCLOS hiện tại trong [0.0, 1.0]. Trả về 0.0 nếu chưa đủ dữ liệu."""
        n = len(self._buffer)
        if n == 0:
            return 0.0
        return self._closed_count / n

    @property
    def is_alert(self) -> bool:
        """True khi PERCLOS vượt ngưỡng cảnh báo."""
        return self.value >= self.alert_threshold

    @property
    def window_fill_ratio(self) -> float:
        """Tỷ lệ cửa sổ đã được lấp đầy (0→1). Dùng để hiển thị tiến trình warm-up."""
        if not self._buffer:
            return 0.0
        elapsed = self._buffer[-1][0] - self._buffer[0][0]
        return min(1.0, elapsed / self.window_seconds)

    def reset(self) -> None:
        """Gọi khi mất khuôn mặt để tránh tích lũy sai."""
        self._buffer.clear()
        self._closed_count = 0
