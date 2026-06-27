"""
test_units.py — Unit tests cho các module tính toán cốt lõi.

Chạy:
    python -m pytest test_units.py -v
    python test_units.py          # chạy trực tiếp không cần pytest

Bao gồm:
  - helpers: calculate_ear, calculate_mar, crop_eye
  - perclos: PerclosCalculator (giá trị, window, reset)
  - adaptive_threshold: AdaptiveEARThreshold (calibration, percentile)
  - fatigue_fusion: FatigueFusion, normalize_ear, normalize_mar
  - head_pose: HeadPoseEstimator (head_drowsiness_score, is_nodding)
"""

import math
import time
import numpy as np
import sys
import os

# Cho phép import trực tiếp từ thư mục PRJ2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import calculate_ear, calculate_mar, crop_eye
from perclos import PerclosCalculator
from adaptive_threshold import AdaptiveEARThreshold
from fatigue_fusion import FatigueFusion, AlertLevel, normalize_ear, normalize_mar
from head_pose import HeadPoseEstimator


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _assert_close(a, b, tol=1e-5, msg=""):
    assert abs(a - b) <= tol, f"{msg} | expected ≈{b:.6f}, got {a:.6f}"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: calculate_ear
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateEAR:
    def test_open_eye(self):
        """Mắt mở hoàn toàn → EAR ≈ 0.3"""
        pts = [
            np.array([0,   0]),   # 0: left corner
            np.array([20, -10]),  # 1: top-left
            np.array([40, -10]),  # 2: top-right
            np.array([60,  0]),   # 3: right corner
            np.array([40,  10]),  # 4: bottom-right
            np.array([20,  10]),  # 5: bottom-left
        ]
        ear = calculate_ear(pts)
        assert ear > 0.25, f"Mắt mở cần EAR > 0.25, got {ear:.3f}"

    def test_closed_eye(self):
        """Mắt đóng → EAR gần 0"""
        pts = [
            np.array([0,  0]),
            np.array([20, 0]),
            np.array([40, 0]),
            np.array([60, 0]),
            np.array([40, 0]),
            np.array([20, 0]),
        ]
        ear = calculate_ear(pts)
        assert ear < 0.05, f"Mắt đóng cần EAR < 0.05, got {ear:.4f}"

    def test_degenerate_horizontal_width_zero(self):
        """Hai góc mắt trùng nhau → trả về 0.0, không crash"""
        pts = [np.array([0, 0])] * 6
        ear = calculate_ear(pts)
        assert ear == 0.0

    def test_symmetry(self):
        """EAR không đổi khi flip ngang"""
        pts = [
            np.array([0,   0]),
            np.array([20, -8]),
            np.array([40, -8]),
            np.array([60,  0]),
            np.array([40,  8]),
            np.array([20,  8]),
        ]
        flipped = [np.array([-p[0], p[1]]) for p in pts]
        # Sau flip, hoán vị thứ tự left/right corner để giữ đúng định nghĩa
        flipped[0], flipped[3] = flipped[3], flipped[0]
        _assert_close(calculate_ear(pts), calculate_ear(flipped), msg="EAR symmetry")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: calculate_mar
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateMAR:
    def _make_mouth(self, open_px=0):
        """8 điểm miệng đơn giản: miệng đóng hoặc mở open_px pixels."""
        return [
            np.array([0,    0]),   # 0: left corner
            np.array([15, -open_px]),  # 1: top-left
            np.array([30, -open_px]),  # 2: top-center
            np.array([45, -open_px]),  # 3: top-right
            np.array([60,   0]),   # 4: right corner
            np.array([45,  open_px]),  # 5: bottom-right
            np.array([30,  open_px]),  # 6: bottom-center
            np.array([15,  open_px]),  # 7: bottom-left
        ]

    def test_closed_mouth(self):
        mar = calculate_mar(self._make_mouth(0))
        assert mar < 0.05, f"Miệng đóng cần MAR ≈ 0, got {mar:.4f}"

    def test_open_mouth(self):
        mar = calculate_mar(self._make_mouth(20))
        assert mar > 0.3, f"Miệng mở cần MAR > 0.3, got {mar:.4f}"

    def test_mar_increases_with_opening(self):
        mar0 = calculate_mar(self._make_mouth(0))
        mar1 = calculate_mar(self._make_mouth(10))
        mar2 = calculate_mar(self._make_mouth(20))
        assert mar0 < mar1 < mar2, "MAR phải tăng khi miệng mở rộng hơn"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: PerclosCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerclosCalculator:
    def test_all_open(self):
        p = PerclosCalculator(window_seconds=10.0)
        for _ in range(100):
            p.update(False)
        _assert_close(p.value, 0.0, msg="Toàn mở → PERCLOS=0")

    def test_all_closed(self):
        p = PerclosCalculator(window_seconds=10.0)
        for _ in range(100):
            p.update(True)
        _assert_close(p.value, 1.0, msg="Toàn đóng → PERCLOS=1")

    def test_half_closed(self):
        p = PerclosCalculator(window_seconds=999.0)
        for i in range(200):
            p.update(i % 2 == 0)   # xen kẽ
        _assert_close(p.value, 0.5, tol=0.02, msg="Xen kẽ → PERCLOS≈0.5")

    def test_alert_threshold(self):
        p = PerclosCalculator(window_seconds=999.0, alert_threshold=0.20)
        for _ in range(15):
            p.update(True)
        for _ in range(85):
            p.update(False)
        assert p.value < 0.20, "15% closed → không alert"
        assert not p.is_alert

        p2 = PerclosCalculator(window_seconds=999.0, alert_threshold=0.20)
        for _ in range(25):
            p2.update(True)
        for _ in range(75):
            p2.update(False)
        assert p2.value > 0.20, "25% closed → alert"
        assert p2.is_alert

    def test_reset(self):
        p = PerclosCalculator(window_seconds=999.0)
        for _ in range(50):
            p.update(True)
        p.reset()
        _assert_close(p.value, 0.0, msg="Sau reset → PERCLOS=0")
        assert len(p._buffer) == 0

    def test_window_fill_ratio(self):
        p = PerclosCalculator(window_seconds=999.0)
        assert p.window_fill_ratio == 0.0
        p.update(False)
        p.update(False)
        # fill_ratio phải [0,1]
        assert 0.0 <= p.window_fill_ratio <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: AdaptiveEARThreshold
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveEARThreshold:
    def test_fallback_before_calibration(self):
        a = AdaptiveEARThreshold(fallback_threshold=0.22, calibration_frames=10)
        assert not a.calibrated
        _assert_close(a.threshold, 0.22, msg="Trước calibrate dùng fallback")

    def test_calibration_completes(self):
        a = AdaptiveEARThreshold(fallback_threshold=0.22, calibration_frames=10,
                                  percentile_rank=15, min_valid_ear=0.18)
        for _ in range(10):
            a.update(0.30, eyes_are_open=True)
        assert a.calibrated, "Sau 10 mẫu phải calibrate xong"

    def test_percentile_calculation(self):
        a = AdaptiveEARThreshold(fallback_threshold=0.22, calibration_frames=100,
                                  percentile_rank=15, min_valid_ear=0.10)
        baseline = np.linspace(0.15, 0.40, 100)
        for v in baseline:
            a.update(float(v), eyes_are_open=True)
        expected = float(np.percentile(baseline, 15))
        _assert_close(a.threshold, expected, tol=1e-4, msg="Percentile 15 đúng")

    def test_skip_closed_eye_samples(self):
        a = AdaptiveEARThreshold(fallback_threshold=0.22, calibration_frames=5,
                                  percentile_rank=50, min_valid_ear=0.10)
        for _ in range(100):
            a.update(0.10, eyes_are_open=False)   # eyes closed → không thu thập
        assert not a.calibrated, "Closed-eye samples phải bị loại"

    def test_min_valid_ear_filter(self):
        a = AdaptiveEARThreshold(fallback_threshold=0.22, calibration_frames=5,
                                  percentile_rank=50, min_valid_ear=0.18)
        for _ in range(100):
            a.update(0.10, eyes_are_open=True)   # ear < min_valid_ear → bị lọc
        assert not a.calibrated, "EAR < min_valid_ear phải bị lọc"

    def test_threshold_not_below_min_valid(self):
        a = AdaptiveEARThreshold(fallback_threshold=0.22, calibration_frames=10,
                                  percentile_rank=1, min_valid_ear=0.18)
        for _ in range(10):
            a.update(0.19, eyes_are_open=True)
        assert a.threshold >= 0.18, "Ngưỡng không được thấp hơn min_valid_ear"

    def test_reset(self):
        a = AdaptiveEARThreshold(fallback_threshold=0.22, calibration_frames=5)
        for _ in range(5):
            a.update(0.30, eyes_are_open=True)
        a.reset()
        assert not a.calibrated
        _assert_close(a.threshold, 0.22, msg="Sau reset về fallback")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: FatigueFusion & normalize_*
# ═══════════════════════════════════════════════════════════════════════════════

class TestFatigueFusion:
    def test_weight_sum_assertion(self):
        try:
            FatigueFusion(w_perclos=0.5, w_ear=0.5, w_mar=0.1, w_head=0.1)
            assert False, "Phải raise AssertionError khi tổng trọng số != 1"
        except AssertionError:
            pass

    def test_all_zero_inputs(self):
        f = FatigueFusion()
        f._ema_score = 0.0
        score = f.compute(0.0, 0.0, 0.0, 0.0)
        assert score < 5.0, "Toàn 0 → score ≈ 0"

    def test_all_one_inputs(self):
        f = FatigueFusion()
        # Warm up EMA
        for _ in range(50):
            f.compute(1.0, 1.0, 1.0, 1.0)
        score = f.last_score
        assert score > 90.0, f"Toàn 1 → score phải tiến đến 100, got {score:.2f}"

    def test_alert_levels(self):
        f = FatigueFusion(yellow_threshold=30, red_threshold=60)
        assert f.classify(10)  == AlertLevel.GREEN
        assert f.classify(30)  == AlertLevel.YELLOW
        assert f.classify(60)  == AlertLevel.RED
        assert f.classify(100) == AlertLevel.RED

    def test_reset(self):
        f = FatigueFusion()
        for _ in range(20):
            f.compute(1.0, 1.0, 1.0, 1.0)
        f.reset()
        _assert_close(f.last_score, 0.0, msg="Sau reset EMA=0")

    def test_ema_smoothing(self):
        """EMA phải làm mịn: sau 1 bước từ 0 lên max, score chưa đạt 100."""
        f = FatigueFusion()
        f._ema_score = 0.0
        score = f.compute(1.0, 1.0, 1.0, 1.0)
        assert score < 50.0, "EMA alpha=0.15 → sau 1 bước score << 100"


class TestNormalizeFunctions:
    def test_normalize_ear_at_threshold(self):
        """EAR = threshold → score ≈ 0.5 (nhưng clip tại 0)."""
        # Khi ear = threshold: (threshold - ear)/threshold = 0 → score=0 (clipped)
        score = normalize_ear(0.22, 0.22)
        _assert_close(score, 0.0, msg="ear==threshold → score=0 (ranh giới)")

    def test_normalize_ear_closed(self):
        score = normalize_ear(0.0, 0.22)
        _assert_close(score, 1.0, msg="ear=0 → score=1.0")

    def test_normalize_ear_wide_open(self):
        score = normalize_ear(0.44, 0.22)   # 2x threshold
        _assert_close(score, 0.0, msg="ear=2×thr → score=0 (clip)")

    def test_normalize_ear_zero_threshold(self):
        score = normalize_ear(0.22, 0.0)
        _assert_close(score, 0.0, msg="threshold=0 → trả về 0.0 (tránh /0)")

    def test_normalize_mar_zero(self):
        _assert_close(normalize_mar(0.0, 0.60), 0.0, msg="mar=0 → score=0")

    def test_normalize_mar_at_threshold(self):
        _assert_close(normalize_mar(0.60, 0.60), 1.0, msg="mar=thr → score=1")

    def test_normalize_mar_clamped(self):
        score = normalize_mar(1.20, 0.60)   # vượt threshold 2x
        _assert_close(score, 1.0, msg="mar > thr → clip tại 1.0")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: HeadPoseEstimator
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeadPoseEstimator:
    def test_is_nodding_true(self):
        h = HeadPoseEstimator(pitch_alert_deg=20.0)
        assert h.is_nodding(25.0)
        assert not h.is_nodding(15.0)

    def test_is_looking_away(self):
        h = HeadPoseEstimator(yaw_ignore_deg=45.0)
        assert h.is_looking_away(50.0)
        assert h.is_looking_away(-50.0)
        assert not h.is_looking_away(30.0)

    def test_head_drowsiness_score_no_yaw(self):
        h = HeadPoseEstimator(pitch_alert_deg=20.0, yaw_ignore_deg=45.0)
        # pitch=0 → score=0
        _assert_close(h.head_drowsiness_score(0.0, 0.0), 0.0, msg="pitch=0 → score=0")
        # pitch=20° (= alert_deg) → score = 20/(2×20) = 0.5
        _assert_close(h.head_drowsiness_score(20.0, 0.0), 0.5, tol=1e-4, msg="pitch=alert_deg → 0.5")
        # pitch=40° → score=1.0
        _assert_close(h.head_drowsiness_score(40.0, 0.0), 1.0, tol=1e-4, msg="pitch=2×alert → 1.0")

    def test_head_drowsiness_score_ignores_when_yaw_large(self):
        h = HeadPoseEstimator(pitch_alert_deg=20.0, yaw_ignore_deg=45.0)
        score = h.head_drowsiness_score(90.0, 50.0)   # ngoảnh đầu → score=0
        _assert_close(score, 0.0, msg="Yaw lớn → score=0 dù pitch cao")

    def test_head_drowsiness_score_clipped(self):
        h = HeadPoseEstimator(pitch_alert_deg=20.0)
        score = h.head_drowsiness_score(-10.0, 0.0)   # pitch âm = ngẩng đầu
        _assert_close(score, 0.0, tol=1e-4, msg="Pitch âm → clip tại 0")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner đơn giản (không cần pytest)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_all():
    test_classes = [
        TestCalculateEAR,
        TestCalculateMAR,
        TestPerclosCalculator,
        TestAdaptiveEARThreshold,
        TestFatigueFusion,
        TestNormalizeFunctions,
        TestHeadPoseEstimator,
    ]

    total = passed = failed = 0
    failures = []

    for cls in test_classes:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith("test_")]
        for method in methods:
            total += 1
            try:
                getattr(obj, method)()
                passed += 1
                print(f"  PASS  {cls.__name__}.{method}")
            except Exception as e:
                failed += 1
                failures.append(f"{cls.__name__}.{method}: {e}")
                print(f"  FAIL  {cls.__name__}.{method}  →  {e}")

    print(f"\n{'='*55}")
    print(f"Tổng: {total} tests | PASS: {passed} | FAIL: {failed}")
    if failures:
        print("Chi tiết lỗi:")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("Tất cả tests đều PASS.")
    print("="*55)
    return failed == 0


if __name__ == "__main__":
    success = _run_all()
    sys.exit(0 if success else 1)
