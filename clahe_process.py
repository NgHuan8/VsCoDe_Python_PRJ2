# clahe_process.py — Xử lý CLAHE cho một ảnh đơn lẻ (dùng để xuất báo cáo)
#
# Cách dùng:
#   python clahe_process.py --input anh_goc.jpg --output anh_clahe.jpg
#
# Tùy chọn:
#   --clip_limit       Độ giới hạn tương phản (mặc định 2.5, giống main.py)
#   --tile_size         Kích thước lưới chia ảnh (mặc định 8, nghĩa là 8x8)
#   --compare           Xuất thêm ảnh ghép trước/sau để minh họa trong báo cáo

import argparse
import os
import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    """
    Áp dụng CLAHE lên ảnh màu (BGR) bằng cách xử lý trên kênh L của không gian màu LAB.
    Giữ nguyên thông tin màu (kênh a, b), chỉ tăng cường độ sáng/tương phản cục bộ.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_ch_eq = clahe.apply(l_ch)

    merged = cv2.merge((l_ch_eq, a_ch, b_ch))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result


def make_side_by_side(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """Ghép ảnh gốc và ảnh đã xử lý cạnh nhau, có nhãn, để dùng minh họa báo cáo."""
    h, w = original.shape[:2]
    combined = np.hstack((original, processed))

    # cv2.putText(combined, "Original", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    # cv2.putText(combined, "CLAHE", (w + 10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Xử lý CLAHE cho một ảnh đơn lẻ")
    parser.add_argument("--input", required=True, help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--output", required=True, help="Đường dẫn ảnh đầu ra (đã xử lý CLAHE)")
    parser.add_argument("--clip_limit", type=float, default=2.5, help="Clip limit cho CLAHE")
    parser.add_argument("--tile_size", type=int, default=8, help="Kích thước tile (NxN)")
    parser.add_argument("--compare", action="store_true",
                         help="Xuất thêm ảnh ghép trước/sau (file _compare)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Không tìm thấy ảnh đầu vào: {args.input}")

    image = cv2.imread(args.input)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {args.input} (định dạng không hỗ trợ?)")

    processed = apply_clahe(image, clip_limit=args.clip_limit, tile_size=args.tile_size)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(args.output, processed)
    print(f"[OK] Đã lưu ảnh CLAHE → {args.output}")

    if args.compare:
        base, ext = os.path.splitext(args.output)
        compare_path = f"{base}_compare{ext}"
        combined = make_side_by_side(image, processed)
        cv2.imwrite(compare_path, combined)
        print(f"[OK] Đã lưu ảnh so sánh → {compare_path}")


if __name__ == "__main__":
    main()