"""
evaluate.py — Đánh giá định lượng mô hình CNN phân loại trạng thái mắt.

Chạy:
    python evaluate.py --data_dir data/val
    python evaluate.py --data_dir data/val --model eye_state_classifier.keras

Đầu ra:
    - Confusion matrix (in ra terminal + lưu ảnh PNG)
    - Classification report: Precision / Recall / F1 / Accuracy
    - ROC-AUC score
    - Phân tích lỗi: các ảnh bị phân loại sai (lưu vào error_samples/)
    - Benchmark latency (ms/frame)
"""

import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use("Agg")                       # không cần display server
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)

# ── Cấu hình mặc định ────────────────────────────────────────────────────────
IMG_SIZE          = 48
DEFAULT_MODEL     = "eye_state_classifier.keras"
DEFAULT_LABELS    = "class_indices.json"
DEFAULT_DATA_DIR  = "data/val"
CLOSED_CLASS_NAME = "sleepy"
BATCH_SIZE        = 64
AUTOTUNE          = tf.data.AUTOTUNE


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate eye-state CNN")
    p.add_argument("--data_dir",  default=DEFAULT_DATA_DIR,  help="Thư mục test (có subfolder theo class)")
    p.add_argument("--model",     default=DEFAULT_MODEL,     help="File .keras model")
    p.add_argument("--labels",    default=DEFAULT_LABELS,    help="File class_indices.json")
    p.add_argument("--out_dir",   default="eval_results",    help="Thư mục lưu kết quả")
    p.add_argument("--n_errors",  type=int, default=20,      help="Số ảnh lỗi lưu vào out_dir/errors/")
    return p.parse_args()


def load_dataset(data_dir: str) -> tuple:
    """Nạp toàn bộ ảnh test vào RAM, trả về (X, y, file_paths, class_names)."""
    class_names = sorted(os.listdir(data_dir))
    class_names = [c for c in class_names if os.path.isdir(os.path.join(data_dir, c))]

    X, y, paths = [], [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32")
            X.append(img[..., np.newaxis])       # (48, 48, 1)
            y.append(label_idx)
            paths.append(fpath)

    return np.array(X, dtype="float32"), np.array(y, dtype="int32"), paths, class_names


def benchmark_latency(model, X: np.ndarray, n_warmup: int = 20) -> dict:
    """Đo latency suy luận (ms/ảnh) sau khi warm-up JIT."""
    # Warm-up
    _ = model.predict(X[:n_warmup], batch_size=BATCH_SIZE, verbose=0)

    t0 = time.perf_counter()
    _ = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    elapsed = time.perf_counter() - t0

    per_image_ms = elapsed / len(X) * 1000
    fps          = len(X) / elapsed
    return {"total_s": elapsed, "per_image_ms": per_image_ms, "fps": fps}


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Eye State CNN")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_roc_curve(y_true, y_scores, closed_idx, out_path):
    """Vẽ ROC curve cho class 'sleepy' (closed eye)."""
    # Đảm bảo y_scores là xác suất của class dương (closed)
    probs = y_scores if closed_idx == 1 else (1.0 - y_scores)
    binary_true = (y_true == closed_idx).astype(int)

    fpr, tpr, _ = roc_curve(binary_true, probs)
    auc = roc_auc_score(binary_true, probs)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — 'sleepy' class")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    return auc


def save_error_samples(X, y_true, y_pred, paths, class_names, out_dir, n=20):
    """Lưu n ảnh bị phân loại sai để phân tích lỗi trực quan."""
    os.makedirs(out_dir, exist_ok=True)
    error_indices = np.where(y_true != y_pred)[0]
    np.random.shuffle(error_indices)
    saved = 0
    for idx in error_indices[:n]:
        img = (X[idx, :, :, 0]).astype("uint8")   # (48,48) grayscale
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        src_fname  = os.path.basename(paths[idx])
        out_fname  = f"true_{true_label}__pred_{pred_label}__{src_fname}"
        cv2.imwrite(os.path.join(out_dir, out_fname), img)
        saved += 1
    print(f"  Saved {saved} error samples -> {out_dir}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Nạp model ─────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] Không tìm thấy model: {args.model}")
        return
    print(f"Nạp model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    # ── 2. Đọc label map ──────────────────────────────────────────────────────
    try:
        with open(args.labels, encoding="utf-8") as f:
            class_indices = json.load(f)
        closed_idx = class_indices[CLOSED_CLASS_NAME]
    except Exception as e:
        print(f"[WARN] Không đọc được {args.labels}: {e}. Dùng closed_idx=1.")
        closed_idx = 1

    # ── 3. Nạp dataset test ───────────────────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Không tìm thấy thư mục test: {args.data_dir}")
        return
    print(f"Nạp dataset test từ: {args.data_dir}")
    X, y_true, paths, class_names = load_dataset(args.data_dir)
    print(f"  Tổng số ảnh test : {len(X)}")
    for i, cn in enumerate(class_names):
        print(f"    {cn}: {np.sum(y_true == i)} ảnh")

    if len(X) == 0:
        print("[ERROR] Dataset rỗng.")
        return

    # ── 4. Dự đoán ────────────────────────────────────────────────────────────
    print("Đang dự đoán...")
    raw_preds = model.predict(X, batch_size=BATCH_SIZE, verbose=1).ravel()  # (N,)
    # raw_preds là sigmoid output: xác suất class index 1
    # class index 1 = class_names[1] theo thứ tự alphabetical
    y_scores = raw_preds                        # prob of class index 1
    y_pred   = (raw_preds >= 0.5).astype(int)  # index 0 or 1

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("="*60)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # AUC
    try:
        auc = roc_auc_score((y_true == closed_idx).astype(int),
                            y_scores if closed_idx == 1 else 1 - y_scores)
        print(f"\nROC-AUC (sleepy class): {auc:.4f}")
    except Exception as e:
        auc = float("nan")
        print(f"[WARN] Không tính được AUC: {e}")

    # ── 6. Benchmark latency ──────────────────────────────────────────────────
    print("\nĐo latency inference...")
    lat = benchmark_latency(model, X)
    print(f"  Latency : {lat['per_image_ms']:.3f} ms/ảnh")
    print(f"  FPS max : {lat['fps']:.1f} fps (batch={BATCH_SIZE})")

    # ── 7. Lưu kết quả ────────────────────────────────────────────────────────
    # Confusion matrix ảnh
    plot_confusion_matrix(
        cm, class_names,
        out_path=os.path.join(args.out_dir, "confusion_matrix.png")
    )
    # ROC curve
    plot_roc_curve(
        y_true, y_scores, closed_idx,
        out_path=os.path.join(args.out_dir, "roc_curve.png")
    )
    # Ảnh lỗi
    save_error_samples(
        X, y_true, y_pred, paths, class_names,
        out_dir=os.path.join(args.out_dir, "errors"),
        n=args.n_errors,
    )

    # Lưu số liệu dạng text
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model     : {args.model}\n")
        f.write(f"Dataset   : {args.data_dir}\n")
        f.write(f"N samples : {len(X)}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write(f"\nROC-AUC (sleepy): {auc:.4f}\n")
        f.write(f"\nLatency: {lat['per_image_ms']:.3f} ms/image | {lat['fps']:.1f} fps\n")
    print(f"\n  Saved summary -> {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()
