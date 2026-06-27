"""
ablation_fusion.py — Ablation study & tối ưu trọng số FatigueFusion.

Mục đích:
  - So sánh 5 cấu hình trọng số: chỉ PERCLOS, chỉ EAR, chỉ Head, full model, và nhiều combo khác
  - Tìm ngưỡng RED/YELLOW tốt nhất qua grid search
  - Xuất bảng kết quả CSV và biểu đồ bar chart

Dữ liệu đầu vào (CSV):
  File CSV có các cột: perclos, ear_score, mar_score, head_score, label
    - perclos/ear_score/mar_score/head_score: float [0,1]
    - label: 0 = tỉnh táo, 1 = mệt mỏi/buồn ngủ

Cách tạo CSV từ session log (chạy main.py và ghi lại):
  Thêm dòng sau vào cuối vòng lặp chính trong main.py:
    import csv
    with open("session_log.csv","a",newline="") as f:
        csv.writer(f).writerow([perclos.value, ear_score, mar_score,
                                 head_score, int(is_drowsy_gt)])

Chạy:
    python ablation_fusion.py --data session_log.csv
    python ablation_fusion.py --data session_log.csv --grid_search
"""

import os
import argparse
import csv
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import f1_score, roc_auc_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn chưa cài. Chỉ tính Accuracy.")

from fatigue_fusion import FatigueFusion, AlertLevel


# ── Các cấu hình trọng số cần thử ─────────────────────────────────────────────
WEIGHT_CONFIGS = {
    "PERCLOS_only":   (1.00, 0.00, 0.00, 0.00),
    "EAR_only":       (0.00, 1.00, 0.00, 0.00),
    "Head_only":      (0.00, 0.00, 0.00, 1.00),
    "No_MAR":         (0.45, 0.30, 0.00, 0.25),
    "No_Head":        (0.50, 0.35, 0.15, 0.00),
    "Equal_weights":  (0.25, 0.25, 0.25, 0.25),
    "Default":        (0.40, 0.25, 0.15, 0.20),   # config.py hiện tại
    "PERCLOS_heavy":  (0.55, 0.20, 0.10, 0.15),
    "EAR_heavy":      (0.30, 0.40, 0.10, 0.20),
}


def load_csv(path: str) -> tuple:
    """Đọc CSV session log, trả về (features_dict, labels)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"File CSV rỗng: {path}")

    perclos   = np.array([float(r["perclos"])   for r in rows])
    ear_score = np.array([float(r["ear_score"]) for r in rows])
    mar_score = np.array([float(r["mar_score"]) for r in rows])
    head_score= np.array([float(r["head_score"])for r in rows])
    labels    = np.array([int(r["label"])        for r in rows])

    print(f"Nạp {len(rows)} mẫu | Buồn ngủ: {labels.sum()} | Tỉnh: {(1-labels).sum()}")
    return {"perclos": perclos, "ear": ear_score, "mar": mar_score, "head": head_score}, labels


def _raw_score(weights: tuple, features: dict, i: int) -> float:
    """
    Tính raw fatigue score [0, 100] KHÔNG qua EMA.
    Dùng cho ablation study vì các mẫu CSV là độc lập (không phải chuỗi thời gian).
    EMA chỉ có ý nghĩa trên video liên tục; reset EMA giữa mỗi mẫu sẽ kéo
    score xuống ~15% giá trị thực (0.15 × raw), không bao giờ vượt ngưỡng RED.
    """
    w_p, w_e, w_m, w_h = weights
    raw = (w_p * features["perclos"][i]
           + w_e * features["ear"][i]
           + w_m * features["mar"][i]
           + w_h * features["head"][i])
    return float(np.clip(raw * 100, 0, 100))


def evaluate_config(features: dict, labels: np.ndarray,
                    weights: tuple, red_th: int = 60) -> dict:
    """Đánh giá một bộ trọng số, trả về dict metrics."""
    preds, scores = [], []
    for i in range(len(labels)):
        score = _raw_score(weights, features, i)
        preds.append(1 if score >= red_th else 0)
        scores.append(score)

    preds  = np.array(preds)
    scores = np.array(scores)

    acc = float((preds == labels).mean())
    tp  = int(((preds == 1) & (labels == 1)).sum())
    fp  = int(((preds == 1) & (labels == 0)).sum())
    fn  = int(((preds == 0) & (labels == 1)).sum())
    tn  = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    result = {"acc": acc, "precision": precision, "recall": recall, "f1": f1,
              "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    if SKLEARN_OK:
        try:
            result["auc"] = roc_auc_score(labels, scores / 100.0)
        except Exception:
            result["auc"] = float("nan")
    return result


def grid_search(features: dict, labels: np.ndarray, out_dir: str):
    """Grid search trên không gian trọng số 4 chiều."""
    print("\nGrid search trọng số (có thể mất vài phút)...")
    step = 0.10
    candidates = np.arange(0.0, 1.0 + step/2, step).round(2)

    best_f1    = -1.0
    best_cfg   = None
    results    = []

    total = 0
    for combo in itertools.product(candidates, repeat=4):
        if abs(sum(combo) - 1.0) > 1e-3:
            continue
        total += 1

    print(f"  Tổng số cấu hình hợp lệ: {total}")
    checked = 0
    for combo in itertools.product(candidates, repeat=4):
        if abs(sum(combo) - 1.0) > 1e-3:
            continue
        m = evaluate_config(features, labels, combo)
        results.append({"weights": combo, **m})
        if m["f1"] > best_f1:
            best_f1  = m["f1"]
            best_cfg = combo
        checked += 1
        if checked % 500 == 0:
            print(f"  {checked}/{total} done...")

    print(f"\nKết quả tốt nhất từ grid search:")
    print(f"  Weights  : PERCLOS={best_cfg[0]}, EAR={best_cfg[1]}, "
          f"MAR={best_cfg[2]}, Head={best_cfg[3]}")
    print(f"  F1       : {best_f1:.4f}")

    # Lưu top-10
    results.sort(key=lambda x: x["f1"], reverse=True)
    out_path = os.path.join(out_dir, "grid_search_top10.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["perclos_w","ear_w","mar_w","head_w","acc","precision","recall","f1","auc"])
        for r in results[:10]:
            w = r["weights"]
            writer.writerow([w[0],w[1],w[2],w[3],
                             f"{r['acc']:.4f}", f"{r['precision']:.4f}",
                             f"{r['recall']:.4f}",  f"{r['f1']:.4f}",
                             f"{r.get('auc', float('nan')):.4f}"])
    print(f"  Top-10 saved -> {out_path}")
    return best_cfg


def plot_comparison(results: dict, out_path: str):
    configs = list(results.keys())
    f1s     = [results[c]["f1"]  for c in configs]
    recalls = [results[c]["recall"] for c in configs]
    precs   = [results[c]["precision"] for c in configs]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(configs)*1.2), 5))
    ax.bar(x - width, f1s,     width, label="F1",        color="steelblue")
    ax.bar(x,         recalls, width, label="Recall",     color="tomato")
    ax.bar(x + width, precs,   width, label="Precision",  color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — FatigueFusion Weight Configurations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Chart saved -> {out_path}")


def print_table(results: dict):
    """In bảng so sánh ra terminal."""
    header = f"{'Config':<18} {'Acc':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'AUC':>6}"
    print("\n" + "="*56)
    print("ABLATION STUDY — FATIGUE FUSION")
    print("="*56)
    print(header)
    print("-"*56)
    for name, m in results.items():
        auc_str = f"{m.get('auc', float('nan')):>6.4f}"
        print(f"{name:<18} {m['acc']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['f1']:>6.4f} {auc_str}")
    print("="*56)


def main():
    parser = argparse.ArgumentParser(description="Ablation study for FatigueFusion")
    parser.add_argument("--data",        default="session_log.csv", help="CSV session log")
    parser.add_argument("--out_dir",     default="ablation_results",  help="Thư mục lưu kết quả")
    parser.add_argument("--grid_search", action="store_true",          help="Bật grid search")
    parser.add_argument("--red_th",      type=int, default=60,         help="Ngưỡng RED (0-100)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        # Tạo dữ liệu giả để demo khi chưa có session log thật
        print(f"[INFO] Không tìm thấy {args.data}. Tạo dữ liệu giả để demo...")
        _generate_demo_csv(args.data)

    os.makedirs(args.out_dir, exist_ok=True)
    features, labels = load_csv(args.data)

    # ── Đánh giá tất cả cấu hình ─────────────────────────────────────────────
    comparison = {}
    for name, weights in WEIGHT_CONFIGS.items():
        comparison[name] = evaluate_config(features, labels, weights, args.red_th)

    print_table(comparison)

    # ── Lưu CSV kết quả ───────────────────────────────────────────────────────
    csv_out = os.path.join(args.out_dir, "ablation_results.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config","acc","precision","recall","f1","auc","tp","fp","fn","tn"])
        for name, m in comparison.items():
            writer.writerow([name, f"{m['acc']:.4f}", f"{m['precision']:.4f}",
                             f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                             f"{m.get('auc', float('nan')):.4f}",
                             m['tp'], m['fp'], m['fn'], m['tn']])
    print(f"\n  CSV saved -> {csv_out}")

    # ── Biểu đồ ──────────────────────────────────────────────────────────────
    plot_comparison(comparison, os.path.join(args.out_dir, "ablation_chart.png"))

    # ── Grid search (tùy chọn) ────────────────────────────────────────────────
    if args.grid_search:
        grid_search(features, labels, args.out_dir)


def _generate_demo_csv(path: str, n: int = 500):
    """Tạo dữ liệu mô phỏng để chạy thử khi chưa có session log thật."""
    import random
    random.seed(42)
    rows = []
    for _ in range(n):
        drowsy = random.random() < 0.35
        if drowsy:
            perclos   = random.gauss(0.35, 0.12)
            ear_score = random.gauss(0.65, 0.15)
            mar_score = random.gauss(0.40, 0.20)
            head_score= random.gauss(0.45, 0.18)
        else:
            perclos   = random.gauss(0.08, 0.06)
            ear_score = random.gauss(0.20, 0.10)
            mar_score = random.gauss(0.10, 0.08)
            head_score= random.gauss(0.10, 0.08)
        rows.append({
            "perclos":   max(0.0, min(1.0, perclos)),
            "ear_score": max(0.0, min(1.0, ear_score)),
            "mar_score": max(0.0, min(1.0, mar_score)),
            "head_score":max(0.0, min(1.0, head_score)),
            "label": int(drowsy),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["perclos","ear_score","mar_score","head_score","label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Demo CSV created: {path} ({n} rows)")


if __name__ == "__main__":
    main()
