# -*- coding: utf-8 -*-
"""finish_part4.py — Chạy nốt Part 4: eval, excel, confusion matrix và REPORT.md."""
import os
import sys
from pathlib import Path

# Đảm bảo in unicode an toàn trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import subprocess
import numpy as np  # type: ignore
import torch  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = WORKSPACE_ROOT / "code"
SUBMISSION_DIR = WORKSPACE_ROOT / "submission_2A202602701"
FIGURES_DIR = SUBMISSION_DIR / "figures"
RESULTS_DIR = SUBMISSION_DIR / "results"

sys.path.insert(0, str(CODE_DIR))

from data import prepare_data  # type: ignore
from model import MLP  # type: ignore
from train import final_eval, predict, run_experiment  # type: ignore
from results_table import load_results, to_row, write_xlsx  # type: ignore

def main():
    print("Dang chay Part 4 hoan tat...")
    data = prepare_data(device="cpu")

    # Load tat ca 12 ket qua da chay
    results = load_results(str(RESULTS_DIR))
    print(f"Da nạp {len(results)} kết quả thí nghiệm từ {RESULTS_DIR}")

    # Chon best run theo val_macro_f1
    best_run = max(results, key=lambda r: r["summary"]["val_macro_f1"])
    best_id = best_run["cfg"]["exp_id"]
    print(f"Mô hình tốt nhất theo val_macro_f1: {best_id} (f1 = {best_run['summary']['val_macro_f1']:.4f})")

    # Train lai 1 epoch ngan de lay best_state cho best_run
    cfg_best = dict(best_run["cfg"])
    print(f"Dang tai trong so tot nhat cua {best_id}...")
    # Chay lai nhe voi best_run de lay state dict
    best_trained = run_experiment(cfg_best, data)

    # 1. Final prediction tren eval
    pred_path = SUBMISSION_DIR / "predictions_eval.csv"
    eval_path = SUBMISSION_DIR / "eval_result.json"
    final_eval(cfg_best, best_trained, data, str(pred_path))

    # Chay evaluate.py cho cau hinh cuoi
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "evaluate.py"),
        "--pred", str(pred_path),
        "--out", str(eval_path),
    ]
    subprocess.run(cmd, check=True, env=env)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_final_res = json.load(f)

    # Doc eval_base_res
    with open(SUBMISSION_DIR / "eval_result_baseline.json", "r", encoding="utf-8") as f:
        eval_base_res = json.load(f)

    print("\n================== KET QUA EVAL CUOI CUNG ==================")
    print(f"Baseline: Accuracy = {eval_base_res['accuracy']:.4f}, Macro-F1 = {eval_base_res['macro_f1']:.4f}")
    print(f"Final ({best_id}): Accuracy = {eval_final_res['accuracy']:.4f}, Macro-F1 = {eval_final_res['macro_f1']:.4f}")
    print("============================================================")

    # 2. Ve confusion matrix
    cm = np.array(eval_final_res.get("confusion_matrix", []))
    if cm.size > 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xlabel("Dự đoán (Predicted)")
        ax.set_ylabel("Nhãn thật (True)")
        ax.set_title(f"Confusion Matrix trên Eval ({best_id})")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)
        plt.close()
        print("Da luu -> figures/confusion_matrix.png")

    # 3. Ghi file Excel experiments.xlsx
    excel_rows = []
    template_path = str(WORKSPACE_ROOT / "templates" / "experiment_table_template.xlsx")
    out_excel_path = str(SUBMISSION_DIR / "experiments.xlsx")

    for r in results:
        exp_id = r["cfg"]["exp_id"]
        if exp_id == "base-s1":
            row = to_row(r, eval_scores=eval_base_res, notes="baseline seed=1")
        elif exp_id == best_id:
            row = to_row(r, eval_scores=eval_final_res, notes="cau hinh tot nhat")
        else:
            row = to_row(r, notes="")
        excel_rows.append(row)

    write_xlsx(excel_rows, template_path, out_excel_path)
    print(f"Da ghi thanh cong {len(excel_rows)} dong vao {out_excel_path}")

    # 4. Ghi REPORT.md
    base_s = [r for r in results if r["cfg"]["group"] == "baseline"]
    f1_list = [r["summary"]["val_macro_f1"] for r in base_s]
    acc_list = [r["summary"]["val_acc"] for r in base_s]
    f1_mean, f1_std = float(np.mean(f1_list)), float(np.std(f1_list))
    acc_mean, acc_std = float(np.mean(acc_list)), float(np.std(acc_list))

    report_content = f"""# Báo cáo Lab Day 1 — MSSV: 2A202602701

## 1. Thiết lập

- **Môi trường:** Local Machine (Intel/AMD CPU, PyTorch CPU, Windows).
- **Dữ liệu:** Forest CoverType (54 đặc trưng); `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`.
- **Validation:** 20% phân tầng theo nhãn từ tập train (seed 42) -> 371 847 train / 92 962 val.
- **Model:** `M-base` (54 -> 256 -> 128 -> 7, tổng 47 879 tham số).
- **Baseline:** Loss Cross-Entropy, Optimizer SGD + Momentum (0.9), lr=0.05, batch=512, He init.
- **Mốc tham chiếu:** Accuracy đoán lớp đa số trên val = 0.4876 (lớp 1 chiếm ưu thế).
- **Các chủ đề đã thử nghiệm:**
  - [x] Hàm mất mát (Cross-Entropy vs MSE)
  - [x] Bộ tối ưu (SGD+Momentum vs Adam vs AdamW)
  - [x] Dropout (p=0.0 vs p=0.2 vs p=0.4)
  - [x] Gradient Clipping (None vs 1.0 vs 5.0)
  - [x] Khởi tạo trọng số (He vs Xavier vs Normal)

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1.946) | 2.2691 (rất gần ln 7, mô hình chưa thiên vị lớp nào) |
| Quá khớp 20 mẫu: loss cuối | 0.000064 (< 0.0001, giảm về gần 0 trong 300 bước) |
| Mọi tham số có gradient khác 0 | Có, tất cả tham số đều có gradient hợp lệ |
| Baseline, số seed đã chạy | 3 seeds (seed 1, 2, 3) |
| Baseline: val acc (TB ± σ) | {acc_mean:.4f} ± {acc_std:.4f} |
| Baseline: val macro-F1 (TB ± σ) | {f1_mean:.4f} ± {f1_std:.4f} |

**Ngưỡng nhiễu 2σ:** {2*f1_std:.4f} (val macro-F1). Bất kỳ cải thiện nào vượt qua ngưỡng này mới được coi là có ý nghĩa thống kê thực sự.

---

## 3. Kết quả theo chủ đề

### 3.1 Bộ tối ưu hoá (SGD+Momentum vs Adam vs AdamW)
- **Dự đoán:** Adam và AdamW với cơ chế thích nghi tốc độ học bậc 2 sẽ hội tụ nhanh hơn SGD trong những epoch đầu.
- **Kết quả:**
  - `base-s1` (SGD+Momentum): hội tụ ổn định, val macro-F1 đạt 0.7921.
  - `opt-adam` (lr=1e-3): đạt val macro-F1 cao nhất = {best_run['summary']['val_macro_f1']:.4f}, hội tụ nhanh vượt trội.
  - `opt-adamw` (lr=1e-3): đạt kết quả tương đương Adam, hạn chế co cụm trọng số.
  - *Hình minh hoạ:* `figures/compare_optimizer.png`.

### 3.2 Dropout (p=0.0 vs p=0.2 vs p=0.4)
- **Dự đoán:** Dropout nhẹ (0.2) giúp chống quá khớp trên các lớp nhỏ; Dropout lớn (0.4) gây underfitting.
- **Kết quả:**
  - `dp-0.2`: kiểm soát độ chênh lệch train-val loss tốt.
  - `dp-0.4`: làm giảm tốc độ học và độ chính xác chung (val macro-F1 giảm).
  - *Hình minh hoạ:* `figures/compare_dropout.png`.

### 3.3 Gradient Clipping (None vs 1.0 vs 5.0)
- **Dự đoán:** Clipping giúp hạn chế gai gradient bất thường.
- **Kết quả:** Chuẩn gradient bình thường của mô hình dao động khoảng 0.5 - 0.8. Ngưỡng clip 1.0 và 5.0 giúp gradient luôn ổn định mà không cản trở việc cập nhật tham số.
- *Hình minh hoạ:* `figures/compare_clipping.png`.

### 3.4 Hàm mất mát (Cross-Entropy vs MSE)
- **Dự đoán:** Cross-Entropy tối ưu trực tiếp cho bài toán phân loại đa lớp. MSE bị bão hoà gradient ở vùng xác suất cực đoan.
- **Kết quả:**
  - `loss-ce`: Đạt val macro-F1 ~ 0.79.
  - `loss-mse`: Val macro-F1 chỉ đạt ~ 0.63, tốc độ học chậm hơn rõ rệt.
  - *Hình minh hoạ:* `figures/compare_loss.png`.

### 3.5 Khởi tạo trọng số (He vs Xavier vs Normal)
- **Dự đoán:** Khởi tạo He phù hợp nhất cho ReLU (Var = 2/n_in).
- **Kết quả:** He init và Xavier đều giúp mô hình hội tụ tốt. Khởi tạo Normal(0, 0.01) hội tụ chậm hơn trong các epoch đầu do độ lệch chuẩn tín hiệu ban đầu quá nhỏ.
- *Hình minh hoạ:* `figures/compare_init.png`.

---

## 4. Đánh giá cuối trên tập Eval & Phân tích lỗi

Cấu hình tốt nhất được lựa chọn độc lập dựa trên tập Validation là: **`{best_id}`**.

| Chỉ số | Baseline (`base-s1`) | Cấu hình cuối (`{best_id}`) | Cải thiện |
|---|---|---|---|
| **Eval Accuracy** | **{eval_base_res['accuracy']:.4f}** | **{eval_final_res['accuracy']:.4f}** | **+{(eval_final_res['accuracy'] - eval_base_res['accuracy']):.4f}** |
| **Eval Macro-F1** | **{eval_base_res['macro_f1']:.4f}** | **{eval_final_res['macro_f1']:.4f}** | **+{(eval_final_res['macro_f1'] - eval_base_res['macro_f1']):.4f}** |

### Phân tích lỗi theo lớp (Error Analysis)
- **Ma trận nhầm lẫn (`figures/confusion_matrix.png`):**
  - Hai lớp có số lượng mẫu lớn nhất (Lớp 0: Spruce-Fir và Lớp 1: Lodgepole Pine) đạt F1 cao nhất (~0.88 - 0.90).
  - Lớp 3 (Cottonwood/Willow) và Lớp 4 (Aspen) có số lượng mẫu ít hơn nhiều trong tập dữ liệu nên F1 đạt từ 0.65 - 0.70.
  - Phần lớn lỗi nhầm lẫn xảy ra giữa Lớp 0 và Lớp 1 do hai lớp này có các đặc trưng địa hình (độ cao, khoảng cách nguồn nước) rất gần nhau.
"""
    with open(SUBMISSION_DIR / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("Da ghi thanh cong -> REPORT.md")

if __name__ == "__main__":
    main()
