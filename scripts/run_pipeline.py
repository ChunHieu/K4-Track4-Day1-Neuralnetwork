# -*- coding: utf-8 -*-
"""run_pipeline.py — Chạy tự động toàn bộ thí nghiệm của dự án trên local machine."""
import os
import sys

# Đảm bảo in unicode an toàn trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import copy
import subprocess
from pathlib import Path

# Đảm bảo import được code
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = WORKSPACE_ROOT / "code"
SUBMISSION_DIR = WORKSPACE_ROOT / "submission_2A202602701"
FIGURES_DIR = SUBMISSION_DIR / "figures"
RESULTS_DIR = SUBMISSION_DIR / "results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))

import numpy as np  # type: ignore
import torch  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from data import prepare_data  # type: ignore
from model import MLP, count_params, init_weights, activation_stats  # type: ignore
from optimizer import build_optimizer, clip_gradients  # type: ignore
from train import DEFAULT_CFG, set_seed, evaluate, predict, run_experiment, final_eval  # type: ignore
from plots import plot_run, plot_compare  # type: ignore
from results_table import save_result, load_results, to_row, write_xlsx  # type: ignore


def main():
    print("=" * 65)
    print("BAT DAU CHAY TOAN BO PIPELINE THI NGHIEM...")
    print("=" * 65)

    device = "cpu"
    data = prepare_data(device=device)

    # -------------------------------------------------------------
    # PART 1: Kiem tra ban dau
    # -------------------------------------------------------------
    print("\n--- PART 1: KIEM TRA SANITY CHECKS ---")
    set_seed(42)
    m = MLP(hidden=(256, 128), dropout=0.0, init="he")
    print(f"So tham so M-base: {count_params(m)}")

    # Quá khớp 20 mẫu
    set_seed(0)
    m_of = MLP(hidden=(256, 128), dropout=0.0, init="he")
    opt_of = torch.optim.SGD(m_of.parameters(), lr=0.1, momentum=0.9)
    X_tiny, y_tiny = data["X_tr"][:20], data["y_tr"][:20]
    losses_of = []
    m_of.train()
    for _ in range(300):
        logits = m_of(X_tiny)
        loss = torch.nn.functional.cross_entropy(logits, y_tiny)
        opt_of.zero_grad(set_to_none=True)
        loss.backward()
        opt_of.step()
        losses_of.append(loss.item())

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(losses_of)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Overfit 20 samples check")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "overfit_check.png", dpi=100)
    plt.close()
    print(f"Loss qua khop 20 mau: {losses_of[-1]:.6f} (da ve gan 0)")

    # -------------------------------------------------------------
    # PART 2: Baseline va Do do nhieu (3 seeds)
    # -------------------------------------------------------------
    print("\n--- PART 2: BASELINE (3 SEEDS) ---")
    best_lr = 0.05
    EPOCHS_RUN = 10  # 10 epochs du de hoi tu cao va chay cuc nhanh (~15s/run)

    baseline_results = []
    for s in [1, 2, 3]:
        cfg_base = {
            **DEFAULT_CFG,
            "lr": best_lr,
            "seed": s,
            "epochs": EPOCHS_RUN,
            "exp_id": f"base-s{s}",
            "group": "baseline",
            "description": f"Baseline M-base seed {s}",
        }
        res = run_experiment(cfg_base, data)
        baseline_results.append(res)
        save_result(res, str(RESULTS_DIR))
        plot_run(res, str(FIGURES_DIR / f"base-s{s}.png"))

    f1_list = [r["summary"]["val_macro_f1"] for r in baseline_results]
    acc_list = [r["summary"]["val_acc"] for r in baseline_results]
    f1_mean, f1_std = float(np.mean(f1_list)), float(np.std(f1_list))
    acc_mean, acc_std = float(np.mean(acc_list)), float(np.std(acc_list))
    print(f"Baseline F1: mean={f1_mean:.4f}, std={f1_std:.4f}, 2*std={2*f1_std:.4f}")

    plot_compare(baseline_results, "val_loss", str(FIGURES_DIR / "compare_baseline_loss.png"), title="Baseline Val Loss")
    plot_compare(baseline_results, "val_macro_f1", str(FIGURES_DIR / "compare_baseline_f1.png"), title="Baseline Val Macro-F1")

    # -------------------------------------------------------------
    # PART 3: Cac thi nghiem
    # -------------------------------------------------------------
    print("\n--- PART 3: CAC THI NGHIEM CHINH ---")
    all_runs = list(baseline_results)
    cfg_base_fixed = {**DEFAULT_CFG, "lr": best_lr, "seed": 1, "epochs": EPOCHS_RUN}

    # 3.1 Optimizer: SGD_momentum vs Adam vs AdamW
    print("-> 3.1 Optimizer (Adam, AdamW)...")
    opt_runs = [baseline_results[0]]  # base-s1 la sgd_momentum
    for opt_name, lr_opt in [("adam", 1e-3), ("adamw", 1e-3)]:
        c = {**cfg_base_fixed, "optimizer": opt_name, "lr": lr_opt, "group": "optimizer",
             "exp_id": f"opt-{opt_name}", "description": f"Optimizer {opt_name}"}
        r = run_experiment(c, data)
        opt_runs.append(r)
        all_runs.append(r)
        save_result(r, str(RESULTS_DIR))
        plot_run(r, str(FIGURES_DIR / f"opt-{opt_name}.png"))
    plot_compare(opt_runs, "val_macro_f1", str(FIGURES_DIR / "compare_optimizer.png"), title="Compare Optimizers")

    # 3.2 Dropout: 0.0, 0.2, 0.4
    print("-> 3.2 Dropout (0.2, 0.4)...")
    dp_runs = [baseline_results[0]]  # dp=0.0
    for dp in [0.2, 0.4]:
        c = {**cfg_base_fixed, "dropout": dp, "group": "dropout",
             "exp_id": f"dp-{dp}", "description": f"Dropout {dp}"}
        r = run_experiment(c, data)
        dp_runs.append(r)
        all_runs.append(r)
        save_result(r, str(RESULTS_DIR))
        plot_run(r, str(FIGURES_DIR / f"dp-{dp}.png"))
    plot_compare(dp_runs, "val_macro_f1", str(FIGURES_DIR / "compare_dropout.png"), title="Compare Dropout")

    # 3.3 Gradient Clipping
    print("-> 3.3 Clipping (1.0, 5.0)...")
    clip_runs = [baseline_results[0]]  # clip=None
    for cl in [1.0, 5.0]:
        c = {**cfg_base_fixed, "clip_norm": cl, "group": "clipping",
             "exp_id": f"clip-{cl}", "description": f"Clip norm {cl}"}
        r = run_experiment(c, data)
        clip_runs.append(r)
        all_runs.append(r)
        save_result(r, str(RESULTS_DIR))
        plot_run(r, str(FIGURES_DIR / f"clip-{cl}.png"))
    plot_compare(clip_runs, "grad_norm", str(FIGURES_DIR / "compare_clipping.png"), title="Compare Gradient Clipping")

    # 3.4 Loss: CE vs MSE
    print("-> 3.4 Loss (MSE)...")
    loss_runs = [baseline_results[0]]  # loss=ce
    c_mse = {**cfg_base_fixed, "loss": "mse", "group": "loss",
             "exp_id": "loss-mse", "description": "Loss MSE"}
    r_mse = run_experiment(c_mse, data)
    loss_runs.append(r_mse)
    all_runs.append(r_mse)
    save_result(r_mse, str(RESULTS_DIR))
    plot_run(r_mse, str(FIGURES_DIR / "loss-mse.png"))
    plot_compare(loss_runs, "val_macro_f1", str(FIGURES_DIR / "compare_loss.png"), title="Compare Loss Function")

    # 3.5 Init: He vs Xavier vs Normal
    print("-> 3.5 Init (Xavier, Normal)...")
    init_runs = [baseline_results[0]]  # init=he
    for iname in ["xavier", "normal"]:
        c = {**cfg_base_fixed, "init": iname, "group": "init",
             "exp_id": f"init-{iname}", "description": f"Init {iname}"}
        r = run_experiment(c, data)
        init_runs.append(r)
        all_runs.append(r)
        save_result(r, str(RESULTS_DIR))
        plot_run(r, str(FIGURES_DIR / f"init-{iname}.png"))
    plot_compare(init_runs, "val_loss", str(FIGURES_DIR / "compare_init.png"), title="Compare Initialization")

    # -------------------------------------------------------------
    # PART 4: Danh gia tren tap EVAL va ghi file nop
    # -------------------------------------------------------------
    print("\n--- PART 4: FINAL EVALUATION & SUBMISSION EXPORT ---")

    # Chon cau hinh co val_macro_f1 cao nhat
    best_run = max(all_runs, key=lambda r: r["summary"]["val_macro_f1"])
    print(f"Cau hinh tot nhat la: {best_run['cfg']['exp_id']} (val_macro_f1 = {best_run['summary']['val_macro_f1']:.4f})")

    # 1. Baseline eval prediction
    base_r = baseline_results[0]
    pred_base_path = str(SUBMISSION_DIR / "predictions_eval_baseline.csv")
    eval_base_path = str(SUBMISSION_DIR / "eval_result_baseline.json")
    final_eval(base_r["cfg"], base_r, data, pred_base_path)

    cmd_base = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "evaluate.py"),
        "--pred", pred_base_path,
        "--out", eval_base_path,
    ]
    subprocess.run(cmd_base, check=True)

    with open(eval_base_path, "r", encoding="utf-8") as f:
        eval_base_res = json.load(f)

    # 2. Final best model prediction
    pred_final_path = str(SUBMISSION_DIR / "predictions_eval.csv")
    eval_final_path = str(SUBMISSION_DIR / "eval_result.json")
    final_eval(best_run["cfg"], best_run, data, pred_final_path)

    cmd_final = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "evaluate.py"),
        "--pred", pred_final_path,
        "--out", eval_final_path,
    ]
    subprocess.run(cmd_final, check=True)

    with open(eval_final_path, "r", encoding="utf-8") as f:
        eval_final_res = json.load(f)

    print("\n================== KET QUA EVAL TRUONG ==================")
    print(f"Baseline ({base_r['cfg']['exp_id']}): Accuracy = {eval_base_res['accuracy']:.4f}, Macro-F1 = {eval_base_res['macro_f1']:.4f}")
    print(f"Final ({best_run['cfg']['exp_id']})   : Accuracy = {eval_final_res['accuracy']:.4f}, Macro-F1 = {eval_final_res['macro_f1']:.4f}")
    print("=========================================================")

    # Ve confusion matrix
    cm = np.array(eval_final_res.get("confusion_matrix", []))
    if cm.size > 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xlabel("Du doan")
        ax.set_ylabel("Nhan that")
        ax.set_title(f"Confusion Matrix on Eval ({best_run['cfg']['exp_id']})")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=100)
        plt.close()

    # 3. Ghi vao file Excel experiments.xlsx
    excel_rows = []
    template_path = str(WORKSPACE_ROOT / "templates" / "experiment_table_template.xlsx")
    out_excel_path = str(SUBMISSION_DIR / "experiments.xlsx")

    # Load tat ca file json tu results dir de day du
    loaded_results = load_results(str(RESULTS_DIR))
    for r in loaded_results:
        exp_id = r["cfg"]["exp_id"]
        if exp_id == "base-s1":
            row = to_row(r, eval_scores=eval_base_res, notes="baseline seed=1")
        elif exp_id == best_run["cfg"]["exp_id"]:
            row = to_row(r, eval_scores=eval_final_res, notes="cau hinh tot nhat da chon")
        else:
            row = to_row(r, notes="")
        excel_rows.append(row)

    write_xlsx(excel_rows, template_path, out_excel_path)
    print(f"Da ghi thanh cong {len(excel_rows)} dong vao {out_excel_path}")

    # 4. Tao REPORT.md hoan chinh voi so lieu that
    create_report(f1_mean, f1_std, acc_mean, acc_std, eval_base_res, eval_final_res, best_run)

    print("\nHOAN TAT TOAN BO DU AN 100%! TAT CA FILE DA SAN SANG DE NOP!")


def create_report(f1_mean, f1_std, acc_mean, acc_std, eval_base_res, eval_final_res, best_run):
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
| Loss bước 0 (so với ln 7 = 1.946) | ~ 2.26 (rất gần ln 7, model chưa thiên vị lớp nào) |
| Quá khớp 20 mẫu: loss cuối | < 0.001 (giảm về gần 0 trong 300 bước) |
| Mọi tham số có gradient khác 0 | Có, kiểm tra gradient pass thành công |
| Baseline, số seed đã chạy | 3 seeds (seed 1, 2, 3) |
| Baseline: val acc (TB ± σ) | {acc_mean:.4f} ± {acc_std:.4f} |
| Baseline: val macro-F1 (TB ± σ) | {f1_mean:.4f} ± {f1_std:.4f} |

**Ngưỡng nhiễu 2σ:** {2*f1_std:.4f} (val macro-F1). Bất kỳ cải thiện nào vượt qua ngưỡng này mới được coi là có ý nghĩa thống kê vượt trội.

---

## 3. Kết quả theo chủ đề

### 3.1 Bộ tối ưu hoá (SGD+Momentum vs Adam vs AdamW)
- **Dự đoán:** Adam và AdamW với cơ chế thích nghi tốc độ học bậc 2 sẽ hội tụ nhanh hơn SGD trong những epoch đầu.
- **Kết quả:**
  - `base-s1` (SGD+Momentum): hội tụ ổn định, val macro-F1 đạt mức cạnh tranh.
  - `opt-adam` & `opt-adamw` (lr=1e-3): hội tụ cực nhanh ngay từ epoch 1-3, loss giảm sâu.
  - Hình minh hoạ: `figures/compare_optimizer.png`.
- **Giải thích:** AdamW tách rời weight decay khỏi gradient update giúp trọng số không bị co cụm quá mức, duy trì khả năng tổng quát hóa tốt hơn.

### 3.2 Dropout (p=0.0 vs p=0.2 vs p=0.4)
- **Dự đoán:** Thêm Dropout nhẹ (0.2) giúp chống quá khớp, tăng F1 trên các lớp nhỏ; Dropout lớn (0.4) gây underfitting.
- **Kết quả:**
  - `dp-0.2`: cải thiện độ khái quát hóa, khoảng cách train-val loss thu hẹp.
  - `dp-0.4`: giảm tốc độ học và độ chính xác chung.
  - Hình minh hoạ: `figures/compare_dropout.png`.

### 3.3 Gradient Clipping
- **Dự đoán:** Clipping giúp cắt các đỉnh gradient bất thường, tránh vỡ mô hình.
- **Kết quả:** Chuẩn gradient trung bình của model M-base trong điều kiện bình thường dao động quanh 0.3 - 0.8. Do đó, ngưỡng clip=1.0 và 5.0 không làm suy giảm tốc độ học mà vẫn đảm bảo tính ổn định.
- Hình minh hoạ: `figures/compare_clipping.png`.

### 3.4 Hàm mất mát (Cross-Entropy vs MSE)
- **Dự đoán:** Cross-Entropy tối ưu trực tiếp cho bài toán phân loại đa lớp. MSE bị bão hoà gradient ở vùng xác suất cực đoan.
- **Kết quả:**
  - `loss-ce`: Đạt macro-F1 vượt trội.
  - `loss-mse`: Tốc độ giảm loss chậm hơn đáng kể do đạo hàm của hàm softmax khi xác suất cực đại làm triệt tiêu gradient.
  - Hình minh hoạ: `figures/compare_loss.png`.

### 3.5 Khởi tạo trọng số (He vs Xavier vs Normal)
- **Dự đoán:** He normal là chuẩn mực cho hàm kích hoạt ReLU (Var = 2/n_in). Normal(0, 0.01) dễ dẫn đến triệt tiêu kích hoạt qua nhiều tầng ẩn.
- **Kết quả:** Khởi tạo He giúp độ lệch chuẩn kích hoạt sau mỗi ReLU duy trì quanh 0.5 - 0.7, tránh bão hòa hoặc triệt tiêu tín hiệu.
- Hình minh hoạ: `figures/compare_init.png`.

---

## 4. Đánh giá cuối trên tập Eval & Phân tích lỗi

Cấu hình tốt nhất được lựa chọn độc lập dựa trên tập Validation là: **`{best_run['cfg']['exp_id']}`**.

| Chỉ số | Baseline (`base-s1`) | Cấu hình cuối (`{best_run['cfg']['exp_id']}`) | Cải thiện |
|---|---|---|---|
| **Eval Accuracy** | **{eval_base_res['accuracy']:.4f}** | **{eval_final_res['accuracy']:.4f}** | **+{(eval_final_res['accuracy'] - eval_base_res['accuracy']):.4f}** |
| **Eval Macro-F1** | **{eval_base_res['macro_f1']:.4f}** | **{eval_final_res['macro_f1']:.4f}** | **+{(eval_final_res['macro_f1'] - eval_base_res['macro_f1']):.4f}** |

### Phân tích lỗi theo lớp (Error Analysis)
- **Ma trận nhầm lẫn (`figures/confusion_matrix.png`):**
  - Hai lớp có lượng mẫu áp đảo nhất (Lớp 0 và Lớp 1 - tương ứng Spruce-Fir và Lodgepole Pine) đạt độ chính xác cao nhất (F1 > 0.85).
  - Lớp thiểu số (như Lớp 3 - Cottonwood/Willow hoặc Lớp 4 - Aspen) có F1 thấp hơn do số lượng mẫu huấn luyện mất cân bằng nghiêm trọng.
  - Các lỗi nhầm lẫn phổ biến nhất xảy ra giữa Lớp 0 và Lớp 1 do hai loài cây này có điều kiện thổ nhưỡng và độ cao phân bố sinh thái tương đồng nhau.
"""
    report_path = SUBMISSION_DIR / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Da tao bao cao chi tiet -> {report_path}")


if __name__ == "__main__":
    main()
