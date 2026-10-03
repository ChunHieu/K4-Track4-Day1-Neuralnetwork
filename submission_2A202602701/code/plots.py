# -*- coding: utf-8 -*-
"""plots.py — Hoàn thiện từ pseudo-code.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mticker  # type: ignore
import numpy as np  # type: ignore


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính, có nhãn trục và chú thích.
    """
    cfg     = result["cfg"]
    hist    = result["history"]
    summary = result["summary"]

    epochs      = hist["epoch"]
    best_epoch  = summary["best_epoch"]
    exp_id      = cfg["exp_id"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"{exp_id}  |  opt={cfg['optimizer']}  lr={cfg['lr']}  "
        f"batch={cfg['batch']}  epochs={cfg['epochs']}  "
        f"dropout={cfg['dropout']}  init={cfg['init']}",
        fontsize=10, fontweight="bold"
    )

    # ---- Ô 1: Train loss và Val loss ----
    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="train_loss", color="#2196F3", linewidth=1.8)
    ax.plot(epochs, hist["val_loss"],   label="val_loss",   color="#FF5722", linewidth=1.8)
    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best={best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss theo epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ---- Ô 2: Val Accuracy và Val macro-F1 ----
    ax = axes[1]
    ax.plot(epochs, hist["val_acc"],       label="val_acc",       color="#4CAF50", linewidth=1.8)
    ax.plot(epochs, hist["val_macro_f1"],  label="val_macro_f1",  color="#9C27B0", linewidth=1.8, linestyle="--")
    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best={best_epoch}")
    # Đường mốc "đoán đa số"
    ax.axhline(0.4876, color="#FF9800", linestyle=":", linewidth=1.2, label="majority≈0.4876")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Val Accuracy & macro-F1")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)

    # ---- Ô 3: Grad norm ----
    ax = axes[2]
    ax.plot(epochs, hist["grad_norm"], label="grad_norm (trước clip)", color="#795548", linewidth=1.8)
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], color="red", linestyle="--", linewidth=1,
                   label=f"clip_norm={cfg['clip_norm']}")
    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best={best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Grad Norm (L2)")
    ax.set_title("Gradient Norm theo epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Thêm annotation tóm tắt
    best_f1  = summary.get("val_macro_f1", 0) or 0
    best_acc = summary.get("val_acc", 0) or 0
    fig.text(0.5, -0.02,
             f"best_epoch={best_epoch}  |  val_macro_f1={best_f1:.4f}  |  val_acc={best_acc:.4f}",
             ha="center", fontsize=9, color="#555555")

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot -> {path}")


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    # Ánh xạ tên metric sang key trong history
    key_map = {
        "val_loss":      "val_loss",
        "train_loss":    "train_loss",
        "val_acc":       "val_acc",
        "val_macro_f1":  "val_macro_f1",
        "grad_norm":     "grad_norm",
    }
    hist_key = key_map.get(metric, metric)

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))
    for i, result in enumerate(results):
        exp_id = result["cfg"]["exp_id"]
        epochs = result["history"]["epoch"]
        values = result["history"].get(hist_key, [])
        if not values:
            continue
        best_e = result["summary"].get("best_epoch", None)
        ax.plot(epochs, values, label=exp_id, color=colors[i], linewidth=1.8)
        # Đánh dấu best_epoch
        if best_e and 1 <= best_e <= len(values):
            ax.scatter([best_e], [values[best_e - 1]],
                       color=colors[i], marker="*", s=120, zorder=5)

    ylabel_map = {
        "val_loss":     "Validation Loss",
        "train_loss":   "Train Loss",
        "val_acc":      "Validation Accuracy",
        "val_macro_f1": "Validation macro-F1",
        "grad_norm":    "Gradient Norm (L2)",
    }
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel_map.get(metric, metric))
    ax.set_title(title or f"So sánh {metric}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved compare plot -> {path}")
