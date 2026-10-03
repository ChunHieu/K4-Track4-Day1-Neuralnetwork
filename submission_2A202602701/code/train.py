# -*- coding: utf-8 -*-
"""train.py — Hoàn thiện từ pseudo-code.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore

from data import iterate_batches  # type: ignore
from model import MLP, EXPECTED_PARAMS, count_params  # type: ignore
from optimizer import build_optimizer, clip_gradients  # type: ignore

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # được chọn bằng val (chạy thử lr=0.01,0.05,0.1)
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Đảm bảo reproducibility (đánh đổi một chút tốc độ)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    n_classes = cm.shape[0]
    f1_scores = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp   # cột c trừ tp
        fn = cm[c, :].sum() - tp   # hàng c trừ tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    preds = []
    N = len(X)
    for i in range(0, N, batch_size):
        xb = X[i : i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    """
    model.eval()
    N = len(X)
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for i in range(0, N, batch_size):
        xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)

        # loss với reduction="sum" để cộng dồn chính xác
        if loss_name == "ce":
            loss_val = F.cross_entropy(logits, yb, reduction="sum")
        elif loss_name == "mse":
            # MSE: one-hot vs softmax(logits), trung bình trên từng mẫu (sum / n_samples)
            one_hot = F.one_hot(yb, num_classes=7).float()
            probs   = torch.softmax(logits, dim=1)
            loss_val = F.mse_loss(probs, one_hot, reduction="sum")
        else:
            raise ValueError(f"loss '{loss_name}' không hợp lệ. Chọn 'ce' hoặc 'mse'")

        total_loss += loss_val.item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(yb.cpu())

    avg_loss = total_loss / N
    preds_all  = torch.cat(all_preds)
    labels_all = torch.cat(all_labels)

    acc = float((preds_all == labels_all).float().mean())

    # Ma trận nhầm lẫn 7×7
    n_classes = 7
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels_all.numpy(), preds_all.numpy()):
        cm[t, p] += 1

    macro_f1 = macro_f1_from_confusion(cm)

    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1, "cm": cm}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa softmax(logit) và one-hot của y (trung bình trên mẫu và lớp).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)   # mean reduction mặc định
    elif loss_name == "mse":
        one_hot = F.one_hot(y, num_classes=7).float()
        probs   = torch.softmax(logits, dim=1)
        return F.mse_loss(probs, one_hot)   # mean reduction
    else:
        raise ValueError(f"loss '{loss_name}' không hợp lệ. Chọn 'ce' hoặc 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất}
    """
    # ---- Bước 0: Khởi tạo ----
    set_seed(cfg["seed"])

    device = data["X_tr"].device
    X_tr, y_tr     = data["X_tr"],  data["y_tr"]
    X_val, y_val   = data["X_val"], data["y_val"]

    hidden  = tuple(cfg["hidden"])
    model   = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], \
        f"Số tham số sai: {count_params(model)} != {EXPECTED_PARAMS[hidden]}"

    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], momentum=cfg["momentum"],
    )

    # Mixed precision
    precision = cfg.get("precision", "fp32")
    use_amp   = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler    = torch.amp.GradScaler(device=str(device)) if precision == "fp16" else None

    # Generator xáo lô (độc lập với seed global)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"])

    # ---- Bước 1: loss TRƯỚC khi huấn luyện ----
    step0_metrics = evaluate(model, X_val, y_val, cfg["loss"])
    step0_loss    = step0_metrics["loss"]
    print(f"[{cfg['exp_id']}] step0_loss = {step0_loss:.4f}  (expected ~ {np.log(7):.4f})")

    # ---- Lịch sử ----
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_acc": [], "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []
    }
    best_val_loss   = float("inf")
    best_epoch      = 0
    best_state      = None
    diverged        = False

    # Tập con cố định để đo train_loss nhanh (~50 000 mẫu cuối mỗi epoch)
    EVAL_TRAIN_N = min(50_000, len(X_tr))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # ---- Bước 2: Vòng lặp huấn luyện ----
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_grad_norms = []
        t_start = time.perf_counter()

        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            # Forward + loss
            if use_amp:
                with torch.autocast(device_type=str(device).split(":")[0], dtype=amp_dtype):
                    logits = model(xb)
                    loss   = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss   = compute_loss(logits, yb, cfg["loss"])

            # Kiểm tra diverge
            if not torch.isfinite(loss):
                print(f"  [WARN] loss = {loss.item()} at epoch {epoch}, early stopping.")
                diverged = True
                break

            optimizer.zero_grad(set_to_none=True)

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
                # Unscale trước khi clip (bắt buộc khi dùng fp16 + clip)
                if cfg.get("clip_norm") is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
                optimizer.step()

            epoch_grad_norms.append(gn)

        if diverged:
            break

        # ---- Cuối epoch: đánh giá ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_epoch = time.perf_counter() - t_start

        # Train loss trên tập con cố định (50 000 mẫu đầu sau khi sắp xếp)
        tr_metrics  = evaluate(model, X_tr[:EVAL_TRAIN_N], y_tr[:EVAL_TRAIN_N], cfg["loss"])
        val_metrics = evaluate(model, X_val, y_val, cfg["loss"])

        avg_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(t_epoch)

        # Lưu best state
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch    = epoch
            best_state    = copy.deepcopy(model.state_dict())

        print(f"  epoch {epoch:2d}/{cfg['epochs']} | "
              f"tr_loss={tr_metrics['loss']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} | "
              f"val_f1={val_metrics['macro_f1']:.4f} | "
              f"gn={avg_gn:.3f} | "
              f"t={t_epoch:.1f}s")

    # ---- Bước 3: Tổng hợp summary ----
    peak_mem_MB = 0.0
    if torch.cuda.is_available():
        peak_mem_MB = torch.cuda.max_memory_allocated(device) / 1024**2

    # Lấy val_acc và val_macro_f1 tại best_epoch
    best_idx = best_epoch - 1 if best_epoch > 0 else -1
    summary = {
        "step0_loss":        step0_loss,
        "best_val_loss":     best_val_loss,
        "best_epoch":        best_epoch,
        "final_train_loss":  history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss":    history["val_loss"][-1]   if history["val_loss"]   else None,
        "val_acc":           history["val_acc"][best_idx]       if history["val_acc"]       else None,
        "val_macro_f1":      history["val_macro_f1"][best_idx]  if history["val_macro_f1"]  else None,
        "time_per_epoch_s":  float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB":       round(peak_mem_MB, 1),
        "diverged":          diverged,
        # eval_acc / eval_macro_f1 điền sau khi chạy evaluate.py
        "eval_acc":          None,
        "eval_macro_f1":     None,
    }

    print(f"[{cfg['exp_id']}] DONE | best_epoch={best_epoch} | "
          f"val_f1={summary['val_macro_f1']:.4f} | val_acc={summary['val_acc']:.4f}")

    return {
        "cfg":        cfg,
        "history":    history,
        "summary":    summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    import pandas as pd
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} predictions -> {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    device = data["X_eval"].device
    model  = MLP(
        hidden=tuple(cfg["hidden"]),
        dropout=cfg["dropout"],
        init=cfg["init"],
    ).to(device)
    model.load_state_dict(result["best_state"])

    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"Run: python scripts/evaluate.py --pred {pred_path} to evaluate")
