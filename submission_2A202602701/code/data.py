# -*- coding: utf-8 -*-
"""data.py — Hoàn thiện từ pseudo-code.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np  # type: ignore
import torch  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_data = np.load(f"{processed_dir}/train.npz")
    eval_data  = np.load(f"{processed_dir}/eval.npz")

    X_train_full = train_data["X"]   # float32, (464809, 54)
    y_train_full = train_data["y"]   # int64,   (464809,)

    X_eval       = eval_data["X"]    # float32, (116203, 54)
    y_eval       = eval_data["y"]    # int64,   (116203,)
    eval_row_id  = eval_data["row_id"]  # int64, dùng để ghép predictions_eval.csv

    # Kiểm tra shape và dtype
    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, \
        f"X_train_full shape sai: {X_train_full.shape}"
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, \
        f"X_eval shape sai: {X_eval.shape}"
    assert X_train_full.dtype == np.float32, f"X_train_full phải là float32, có {X_train_full.dtype}"
    assert X_eval.dtype == np.float32, f"X_eval phải là float32, có {X_eval.dtype}"
    assert y_train_full.dtype == np.int64, f"y_train_full phải là int64, có {y_train_full.dtype}"
    assert y_eval.dtype == np.int64, f"y_eval phải là int64, có {y_eval.dtype}"
    assert y_train_full.min() >= 0 and y_train_full.max() <= 6, \
        f"Nhãn train ngoài khoảng 0..6: [{y_train_full.min()}, {y_train_full.max()}]"
    assert y_eval.min() >= 0 and y_eval.max() <= 6, \
        f"Nhãn eval ngoài khoảng 0..6: [{y_eval.min()}, {y_eval.max()}]"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=val_fraction,
        stratify=y,
        random_state=seed,
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    → Tính trên toàn bộ dữ liệu (bao gồm val/eval) là "data leakage":
      mô hình sẽ gián tiếp "biết" thông tin từ tập kiểm tra/validation,
      dẫn đến đánh giá lạc quan và không phản ánh đúng khả năng tổng quát hoá.
    """
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)   # shape (10,)
    std  = X_tr[:, :N_NUMERIC].std(axis=0)    # shape (10,)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_out = X.copy()  # tạo bản sao, không sửa X gốc
    # Xử lý std = 0 bằng cách thay bằng 1 (tránh chia 0; cột hằng số → sau chuẩn hoá vẫn = 0)
    safe_std = np.where(std == 0, 1.0, std)
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / safe_std
    return X_out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    # Bước 1: Nạp dữ liệu
    X_train_full, y_train_full, X_eval_np, y_eval_np, eval_row_id = load_split(processed_dir)

    # Bước 2: Tách validation từ train
    X_tr_np, y_tr_np, X_val_np, y_val_np = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    # Bước 3: Fit standardizer CHỈ trên X_tr (sau khi tách val)
    mean, std = fit_standardizer(X_tr_np)

    # Bước 4: Áp dụng cùng mean/std cho cả 3 tập
    X_tr_np  = apply_standardizer(X_tr_np,  mean, std)
    X_val_np = apply_standardizer(X_val_np, mean, std)
    X_eval_np = apply_standardizer(X_eval_np, mean, std)

    # Bước 5: Chuyển sang tensor và đưa lên device
    def to_tensor_X(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def to_tensor_y(arr):
        return torch.tensor(arr, dtype=torch.int64, device=device)

    X_tr   = to_tensor_X(X_tr_np)
    y_tr   = to_tensor_y(y_tr_np)
    X_val  = to_tensor_X(X_val_np)
    y_val  = to_tensor_y(y_val_np)
    X_eval = to_tensor_X(X_eval_np)
    y_eval = to_tensor_y(y_eval_np)

    # Bước 6: In thông tin kiểm tra
    print("=" * 55)
    print("  Dataset shapes after normalization:")
    print(f"    X_tr   : {tuple(X_tr.shape)},  y_tr   : {tuple(y_tr.shape)}")
    print(f"    X_val  : {tuple(X_val.shape)},  y_val  : {tuple(y_val.shape)}")
    print(f"    X_eval : {tuple(X_eval.shape)},  y_eval : {tuple(y_eval.shape)}")
    print()

    # Kiểm tra chuẩn hoá trên X_tr (10 cột số): mean ≈ 0, std ≈ 1
    x_tr_num = X_tr[:, :N_NUMERIC].cpu().numpy()
    print("  Normalization check (10 numeric features on X_tr):")
    print(f"    mean : {x_tr_num.mean(axis=0).round(4)}")
    print(f"    std  : {x_tr_num.std(axis=0).round(4)}")
    print()

    # Accuracy chiến lược "luôn đoán lớp đa số" trên val
    majority_class = int(torch.bincount(y_tr).argmax())
    majority_acc   = float((y_val == majority_class).float().mean())
    print(f"  Majority class baseline ({majority_class}) on val:")
    print(f"    Accuracy = {majority_acc:.4f}  (baseline ~ 0.4876)")
    print("=" * 55)

    return {
        "X_tr":       X_tr,
        "y_tr":       y_tr,
        "X_val":      X_val,
        "y_val":      y_val,
        "X_eval":     X_eval,
        "y_eval":     y_eval,
        "eval_row_id": eval_row_id,   # numpy array, giữ đúng thứ tự cho predictions_eval.csv
        "mean":       mean,            # numpy array (10,), lưu để có thể dùng lại
        "std":        std,             # numpy array (10,)
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]

    Xử lý batch cuối:
      - Batch cuối có thể nhỏ hơn batch_size (số mẫu không chia hết).
      - Ta GIỮ nguyên batch cuối (drop_last=False) để không bỏ sót mẫu nào.
      - Trong huấn luyện, batch nhỏ hơn làm ước lượng gradient ít chính xác hơn một chút
        nhưng không đáng kể vì mỗi epoch chỉ có 1 batch cuối nhỏ.
    """
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)

    for i in range(0, N, batch_size):
        idx = perm[i : i + batch_size]
        yield X[idx], y[idx]
