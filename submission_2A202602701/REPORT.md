# Báo cáo Lab Day 1 — MSSV: 2A202602701

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
| Baseline: val acc (TB ± σ) | 0.8801 ± 0.0031 |
| Baseline: val macro-F1 (TB ± σ) | 0.7965 ± 0.0061 |

**Ngưỡng nhiễu 2σ:** 0.0121 (val macro-F1). Bất kỳ cải thiện nào vượt qua ngưỡng này mới được coi là có ý nghĩa thống kê thực sự.

---

## 3. Kết quả theo chủ đề

### 3.1 Bộ tối ưu hoá (SGD+Momentum vs Adam vs AdamW)
- **Dự đoán:** Adam và AdamW với cơ chế thích nghi tốc độ học bậc 2 sẽ hội tụ nhanh hơn SGD trong những epoch đầu.
- **Kết quả:**
  - `base-s1` (SGD+Momentum): hội tụ ổn định, val macro-F1 đạt 0.7921.
  - `opt-adam` (lr=1e-3): đạt val macro-F1 cao nhất = 0.8092, hội tụ nhanh vượt trội.
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

Cấu hình tốt nhất được lựa chọn độc lập dựa trên tập Validation là: **`opt-adam`**.

| Chỉ số | Baseline (`base-s1`) | Cấu hình cuối (`opt-adam`) | Cải thiện |
|---|---|---|---|
| **Eval Accuracy** | **0.8768** | **0.8799** | **+0.0031** |
| **Eval Macro-F1** | **0.7898** | **0.8119** | **+0.0221** |

### Phân tích lỗi theo lớp (Error Analysis)
- **Ma trận nhầm lẫn (`figures/confusion_matrix.png`):**
  - Hai lớp có số lượng mẫu lớn nhất (Lớp 0: Spruce-Fir và Lớp 1: Lodgepole Pine) đạt F1 cao nhất (~0.88 - 0.90).
  - Lớp 3 (Cottonwood/Willow) và Lớp 4 (Aspen) có số lượng mẫu ít hơn nhiều trong tập dữ liệu nên F1 đạt từ 0.65 - 0.70.
  - Phần lớn lỗi nhầm lẫn xảy ra giữa Lớp 0 và Lớp 1 do hai lớp này có các đặc trưng địa hình (độ cao, khoảng cách nguồn nước) rất gần nhau.
