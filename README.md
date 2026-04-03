# titans-torch

> PyTorch implementation of **"Titans: Learning to Memorize at Test Time"**
> — Behrouz, Zhong & Daliri, arXiv:2501.00663, 2025.

Titans là kiến trúc kết hợp **short-term attention** và **Neural Long-Term Memory (LMM)** — một module bộ nhớ dạng MLP có thể được cập nhật tại test time thông qua inner-loop gradient descent, cho phép mô hình thích nghi với chuỗi mới mà không cần huấn luyện lại.

---

## Mục lục

- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Sử dụng nhanh](#sử-dụng-nhanh)
- [API Reference](#api-reference)
  - [NeuralLongTermMemory](#neurallongtermmemory)
  - [PersistentMemory](#persistentmemory)
  - [TitansMAC](#titansmac)
  - [TitansMAG](#titansmag)
  - [TitansMAL](#titansmal)
- [Ví dụ training](#ví-dụ-training)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Tham khảo](#tham-khảo)
- [License](#license)

---

## Tính năng

- ✅ **Neural Long-Term Memory (LMM)** với inner-loop differentiable — huấn luyện end-to-end
- ✅ **Test-time adaptation** — cập nhật memory theo từng token mà không cần giữ computation graph
- ✅ **3 kiến trúc Titans**: MAC · MAG · MAL
- ✅ **Persistent Memory** — learnable token bank prepend vào sequence
- ✅ **Data-dependent coefficients** (α, θ, η) điều khiển learning rate và momentum
- ✅ **1D depthwise conv** tùy chọn trên K/V/Q
- ✅ `num_heads` và `dropout` hoàn toàn cấu hình được

---

## Cài đặt

**Yêu cầu:** Python ≥ 3.9 · PyTorch ≥ 2.0.0

```bash
# Cài từ source
git clone https://github.com/buiquangdat1710/titans-torch.git
cd titans-pytorch
pip install -e .
```

```bash
# Hoặc khi đã publish lên PyPI
pip install titans-torch
```

---

## Kiến trúc tổng quan

```
                        ┌─────────────────────────────────────┐
                        │        Neural Long-Term Memory       │
                        │                                      │
                        │   W_K, W_V, W_Q  (+ optional conv)  │
                        │           ↓                          │
                        │       Memory MLP                     │
                        │     M(k) ≈ v  (associative)         │
                        │           ↓                          │
                        │   Inner-loop gradient descent        │
                        │   S_t = η·S_{t-1} − θ·∇L            │
                        │   W_t = (1−α)·W_{t-1} + S_t         │
                        └─────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  MAC     │    │  MAG     │    │  MAL     │
  │          │    │          │    │          │
  │ persist  │    │ persist  │    │ persist  │
  │    ‖     │    │    +     │    │    +     │
  │ h_t=M(q) │    │  SWAttn  │    │   LMM    │
  │    ‖     │    │  ──┬──   │    │    ↓     │
  │ segment  │    │  gate    │    │  SWAttn  │
  │    ↓     │    │  ──┴──   │    │          │
  │  Attn    │    │   LMM    │    │          │
  │    ↓     │    │          │    │          │
  │  y⊗M(y)  │    │  out     │    │   out    │
  └──────────┘    └──────────┘    └──────────┘
  Memory as       Memory as       Memory as
  Context         Gate            Layer
```

---

## Sử dụng nhanh

```python
import torch
from titans-torch import TitansMAC, TitansMAG, TitansMAL, NeuralLongTermMemory

batch, seq_len, dim = 2, 16, 64
x = torch.randn(batch, seq_len, dim)

# Memory as a Context
model = TitansMAC(input_dim=dim, num_heads=4, dropout=0.1)
out = model(x, return_all=True)    # (2, 16, 64)

# Memory as a Gate
model = TitansMAG(input_dim=dim, num_heads=4, dropout=0.0)
out = model(x, return_all=True)    # (2, 16, 64)

# Memory as a Layer
model = TitansMAL(input_dim=dim, num_heads=4, dropout=0.1)
out = model(x, return_all=True)    # (2, 16, 64)

# Standalone LMM — test-time update
lmm = NeuralLongTermMemory(input_dim=dim)
out = lmm.test_time_update(x, return_all_outputs=True)   # (2, 16, 64)
```

---

## API Reference

### `NeuralLongTermMemory`

```python
NeuralLongTermMemory(
    input_dim: int,
    mem_dim: int = None,       # mặc định = input_dim
    num_layers: int = 2,
    hidden_dim: int = 32,
    use_conv: bool = True,
    conv_kernel: int = 3,
)
```

| Method | Mô tả |
|--------|-------|
| `forward_trainable(x, chunk_size=4, return_all_outputs=False)` | Inner-loop differentiable, dùng cho outer-loop training |
| `forward(x, return_all_outputs=False)` | Inference thuần túy, không cập nhật memory |
| `test_time_update(x, return_all_outputs=False)` | Cập nhật tuần tự tại test time, không giữ computation graph |

**Inner-loop update rule:**

```
S_t  =  η_t · S_{t-1}  −  θ_t · ∇L(W; k_t, v_t)
W_t  =  (1 − α_t) · W_{t-1}  +  S_t
```

Các hệ số α, θ, η được tạo ra từ dữ liệu (data-dependent) qua các MLP nhỏ.

---

### `PersistentMemory`

```python
PersistentMemory(
    num_tokens: int,
    dim: int,
    init_scale: float = 0.02,
)
```

Prepend `num_tokens` learnable token vào đầu sequence trước khi đưa vào attention.

```python
pm = PersistentMemory(num_tokens=8, dim=64)
x_extended = pm(x)   # (batch, 8 + seq_len, 64)
```

---

### `TitansMAC`

**Memory as a Context** — truy xuất thông tin lịch sử từ LMM, đưa vào context của attention.

```python
TitansMAC(
    input_dim: int,
    num_persistent: int = 8,
    mem_dim: int = None,
    num_memory_layers: int = 2,
    hidden_memory_dim: int = 32,
    chunk_size: int = 4,
    use_conv: bool = True,
    num_heads: int = 4,       # số attention head
    dropout: float = 0.1,     # dropout rate
)
```

**Forward flow (mỗi segment):**

```
q = W_Q(S^(t))
h_t = M_{t-1}*(q)                          # retrieve từ LMM
input = [persistent ‖ h_t ‖ S^(t)]
y_t = CausalAttention(input)
M_t = update(M_{t-1}, y_t)                 # forward_trainable
o_t = y_t ⊗ M_t*(y_t)                      # Hadamard gating
```

---

### `TitansMAG`

**Memory as a Gate** — kết hợp sliding-window attention và LMM qua learned gate.

```python
TitansMAG(
    input_dim: int,
    num_persistent: int = 8,
    mem_dim: int = None,
    num_memory_layers: int = 2,
    hidden_memory_dim: int = 32,
    window_size: int = 4,
    use_conv: bool = True,
    num_heads: int = 4,       # số attention head
    dropout: float = 0.1,     # dropout rate
)
```

**Forward flow:**

```
attn_out   = SlidingWindowAttention([persistent ‖ x])
memory_out = LMM.forward_trainable(x)
gate       = sigmoid([attn_out ‖ memory_out] · W)
out        = gate ⊙ memory_out + (1 − gate) ⊙ attn_out
```

---

### `TitansMAL`

**Memory as a Layer** — LMM xử lý trước, kết quả đưa vào sliding-window attention.

```python
TitansMAL(
    input_dim: int,
    num_persistent: int = 4,
    mem_dim: int = None,
    num_memory_layers: int = 2,
    hidden_memory_dim: int = 32,
    window_size: int = 4,
    use_conv: bool = True,
    num_heads: int = 4,       # số attention head
    dropout: float = 0.1,     # dropout rate
)
```

**Forward flow:**

```
x_persist = [persistent ‖ x]
LMM.forward_trainable(x_persist)           # update memory
out = SlidingWindowAttention(x_persist)    # attention trên full context
```

---

## Ví dụ training

```python
import torch
import torch.nn.functional as F
from titans-torch import TitansMAL

model = TitansMAL(
    input_dim=64,
    num_persistent=4,
    num_heads=4,
    dropout=0.1,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    x      = torch.randn(4, 32, 64)
    target = torch.randn(4, 32, 64)

    optimizer.zero_grad()
    out  = model(x, return_all=True)
    loss = F.mse_loss(out, target)
    loss.backward()
    optimizer.step()

    if (step + 1) % 20 == 0:
        print(f"Step {step+1:3d}  loss={loss.item():.4f}")
```

**Test-time adaptation với LMM:**

```python
from titans-torch import NeuralLongTermMemory

lmm = NeuralLongTermMemory(input_dim=64)
lmm.eval()

x_new = torch.randn(2, 128, 64)
# Cập nhật memory tại test time — không cần gradient
with torch.no_grad():
    out = lmm.test_time_update(x_new, return_all_outputs=True)
```

---

## Cấu trúc thư mục

```
titans-torch/
├── __init__.py        # Public exports
├── memory.py          # NeuralLongTermMemory · PersistentMemory
├── mac.py          # TitansMAC 
├── mag.py          # TitansMAG 
├── mal.py          # TitansMAL 

setup.py
README.md
MANIFEST.in
CHANGELOG.txt
LICENSE.txt
```

---

## Tham khảo

```bibtex
@article{behrouz2025titans,
  title   = {Titans: Learning to Memorize at Test Time},
  author  = {Behrouz, Ali and Zhong, Peilin and Daliri, Majid},
  journal = {arXiv preprint arXiv:2501.00663},
  year    = {2025},
  url     = {https://arxiv.org/abs/2501.00663}
}
```

---

## License

Phát hành theo giấy phép **MIT** — xem [`LICENSE.txt`](LICENSE.txt) để biết thêm chi tiết.