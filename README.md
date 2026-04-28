# titans-torch

> A PyTorch implementation of **"Titans: Learning to Memorize at Test Time"**
> — Behrouz, Zhong & Daliri, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663), 2025.

[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D2.0-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)

Titans is a hybrid architecture combining **short-term attention** with a **Neural Long-Term Memory (LMM)** module — a learnable MLP-based memory that is updated at test time via inner-loop gradient descent, enabling the model to adapt to new sequences without retraining.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [NeuralLongTermMemory](#neurallongtermmemory)
  - [PersistentMemory](#persistentmemory)
  - [TitansMAC](#titansmac)
  - [TitansMAG](#titansmag)
  - [TitansMAL](#titansmal)
- [Training Examples](#training-examples)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Features

- ✅ **Neural Long-Term Memory (LMM)** with differentiable inner-loop — fully end-to-end trainable
- ✅ **Test-time adaptation** — sequential memory updates per token without retaining the computation graph
- ✅ **Three Titans variants**: MAC · MAG · MAL
- ✅ **Persistent Memory** — learnable token bank prepended to the input sequence
- ✅ **Data-dependent coefficients** (α, θ, η) for adaptive learning rate and momentum control
- ✅ Optional **1D depthwise convolution** on keys, values, and queries
- ✅ Fully configurable `num_heads` and `dropout`

---

## Installation

**Requirements:** Python ≥ 3.9 · PyTorch ≥ 2.0.0

```bash
# Install from source
git clone https://github.com/buiquangdat1710/titans-torch.git
cd titans-torch
pip install -e .
```

```bash
# Install from PyPI (once published)
pip install titans-torch
```

---

## Architecture Overview

```
                        ┌─────────────────────────────────────┐
                        │        Neural Long-Term Memory       │
                        │                                      │
                        │   W_K, W_V, W_Q  (+ optional conv)  │
                        │               ↓                      │
                        │          Memory MLP                  │
                        │       M(k) ≈ v  (associative)        │
                        │               ↓                      │
                        │    Inner-loop gradient descent       │
                        │    S_t = η·S_{t-1} − θ·∇L           │
                        │    W_t = (1−α)·W_{t-1} + S_t        │
                        └─────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │   MAC    │    │   MAG    │    │   MAL    │
  │          │    │          │    │          │
  │ persist  │    │ persist  │    │ persist  │
  │    ‖     │    │    +     │    │    +     │
  │ h_t=M(q) │    │  SWAttn  │    │   LMM    │
  │    ‖     │    │  ──┬──   │    │    ↓     │
  │ segment  │    │  gate    │    │  SWAttn  │
  │    ↓     │    │  ──┴──   │    │          │
  │  Attn    │    │   LMM    │    │          │
  │    ↓     │    │          │    │          │
  │  y⊗M(y)  │    │   out    │    │   out    │
  └──────────┘    └──────────┘    └──────────┘
  Memory as       Memory as       Memory as
  Context         Gate            Layer
```

---

## Quick Start

```python
import torch
from titans_torch import TitansMAC, TitansMAG, TitansMAL, NeuralLongTermMemory

batch, seq_len, dim = 2, 16, 64
x = torch.randn(batch, seq_len, dim)

# Memory as a Context
model = TitansMAC(input_dim=dim, num_heads=4, dropout=0.1)
out = model(x, return_all=True)    # → (2, 16, 64)

# Memory as a Gate
model = TitansMAG(input_dim=dim, num_heads=4, dropout=0.0)
out = model(x, return_all=True)    # → (2, 16, 64)

# Memory as a Layer
model = TitansMAL(input_dim=dim, num_heads=4, dropout=0.1)
out = model(x, return_all=True)    # → (2, 16, 64)

# Standalone LMM — test-time adaptation
lmm = NeuralLongTermMemory(input_dim=dim)
out = lmm.test_time_update(x, return_all_outputs=True)   # → (2, 16, 64)
```

---

## API Reference

### `NeuralLongTermMemory`

A standalone Neural Long-Term Memory module with inner-loop gradient-based updates.

```python
NeuralLongTermMemory(
    input_dim: int,
    mem_dim: int = None,       # defaults to input_dim
    num_layers: int = 2,
    hidden_dim: int = 32,
    use_conv: bool = True,
    conv_kernel: int = 3,
)
```

| Method | Description |
|--------|-------------|
| `forward_trainable(x, chunk_size=4, return_all_outputs=False)` | Differentiable inner-loop; used during outer-loop training |
| `forward(x, return_all_outputs=False)` | Pure inference; memory weights are not updated |
| `test_time_update(x, return_all_outputs=False)` | Sequential test-time update; no computation graph retained |

**Inner-loop update rule:**

```
S_t  =  η_t · S_{t-1}  −  θ_t · ∇L(W; k_t, v_t)
W_t  =  (1 − α_t) · W_{t-1}  +  S_t
```

The coefficients α, θ, and η are data-dependent, generated by lightweight MLPs conditioned on the input.

---

### `PersistentMemory`

A bank of learnable tokens prepended to the input sequence before attention.

```python
PersistentMemory(
    num_tokens: int,
    dim: int,
    init_scale: float = 0.02,
)
```

```python
pm = PersistentMemory(num_tokens=8, dim=64)
x_extended = pm(x)   # → (batch, 8 + seq_len, 64)
```

---

### `TitansMAC`

**Memory as a Context** — retrieves historical information from LMM and injects it into the attention context.

```python
TitansMAC(
    input_dim: int,
    num_persistent: int = 8,
    mem_dim: int = None,
    num_memory_layers: int = 2,
    hidden_memory_dim: int = 32,
    chunk_size: int = 4,
    use_conv: bool = True,
    num_heads: int = 4,
    dropout: float = 0.1,
)
```

**Forward flow (per segment):**

```
q      = W_Q(S^(t))
h_t    = M_{t-1}*(q)                        # retrieve from LMM
input  = [persistent ‖ h_t ‖ S^(t)]
y_t    = CausalAttention(input)
M_t    = update(M_{t-1}, y_t)               # forward_trainable
o_t    = y_t ⊗ M_t*(y_t)                    # Hadamard gating
```

---

### `TitansMAG`

**Memory as a Gate** — combines sliding-window attention and LMM outputs through a learned gate.

```python
TitansMAG(
    input_dim: int,
    num_persistent: int = 8,
    mem_dim: int = None,
    num_memory_layers: int = 2,
    hidden_memory_dim: int = 32,
    window_size: int = 4,
    use_conv: bool = True,
    num_heads: int = 4,
    dropout: float = 0.1,
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

**Memory as a Layer** — LMM processes the input first; its output is then passed to sliding-window attention.

```python
TitansMAL(
    input_dim: int,
    num_persistent: int = 4,
    mem_dim: int = None,
    num_memory_layers: int = 2,
    hidden_memory_dim: int = 32,
    window_size: int = 4,
    use_conv: bool = True,
    num_heads: int = 4,
    dropout: float = 0.1,
)
```

**Forward flow:**

```
x_persist = [persistent ‖ x]
LMM.forward_trainable(x_persist)            # update memory state
out = SlidingWindowAttention(x_persist)     # attend over full context
```

---

## Training Examples

**Standard training loop:**

```python
import torch
import torch.nn.functional as F
from titans_torch import TitansMAL

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

**Test-time adaptation with LMM:**

```python
from titans_torch import NeuralLongTermMemory

lmm = NeuralLongTermMemory(input_dim=64)
lmm.eval()

x_new = torch.randn(2, 128, 64)

# Update memory at test time — no gradients required
with torch.no_grad():
    out = lmm.test_time_update(x_new, return_all_outputs=True)
```

---

## Project Structure

```
titans-torch/
├── __init__.py        # Public exports
├── memory.py          # NeuralLongTermMemory, PersistentMemory
├── mac.py             # TitansMAC (Memory as Context)
├── mag.py             # TitansMAG (Memory as Gate)
├── mal.py             # TitansMAL (Memory as Layer)
setup.py
README.md
MANIFEST.in
CHANGELOG.txt
LICENSE.txt
```

---

## Citation

If you use this code in your research, please cite the original paper:

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

Released under the **MIT License** — see [`LICENSE.txt`](LICENSE.txt) for details.
