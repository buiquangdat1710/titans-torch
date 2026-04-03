import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from typing import Tuple, Optional


class NeuralLongTermMemory(nn.Module):
    """
    Neural Long-Term Memory (LMM) với inner-loop differentiable.
    Hỗ trợ training end-to-end (outer loop) và test-time update.
    """

    def __init__(
        self,
        input_dim: int,
        mem_dim: Optional[int] = None,
        num_layers: int = 2,
        hidden_dim: int = 32,
        use_conv: bool = True,
        conv_kernel: int = 3,
        coeff_hidden_dim: int = 16,      # hidden size của alpha/theta/eta nets
        chunk_size: int = 4,             # kích thước chunk mặc định cho forward_trainable
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mem_dim = mem_dim if mem_dim is not None else input_dim
        self.num_layers = num_layers
        self.use_conv = use_conv
        self.chunk_size = chunk_size

        # Projections
        self.W_K = nn.Linear(input_dim, self.mem_dim, bias=False)
        self.W_V = nn.Linear(input_dim, self.mem_dim, bias=False)
        self.W_Q = nn.Linear(input_dim, self.mem_dim, bias=False)

        # Optional depthwise 1D convolutions
        if use_conv:
            self.conv_k = nn.Conv1d(
                self.mem_dim, self.mem_dim, conv_kernel,
                padding=conv_kernel // 2, groups=self.mem_dim
            )
            self.conv_v = nn.Conv1d(
                self.mem_dim, self.mem_dim, conv_kernel,
                padding=conv_kernel // 2, groups=self.mem_dim
            )
            self.conv_q = nn.Conv1d(
                self.mem_dim, self.mem_dim, conv_kernel,
                padding=conv_kernel // 2, groups=self.mem_dim
            )

        # Deep memory MLP
        layers = []
        in_dim = self.mem_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else self.mem_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.memory_mlp = nn.Sequential(*layers)

        # Data‑dependent coefficient nets (dùng coeff_hidden_dim thay vì 16 cứng)
        self.alpha_net = nn.Sequential(
            nn.Linear(input_dim, coeff_hidden_dim),
            nn.SiLU(),
            nn.Linear(coeff_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.theta_net = nn.Sequential(
            nn.Linear(input_dim, coeff_hidden_dim),
            nn.SiLU(),
            nn.Linear(coeff_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.eta_net = nn.Sequential(
            nn.Linear(input_dim, coeff_hidden_dim),
            nn.SiLU(),
            nn.Linear(coeff_hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(self.mem_dim)

    def _get_coeffs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = self.alpha_net(x).squeeze(-1)
        theta = self.theta_net(x).squeeze(-1)
        eta = self.eta_net(x).squeeze(-1)
        return alpha, theta, eta

    def _apply_conv(self, conv_layer, x_seq):
        x_conv = x_seq.transpose(1, 2)
        x_conv = conv_layer(x_conv)
        return x_conv.transpose(1, 2)

    def _get_params_flat(self) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in self.memory_mlp.parameters()])

    def _set_params_flat(self, flat_params: torch.Tensor):
        offset = 0
        for p in self.memory_mlp.parameters():
            numel = p.numel()
            p.data = flat_params[offset:offset + numel].reshape(p.shape)
            offset += numel

    def _loss_fn(self, flat_params: torch.Tensor, kt: torch.Tensor, vt: torch.Tensor) -> torch.Tensor:
        offset = 0
        params_dict = {}
        for name, p in self.memory_mlp.named_parameters():
            numel = p.numel()
            params_dict[name] = flat_params[offset:offset + numel].reshape(p.shape)
            offset += numel
        out = functional_call(self.memory_mlp, params_dict, kt)
        return ((out - vt) ** 2).mean()

    def forward_trainable(
        self,
        x: torch.Tensor,
        chunk_size: Optional[int] = None,
        return_all_outputs: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass với inner-loop differentiable (dùng trong training).
        chunk_size: nếu None thì dùng giá trị mặc định từ __init__.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        batch, seq_len, _ = x.shape
        device = x.device

        k = self.W_K(x)
        v = self.W_V(x)
        q = self.W_Q(x)

        if self.use_conv:
            k = self._apply_conv(self.conv_k, k)
            v = self._apply_conv(self.conv_v, v)
            q = self._apply_conv(self.conv_q, q)

        k = self.norm(k)
        q = self.norm(q)

        flat_params = self._get_params_flat().detach().requires_grad_(True)
        S = torch.zeros_like(flat_params)

        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        outputs = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            k_chunk = k[:, start:end, :]
            v_chunk = v[:, start:end, :]
            q_chunk = q[:, start:end, :]
            x_chunk = x[:, start:end, :]

            for t in range(chunk_size):
                if start + t >= seq_len:
                    break
                kt = k_chunk[:, t, :]
                vt = v_chunk[:, t, :]
                qt = q_chunk[:, t, :]
                xt = x_chunk[:, t, :]

                # Retrieve output with current parameters
                def forward_mem(params, inp):
                    offset = 0
                    params_dict = {}
                    for name, p in self.memory_mlp.named_parameters():
                        numel = p.numel()
                        params_dict[name] = params[offset:offset + numel].reshape(p.shape)
                        offset += numel
                    return functional_call(self.memory_mlp, params_dict, inp)

                y_t = forward_mem(flat_params, qt)
                outputs.append(y_t)

                loss = self._loss_fn(flat_params, kt, vt)
                grad_flat = torch.autograd.grad(loss, flat_params, create_graph=True)[0]

                alpha, theta, eta = self._get_coeffs(xt)
                alpha = alpha.mean()
                theta = theta.mean()
                eta = eta.mean()

                S = eta * S - theta * grad_flat
                flat_params = (1 - alpha) * flat_params + S

        outputs = torch.stack(outputs, dim=1)
        self._set_params_flat(flat_params.detach())
        if return_all_outputs:
            return outputs
        else:
            return outputs[:, -1, :]

    def forward(self, x: torch.Tensor, return_all_outputs: bool = False) -> torch.Tensor:
        """Forward pass không cập nhật inner (inference)."""
        batch, seq_len, _ = x.shape
        k = self.W_K(x)
        v = self.W_V(x)
        q = self.W_Q(x)
        if self.use_conv:
            k = self._apply_conv(self.conv_k, k)
            v = self._apply_conv(self.conv_v, v)
            q = self._apply_conv(self.conv_q, q)
        k = self.norm(k)
        q = self.norm(q)

        q_flat = q.view(-1, self.mem_dim)
        y_flat = self.memory_mlp(q_flat)
        y = y_flat.view(batch, seq_len, self.mem_dim)

        if return_all_outputs:
            return y
        else:
            return y[:, -1, :]

    def test_time_update(self, x: torch.Tensor, return_all_outputs: bool = False):
        """Thích nghi memory trên chuỗi mới mà không lưu đồ thị gradient."""
        batch, seq_len, _ = x.shape
        k = self.W_K(x)
        v = self.W_V(x)
        q = self.W_Q(x)
        if self.use_conv:
            k = self._apply_conv(self.conv_k, k)
            v = self._apply_conv(self.conv_v, v)
            q = self._apply_conv(self.conv_q, q)
        k = self.norm(k)
        q = self.norm(q)

        flat_params = self._get_params_flat().detach()
        S = torch.zeros_like(flat_params)
        outputs = []

        for t in range(seq_len):
            kt = k[:, t, :].detach()
            vt = v[:, t, :].detach()
            qt = q[:, t, :].detach()
            xt = x[:, t, :].detach()

            with torch.no_grad():
                y_t = functional_call(
                    self.memory_mlp, dict(self.memory_mlp.named_parameters()), qt
                )
            outputs.append(y_t)

            params_grad = flat_params.detach().requires_grad_(True)
            loss = self._loss_fn(params_grad, kt, vt)
            grad_flat = torch.autograd.grad(loss, params_grad, create_graph=False)[0]

            alpha, theta, eta = self._get_coeffs(xt)
            alpha = alpha.mean().item()
            theta = theta.mean().item()
            eta = eta.mean().item()

            S = eta * S - theta * grad_flat.detach()
            flat_params = (1 - alpha) * flat_params + S
            self._set_params_flat(flat_params.detach())

        outputs = torch.stack(outputs, dim=1)
        if return_all_outputs:
            return outputs
        else:
            return outputs[:, -1, :]


class PersistentMemory(nn.Module):
    def __init__(self, num_tokens: int, dim: int, init_scale: float = 0.02):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.tokens = nn.Parameter(torch.empty(num_tokens, dim))
        nn.init.normal_(self.tokens, mean=0.0, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        persistent = self.tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([persistent, x], dim=1)