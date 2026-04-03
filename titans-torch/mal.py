import torch
import torch.nn as nn
from .memory import NeuralLongTermMemory, PersistentMemory


class TitansMAG(nn.Module):
    """
    Memory as a Gate (MAG) – kết hợp sliding window attention và LMM qua gating.
    """
    def __init__(
        self,
        input_dim: int,
        num_persistent: int = 8,
        mem_dim: Optional[int] = None,
        num_memory_layers: int = 2,
        hidden_memory_dim: int = 32,
        window_size: int = 4,
        use_conv: bool = True,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        coeff_hidden_dim: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mem_dim = mem_dim if mem_dim is not None else input_dim

        self.persistent = PersistentMemory(num_persistent, input_dim)
        self.long_term_memory = NeuralLongTermMemory(
            input_dim=input_dim,
            mem_dim=self.mem_dim,
            num_layers=num_memory_layers,
            hidden_dim=hidden_memory_dim,
            use_conv=use_conv,
            coeff_hidden_dim=coeff_hidden_dim,
        )
        self.short_term_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=attention_dropout,
        )
        self.window_size = window_size

        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        x_persist = self.persistent(x)
        batch, seq_len, dim = x_persist.shape

        # Sliding window causal mask
        mask = torch.ones(seq_len, seq_len, device=x.device).triu(diagonal=1).bool()
        for i in range(seq_len):
            for j in range(max(0, i - self.window_size), i):
                mask[i, j] = False

        attn_out, _ = self.short_term_attention(
            x_persist, x_persist, x_persist, attn_mask=mask
        )
        attn_out = attn_out[:, self.persistent.num_tokens:, :]  # bỏ persistent

        memory_out = self.long_term_memory.forward_trainable(
            x, return_all_outputs=return_all
        )

        gate_value = self.gate(torch.cat([attn_out, memory_out], dim=-1))
        out = gate_value * memory_out + (1 - gate_value) * attn_out
        return out