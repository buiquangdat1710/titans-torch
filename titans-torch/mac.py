import torch
import torch.nn as nn
from .memory import NeuralLongTermMemory, PersistentMemory


class TitansMAC(nn.Module):
    """
    Memory as a Context (MAC) – đúng theo công thức (21)-(25).
    """
    def __init__(
        self,
        input_dim: int,
        num_persistent: int = 8,
        mem_dim: Optional[int] = None,
        num_memory_layers: int = 2,
        hidden_memory_dim: int = 32,
        chunk_size: int = 4,
        use_conv: bool = True,
        num_heads: int = 4,               # số head cho multihead attention
        attention_dropout: float = 0.1,   # dropout trong attention
        coeff_hidden_dim: int = 16,       # hidden size cho alpha/theta/eta nets (trong LMM)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mem_dim = mem_dim if mem_dim is not None else input_dim
        self.chunk_size = chunk_size

        self.persistent = PersistentMemory(num_persistent, input_dim)
        self.W_Q = nn.Linear(input_dim, self.mem_dim, bias=False)

        self.long_term_memory = NeuralLongTermMemory(
            input_dim=input_dim,
            mem_dim=self.mem_dim,
            num_layers=num_memory_layers,
            hidden_dim=hidden_memory_dim,
            use_conv=use_conv,
            coeff_hidden_dim=coeff_hidden_dim,
            chunk_size=chunk_size,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=attention_dropout,
        )

        if self.mem_dim != self.input_dim:
            self.proj_h = nn.Linear(self.mem_dim, self.input_dim)
        else:
            self.proj_h = nn.Identity()

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        persistent_tokens = self.persistent.tokens.unsqueeze(0).expand(batch, -1, -1)

        num_segments = (seq_len + self.chunk_size - 1) // self.chunk_size
        outputs = []

        for seg_idx in range(num_segments):
            start = seg_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            S = x[:, start:end, :]

            # 1. Retrieve historical info h_t = M_{t-1}^*(q_t)
            q = self.W_Q(S)
            h = self.long_term_memory.forward(q, return_all_outputs=True)
            h = self.proj_h(h)

            # 2. Attention input: [persistent] || h_t || S^{(t)}
            attn_input = torch.cat([persistent_tokens, h, S], dim=1)

            # 3. Causal mask (Figure 3a)
            L = attn_input.size(1)
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            y_t, _ = self.attention(attn_input, attn_input, attn_input, attn_mask=causal_mask)

            # 4. Update memory: M_t = M_{t-1}(y_t)
            self.long_term_memory.forward_trainable(
                y_t, chunk_size=self.chunk_size, return_all_outputs=False
            )

            # 5. Inference on updated memory: M_t^*(y_t)
            memory_out = self.long_term_memory.forward(y_t, return_all_outputs=True)
            if memory_out.shape[-1] != self.input_dim:
                memory_out = self.proj_h(memory_out)

            # 6. Output o_t = y_t ⊗ M_t^*(y_t)
            o_t = y_t * memory_out

            # Extract segment part (drop persistent and h)
            out_segment = o_t[:, -S.shape[1]:, :]
            outputs.append(out_segment)

        final_out = torch.cat(outputs, dim=1)
        if return_all:
            return final_out
        else:
            return final_out[:, -1, :]