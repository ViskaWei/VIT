
from typing import Optional
import torch
import torch.nn as nn

from .pipeline import KernelPCAState


class KPCAWarmSelfAttention(nn.Module):
    """Multi-head self-attention whose Q/K are built from fixed KPCA features.
    - Compute Z = KPCA.transform(x) on the last feature dimension
    - Small trainable linear adapters map Z to per-head Q/K
    - V and output projections are standard linear layers
    Note: Ensure the KPCA state was fitted on vectors with the SAME dimension
    as the feature dimension of x at this layer (e.g., token dim)."""
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 kpca_state: KernelPCAState,
                 qk_dim: Optional[int] = None,
                 bias: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.state = kpca_state
        self.qk_input_dim = kpca_state.r
        if qk_dim is None:
            qk_dim = max(1, (self.qk_input_dim + num_heads - 1) // num_heads)
        self.d_k = qk_dim

        # Adapters from KPCA features to per-head subspaces
        self.q_adapter = nn.Linear(self.qk_input_dim, num_heads * self.d_k, bias=bias)
        self.k_adapter = nn.Linear(self.qk_input_dim, num_heads * self.d_k, bias=bias)

        # Initialize adapters close to identity (best-effort)
        with torch.no_grad():
            Wq = self.q_adapter.weight.view(num_heads, self.d_k, self.qk_input_dim)
            Wk = self.k_adapter.weight.view(num_heads, self.d_k, self.qk_input_dim)
            Wq.zero_(); Wk.zero_()
            step = max(1, self.qk_input_dim // self.d_k)
            for h in range(num_heads):
                for i in range(self.d_k):
                    j = min(i * step, self.qk_input_dim - 1)
                    Wq[h, i, j] = 1.0
                    Wk[h, i, j] = 1.0
            if bias:
                self.q_adapter.bias.zero_()
                self.k_adapter.bias.zero_()

        # Value and output projections (standard)
        self.v_proj = nn.Linear(dim, num_heads * self.d_k, bias=bias)
        self.out_proj = nn.Linear(num_heads * self.d_k, dim, bias=bias)

    def set_qk_requires_grad(self, flag: bool):
        for p in self.q_adapter.parameters():
            p.requires_grad = flag
        for p in self.k_adapter.parameters():
            p.requires_grad = flag

    def forward(self, x, need_weights: bool = False):
        """x: [B, L, D] -> out: [B, L, D]"""
        B, L, D = x.shape

        Z = self.state.transform(x)          # [B, L, r]
        Q = self.q_adapter(Z)                # [B, L, H*d_k]
        K = self.k_adapter(Z)
        V = self.v_proj(x)

        H, d_k = self.num_heads, self.d_k
        Q = Q.view(B, L, H, d_k).transpose(1, 2)   # [B,H,L,d_k]
        K = K.view(B, L, H, d_k).transpose(1, 2)
        V = V.view(B, L, H, d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B,H,L,L]
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, V)                              # [B,H,L,d_k]

        out = attn_out.transpose(1, 2).contiguous().view(B, L, H * d_k)
        out = self.out_proj(out)  # [B, L, D]
        if need_weights:
            return out, attn_probs
        return out
