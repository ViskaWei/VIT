from __future__ import annotations

import math

import torch
import torch.nn as nn


__all__ = ["PrefilledAttention"]


class PrefilledAttention(nn.Module):
    """Attention layer with query/key projections prefilled from a basis matrix.
    
    Args:
        input_dim: Input feature dimension D
        eigvecs: Eigenvector matrix for prefilling Q/K
        r: Rank for Q/K projections. If None, uses eigvecs.shape[1].
           By default, uses low-rank (D x r) if r < D, otherwise full-rank (D x D).
        low_rank: If specified, forces low-rank (True) or full-rank (False) mode.
                  If None (default), automatically uses low-rank when r < D.
    """

    def __init__(
        self, 
        input_dim: int, 
        eigvecs: torch.Tensor, 
        r: int | None = None,
        low_rank: bool | None = None
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.r = r if r is not None else eigvecs.shape[1]
        # Auto-determine low_rank: use low-rank if r < input_dim (unless explicitly overridden)
        self.low_rank = low_rank if low_rank is not None else (self.r < input_dim)
        
        # Initialize Q, K, V projections
        if self.low_rank:
            # Low-rank: Q and K are (input_dim x r)
            self.q_lin = nn.Linear(input_dim, self.r, bias=False)
            self.k_lin = nn.Linear(input_dim, self.r, bias=False)
        else:
            # Full-rank: Q and K are (input_dim x input_dim)
            self.q_lin = nn.Linear(input_dim, input_dim, bias=False)
            self.k_lin = nn.Linear(input_dim, input_dim, bias=False)
        
        self.v_lin = nn.Linear(input_dim, input_dim, bias=False)

        # Prefill Q and K with PCA/ZCA eigenvectors
        V = eigvecs[:, :self.r].t().contiguous()  # (r, input_dim)
        if self.low_rank:
            # For low-rank: directly use V^T as weights (r x input_dim)
            self._prefill_linear_lowrank(self.q_lin, V)
            self._prefill_linear_lowrank(self.k_lin, V)
        else:
            # For full-rank: prefill first r rows (input_dim x input_dim)
            self._prefill_linear(self.q_lin, V)
            self._prefill_linear(self.k_lin, V)

        # Initialize V projection
        nn.init.kaiming_uniform_(self.v_lin.weight, a=math.sqrt(5))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For 2D input (batch, features), just apply Q projection
        if x.dim() == 2:
            return self.q_lin(x)
        
        # For 3D input, apply full attention
        q = self.q_lin(x)  # (batch, seq, r) if low_rank else (batch, seq, input_dim)
        k = self.k_lin(x)  # (batch, seq, r) if low_rank else (batch, seq, input_dim)
        v = self.v_lin(x)  # (batch, seq, input_dim)
        
        # Compute attention: Q @ K^T
        # Low-rank: (batch, seq, r) @ (batch, r, seq) = (batch, seq, seq)
        # Full-rank: (batch, seq, D) @ (batch, D, seq) = (batch, seq, seq)
        scale = self.r if self.low_rank else self.input_dim
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (scale ** 0.5)
        attn_probs = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_probs, v)
        return attn_output

    def set_qk_trainable(self, trainable: bool = True) -> None:
        """Freeze/unfreeze Q and K projections"""
        for param in self.q_lin.parameters():
            param.requires_grad = trainable
        for param in self.k_lin.parameters():
            param.requires_grad = trainable

    @staticmethod
    def _prefill_linear(layer: nn.Linear, basis: torch.Tensor) -> None:
        """Copy ``basis`` rows into ``layer`` weight without gradients.
        For full-rank layers (input_dim x input_dim).
        """
        rows = basis.shape[0]
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[:rows, :].copy_(basis)

    @staticmethod
    def _prefill_linear_lowrank(layer: nn.Linear, basis: torch.Tensor) -> None:
        """Copy entire ``basis`` matrix into low-rank ``layer`` weight.
        For low-rank layers (input_dim x r), basis should be (r x input_dim).
        """
        with torch.no_grad():
            layer.weight.copy_(basis)
