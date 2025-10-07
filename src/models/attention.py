from __future__ import annotations

import math

import torch
import torch.nn as nn


__all__ = ["PrefilledAttention"]


class PrefilledAttention(nn.Module):
    """Attention layer with query/key projections prefilled from a basis matrix."""

    def __init__(self, input_dim: int, eigvecs: torch.Tensor, r: int | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.r = r if r is not None else eigvecs.shape[1]
        
        # Initialize Q, K, V projections
        self.q_lin = nn.Linear(input_dim, input_dim, bias=False)
        self.k_lin = nn.Linear(input_dim, input_dim, bias=False)
        self.v_lin = nn.Linear(input_dim, input_dim, bias=False)

        # Prefill Q and K with PCA/ZCA eigenvectors
        V = eigvecs[:, :self.r].t().contiguous()  # (r, input_dim)
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
        q = self.q_lin(x)
        k = self.k_lin(x)
        v = self.v_lin(x)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.input_dim ** 0.5)
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
        """Copy ``basis`` rows into ``layer`` weight without gradients."""
        rows = basis.shape[0]
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[:rows, :].copy_(basis)
