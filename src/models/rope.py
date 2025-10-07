from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["RotaryPositionEmbedding"]


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for 1D sequences.
    
    RoPE applies rotation matrices to query and key embeddings based on position,
    providing better length generalization than learned position embeddings.
    
    Args:
        dim: Hidden dimension (must be even)
        max_seq_len: Maximum sequence length to precompute
        base: Base for the geometric progression of frequencies (default: 10000)
    
    References:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the rotation frequencies
        # inv_freq shape: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin for max_seq_len positions
        self._precompute_freqs(max_seq_len)

    def _precompute_freqs(self, seq_len: int) -> None:
        """Precompute cos and sin values for all positions up to seq_len."""
        # t shape: (seq_len,)
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        
        # freqs shape: (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        
        # emb shape: (seq_len, dim)
        # Interleave to match (x0, x1, x2, x3, ...) -> (x0, x1, x2, x3, ...)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_seq_len = seq_len

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input.
        
        For input [x0, x1, x2, x3, ...], returns [-x_(dim/2), -x_(dim/2+1), ..., x0, x1, ...]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, heads, seq_len, dim)
            cos: Cosine values of shape (seq_len, dim)
            sin: Sine values of shape (seq_len, dim)
            
        Returns:
            Rotated tensor with the same shape as input
        """
        # Handle different input shapes
        if x.dim() == 3:
            # Shape: (batch, seq_len, dim)
            cos = cos[:x.shape[1], :]  # (seq_len, dim)
            sin = sin[:x.shape[1], :]  # (seq_len, dim)
            # Expand for broadcasting: (1, seq_len, dim)
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        elif x.dim() == 4:
            # Shape: (batch, heads, seq_len, dim)
            cos = cos[:x.shape[2], :]  # (seq_len, dim)
            sin = sin[:x.shape[2], :]  # (seq_len, dim)
            # Expand for broadcasting: (1, 1, seq_len, dim)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Input must be 3D or 4D tensor, got shape {x.shape}")
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, heads, seq_len, dim)
            
        Returns:
            Tensor with rotary position embeddings applied
        """
        seq_len = x.shape[-2] if x.dim() == 4 else x.shape[1]
        
        # Extend cache if needed
        if seq_len > self._cached_seq_len:
            self._precompute_freqs(seq_len)
        
        return self.apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)

    def forward_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors separately.
        
        This is useful when you want to apply RoPE in the attention mechanism.
        
        Args:
            q: Query tensor
            k: Key tensor
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        q_rotated = self.forward(q)
        k_rotated = self.forward(k)
        return q_rotated, k_rotated
