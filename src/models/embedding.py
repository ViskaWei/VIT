from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .rope import RotaryPositionEmbedding
from .tokenization import Conv1DPatchTokenizer, SlidingWindowTokenizer


__all__ = ["SpectraEmbeddings"]


class SpectraEmbeddings(nn.Module):
    """Patch + positional embeddings for 1D spectral inputs
    
    Supports different position encoding types:
    - None or 'none': No position encoding (default)
    - 'rope': Rotary Position Embedding (better length generalization)
    - 'learned': Learned absolute position embeddings (original implementation)
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        stride_size = getattr(config, "stride_size", None)
        stride = stride_size if stride_size and stride_size > 0 else int(config.stride_ratio * config.patch_size)
        
        if config.proj_fn == "SW":
            self.patch_embeddings = SlidingWindowTokenizer(
                input_length=config.image_size,
                patch_size=config.patch_size,
                hidden_size=config.hidden_size,
                stride=stride,
            )
        elif config.proj_fn in ("C1D", "CNN"):
            self.patch_embeddings = Conv1DPatchTokenizer(
                input_length=config.image_size,
                patch_size=config.patch_size,
                hidden_size=config.hidden_size,
                stride=stride,
            )
        else:
            raise ValueError(f"Unsupported proj_fn '{config.proj_fn}'")

        self.num_patches = self.patch_embeddings.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position encoding setup
        self.pos_encoding_type = getattr(config, "pos_encoding_type", None)
        
        if self.pos_encoding_type == "rope":
            # RoPE: Rotary Position Embedding (default)
            max_seq_len = getattr(config, "max_position_embeddings", self.num_patches + 1)
            rope_base = getattr(config, "rope_base", 10000.0)
            self.rope = RotaryPositionEmbedding(
                dim=config.hidden_size,
                max_seq_len=max_seq_len,
                base=rope_base
            )
            self.position_embeddings = None
        elif self.pos_encoding_type == "learned":
            # Learned absolute position embeddings (original)
            self.position_embeddings = nn.Parameter(
                torch.randn(1, self.num_patches + 1, config.hidden_size)
            )
            self.rope = None
        elif self.pos_encoding_type == "none" or self.pos_encoding_type is None:
            # No position encoding (default)
            self.position_embeddings = None
            self.rope = None
        else:
            raise ValueError(
                f"Unsupported pos_encoding_type '{self.pos_encoding_type}'. "
                f"Choose from: 'rope', 'learned', 'none', or None"
            )

    def forward(
        self,
        x: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        tokens = self.patch_embeddings(x)
        batch_size, _, _ = tokens.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        
        # Apply position encoding based on type
        if self.pos_encoding_type == "rope":
            # RoPE is applied in the attention mechanism, not here
            # We just pass through the tokens
            pass
        elif self.pos_encoding_type == "learned":
            # Add learned position embeddings
            tokens = tokens + self.position_embeddings
        # elif self.pos_encoding_type == "none": just pass through
        
        return self.dropout(tokens)

    def set_patch_proj_trainable(self, trainable: bool = True) -> None:
        if hasattr(self.patch_embeddings, "projection"):
            for param in self.patch_embeddings.projection.parameters():
                param.requires_grad = trainable

