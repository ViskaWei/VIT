from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTSelfAttention

from .rope import RotaryPositionEmbedding


__all__ = ["ViTSelfAttentionWithRoPE"]


class ViTSelfAttentionWithRoPE(ViTSelfAttention):
    """ViT Self-Attention with RoPE support.
    
    This module extends HuggingFace's ViTSelfAttention to support Rotary Position Embeddings.
    When RoPE is enabled, it applies rotary position encodings to query and key tensors.
    """
    
    def __init__(self, config, use_rope: bool = True) -> None:
        super().__init__(config)
        
        self.use_rope = use_rope
        
        if self.use_rope:
            # Initialize RoPE
            head_dim = self.attention_head_size
            max_seq_len = getattr(config, "max_position_embeddings", 512)
            rope_base = getattr(config, "rope_base", 10000.0)
            
            # RoPE is applied per head, so dim = head_dim
            self.rope = RotaryPositionEmbedding(
                dim=head_dim,
                max_seq_len=max_seq_len,
                base=rope_base
            )
        else:
            self.rope = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional RoPE application."""
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        # Generate Q, K, V
        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)
        
        # Apply RoPE to query and key if enabled
        if self.use_rope and self.rope is not None:
            query_layer, key_layer = self.rope.forward_qk(query_layer, key_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Normalize to get attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        dropout_prob = 0.0 if not self.training else self.dropout_prob
        attention_probs_dropped = nn.functional.dropout(attention_probs, p=dropout_prob, training=self.training)
        
        # Apply head mask if provided
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        
        # Compute context
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # Always return both context and attention_probs (as expected by ViTAttention wrapper)
        return context_layer, attention_probs
