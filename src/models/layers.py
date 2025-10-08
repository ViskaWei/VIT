from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["PrefilledLinear"]


class PrefilledLinear(nn.Module):
    """Linear layer with weight initialized from a matrix (ZCA, PCA, etc.)
    
    Supports optional bias for centering operations (e.g., ZCA whitening).
    The bias is computed as -mean @ matrix to implement (x - mean) @ matrix.T
    """

    def __init__(self, matrix: torch.Tensor, bias: torch.Tensor | None = None, freeze: bool = True) -> None:
        super().__init__()
        weight = matrix.to(torch.float32)
        
        if freeze:
            # Register as buffer (not counted as parameter)
            self.register_buffer('weight', weight)
            self._is_frozen = True
        else:
            # Register as parameter (trainable)
            self.weight = nn.Parameter(weight)
            self._is_frozen = False
        
        # Handle bias (for centering in ZCA/PCA)
        if bias is not None:
            bias = bias.to(torch.float32)
            if freeze:
                self.register_buffer('bias', bias)
            else:
                self.bias = nn.Parameter(bias)
        else:
            self.register_buffer('bias', None)

    def freeze(self, freeze: bool = True) -> None:
        if freeze and not self._is_frozen:
            # Convert parameter to buffer
            weight_data = self.weight.data.clone()
            del self.weight
            self.register_buffer('weight', weight_data)
            if hasattr(self, 'bias') and isinstance(self.bias, nn.Parameter):
                bias_data = self.bias.data.clone()
                del self.bias
                self.register_buffer('bias', bias_data)
            self._is_frozen = True
        elif not freeze and self._is_frozen:
            # Convert buffer to parameter
            weight_data = self.weight.clone()
            delattr(self, 'weight')
            self.weight = nn.Parameter(weight_data)
            if self.bias is not None:
                bias_data = self.bias.clone()
                delattr(self, 'bias')
                self.bias = nn.Parameter(bias_data)
            self._is_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, bias=self.bias)
