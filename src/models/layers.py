from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["PrefilledLinear"]


class PrefilledLinear(nn.Module):
    """Linear layer with weight initialized from a matrix (ZCA, PCA, etc.)"""

    def __init__(self, matrix: torch.Tensor, freeze: bool = True) -> None:
        super().__init__()
        weight = matrix.to(torch.float32)
        self.lin = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(weight)
        self.freeze(freeze)

    def freeze(self, freeze: bool = True) -> None:
        for param in self.lin.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)
