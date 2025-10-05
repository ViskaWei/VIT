from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


__all__ = [
    "PatchTokenizer",
    "SlidingWindowTokenizer",
    "Conv1DPatchTokenizer",
    "LinearPatchTokenizer",
    "TokenizerFactory",
]


@runtime_checkable
class PatchTokenizer(Protocol):
    """Interface for modules converting 1D inputs into patch tokens."""

    num_patches: int
    patch_size: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class SlidingWindowTokenizer(nn.Module):
    """Tokenise 1D signals via unfold + Linear projection."""

    def __init__(self, input_length: int, patch_size: int, hidden_size: int, stride: int | None = None) -> None:
        super().__init__()
        stride_size = stride if stride and stride > 0 else int(patch_size)
        self.image_size = input_length
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.num_patches = math.ceil((self.image_size - self.patch_size) / self.stride_size) + 1
        self.projection = nn.Linear(self.patch_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride_size)
        if patches.size(1) < self.num_patches:
            padding = torch.zeros(batch_size, self.num_patches - patches.size(1), self.patch_size, device=x.device)
            patches = torch.cat([patches, padding], dim=1)
        patches = patches.contiguous().reshape(batch_size, self.num_patches, self.patch_size)
        return self.projection(patches)


class Conv1DPatchTokenizer(nn.Module):
    """Tokenise with a Conv1d kernel applied over the 1D signal."""

    def __init__(self, input_length: int, patch_size: int, hidden_size: int, stride: int | None = None) -> None:
        super().__init__()
        stride_size = stride if stride and stride > 0 else int(patch_size)
        self.image_size = input_length
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.num_channels = 1
        self.num_patches = ((self.image_size - self.patch_size) // self.stride_size) + 1
        self.projection = nn.Conv1d(self.num_channels, hidden_size, kernel_size=self.patch_size, stride=self.stride_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, self.image_size)
        x = self.projection(x)
        return x.transpose(1, 2)


class LinearPatchTokenizer(nn.Module):
    """Reshape-based tokenizer mirroring the original SpectraTransformer patch embed."""

    def __init__(self, input_length: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        if input_length % patch_size != 0:
            raise ValueError(f"input_length={input_length} must be divisible by patch_size={patch_size}")
        self.input_length = input_length
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = input_length // patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("LinearPatchTokenizer expects input of shape (batch, length)")
        bsz, length = x.shape
        if length != self.input_length:
            raise ValueError(f"Expected input length {self.input_length}, received {length}")
        patches = x.view(bsz, self.num_patches, self.patch_size)
        return self.projection(patches)


@dataclass
class TokenizerFactory:
    """Helper to create patch tokenizers from a config-like object."""

    input_length: int
    patch_size: int
    hidden_size: int
    stride_size: int | None = None

    def sliding_window(self) -> SlidingWindowTokenizer:
        return SlidingWindowTokenizer(
            input_length=self.input_length,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            stride=self.stride_size,
        )

    def conv1d(self) -> Conv1DPatchTokenizer:
        return Conv1DPatchTokenizer(
            input_length=self.input_length,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            stride=self.stride_size,
        )

    def linear(self, embed_dim: int | None = None) -> LinearPatchTokenizer:
        dim = embed_dim if embed_dim is not None else self.hidden_size
        return LinearPatchTokenizer(
            input_length=self.input_length,
            patch_size=self.patch_size,
            embed_dim=dim,
        )
