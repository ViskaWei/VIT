"""SpectraTransformer integrated with shared tokenisation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from .layers import ZCALinear
from .tokenization import LinearPatchTokenizer

TensorLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]


__all__ = ["SpectraTransformer", "SpectraTransformerConfig"]


@dataclass
class SpectraTransformerConfig:
    input_dim: int
    num_targets: int = 3
    patch_size: int = 16
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    projector_dim: Optional[int] = None


class SpectraTransformer(nn.Module):
    """Transformer encoder tailored for spectral regression tasks."""

    def __init__(self, config: SpectraTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.project_dim = config.projector_dim or config.input_dim
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        self.dropout = config.dropout

        self.projector: nn.Module = self._build_linear_projector(self.project_dim, freeze=True)
        self.patch_embed = LinearPatchTokenizer(self.project_dim, self.patch_size, self.embed_dim)
        self.positional = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        self.pos_dropout = nn.Dropout(self.dropout)

        ff_dim = int(self.embed_dim * config.mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=ff_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, config.num_targets)
        self._reset_parameters()

    # ------------------------------------------------------------------
    # Projector helpers
    # ------------------------------------------------------------------
    def _build_linear_projector(self, out_dim: int, freeze: bool) -> nn.Linear:
        layer = nn.Linear(self.input_dim, out_dim, bias=False)
        if out_dim == self.input_dim:
            with torch.no_grad():
                layer.weight.copy_(torch.eye(self.input_dim))
        else:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        for param in layer.parameters():
            param.requires_grad = not freeze
        return layer

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _reconfigure_patch_embed(self, new_dim: int) -> None:
        self.project_dim = new_dim
        self.patch_embed = LinearPatchTokenizer(self.project_dim, self.patch_size, self.embed_dim)
        self.positional = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        self._reset_parameters()

    # ------------------------------------------------------------------
    # External preconditioning management
    # ------------------------------------------------------------------
    def load_preconditioning(
        self,
        matrix: TensorLike,
        freeze: bool = True,
        assume_aligned: bool = False,
    ) -> None:
        tensor = torch.as_tensor(matrix, dtype=self.head.weight.dtype)
        if tensor.dim() != 2:
            raise ValueError("Preconditioning matrix must be 2D")
        out_dim, in_dim = tensor.shape
        if in_dim != self.input_dim:
            raise ValueError(
                f"Preconditioning matrix expects input_dim={self.input_dim}, received {in_dim}"
            )
        if not assume_aligned:
            self._reconfigure_patch_embed(out_dim)
        projector = ZCALinear(tensor, freeze=freeze)
        self.projector = projector

    def load_from_zca(self, zca, mode: str = "whitening", freeze: bool = True) -> None:
        if mode not in {"whitening", "projector"}:
            raise ValueError("mode must be 'whitening' or 'projector'")
        matrix = zca.whitening_matrix if mode == "whitening" else zca.projector_matrix
        self.load_preconditioning(matrix, freeze=freeze)

    def reset_preconditioning(self, freeze: bool = True) -> None:
        self.projector = self._build_linear_projector(self.config.projector_dim or self.input_dim, freeze=freeze)
        self._reconfigure_patch_embed(self.config.projector_dim or self.input_dim)

    # ------------------------------------------------------------------
    # Forward + feature utilities
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"SpectraTransformer expects input of shape (batch, {self.input_dim})"
            )
        x = self.projector(x)
        x = self.patch_embed(x)
        x = x + self.positional
        x = self.pos_dropout(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return x

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
