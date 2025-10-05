"""Deprecated compatibility wrapper for legacy imports.

Prefer importing from ``src.models`` directly.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.model` is deprecated; import from `src.models` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.models import (
    GlobalAttnViT,
    PreconditionedViT,
    MyViT,
    SpectraTransformer,
    SpectraTransformerConfig,
    ZCALinear,
    get_model,
    get_pca_config,
    get_vit_config,
    get_vit_pretrain_model,
)
from src.models.embedding import MyEmbeddings, SpectraEmbeddings, apply_patch_embed_pca
from src.models.layers import complete_with_orthogonal, load_basis_matrix

_apply_embed_pca = apply_patch_embed_pca
_load_V_matrix = load_basis_matrix

__all__ = [
    "complete_with_orthogonal",
    "load_basis_matrix",
    "apply_patch_embed_pca",
    "get_model",
    "get_vit_pretrain_model",
    "get_pca_config",
    "get_vit_config",
    "MyViT",
    "PreconditionedViT",
    "GlobalAttnViT",
    "SpectraTransformer",
    "SpectraTransformerConfig",
    "SpectraEmbeddings",
    "MyEmbeddings",
    "ZCALinear",
]
