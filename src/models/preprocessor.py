from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from .attention import GlobalAttentionLayer
from .layers import ZCALinear


__all__ = [
    "IdentityPreprocessor",
    "ZCAPreprocessor",
    "build_preprocessor",
]


class IdentityPreprocessor(nn.Module):
    """No-op preprocessor used when no conditioning is requested."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x


class ZCAPreprocessor(nn.Module):
    """Linear ZCA-style projection with optional Bias."""

    def __init__(
        self,
        matrix: torch.Tensor,
        *,
        freeze: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.linear = ZCALinear(matrix, freeze=freeze)
        if bias is not None:
            bias = bias.to(dtype=self.linear.lin.weight.dtype)
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.linear(x)
        bias = getattr(self, "bias", None)
        if isinstance(bias, torch.Tensor):
            x = x + bias
        return x

    def freeze(self, freeze: bool = True) -> None:
        self.linear.freeze(freeze)


def _as_tensor(val) -> Optional[torch.Tensor]:
    if isinstance(val, torch.Tensor):
        return val
    if hasattr(val, "__array__"):
        import numpy as np

        arr = np.asarray(val)
        return torch.from_numpy(arr)
    return None


def _select_candidate(
    stats: dict,
    keys: Iterable[str],
    *,
    input_dim: int,
) -> Optional[torch.Tensor]:
    for key in keys:
        if key not in stats:
            continue
        cand = _as_tensor(stats[key])
        if cand is None or cand.dim() != 2:
            continue
        if cand.shape[1] != input_dim:
            continue
        return cand
    return None


def extract_linear_projection(
    pca_stats,
    *,
    input_dim: int,
    uv_key: Optional[str] = None,
    allow_rectangular: bool = False,
) -> Optional[torch.Tensor]:
    if isinstance(pca_stats, torch.Tensor):
        mat = pca_stats
        if mat.dim() == 2 and mat.shape[1] == input_dim:
            return mat
        return None
    if not isinstance(pca_stats, dict):
        return None

    keys = []
    if isinstance(uv_key, str) and uv_key:
        keys.append(uv_key)
    keys.extend(
        [
            "matrix",
            "weight",
            "linear_weight",
            "preconditioner",
            "projector",
            "projector_matrix",
            "whitening",
            "whitening_matrix",
            "zca",
            "P",
            "transform",
        ]
    )
    mat = _select_candidate(pca_stats, keys, input_dim=input_dim)
    if mat is None:
        # Fall back to more generic PCA keys if they are square.
        fallback = _select_candidate(
            pca_stats,
            ["V", "components", "components_", "Ut", "U"],
            input_dim=input_dim,
        )
        if fallback is not None:
            mat = fallback
    if mat is None:
        return None
    if not allow_rectangular and mat.shape[0] != input_dim:
        return None
    return mat


def extract_bias_vector(pca_stats, *, input_dim: int) -> Optional[torch.Tensor]:
    if isinstance(pca_stats, dict):
        bias = pca_stats.get("bias", None)
        if bias is None:
            bias = pca_stats.get("mean", None)
        bias_tensor = _as_tensor(bias)
        if isinstance(bias_tensor, torch.Tensor) and bias_tensor.dim() == 1 and bias_tensor.shape[0] == input_dim:
            return bias_tensor
    return None


def build_preprocessor(
    kind: str | None,
    *,
    input_dim: int,
    pca_stats=None,
    uv_key: Optional[str] = None,
    use_input_bias: bool = False,
    freeze: bool = True,
    allow_rectangular: bool = False,
    **kwargs,
) -> nn.Module:
    """Factory that returns an input preprocessor module."""

    if kind is None or kind == "identity":
        return IdentityPreprocessor()

    if kind == "global_attention":
        return GlobalAttentionLayer(
            input_dim=input_dim,
            pca_stats=pca_stats,
            uv_key=uv_key,
            use_pca_bias=use_input_bias,
            **kwargs,
        )

    if kind == "zca_linear":
        matrix = extract_linear_projection(
            pca_stats,
            input_dim=input_dim,
            uv_key=uv_key,
            allow_rectangular=allow_rectangular,
        )
        if matrix is None:
            raise ValueError(
                "Could not find a suitable projection matrix in PCA stats for `zca_linear` preprocessor"
            )
        bias = extract_bias_vector(pca_stats, input_dim=input_dim) if use_input_bias else None
        return ZCAPreprocessor(matrix, freeze=freeze, bias=bias)

    raise ValueError(f"Unsupported preprocessor kind '{kind}'")
