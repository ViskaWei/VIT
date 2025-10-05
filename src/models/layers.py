from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


__all__ = [
    "complete_with_orthogonal",
    "load_basis_matrix",
    "ZCALinear",
]


def complete_with_orthogonal(U: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Expand a basis ``U`` to ``out_dim`` with an orthogonal completion."""
    in_dim, r = U.shape
    if r >= out_dim:
        return U[:, :out_dim]
    rand_mat = torch.randn(in_dim, out_dim - r, device=U.device, dtype=U.dtype)
    A = torch.cat([U, rand_mat], dim=1)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :out_dim]


def load_basis_matrix(
    pth: str,
    patch_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    basis_key: str | None = None,
) -> torch.Tensor | None:
    """Load a PCA-style basis matrix from ``pth`` and return shape ``(patch_dim, k)``."""

    def _as_patch_by_k(t: torch.Tensor) -> torch.Tensor | None:
        if t.dim() != 2:
            return None
        if t.shape[0] == patch_dim:
            return t
        if t.shape[1] == patch_dim:
            return t.t()
        return None

    try:
        obj = torch.load(pth, weights_only=True, map_location="cpu")
    except Exception:
        try:
            obj = torch.load(pth, map_location="cpu")
        except Exception as exc:
            print(f"[embed-warmup] Failed to load PCA file '{pth}': {exc}")
            return None

    V_mat: torch.Tensor | None = None
    if isinstance(obj, torch.Tensor):
        V_mat = _as_patch_by_k(obj)
    elif isinstance(obj, dict):
        if isinstance(basis_key, str) and basis_key:
            key = basis_key.strip()
            lowered = key.lower()
            if lowered == "vt":
                candidates = ["Vt", "Vh", "vh", "V", "components", "components_"]
            elif lowered == "v":
                candidates = ["V", "components", "components_", "Vt", "Vh", "vh"]
            elif lowered == "ut":
                candidates = ["Ut", "U"]
            elif lowered == "u":
                candidates = ["U", "Ut"]
            else:
                candidates = [key]
            for cand_key in candidates:
                if cand_key in obj and isinstance(obj[cand_key], torch.Tensor):
                    V_mat = _as_patch_by_k(obj[cand_key])
                    if V_mat is not None:
                        break
        if V_mat is None:
            for k in (
                "V",
                "components",
                "components_",
                "eigvecs",
                "eigvec",
                "basis",
                "Vh",
                "vh",
                "U",
                "scores",
                "A",
            ):
                if k in obj and isinstance(obj[k], torch.Tensor):
                    V_mat = _as_patch_by_k(obj[k])
                    if V_mat is not None:
                        break
        if V_mat is None:
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    cand = _as_patch_by_k(v)
                    if cand is not None:
                        V_mat = cand
                        break
    else:
        try:
            if isinstance(basis_key, str) and basis_key:
                maybe = getattr(obj, basis_key, None)
                if isinstance(maybe, torch.Tensor):
                    V_mat = _as_patch_by_k(maybe)
            if V_mat is None:
                for k in ("V", "U"):
                    maybe = getattr(obj, k, None)
                    if isinstance(maybe, torch.Tensor):
                        V_mat = _as_patch_by_k(maybe)
                        if V_mat is not None:
                            break
        except Exception:
            V_mat = None

    if V_mat is None:
        print(f"[embed-warmup] No usable V/components found in '{pth}' for patch_dim={patch_dim}")
        return None

    return V_mat.to(device=device, dtype=dtype, copy=False)


class ZCALinear(nn.Module):
    """Linear layer initialised with a ZCA/ZCA-like projection matrix."""

    def __init__(self, P: np.ndarray | torch.Tensor, freeze: bool = True):
        super().__init__()
        weight = torch.as_tensor(P, dtype=torch.float32)
        self.lin = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(weight)
        self.freeze(freeze)

    @classmethod
    def identity(cls, dim: int, freeze: bool = True) -> "ZCALinear":
        eye = np.eye(dim, dtype=np.float32)
        return cls(eye, freeze=freeze)

    def freeze(self, freeze: bool = True) -> None:
        for param in self.lin.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)

    def update(self, P: np.ndarray | torch.Tensor, freeze: bool | None = None) -> None:
        weight = torch.as_tensor(P, dtype=self.lin.weight.dtype, device=self.lin.weight.device)
        if weight.shape != self.lin.weight.shape:
            raise ValueError(
                f"Shape mismatch updating ZCALinear: got {tuple(weight.shape)}, expect {tuple(self.lin.weight.shape)}"
            )
        with torch.no_grad():
            self.lin.weight.copy_(weight)
        if freeze is not None:
            self.freeze(freeze)
