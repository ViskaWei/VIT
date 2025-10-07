from __future__ import annotations

import torch
import torch.nn as nn

from .layers import PrefilledLinear


__all__ = ["LinearPreprocessor", "compute_zca_matrix", "compute_pca_matrix"]


def compute_zca_matrix(
    eigvecs: torch.Tensor,
    eigvals: torch.Tensor,
    eps: float = 1e-5,
    r: int | None = None,
    shrinkage: float = 0.0,
) -> torch.Tensor:
    """Compute ZCA whitening matrix P = V @ D^(-1/2) @ V.T
    
    Args:
        eigvecs: Eigenvectors (D, D) sorted by eigenvalues descending
        eigvals: Eigenvalues (D,) sorted descending
        eps: Regularization for numerical stability
        r: Number of components for low-rank ZCA. If None, full-rank
        shrinkage: Shrinkage parameter in [0, 1]. 0 = no shrinkage, 1 = identity
    
    Returns:
        ZCA matrix P of shape (D, D) for full-rank or low-rank approximation
    """
    if r is None:
        # Full-rank ZCA
        if shrinkage > 0.0:
            avg = eigvals.mean()
            eigvals_hat = (1.0 - shrinkage) * eigvals + shrinkage * avg
        else:
            eigvals_hat = eigvals
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals_hat + eps))
        P = eigvecs @ D_inv_sqrt @ eigvecs.t()
    else:
        # Low-rank ZCA: P = (Vr * inv_sqrt_r) @ Vr.T
        if shrinkage > 0.0:
            avg = eigvals.mean()
            eigvals_hat = (1.0 - shrinkage) * eigvals + shrinkage * avg
        else:
            eigvals_hat = eigvals
        Vr = eigvecs[:, :r]
        inv_sqrt_r = torch.rsqrt(eigvals_hat[:r] + eps)
        P = (Vr * inv_sqrt_r) @ Vr.t()
    
    return P


def compute_pca_matrix(eigvecs: torch.Tensor, r: int) -> torch.Tensor:
    """Compute PCA projection matrix P = V[:, :r].T
    
    Args:
        eigvecs: Eigenvectors (D, D) sorted by eigenvalues descending
        r: Number of components to keep
    
    Returns:
        PCA matrix P of shape (r, D)
    """
    return eigvecs[:, :r].t()


class LinearPreprocessor(nn.Module):
    """Linear preprocessing: x -> P @ x
    
    Unified class for both ZCA whitening and PCA projection.
    - ZCA: P is (D, D), output is (batch, D)
    - PCA: P is (r, D), output is (batch, r)
    
    Use compute_zca_matrix() or compute_pca_matrix() to create P.
    """

    def __init__(self, matrix: torch.Tensor, freeze: bool = True) -> None:
        super().__init__()
        self.linear = PrefilledLinear(matrix, freeze=freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def freeze(self, freeze: bool = True) -> None:
        self.linear.freeze(freeze)
