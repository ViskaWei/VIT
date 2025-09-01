
# import torch

# def _center_gram(K: torch.Tensor) -> torch.Tensor:
#     n = K.shape[0]
#     one_n = torch.full((n, n), 1.0 / n, dtype=K.dtype, device=K.device)
#     return K - one_n @ K - K @ one_n + one_n @ K @ one_n

# def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
#     """Linear CKA between (n,d1) and (n,d2). Returns scalar tensor."""
#     if X.ndim > 2: X = X.flatten(1)
#     if Y.ndim > 2: Y = Y.flatten(1)
#     Kx = X @ X.T
#     Ky = Y @ Y.T
#     Kxc = _center_gram(Kx)
#     Kyc = _center_gram(Ky)
#     num = (Kxc * Kyc).sum()
#     den = torch.linalg.norm(Kxc) * torch.linalg.norm(Kyc) + 1e-12
#     return num / den


# =============================
# cka.py
# =============================
from __future__ import annotations
from typing import Optional
import torch

def _gram_linear(X: torch.Tensor) -> torch.Tensor:
    return X @ X.T

def _center_gram(G: torch.Tensor) -> torch.Tensor:
    n = G.size(0)
    unit = torch.ones((n, n), device=G.device, dtype=G.dtype)
    I = torch.eye(n, device=G.device, dtype=G.dtype)
    H = I - unit / n
    return H @ G @ H

def linear_cka(A: torch.Tensor, B: torch.Tensor, debiased: bool = True) -> torch.Tensor:
    """Linear CKA between feature matrices A,B with rows as samples.
    A,B: [n, d]
    """
    GA = _center_gram(_gram_linear(A))
    GB = _center_gram(_gram_linear(B))
    # Frobenius inner products
    hsic = (GA * GB).sum()
    normA = torch.linalg.matrix_norm(GA, ord='fro')
    normB = torch.linalg.matrix_norm(GB, ord='fro')
    cka = hsic / (normA * normB + 1e-12)
    if not debiased:
        return cka
    # Simple shrinkage towards zero for small-sample bias (lightweight surrogate)
    n = A.size(0)
    bias = 1.0 - 2.0 / (n - 1)
    return cka * bias


def rbf_cka(A: torch.Tensor, B: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    # median heuristic
    with torch.no_grad():
        d2 = torch.cdist(A, A) ** 2
        med2 = d2.median()
    if gamma is None:
        gamma = 0.5 / (med2 + 1e-8)
    KA = torch.exp(-gamma * torch.cdist(A, A) ** 2)
    KB = torch.exp(-gamma * torch.cdist(B, B) ** 2)
    KA = _center_gram(KA)
    KB = _center_gram(KB)
    hsic = (KA * KB).sum()
    normA = torch.linalg.matrix_norm(KA, ord='fro')
    normB = torch.linalg.matrix_norm(KB, ord='fro')
    return hsic / (normA * normB + 1e-12)


@torch.no_grad()
def compute_cka(
    reps_A: torch.Tensor,
    reps_B: torch.Tensor,
    kernel: str = 'linear',
    debiased: bool = True,
) -> float:
    """Public API for CKA between two representation matrices.
    reps_A, reps_B: [n, d]
    """
    assert reps_A.shape[0] == reps_B.shape[0], "CKA requires same #samples"
    if kernel == 'linear':
        return float(linear_cka(reps_A, reps_B, debiased=debiased).item())
    elif kernel == 'rbf':
        return float(rbf_cka(reps_A, reps_B).item())
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")
