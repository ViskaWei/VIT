
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

KernelName = Literal["rbf", "poly", "linear"]


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean distances between rows of x and y.
    x: [N, D], y: [M, D] -> [N, M]"""
    x2 = (x * x).sum(dim=1, keepdim=True)        # [N, 1]
    y2 = (y * y).sum(dim=1, keepdim=True).T      # [1, M]
    xy = x @ y.T                                  # [N, M]
    return torch.clamp(x2 + y2 - 2.0 * xy, min=0.0)


def kernel(x: torch.Tensor,
           y: torch.Tensor,
           name: KernelName = "rbf",
           gamma: Optional[float] = None,
           degree: int = 3,
           coef0: float = 1.0) -> torch.Tensor:
    """Compute kernel matrix K_ij = k(x_i, y_j).
    - rbf:  exp(-gamma * ||x - y||^2), default gamma = 1 / median(||x - y||^2)
    - poly: (gamma * x·y + coef0)^degree
    - linear: x·y"""
    if name == "linear":
        return x @ y.T
    elif name == "poly":
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        return (gamma * (x @ y.T) + coef0) ** degree
    elif name == "rbf":
        if gamma is None:
            with torch.no_grad():
                xs = x[::max(1, x.shape[0] // 4096)]
                ys = y[::max(1, y.shape[0] // 4096)]
                d2 = _pairwise_sq_dists(xs, ys).flatten()
                med = torch.median(d2)
                gamma = 1.0 / (med + 1e-8)
        return torch.exp(-gamma * _pairwise_sq_dists(x, y))
    else:
        raise ValueError(f"Unknown kernel: {name}")


def center_gram_train(K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Center Gram matrix for training (landmark) set.
    Returns:
      Kc: centered Gram matrix
      row_means: per-column mean of original K, shape [M]
      K_mean: scalar mean of all entries of K"""
    M = K.shape[0]
    row_means = K.mean(dim=0)                     # [M]
    K_mean = row_means.mean().item()              # scalar
    ones = torch.ones((M, M), dtype=K.dtype, device=K.device) / M
    Kc = K - ones @ K - K @ ones + ones @ K @ ones
    Kc = 0.5 * (Kc + Kc.T)                        # numerical symmetrization
    return Kc, row_means, K_mean


def center_kvec_test(k_xy: torch.Tensor,
                     row_means: torch.Tensor,
                     K_mean: float) -> torch.Tensor:
    """Center kernel vector between test batch X and landmark set Y.
      k_xy: [B, M] with entries k(x_b, y_m)
      row_means: [M] mean over rows/cols of K(Y,Y)
      K_mean: scalar mean of K(Y,Y)"""
    mean_b = k_xy.mean(dim=1, keepdim=True)              # [B, 1]
    k_c = k_xy - mean_b - row_means.unsqueeze(0) + K_mean
    return k_c


@dataclass
class KernelPCAState:
    landmarks: torch.Tensor       # [M, D]
    A: torch.Tensor               # [M, r]  == eigvecs / sqrt(eigvals)
    row_means: torch.Tensor       # [M]
    K_mean: float                 # scalar
    kernel_name: KernelName
    gamma: Optional[float] = None
    degree: int = 3
    coef0: float = 1.0
    r: int = 32

    def to(self, device: torch.device) -> "KernelPCAState":
        self.landmarks = self.landmarks.to(device)
        self.A = self.A.to(device)
        self.row_means = self.row_means.to(device)
        return self

    def cpu(self) -> "KernelPCAState":
        return self.to(torch.device("cpu"))

    @torch.no_grad()
    def transform(self, X: torch.Tensor, chunk: int = 0) -> torch.Tensor:
        """Compute KPCA coordinates for X using Nyström approximation and
        training centering statistics. X: [..., D] -> Z: [..., r]"""
        shape = X.shape
        D = shape[-1]
        Xf = X.reshape(-1, D)
        if chunk and Xf.shape[0] > chunk:
            outs = []
            for i in range(0, Xf.shape[0], chunk):
                outs.append(self.transform(Xf[i:i+chunk], chunk=0))
            return torch.cat(outs, dim=0).reshape(*shape[:-1], -1)

        Kxy = kernel(Xf, self.landmarks, self.kernel_name, self.gamma, self.degree, self.coef0)  # [N, M]
        Kxy_c = center_kvec_test(Kxy, self.row_means, self.K_mean)                                # [N, M]
        Z = Kxy_c @ self.A                                                                        # [N, r]
        return Z.reshape(*shape[:-1], self.A.shape[1])

    def save(self, path: str):
        torch.save({
            "landmarks": self.landmarks.cpu(),
            "A": self.A.cpu(),
            "row_means": self.row_means.cpu(),
            "K_mean": self.K_mean,
            "kernel_name": self.kernel_name,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "r": self.r,
        }, path)

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> "KernelPCAState":
        obj = torch.load(path, map_location=map_location)
        return KernelPCAState(
            landmarks=obj["landmarks"],
            A=obj["A"],
            row_means=obj["row_means"],
            K_mean=obj["K_mean"],
            kernel_name=obj["kernel_name"],
            gamma=obj.get("gamma", None),
            degree=obj.get("degree", 3),
            coef0=obj.get("coef0", 1.0),
            r=obj.get("r", obj["A"].shape[1]),
        )


@torch.no_grad()
def fit_kpca_nystrom(X: torch.Tensor,
                     r: int,
                     m_landmarks: Optional[int] = None,
                     kernel_name: KernelName = "rbf",
                     gamma: Optional[float] = None,
                     degree: int = 3,
                     coef0: float = 1.0,
                     seed: int = 0,
                     device: Optional[torch.device] = None) -> KernelPCAState:
    """Fit KPCA with Nyström approximation on X of shape [N, D].
    Steps:
      1) Sample M landmarks (uniform without replacement).
      2) Compute K = k(L, L), center it.
      3) Eigendecompose centered K, take top-r components.
      4) Store A = eigvecs / sqrt(eigvals), row/overall means for test centering."""
    if device is None:
        device = X.device
    N, D = X.shape
    if m_landmarks is None or m_landmarks > N:
        m_landmarks = N

    # sample landmarks
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(N, generator=g)[:m_landmarks]
    L = X[idx].to(device)  # [M, D]

    # compute K and center
    K = kernel(L, L, kernel_name, gamma, degree, coef0)
    Kc, row_means, K_mean = center_gram_train(K)

    # eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(Kc)      # ascending
    eigvals = eigvals.clamp(min=1e-9)
    top = min(r, eigvals.numel())
    eigvals_top = eigvals[-top:]
    eigvecs_top = eigvecs[:, -top:]

    # projection = Kc @ (eigvec / sqrt(eigval))
    A = eigvecs_top / torch.sqrt(eigvals_top).unsqueeze(0)

    state = KernelPCAState(landmarks=L, A=A, row_means=row_means, K_mean=float(K_mean),
                           kernel_name=kernel_name, gamma=gamma, degree=degree, coef0=coef0, r=top)
    return state
