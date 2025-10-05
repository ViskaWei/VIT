"""Unified spectral preprocessing utilities.

This module centralises data loading and common preprocessing routines such as
PCA, KPCA, PCP, CKA, and ZCA so that experiments share a single, well-tested
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple
import warnings

import h5py
import numpy as np
import torch


Tensor = torch.Tensor
KernelName = Literal["rbf", "poly", "linear"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_spectra(
    file_path: str | Path,
    *,
    dataset_key: str = "dataset/arrays/flux/value",
    wave_key: str = "spectrumdataset/wave",
    error_key: Optional[str] = "dataset/arrays/error/value",
    num_samples: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tensor]:
    """Load spectra, wavelength grid, and optional errors from an HDF5 file."""

    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        flux = torch.tensor(f[dataset_key][:num_samples], dtype=dtype)
        wave = torch.tensor(f[wave_key][()], dtype=dtype)
        payload: Dict[str, Tensor] = {"flux": flux, "wave": wave}
        if error_key and error_key in f:
            payload["error"] = torch.tensor(f[error_key][:num_samples], dtype=dtype)
    return payload


def _sorted_eigh_sym(cov: Tensor) -> Tuple[Tensor, Tensor]:
    cov_sym = 0.5 * (cov + cov.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(cov_sym)
    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx]


def ensure_covariance(
    data: Tensor,
    cov_path: Optional[Path],
    *,
    allow_compute: bool = True,
    wave: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Return covariance stats, computing and persisting them if needed.

    Parameters
    ----------
    data:
        Input spectra arranged as `[num_samples, num_pixels]`.
    cov_path:
        Filesystem location to persist covariance statistics.
    allow_compute:
        When ``True`` the covariance is recomputed if missing; otherwise it must exist.
    wave:
        Optional wavelength grid matching the covariance dimension. When provided the
        covariance heatmap X/Y tick labels are annotated with the corresponding
        wavelengths for readability.
    """
    # Import here to avoid circular dependency
    from src.prepca.preprocessor_utils import load_or_compute_covariance
    
    if cov_path is None:
        if not allow_compute:
            raise FileNotFoundError("cov_path must be provided when allow_compute=False")
        # Use default path if computing
        cov_path = Path("data/pca/covariance_stats.pt")

    cov_path = Path(cov_path)
    
    # Use the new utility function
    if cov_path.exists():
        # Load existing
        cov_stats = load_or_compute_covariance(cov_path=cov_path)
    elif allow_compute:
        # Compute and save
        cov_stats = load_or_compute_covariance(
            cov_path=None,
            data=data,
            save_path=cov_path,
            wave=wave
        )
    else:
        raise FileNotFoundError(f"Covariance file {cov_path} not found and computation disabled")
    
    # Return in the old dict format for backward compatibility
    return {
        "mean": cov_stats.mean,
        "cov": cov_stats.cov,
        "num_samples": torch.tensor(cov_stats.num_samples),
        "eigvals": cov_stats.eigvals,
        "eigvecs": cov_stats.eigvecs,
    }


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def compute_pca(
    spectra: Tensor,
    *,
    patch_size: int,
    step: Optional[int] = None,
    limit: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """Compute PCA basis over flattened spectral patches."""

    if spectra.ndim != 2:
        raise ValueError(f"Expected [N, L] tensor, got {tuple(spectra.shape)}")
    if limit is not None and 0 < limit < spectra.shape[0]:
        spectra = spectra[:limit]

    step = int(step) if (step is not None and int(step) > 0) else int(patch_size)
    patches = spectra.unfold(1, patch_size, step).contiguous().view(-1, patch_size)

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    patches = patches.to(dev)
    with torch.no_grad():
        try:
            U, S, V = torch.pca_lowrank(patches, q=patch_size, center=True)
        except RuntimeError:
            patches = patches.to(torch.device("cpu"))
            U, S, V = torch.pca_lowrank(patches, q=patch_size, center=True)

    V = V.contiguous().cpu()
    S = S[:patch_size].contiguous().cpu()
    U = U[:, :patch_size].contiguous().cpu()
    mean = patches.mean(dim=0).cpu()
    evr = (S ** 2)
    evr = evr / evr.sum() if float(evr.sum()) > 0 else evr

    return {
        "components": V,
        "scores": U,
        "singular_values": S,
        "mean": mean,
        "explained_variance_ratio": evr,
        "patch_size": torch.tensor(patch_size),
        "step": torch.tensor(step),
        "num_patches": torch.tensor(patches.shape[0]),
    }


# ---------------------------------------------------------------------------
# KPCA (Nyström)
# ---------------------------------------------------------------------------
def _pairwise_sq_dists(x: Tensor, y: Tensor) -> Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    xy = x @ y.T
    return torch.clamp(x2 + y2 - 2.0 * xy, min=0.0)


def _kernel(
    x: Tensor,
    y: Tensor,
    *,
    name: KernelName = "rbf",
    gamma: Optional[float] = None,
    degree: int = 3,
    coef0: float = 1.0,
) -> Tensor:
    if name == "linear":
        return x @ y.T
    if name == "poly":
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        return (gamma * (x @ y.T) + coef0) ** degree
    if name == "rbf":
        if gamma is None:
            with torch.no_grad():
                xs = x[::max(1, x.shape[0] // 4096)]
                ys = y[::max(1, y.shape[0] // 4096)]
                d2 = _pairwise_sq_dists(xs, ys).flatten()
                med = torch.median(d2)
                gamma = 1.0 / (med + 1e-8)
        return torch.exp(-gamma * _pairwise_sq_dists(x, y))
    raise ValueError(f"Unknown kernel: {name}")


def _center_gram_train(K: Tensor) -> Tuple[Tensor, Tensor, float]:
    M = K.shape[0]
    row_means = K.mean(dim=0)
    K_mean = row_means.mean().item()
    ones = torch.ones((M, M), dtype=K.dtype, device=K.device) / M
    Kc = K - ones @ K - K @ ones + ones @ K @ ones
    Kc = 0.5 * (Kc + Kc.T)
    return Kc, row_means, K_mean


def _center_kvec_test(k_xy: Tensor, row_means: Tensor, K_mean: float) -> Tensor:
    mean_b = k_xy.mean(dim=1, keepdim=True)
    return k_xy - mean_b - row_means.unsqueeze(0) + K_mean


@dataclass
class KernelPCAState:
    landmarks: Tensor
    A: Tensor
    row_means: Tensor
    K_mean: float
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
    def transform(self, X: Tensor, chunk: int = 0) -> Tensor:
        shape = X.shape
        D = shape[-1]
        Xf = X.reshape(-1, D)
        if chunk and Xf.shape[0] > chunk:
            outs = [self.transform(Xf[i:i + chunk], chunk=0) for i in range(0, Xf.shape[0], chunk)]
            return torch.cat(outs, dim=0).reshape(*shape[:-1], -1)
        Kxy = _kernel(Xf, self.landmarks, name=self.kernel_name, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        Kxy_c = _center_kvec_test(Kxy, self.row_means, self.K_mean)
        Z = Kxy_c @ self.A
        return Z.reshape(*shape[:-1], self.A.shape[1])

    def save(self, path: str) -> None:
        torch.save(
            {
                "landmarks": self.landmarks.cpu(),
                "A": self.A.cpu(),
                "row_means": self.row_means.cpu(),
                "K_mean": self.K_mean,
                "kernel_name": self.kernel_name,
                "gamma": self.gamma,
                "degree": self.degree,
                "coef0": self.coef0,
                "r": self.r,
            },
            path,
        )

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
def compute_kernel_pca(
    spectra: Tensor,
    *,
    r: int,
    landmarks: Optional[int] = None,
    kernel_name: KernelName = "rbf",
    gamma: Optional[float] = None,
    degree: int = 3,
    coef0: float = 1.0,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> KernelPCAState:
    device = device or spectra.device
    N, _ = spectra.shape
    m_landmarks = min(landmarks or N, N)
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(N, generator=g)[:m_landmarks]
    L = spectra[idx].to(device)
    K = _kernel(L, L, name=kernel_name, gamma=gamma, degree=degree, coef0=coef0)
    Kc, row_means, K_mean = _center_gram_train(K)
    eigvals, eigvecs = torch.linalg.eigh(Kc)
    eigvals = eigvals.clamp(min=1e-9)
    top = min(r, eigvals.numel())
    eigvals_top = eigvals[-top:]
    eigvecs_top = eigvecs[:, -top:]
    A = eigvecs_top / torch.sqrt(eigvals_top).unsqueeze(0)
    return KernelPCAState(
        landmarks=L,
        A=A,
        row_means=row_means,
        K_mean=float(K_mean),
        kernel_name=kernel_name,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        r=top,
    )


# ---------------------------------------------------------------------------
# ZCA Whitening
# ---------------------------------------------------------------------------
@dataclass
class ZCAState:
    mean: Tensor
    whitening: Tensor
    covariance: Tensor
    eigenvectors: Tensor
    eigenvalues: Tensor
    projector: Optional[Tensor]
    metadata: Dict[str, object]


class ZCAWhitening:
    def __init__(
        self,
        *,
        gamma: float = 0.0,
        eps: float = 1e-5,
        rank: Optional[int] = None,
        alpha: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.rank = rank
        self.alpha = float(alpha)
        self.device = device
        self.dtype = dtype
        self._state: Optional[ZCAState] = None

    def fit(self, data: Tensor) -> "ZCAWhitening":
        if data.ndim != 2:
            raise ValueError("Expected 2D tensor for ZCA fit")
        x = data.to(device=self.device, dtype=self.dtype)
        if x.shape[0] < 2:
            raise ValueError("Need at least two samples to compute covariance")
        mean = x.mean(dim=0)
        centered = x - mean
        cov = centered.t().matmul(centered) / (x.shape[0] - 1)
        if self.gamma > 0:
            diag = torch.diag(torch.diag(cov))
            cov = (1.0 - self.gamma) * cov + self.gamma * diag
        d = cov.shape[0]
        cov = cov + self.eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=self.eps)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        inv_sqrt = torch.rsqrt(eigvals)
        whitening = eigvecs @ torch.diag(inv_sqrt) @ eigvecs.t()
        projector = None
        if self.rank is not None:
            r = max(1, min(int(self.rank), d))
            lead_vecs = eigvecs[:, :r]
            lead_vals = eigvals[:r]
            inv_sqrt_r = torch.rsqrt(lead_vals)
            low_rank_whiten = lead_vecs @ torch.diag(inv_sqrt_r) @ lead_vecs.t()
            projector = lead_vecs.t().contiguous()
            identity = torch.eye(d, device=lead_vecs.device, dtype=lead_vecs.dtype)
            whitening = low_rank_whiten + self.alpha * (identity - lead_vecs @ lead_vecs.t())
        self._state = ZCAState(
            mean=mean,
            whitening=whitening,
            covariance=cov,
            eigenvectors=eigvecs,
            eigenvalues=eigvals,
            projector=projector,
            metadata={
                "gamma": self.gamma,
                "eps": self.eps,
                "rank": self.rank,
                "alpha": self.alpha,
                "dtype": str(self.dtype).split(".")[-1],
            },
        )
        return self

    def fit_transform(self, data: Tensor) -> Tensor:
        return self.fit(data).transform(data)

    def _require_state(self) -> ZCAState:
        if self._state is None:
            raise RuntimeError("ZCAWhitening has not been fitted yet")
        return self._state

    def transform(self, data: Tensor) -> Tensor:
        state = self._require_state()
        x = data.to(device=state.mean.device, dtype=state.mean.dtype)
        return (x - state.mean) @ state.whitening.t()

    def inverse_transform(self, data: Tensor) -> Tensor:
        state = self._require_state()
        eigvecs = state.eigenvectors
        eigvals = state.eigenvalues
        sqrt_vals = torch.sqrt(eigvals)
        dewhitener = eigvecs @ torch.diag(sqrt_vals) @ eigvecs.t()
        x = data.to(device=state.mean.device, dtype=state.mean.dtype)
        return x @ dewhitener.t() + state.mean

    def project(self, data: Tensor) -> Tensor:
        state = self._require_state()
        if state.projector is None:
            raise RuntimeError("No projector available; fit with rank != None to enable")
        x = data.to(device=state.mean.device, dtype=state.mean.dtype)
        return (x - state.mean) @ state.projector.t()

    def save(self, path: str | Path) -> None:
        state = self._require_state()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": state.mean.detach().cpu(),
            "whitening": state.whitening.detach().cpu(),
            "covariance": state.covariance.detach().cpu(),
            "eigenvectors": state.eigenvectors.detach().cpu(),
            "eigenvalues": state.eigenvalues.detach().cpu(),
            "projector": None if state.projector is None else state.projector.detach().cpu(),
            "metadata": state.metadata,
        }
        torch.save(payload, target)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: Optional[torch.device | str] = None,
    ) -> "ZCAWhitening":
        payload = torch.load(path, map_location=map_location)
        metadata = payload.get("metadata", {})
        dtype_name = metadata.get("dtype", "float32")
        dtype = getattr(torch, dtype_name)
        obj = cls(
            gamma=metadata.get("gamma", 0.0),
            eps=metadata.get("eps", 1e-5),
            rank=metadata.get("rank", None),
            alpha=metadata.get("alpha", 0.0),
            dtype=dtype,
        )
        obj._state = ZCAState(
            mean=payload["mean"],
            whitening=payload["whitening"],
            covariance=payload["covariance"],
            eigenvectors=payload["eigenvectors"],
            eigenvalues=payload["eigenvalues"],
            projector=payload.get("projector", None),
            metadata=metadata,
        )
        return obj

    @property
    def whitening_matrix(self) -> Tensor:
        return self._require_state().whitening


# ---------------------------------------------------------------------------
# PCP (Principal Component Pursuit)
# ---------------------------------------------------------------------------
def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def _svt(M: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return U, s_thresh, Vt


def compute_pcp(
    spectra: Tensor,
    *,
    lambda_: Optional[float] = None,
    mu: Optional[float] = None,
    tol: float = 1e-7,
    max_iter: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Iterable[float]]]:
    D = spectra.detach().cpu().numpy().astype(np.float32, copy=True)
    n, m = D.shape
    normD = np.linalg.norm(D, ord="fro")
    if lambda_ is None:
        lambda_ = 1.0 / np.sqrt(max(n, m))
    L = np.zeros_like(D)
    S = np.zeros_like(D)
    Y = np.zeros_like(D)
    if mu is None:
        x = np.random.randn(m).astype(D.dtype)
        for _ in range(5):
            x = D.T @ (D @ x)
            x /= np.linalg.norm(x) + 1e-12
        spectral = np.sqrt(np.linalg.norm(D @ x))
        mu = 1.25 / (spectral + 1e-12)
    mu_bar = mu * 1e7
    rho = 1.5
    history = {"primal_resid": [], "rank": [], "nnz": [], "iters": 0}
    for k in range(1, max_iter + 1):
        M = D - S + (1.0 / mu) * Y
        U, s, Vt = _svt(M, 1.0 / mu)
        L = (U * s) @ Vt
        r = D - L + (1.0 / mu) * Y
        S = _soft_threshold(r, lambda_ / mu)
        R = D - L - S
        Y = Y + mu * R
        primal_resid = np.linalg.norm(R, ord="fro") / (normD + 1e-12)
        rank = int((s > 0).sum())
        nnz = int((np.abs(S) > 0).sum())
        history["primal_resid"].append(float(primal_resid))
        history["rank"].append(rank)
        history["nnz"].append(nnz)
        history["iters"] = k
        if verbose and (k % 10 == 0 or primal_resid < tol):
            print(f"[PCP] iter={k:4d} resid={primal_resid:.3e} rank={rank} nnz={nnz} mu={mu:.3e}")
        if primal_resid < tol:
            break
        mu = min(mu * rho, mu_bar)
    return L, S, history


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------
def _gram_linear(X: Tensor) -> Tensor:
    return X @ X.T


def _center_gram(G: Tensor) -> Tensor:
    n = G.size(0)
    unit = torch.ones((n, n), device=G.device, dtype=G.dtype)
    I = torch.eye(n, device=G.device, dtype=G.dtype)
    H = I - unit / n
    return H @ G @ H


def compute_cka(A: Tensor, B: Tensor, *, kernel: str = "linear", debiased: bool = True) -> Tensor:
    if A.shape[0] != B.shape[0]:
        raise ValueError("CKA requires both inputs to have the same number of samples")
    if kernel == "linear":
        GA = _center_gram(_gram_linear(A))
        GB = _center_gram(_gram_linear(B))
    elif kernel == "rbf":
        KA = torch.exp(-torch.cdist(A, A) ** 2)
        KB = torch.exp(-torch.cdist(B, B) ** 2)
        GA = _center_gram(KA)
        GB = _center_gram(KB)
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")
    hsic = (GA * GB).sum()
    normA = torch.linalg.matrix_norm(GA, ord="fro")
    normB = torch.linalg.matrix_norm(GB, ord="fro")
    cka = hsic / (normA * normB + 1e-12)
    if not debiased:
        return cka
    n = A.size(0)
    bias = 1.0 - 2.0 / (n - 1)
    return cka * bias


# ---------------------------------------------------------------------------
# Pipeline façade
# ---------------------------------------------------------------------------
class PreprocessingPipeline:
    """Convenience façade exposing multiple preprocessing routines."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        dataset_key: str = "dataset/arrays/flux/value",
        wave_key: str = "spectrumdataset/wave",
        error_key: Optional[str] = "dataset/arrays/error/value",
        num_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        data = load_spectra(
            file_path,
            dataset_key=dataset_key,
            wave_key=wave_key,
            error_key=error_key,
            num_samples=num_samples,
            dtype=dtype,
        )
        self.flux = data["flux"]
        self.wave = data["wave"]
        self.error = data.get("error")

    def run(self, method: str, **kwargs):
        method = method.lower()
        if method == "pca":
            result = compute_pca(self.flux, **kwargs)
            result["wave"] = self.wave
            return result
        if method == "kpca":
            state = compute_kernel_pca(self.flux, **kwargs)
            return {"wave": self.wave, "state": state}
        if method == "zca":
            zca = ZCAWhitening(**kwargs).fit(self.flux)
            return {"wave": self.wave, "zca": zca}
        if method == "pcp":
            L, S, history = compute_pcp(self.flux, **kwargs)
            return {"wave": self.wave, "low_rank": L, "sparse": S, "history": history}
        if method == "cka":
            other = kwargs.get("other")
            if other is None:
                raise ValueError("'cka' requires 'other' tensor via kwargs")
            score = compute_cka(self.flux, other, kernel=kwargs.get("kernel", "linear"), debiased=kwargs.get("debiased", True))
            return {"cka": score}
        raise ValueError(f"Unknown preprocessing method '{method}'")


__all__ = [
    "PreprocessingPipeline",
    "KernelPCAState",
    "ZCAWhitening",
    "ZCAState",
    "load_spectra",
    "ensure_covariance",
    "compute_pca",
    "compute_kernel_pca",
    "compute_pcp",
    "compute_cka",
]
