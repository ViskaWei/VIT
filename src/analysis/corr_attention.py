from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PCAResult:
    components: torch.Tensor  # (D, r)
    singular_values: torch.Tensor  # (r,)
    explained_variance: torch.Tensor  # (r,)
    explained_variance_ratio: torch.Tensor  # (r,)
    mean: torch.Tensor  # (D,)
    scale: torch.Tensor | None  # (D,) when column-standardized
    n_samples: int


@dataclass
class SpectralLine:
    name: str
    wavelength: float
    window: float = 1.0  # +/- window in Angstrom for highlighting


DEFAULT_LINES: tuple[SpectralLine, ...] = (
    # Balmer (may fall outside current wavelength coverage but kept for completeness)
    SpectralLine("Halpha", 6562.8, 3.0),
    SpectralLine("Hbeta", 4861.3, 3.0),
    SpectralLine("Hgamma", 4340.5, 3.0),
    SpectralLine("Hdelta", 4101.7, 3.0),
    # Paschen series prominent in the NIR window
    SpectralLine("Pa12", 8750.5, 3.0),
    SpectralLine("Pa13", 8665.0, 3.0),
    SpectralLine("Pa14", 8598.4, 3.0),
    SpectralLine("Pa15", 8545.4, 3.0),
    SpectralLine("Pa16", 8502.5, 3.0),
    SpectralLine("Pa17", 8467.3, 3.0),
    SpectralLine("Pa18", 8437.9, 3.0),
    # Ca II triplet
    SpectralLine("CaII-8498", 8498.02, 2.0),
    SpectralLine("CaII-8542", 8542.09, 2.0),
    SpectralLine("CaII-8662", 8662.14, 2.0),
)


def load_flux_matrix(
    h5_path: str,
    limit: int | None = None,
    clip_min: float | None = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load flux matrix (N, D) and wavelength grid (D,) from BOSZ-style HDF5."""
    with h5py.File(h5_path, "r") as handle:
        flux_ds = handle["dataset/arrays/flux/value"]
        wave_ds = handle["spectrumdataset/wave"]
        if limit is not None:
            flux_np = flux_ds[:limit]
        else:
            flux_np = flux_ds[:]
        wave = torch.from_numpy(wave_ds[:]).to(torch.float32)
    flux = torch.from_numpy(np.asarray(flux_np, dtype=np.float32))
    if clip_min is not None:
        flux = flux.clamp_min(clip_min)
    return flux, wave


def standardize_columns(
    X: torch.Tensor,
    center: bool = True,
    scale: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Zero-center (and optionally unit-scale) columns of X."""
    if X.dim() != 2:
        raise ValueError("Input must be 2D (N, D)")
    if center:
        mean = X.mean(dim=0)
        Xc = X - mean
    else:
        mean = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
        Xc = X
    scale_vec: torch.Tensor | None = None
    if scale:
        std = Xc.std(dim=0, unbiased=False)
        std = torch.where(std < eps, torch.ones_like(std), std)
        Xc = Xc / std
        scale_vec = std
    return Xc, mean, scale_vec


def compute_pca(
    X: torch.Tensor,
    k_max: int | None = None,
    center: bool = True,
    scale: bool = False,
) -> tuple[PCAResult, torch.Tensor]:
    """Compute PCA via SVD on X and return PCAResult plus centered data."""
    Xc, mean, scale_vec = standardize_columns(X, center=center, scale=scale)
    n, d = Xc.shape
    if k_max is None:
        k_max = min(n, d)
    k_max = min(k_max, min(n, d))
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    # torch.linalg.svd returns Vh with shape (min(n,d), d)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    S = S[:k_max]
    V = Vh[:k_max].T[:, :k_max]
    # Explained variance (per component)
    denom = max(n - 1, 1)
    explained_var = (S**2) / denom
    total_var = explained_var.sum().clamp(min=1e-12)
    explained_ratio = explained_var / total_var
    pca = PCAResult(
        components=V,
        singular_values=S,
        explained_variance=explained_var,
        explained_variance_ratio=explained_ratio,
        mean=mean,
        scale=scale_vec,
        n_samples=n,
    )
    return pca, Xc


def covariance_from_pca(
    pca: PCAResult,
    k: int | None = None,
    normalization: str = "sample",
) -> torch.Tensor:
    """Reconstruct covariance (or Gram) matrix from top-k components."""
    if k is None:
        k = pca.components.shape[1]
    if k <= 0 or k > pca.components.shape[1]:
        raise ValueError("k must be between 1 and available components")
    V = pca.components[:, :k]
    S = pca.singular_values[:k]
    cov = V @ torch.diag(S**2) @ V.T
    if normalization == "sample":
        cov = cov / max(pca.n_samples - 1, 1)
    elif normalization == "population":
        cov = cov / max(pca.n_samples, 1)
    elif normalization == "none":
        pass
    else:
        raise ValueError("normalization must be 'sample', 'population', or 'none'")
    return cov

def covariance_series(
    pca: PCAResult,
    ranks: Sequence[int],
    normalization: str = "sample",
    zero_diagonal: bool = False,
) -> dict[int, torch.Tensor]:
    mats: dict[int, torch.Tensor] = {}
    for k in ranks:
        cov = covariance_from_pca(pca, k=k, normalization=normalization)
        if zero_diagonal:
            cov = zero_diag(cov)
        mats[int(k)] = cov
    return mats


def correlation_from_covariance(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert covariance matrix to correlation matrix."""
    if cov.dim() != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    diag = torch.diagonal(cov)
    diag = torch.where(diag.abs() < eps, torch.full_like(diag, eps), diag)
    inv_std = diag.rsqrt()
    corr = cov * inv_std.unsqueeze(0) * inv_std.unsqueeze(1)
    return corr


def zero_diag(mat: torch.Tensor) -> torch.Tensor:
    return mat - torch.diag(torch.diagonal(mat))


def row_softmax(mat: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = mat / temperature
    scaled = scaled - scaled.max(dim=1, keepdim=True).values
    exp = scaled.exp()
    denom = exp.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return exp / denom


def downsample_matrix(mat: torch.Tensor, target_size: int) -> torch.Tensor:
    """Average-pool square matrix to target_size x target_size."""
    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square")
    n = mat.shape[0]
    if n % target_size != 0:
        raise ValueError("target_size must divide matrix dimension")
    factor = n // target_size
    tensor = mat.unsqueeze(0).unsqueeze(0)
    pooled = F.avg_pool2d(tensor, kernel_size=factor, stride=factor)
    return pooled.squeeze(0).squeeze(0)


def softmax_and_downsample(
    mat: torch.Tensor,
    temperature: float,
    downsample_to: int | None = None,
) -> torch.Tensor:
    prob = row_softmax(mat, temperature=temperature)
    if downsample_to is not None:
        prob = downsample_matrix(prob, downsample_to)
    return prob

def softmax_series(
    mats: dict[int, torch.Tensor],
    temperature: float,
    downsample_to: int | None = None,
    zero_diagonal: bool = False,
) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for k, mat in mats.items():
        base = mat
        if zero_diagonal:
            base = zero_diag(base)
        prob = row_softmax(base, temperature=temperature)
        if downsample_to is not None:
            prob = downsample_matrix(prob, downsample_to)
        out[k] = prob
    return out


def find_spectral_lines(
    wavelengths: torch.Tensor,
    lines: Iterable[SpectralLine] = DEFAULT_LINES,
) -> list[dict]:
    """Map spectral features to nearest indices within the wavelength grid."""
    wave = wavelengths.detach().cpu().numpy()
    results: list[dict] = []
    n = wave.shape[0]
    for line in lines:
        idx = int(np.clip(np.searchsorted(wave, line.wavelength), 0, n - 1))
        center_lambda = wave[idx]
        within = abs(center_lambda - line.wavelength) <= line.window
        if not within:
            continue
        lower = line.wavelength - line.window / 2
        upper = line.wavelength + line.window / 2
        start = int(np.clip(np.searchsorted(wave, lower), 0, n - 1))
        end = int(np.clip(np.searchsorted(wave, upper), 0, n - 1))
        results.append(
            {
                "name": line.name,
                "wavelength": line.wavelength,
                "index": idx,
                "start": min(start, end),
                "end": max(start, end),
            }
        )
    return results

def patch_to_wavelength_mapping(
    signal_length: int,
    patch_size: int,
    stride: int,
    num_patches: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    '''Build D x P matrix distributing patch tokens over wavelength bins.'''
    if signal_length <= 0 or patch_size <= 0 or stride <= 0 or num_patches <= 0:
        raise ValueError('Invalid patch configuration')
    idx = torch.arange(signal_length, device=device)
    windows = idx.unfold(0, patch_size, stride)
    if windows.size(0) < num_patches:
        tail = idx[-patch_size:]
        repeat = num_patches - windows.size(0)
        pad = tail.repeat(repeat, 1)
        windows = torch.cat([windows, pad], dim=0)
    elif windows.size(0) > num_patches:
        windows = windows[:num_patches]
    mapping = torch.zeros(signal_length, windows.size(0), device=device, dtype=torch.float32)
    rows = windows.reshape(-1)
    cols = torch.arange(windows.size(0), device=device).repeat_interleave(windows.size(1))
    mapping.index_put_((rows, cols), torch.ones_like(rows, dtype=torch.float32), accumulate=True)
    coverage = mapping.sum(dim=1, keepdim=True).clamp_min(1.0)
    mapping = mapping / coverage
    return mapping


def project_patch_attention(
    attn: torch.Tensor,
    mapping: torch.Tensor,
    drop_cls: bool = True,
) -> torch.Tensor:
    '''Project (B,H,T,T) or (T,T) attention onto wavelength grid using mapping.'''
    if attn.dim() == 4:
        if drop_cls:
            attn = attn[:, :, 1:, 1:]
        attn_flat = attn.mean(dim=(0, 1))
    elif attn.dim() == 2:
        attn_flat = attn
        if drop_cls and attn_flat.shape[0] == mapping.shape[1] + 1:
            attn_flat = attn_flat[1:, 1:]
    else:
        raise ValueError('attention tensor must be (B,H,T,T) or (T,T)')
    if attn_flat.shape[0] != mapping.shape[1]:
        raise ValueError('Attention token dimension and mapping columns must match')
    return mapping @ attn_flat @ mapping.T


def rowwise_correlation(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if A.shape != B.shape:
        raise ValueError('Matrices must share shape')
    A_c = A - A.mean(dim=1, keepdim=True)
    B_c = B - B.mean(dim=1, keepdim=True)
    denom = (A_c.norm(dim=1) * B_c.norm(dim=1)).clamp_min(eps)
    return (A_c * B_c).sum(dim=1) / denom


def topk_overlap(
    A: torch.Tensor,
    B: torch.Tensor,
    k: int,
    ignore_diagonal: bool = True,
) -> torch.Tensor:
    if A.shape != B.shape:
        raise ValueError('Matrices must share shape')
    if k <= 0 or k >= A.shape[1]:
        raise ValueError('k must be between 1 and matrix dimension - 1')
    if ignore_diagonal:
        diag = torch.arange(A.shape[0])
        A = A.clone()
        B = B.clone()
        A[diag, diag] = float('-inf')
        B[diag, diag] = float('-inf')
    topA = torch.topk(A, k=k, dim=1).indices
    topB = torch.topk(B, k=k, dim=1).indices
    overlaps = []
    for idx_row in range(A.shape[0]):
        overlaps.append(torch.isin(topA[idx_row], topB[idx_row]).float().mean())
    return torch.stack(overlaps)


def cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> float:
    flat_a = A.flatten()
    flat_b = B.flatten()
    return torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0).item()


def frobenius_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    return torch.linalg.matrix_norm(A - B, ord='fro').item()


def cka_similarity(A: torch.Tensor, B: torch.Tensor) -> float:
    from src.cka import compute_cka

    return compute_cka(A, B, kernel='linear', debiased=True)


def attention_from_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    num_heads: int,
    temperature: float | None = None,
) -> torch.Tensor:
    """Compute attention weights from saved query/key activations."""
    if query.shape != key.shape:
        raise ValueError("query and key must have same shape")
    if query.dim() != 3:
        raise ValueError("Expected (B, T, D) tensors")
    bsz, tokens, dim = query.shape
    if dim % num_heads != 0:
        raise ValueError("Hidden size must be divisible by num_heads")
    head_dim = dim // num_heads
    q = query.view(bsz, tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    k = key.view(bsz, tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if temperature is not None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        attn_scores = attn_scores / temperature
    attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values
    probs = attn_scores.exp()
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs / denom


def aggregate_heads(attn: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Aggregate head dimension (B, H, T, T) into (T, T)."""
    if attn.dim() != 4:
        raise ValueError("Expected attention tensor with shape (B, H, T, T)")
    if mode == "mean":
        return attn.mean(dim=(0, 1))
    if mode == "max":
        return attn.max(dim=1).values.mean(dim=0)
    if mode == "first":
        return attn[0, 0]
    raise ValueError("Unsupported aggregation mode")


__all__ = [
    "PCAResult",
    "SpectralLine",
    "DEFAULT_LINES",
    "load_flux_matrix",
    "standardize_columns",
    "compute_pca",
    "covariance_from_pca",
    "correlation_from_covariance",
    "zero_diag",
    "row_softmax",
    "downsample_matrix",
    "softmax_and_downsample",
    "covariance_series",
    "softmax_series",
    "find_spectral_lines",
    "attention_from_qk",
    "aggregate_heads",
    "patch_to_wavelength_mapping",
    "project_patch_attention",
    "rowwise_correlation",
    "topk_overlap",
    "cosine_similarity",
    "frobenius_distance",
    "cka_similarity",
]
