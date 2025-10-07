"""Utilities shared across preprocessor fitting scripts.

This module centralises loading and validation of covariance statistics so that
scripts can reuse the same logic when preparing linear preprocessing operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import torch

from evals.zca_util import zca_report as compute_zca_report

Tensor = torch.Tensor


@dataclass
class CovarianceStats:
    """Container for the covariance statistics required by preprocessors."""

    mean: Tensor
    cov: Tensor
    num_samples: int
    eigvals: Tensor
    eigvecs: Tensor
    source_path: Optional[Path] = None

    def to(self, dtype: torch.dtype) -> "CovarianceStats":
        return CovarianceStats(
            mean=self.mean.to(dtype),
            cov=self.cov.to(dtype),
            num_samples=self.num_samples,
            eigvals=self.eigvals.to(dtype),
            eigvecs=self.eigvecs.to(dtype),
            source_path=self.source_path,
        )


def _sorted_eigh_sym(cov: Tensor) -> Tuple[Tensor, Tensor]:
    cov_sym = 0.5 * (cov + cov.t())
    eigvals, eigvecs = torch.linalg.eigh(cov_sym)
    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx], cov_sym


def _tensor_num_samples(value: Optional[Tensor]) -> int:
    if value is None:
        return -1
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def load_covariance_stats(cov_path: str | Path) -> CovarianceStats:
    path = Path(cov_path)
    if not path.exists():
        raise FileNotFoundError(f"Covariance file not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Covariance payload at {path} is not a dictionary")
    required = {"mean", "cov", "eigvals", "eigvecs"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"Covariance payload at {path} missing keys: {missing}")
    mean = payload["mean"].detach().clone()
    cov = payload["cov"].detach().clone()
    eigvals = payload["eigvals"].detach().clone()
    eigvecs = payload["eigvecs"].detach().clone()
    # Ensure eigenpairs exist even if the stored payload predates their inclusion.
    if eigvals.numel() == 0 or eigvecs.numel() == 0:
        eigvals, eigvecs, cov = _sorted_eigh_sym(cov)
    num_samples = _tensor_num_samples(payload.get("num_samples"))
    return CovarianceStats(
        mean=mean,
        cov=cov,
        num_samples=num_samples,
        eigvals=eigvals,
        eigvecs=eigvecs,
        source_path=path,
    )


def ensure_covariance_stats(
    data: Tensor,
    cov_path: str | Path,
    *,
    wave: Optional[Tensor] = None,
) -> CovarianceStats:
    """Legacy wrapper for load_or_compute_covariance for backward compatibility."""
    return load_or_compute_covariance(cov_path, data=data, save_path=cov_path, wave=wave)


def compute_whitening_metrics(
    matrix: Tensor,
    cov_used: Optional[Tensor],
    projector: Tensor,
    *,
    complement_scale: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    if cov_used is None:
        return None
    dim = matrix.shape[0]
    target = projector
    if complement_scale is not None:
        eye = torch.eye(dim, dtype=matrix.dtype, device=matrix.device)
        resid = eye - projector
        target = projector + (complement_scale ** 2) * (resid @ cov_used @ resid)
    diff = matrix.t() @ cov_used @ matrix - target
    max_abs = float(diff.abs().max().item())
    denom = float(target.abs().max().item())
    rel = max_abs / max(denom, 1e-12)
    return {"max_abs": max_abs, "rel_max_abs": rel}


def zca_self_check(
    P: Tensor,
    cov_hat: Tensor,
    *,
    eps: float = 0.0,
    lowrank: bool = False,
    Vr: Optional[Tensor] = None,
) -> Tuple[float, float, float]:
    Ihat = P.t() @ (cov_hat + eps * torch.eye(cov_hat.shape[0], dtype=cov_hat.dtype, device=cov_hat.device)) @ P
    if lowrank:
        if Vr is None:
            raise ValueError("Vr basis must be provided for low-rank checks")
        Ihat = Vr.t() @ Ihat @ Vr
        I = torch.eye(Vr.shape[1], dtype=P.dtype, device=P.device)
    else:
        I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
    rel = torch.linalg.norm(Ihat - I, "fro") / torch.linalg.norm(I, "fro")
    if float(rel) >= 0.1:
        raise AssertionError(f"ZCA self-check failed: rel={float(rel):.3e}")
    lam = torch.linalg.eigvalsh(cov_hat).clamp_min(1e-18)
    cond_before = float(lam.max() / lam.min())
    lam2 = torch.linalg.eigvalsh(0.5 * (Ihat + Ihat.t())).clamp_min(1e-18)
    cond_after = float(lam2.max() / lam2.min())
    if abs(cond_after) - 1 >= 0.1:
        raise AssertionError(f"ZCA self-check failed: cond_after={cond_after:.3e}")
    return float(rel), cond_before, cond_after


def compute_zca_diagnostics(
    flux: Optional[Tensor],
    mean: Tensor,
    cov_for_report: Tensor,
    matrix: Tensor,
    shrinkage: float,
) -> Optional[Dict[str, float]]:
    if flux is None:
        return None
    report = compute_zca_report(
        flux,
        mean,
        cov_for_report,
        matrix,
        float(shrinkage),
    )
    return {key: float(value) for key, value in report.items()}


def _sym_limits_from_percentiles(A: np.ndarray, clip: float = 1.0) -> Tuple[float, float]:
    """Calculate symmetric colormap limits from percentiles."""
    lo = float(np.nanpercentile(A, clip))
    hi = float(np.nanpercentile(A, 100 - clip))
    vmax = max(abs(lo), abs(hi))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(np.abs(A)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax


def plot_covariance_heatmap(
    cov: Tensor,
    output_path: Path,
    wave: Optional[Tensor] = None,
) -> None:
    """Plot and save covariance matrix heatmap.
    
    Parameters
    ----------
    cov : Tensor
        Covariance matrix to plot
    output_path : Path
        Path where heatmap image will be saved
    wave : Optional[Tensor]
        Optional wavelength grid for axis labels
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib import colors

        cov_np = cov.detach().cpu().numpy()
        vmin, vmax = _sym_limits_from_percentiles(cov_np, clip=3.0)
        if vmin >= vmax:
            vmax = vmin + 1e-6
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(cov_np, cmap="magma", aspect="auto", norm=norm)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        ax.set_title("Covariance Heatmap")

        tick_wave_np: Optional[np.ndarray] = None
        if wave is not None:
            wave_view = wave.detach().cpu().reshape(-1)
            if wave_view.numel() == cov_np.shape[0]:
                tick_wave_np = wave_view.numpy()
            else:
                warnings.warn(
                    "Provided wavelength grid does not match covariance dimension; using pixel indices instead",
                    RuntimeWarning,
                )

        if tick_wave_np is not None:
            max_ticks = 10 if tick_wave_np.size >= 10 else tick_wave_np.size
            idx = np.linspace(0, tick_wave_np.size - 1, max_ticks, dtype=int)
            ax.set_xticks(idx)
            ax.set_yticks(idx)
            tick_labels = [f"{tick_wave_np[i]:.0f}" for i in idx]
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            ax.set_yticklabels(tick_labels)
            ax.set_xlabel("Wavelength")
            ax.set_ylabel("Wavelength")
        else:
            ax.set_xlabel("Wavelength idx")
            ax.set_ylabel("Wavelength idx")

        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved covariance heatmap at {output_path}")
    except Exception as exc:
        warnings.warn(f"Failed to save covariance heatmap at {output_path}: {exc}")


def plot_eigenvalue_spectrum(
    eigvals: Tensor,
    output_path: Path,
    num_samples: Optional[int] = None,
) -> None:
    """Plot eigenvalue spectrum and remaining variance.
    
    Parameters
    ----------
    eigvals : Tensor
        Eigenvalues sorted in descending order
    output_path : Path
        Path where plot will be saved
    num_samples : Optional[int]
        Number of samples used (for plot title)
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        eigvals_np = eigvals.detach().cpu().numpy()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Plot 1: Normalized eigenvalue spectrum (log-log)
        eigval_ratio = eigvals / eigvals.sum()
        eigval_ratio_np = eigval_ratio.detach().cpu().numpy()
        
        ax1.plot(np.arange(len(eigval_ratio_np)), eigval_ratio_np, 'o-', 
                color='k', ms=2, lw=0.8, label='Eigenvalue ratio')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylabel('Eigenvalue Ratio')
        ax1.set_xlabel('Component Index')
        ax1.set_title('Eigenvalue Spectrum (Normalized)')

        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Remaining unexplained variance
        cumsum_ratio = torch.cumsum(eigvals, dim=0) / eigvals.sum()
        remaining_variance = 1 - cumsum_ratio
        remaining_np = remaining_variance.detach().cpu().numpy()

        ax2.plot(remaining_np, 'r-', lw=1.5, label='Remaining Var')
        ax2.set_yscale('log')
        ax2.set_ylabel(r'$1 \;- \; \sum^i \lambda_j \quad / \quad \sum^N \lambda_j$')
        ax2.set_xlabel('i-th component')
        caption = r'The plot shows $1 - \sum^i \lambda_j \; / \; \sum^N \lambda_j$, i.e., the remaining variance not captured by the first $i$ components. Vertical lines indicate the number of components needed to explain: '

        # Mark variance thresholds
        for i in range(1, 4):
            ratio = 10 ** (-i)
            idx = torch.argmin(torch.abs(remaining_variance - ratio)).item()
            explained_pct = 100 - 100 * ratio
            ax2.axvline(idx, color='b', linestyle='--', alpha=0.7, 
                       label=f'i={idx} explained {explained_pct:.1f}%')
            
            # Add annotation at intersection point - only show the ratio value
            y_val = remaining_variance[idx].item()
            ax2.plot(idx, y_val, 'o', color='gold', markersize=6, zorder=5)
            ax2.annotate(f'({idx}, {ratio})', 
                        xy=(idx, y_val), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1))
            
            caption += f' {explained_pct:.1f}% ({idx}), '
        
        title_suffix = f' ({num_samples:,} samples)' if num_samples else ''
        ax2.set_title(f'Remaining Unexplained Variance{title_suffix}')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        # ax2.set_xlim(0.1, len(eigvals_np))
        ax2.text(0.5, -0.2, caption, ha='center', va='top', fontsize=9, transform=ax2.transAxes, wrap=True)
        ax2.set_xscale('log')
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved eigenvalue spectrum plot at {output_path}")
    except Exception as exc:
        warnings.warn(f"Failed to save eigenvalue spectrum at {output_path}: {exc}")


def compute_covariance_stats(
    data: Tensor,
    save_path: Optional[Path] = None,
    wave: Optional[Tensor] = None,
    src_path: Optional[str | Path] = None,
) -> CovarianceStats:
    """Compute covariance statistics from data and optionally save them.
    
    Parameters
    ----------
    data : Tensor
        Input data arranged as [num_samples, num_features]
    save_path : Optional[Path]
        Path to save computed statistics. If None, uses default location.
    wave : Optional[Tensor]
        Optional wavelength grid for heatmap plotting
    src_path : Optional[str | Path]
        Source file path where the data came from (for metadata)
    
    Returns
    -------
    CovarianceStats
        Computed covariance statistics
    """
    data = data.to(torch.float32)
    mean = data.mean(dim=0, keepdim=False)
    centered = data - mean
    cov = centered.t().matmul(centered) / (centered.shape[0] - 1)
    eigvals, eigvecs, cov = _sorted_eigh_sym(cov)  # Ensure symmetry
    eigvals = eigvals.to(cov.dtype)
    eigvecs = eigvecs.to(cov.dtype)
    
    num_samples = data.shape[0]
    
    stats = CovarianceStats(
        mean=mean,
        cov=cov,
        num_samples=num_samples,
        eigvals=eigvals,
        eigvecs=eigvecs,
        source_path=save_path,
    )
    
    # Save if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "mean": mean.cpu(),
            "cov": cov.cpu(),
            "num_samples": torch.tensor(num_samples),
            "eigvals": eigvals.cpu(),
            "eigvecs": eigvecs.cpu(),
        }
        
        # Add source path metadata if provided
        if src_path is not None:
            payload["src_path"] = str(src_path)
        
        torch.save(payload, save_path)
        print(f"Saved covariance statistics to {save_path}")
        if src_path:
            print(f"  Source data: {src_path}")
        
        # Plot heatmap
        heatmap_path = save_path.with_name(f"{save_path.stem}_heatmap.png")
        plot_covariance_heatmap(cov, heatmap_path, wave=wave)
        
        # Plot eigenvalue spectrum
        eigval_path = save_path.with_name(f"{save_path.stem}_eigenvalues.png")
        plot_eigenvalue_spectrum(eigvals, eigval_path, num_samples=num_samples)
    
    return stats


def load_or_compute_covariance(
    cov_path: Optional[str | Path],
    data: Optional[Tensor] = None,
    save_path: Optional[str | Path] = None,
    wave: Optional[Tensor] = None,
    src_path: Optional[str | Path] = None,
) -> CovarianceStats:
    """Load covariance statistics from file or compute from data.
    
    Logic:
    1. If cov_path is provided and exists: load from file (no recomputation)
    2. If cov_path is not provided or doesn't exist: compute from data and save
    
    Parameters
    ----------
    cov_path : Optional[str | Path]
        Path to existing covariance file. If exists, data is loaded from here.
    data : Optional[Tensor]
        Input data for computing covariance if cov_path doesn't exist.
    save_path : Optional[str | Path]
        Path to save computed covariance. If None, defaults to cov_path.
    wave : Optional[Tensor]
        Optional wavelength grid for heatmap plotting.
    src_path : Optional[str | Path]
        Source file path where the data came from (for metadata).
    
    Returns
    -------
    CovarianceStats
        Loaded or computed covariance statistics.
    """
    # Try to load from cov_path if provided
    if cov_path is not None:
        cov_path = Path(cov_path)
        if cov_path.exists():
            print(f"Loading covariance statistics from {cov_path}")
            return load_covariance_stats(cov_path)
    
    # If we reach here, we need to compute
    if data is None:
        raise ValueError("Data must be provided when covariance file doesn't exist or cov_path is None")
    
    # Determine save location
    target_path = None
    if save_path is not None:
        target_path = Path(save_path)
    elif cov_path is not None:
        target_path = Path(cov_path)
    else:
        # Default location
        target_path = Path("data/pca/covariance_stats.pt")
    
    print(f"Computing covariance statistics from data...")
    return compute_covariance_stats(data, save_path=target_path, wave=wave, src_path=src_path)
