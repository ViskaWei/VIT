"""Utility to fit linear preprocessing operators for ViT ablations.

This script reuses the project configuration to load the training spectra,
computes the required statistics (mean, covariance, supervised cross-covariance),
and writes a `.pt` artifact containing the affine map expected by the
`zca_linear`/`linear` preprocessor.

The saved payload always includes:

```
{
    "matrix": torch.Tensor[out_dim, in_dim],  # transposed weight for nn.Linear
    "bias": torch.Tensor[out_dim],            # optional bias, defaults to mean shift
    "mean": torch.Tensor[in_dim],
    "cov": torch.Tensor[in_dim, in_dim],
    "metadata": {...},
}
```

During training the module applies `y = x @ matrix.T + bias`, therefore
choosing `matrix = M.T` and `bias = -(mean @ M)` realises the usual
`(x - mean) @ M` pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.utils import load_config
from src.dataloader.spec_datasets import ClassSpecDataset, RegSpecDataset
from src.prepca.preprocessor_utils import (
    CovarianceStats,
    compute_whitening_metrics,
    compute_zca_diagnostics,
    load_or_compute_covariance,
    zca_self_check,
)


Tensor = torch.Tensor


def _normalise_task(config: Dict) -> str:
    model_cfg = config.get("model", {}) or {}
    task = (model_cfg.get("task_type") or model_cfg.get("task") or "cls").lower()
    return "reg" if task.startswith("reg") else "cls"


def _select_dataset(config: Dict) -> Tuple[torch.utils.data.Dataset, str]:
    task = _normalise_task(config)
    dataset_cls = RegSpecDataset if task == "reg" else ClassSpecDataset
    dataset = dataset_cls.from_config(config)
    dataset.load_data(stage="train")
    return dataset, task


def _ensure_limit(tensor: Tensor, limit: Optional[int]) -> Tensor:
    if limit is None or limit <= 0 or limit >= tensor.shape[0]:
        return tensor
    return tensor[:limit]


def _get_label_tensor(dataset, task: str, num_labels: int, limit: Optional[int]) -> Tensor:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise RuntimeError("Dataset does not expose labels; required for supervised preprocs")
    labels = _ensure_limit(labels, limit)
    if task == "cls":
        labels = labels.reshape(-1).long()
        num_classes = int(num_labels)
        if labels.numel() == 0:
            raise RuntimeError("No labels available to compute supervised projector")
        max_label = int(labels.max().item())
        if num_classes <= max_label:
            num_classes = max_label + 1
        return F.one_hot(labels, num_classes=num_classes).to(torch.float64)
    if labels.dim() == 1:
        labels = labels.unsqueeze(-1)
    return labels.to(torch.float64)


def _compute_cross_stats(
    flux: Tensor,
    targets: Tensor,
    mean_x: Tensor,
    batch_size: int = 1024,
) -> Tuple[Tensor, Tensor, Tensor]:
    n = flux.shape[0]
    t_dim = targets.shape[1]
    mean_y = torch.zeros(t_dim, dtype=torch.float64)
    cov_y = torch.zeros(t_dim, t_dim, dtype=torch.float64)
    cov_xy = torch.zeros(flux.shape[1], t_dim, dtype=torch.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_chunk = flux[start:end].to(torch.float64)
        y_chunk = targets[start:end]
        mean_y += y_chunk.sum(dim=0)
        cov_y += y_chunk.t().matmul(y_chunk)
        cov_xy += x_chunk.t().matmul(y_chunk)

    mean_y = mean_y / n
    cov_y = (cov_y - n * torch.outer(mean_y, mean_y)) / max(n - 1, 1)
    cov_xy = (cov_xy - n * torch.outer(mean_x, mean_y)) / max(n - 1, 1)
    return mean_y, cov_y, cov_xy


def _scaled_outer(U: Tensor, scale: Tensor) -> Tensor:
    scaled = U * scale.unsqueeze(0)
    return scaled @ U.t()


def _symmetrize(mat: Tensor) -> Tensor:
    return 0.5 * (mat + mat.t())


def _shrink_covariance(cov: Tensor, shrinkage: float) -> Tensor:
    if shrinkage <= 0.0:
        return cov
    dim = cov.shape[0]
    tr_over_dim = torch.trace(cov) / float(dim)
    eye = torch.eye(dim, dtype=cov.dtype, device=cov.device)
    return (1.0 - shrinkage) * cov + shrinkage * tr_over_dim * eye


def _zca_matrix(
    cov: torch.Tensor,
    eps: float,
    shrinkage: float,
    rank: Optional[int],
    perp_mode: str,
    perp_scale: Optional[float],
    *,
    eigvals: Optional[Tensor] = None,
    eigvecs: Optional[Tensor] = None,
):
    # 1) 对称化 + eigh（降序）
    if eigvals is None or eigvecs is None:
        cov = 0.5 * (cov + cov.t())
        eigvals_raw, eigvecs_raw = torch.linalg.eigh(cov)
        idx = torch.argsort(eigvals_raw, descending=True)
        eigvals = eigvals_raw[idx]
        eigvecs = eigvecs_raw[:, idx]
    else:
        eigvals = eigvals.to(torch.float64)
        eigvecs = eigvecs.to(torch.float64)

    # 2) 对特征值做收缩（等价于 (1-γ)C + γ·avg·I）
    if shrinkage > 0.0:
        avg = eigvals.mean()
        eigvals_hat = (1.0 - shrinkage) * eigvals + shrinkage * avg
    else:
        eigvals_hat = eigvals

    # 3) 显式构造 cov_hat（自检要用它）
    cov_hat = eigvecs @ torch.diag(eigvals_hat) @ eigvecs.t()

    # 4) ZCA：只用一次 inv_sqrt（单边缩放），千万别“平方”成 λ^{-1}
    inv_sqrt = torch.rsqrt(eigvals_hat + eps)

    D = eigvals_hat.numel()
    if rank is None or rank <= 0 or rank >= D:
        # 全秩
        P = (eigvecs * inv_sqrt) @ eigvecs.t()     # = V diag((λ+eps)^-1/2) V^T
        projector = eigvecs @ eigvecs.t()          # = I
        (rel, cond0, cond1) = zca_self_check(P, cov_hat, eps=0.0)
        print(f"[zca] self-check: rel_err={rel:.3e}, cond_before={cond0:.3e}, cond_after={cond1:.3e}")
        return P, projector, eigvals_hat, cov_hat, None

    # 5) 低秩
    r = int(rank)
    Vr = eigvecs[:, :r]
    inv_sqrt_r = torch.rsqrt(eigvals_hat[:r] + eps)
    P = (Vr * inv_sqrt_r) @ Vr.t()
    projector = Vr @ Vr.t()

    # 余空间缩放
    if perp_scale is not None:
        s = float(perp_scale)
    else:
        mode = (perp_mode or "zero").lower()
        if mode == "identity":
            s = 1.0
        elif mode == "avg" and r < D - 1:
            resid = eigvals_hat[r:]
            s = float(torch.rsqrt(resid.mean() + eps))
        else:
            s = 0.0
    if s != 0.0:
        I = torch.eye(D, dtype=cov.dtype, device=cov.device)
        P = P + s * (I - projector)
    # test: P.T x C x P − I | ​/ I ​--> 0 (whitening self-consistency)
    (rel, cond0, cond1) = zca_self_check(P, cov_hat, eps=0.0, lowrank=True, Vr=Vr)

    return P, projector, eigvals_hat, cov_hat, s


def _projector_from_basis(
    basis: Tensor,
    scale: Optional[Tensor] = None,
) -> Tensor:
    if scale is None:
        return basis @ basis.t()
    scaled = basis * scale.unsqueeze(0)
    return scaled @ basis.t()


def _pls_matrix(
    cov_xy: Tensor,
    rank: int,
    eps: float,
    whiten: bool,
) -> Tuple[Tensor, Tensor]:
    U, S, _Vh = torch.linalg.svd(cov_xy, full_matrices=False)
    if rank <= 0 or rank > U.shape[1]:
        rank = U.shape[1]
    U = U[:, :rank]
    S = S[:rank]
    if whiten:
        scale = torch.rsqrt(S + eps)
        matrix = _projector_from_basis(U, scale)
    else:
        matrix = _projector_from_basis(U)
    return matrix, S


def _cca_matrix(
    cov_x: Tensor,
    cov_y: Tensor,
    cov_xy: Tensor,
    rank: int,
    eps: float,
    whiten: bool,
) -> Tuple[Tensor, Tensor]:
    cov_x = cov_x + eps * torch.eye(cov_x.shape[0], dtype=cov_x.dtype)
    cov_y = cov_y + eps * torch.eye(cov_y.shape[0], dtype=cov_y.dtype)
    evals_x, evecs_x = torch.linalg.eigh(cov_x)
    evals_y, evecs_y = torch.linalg.eigh(cov_y)
    cov_x_inv_sqrt = _scaled_outer(evecs_x, torch.rsqrt(evals_x + eps))
    cov_y_inv_sqrt = _scaled_outer(evecs_y, torch.rsqrt(evals_y + eps))
    M = cov_x_inv_sqrt @ cov_xy @ cov_y_inv_sqrt
    U, S, _Vh = torch.linalg.svd(M, full_matrices=False)
    if rank <= 0 or rank > U.shape[1]:
        rank = U.shape[1]
    U = U[:, :rank]
    S = S[:rank]
    Wx = cov_x_inv_sqrt @ U
    if whiten:
        scale = torch.rsqrt(S + eps)
        matrix = _projector_from_basis(Wx, scale)
    else:
        matrix = _projector_from_basis(Wx)
    return matrix, S


def _random_rotation(dim: int, seed: int) -> Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(mat)
    d = torch.sign(torch.diagonal(r))
    return q * d


def _compose_bias(mean: Tensor, matrix: Tensor) -> Tensor:
    return -mean @ matrix


def fit_preprocessor(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_cfg = config.get("data", {}) or {}
    cov_path_cfg = data_cfg.get("cov_path")
    cov_save_path_cfg = data_cfg.get("save_path")

    cov_stats: Optional[CovarianceStats] = None
    cov_stats_path: Optional[Path] = None

    dataset = None
    flux: Optional[Tensor] = None
    wave: Optional[Tensor] = None
    stats_from_dataset = False

    requires_targets = args.mode in {"pls", "cca"}
    requires_flux_for_limit = args.limit is not None and args.limit > 0
    num_labels = int(config.get("model", {}).get("num_labels", 1) or 1)

    # Try to load covariance from cov_path if provided
    if cov_path_cfg:
        cov_path = Path(cov_path_cfg)
        if cov_path.exists():
            try:
                cov_stats = load_or_compute_covariance(cov_path_cfg)
                cov_stats_path = cov_path
                print(f"[fit] Loaded covariance statistics from {cov_stats_path}")
            except Exception as exc:
                raise RuntimeError(f"Failed to load covariance stats at {cov_path_cfg}: {exc}") from exc

    # If covariance not loaded, need to compute it from dataset
    if cov_stats is None:
        dataset, task = _select_dataset(config)
        dataset.load_data(stage="train")
        wave = dataset.wave.detach().cpu() if hasattr(dataset, "wave") else None
        flux_cpu = dataset.flux.detach().cpu().view(dataset.num_samples, -1)
        flux_cpu = _ensure_limit(flux_cpu, args.limit)
        
        # Determine save location
        save_path = cov_save_path_cfg or cov_path_cfg
        if save_path is None:
            save_path = Path(args.output).with_name("cov.pt")
        
        cov_stats = load_or_compute_covariance(
            cov_path=None,  # Force computation since we didn't load
            data=flux_cpu,
            save_path=save_path,
            wave=wave
        )
        cov_stats_path = Path(save_path)
        flux = flux_cpu.to(torch.float64)
        stats_from_dataset = True
        print(f"[fit] Computed covariance statistics and saved to {cov_stats_path}")
    else:
        task = _normalise_task(config)

    # Load dataset if needed for targets or flux limit
    if dataset is None and (requires_targets or requires_flux_for_limit):
        try:
            dataset, task = _select_dataset(config)
            dataset.load_data(stage="train")
        except Exception as exc:
            raise RuntimeError(
                "Selected preprocessing mode requires dataset access but data.file_path is missing; please provide it in the config."
            ) from exc

    if dataset is not None and flux is None:
        wave = dataset.wave.detach().cpu() if hasattr(dataset, "wave") else wave
        flux_cpu = dataset.flux.detach().cpu().view(dataset.num_samples, -1)
        flux_cpu = _ensure_limit(flux_cpu, args.limit)
        flux = flux_cpu.to(torch.float64)

    if flux is None and requires_targets:
        raise RuntimeError(
            "PLS/CCA modes require dataset access to compute targets; please provide data.file_path in the config."
        )

    stats64 = cov_stats.to(torch.float64)
    mean_x = stats64.mean
    cov_x = stats64.cov
    eigvals_stats = stats64.eigvals
    eigvecs_stats = stats64.eigvecs
    order = torch.argsort(eigvals_stats, descending=True)
    eigvals_sorted = eigvals_stats[order]
    eigvecs_sorted = eigvecs_stats[:, order]
    n_samples = cov_stats.num_samples
    if n_samples <= 0 and flux is not None:
        n_samples = flux.shape[0]
    dim = cov_x.shape[0]

    if requires_targets and dataset is not None:
        targets = _get_label_tensor(dataset, task, num_labels, args.limit)
        mean_y, cov_y, cov_xy = _compute_cross_stats(flux, targets, mean_x)
    else:
        targets = None
        mean_y = cov_y = cov_xy = None

    eps = float(args.eps)
    shrinkage = float(args.shrinkage)
    rank = args.rank
    if isinstance(rank, int) and rank > dim:
        rank = dim

    cov_used: Optional[Tensor] = None
    complement_scale: Optional[float] = None
    projector: Optional[Tensor] = None
    spectrum: Optional[Tensor] = None

    if args.mode == "center":
        matrix = torch.eye(dim, dtype=torch.float64)
    elif args.mode == "standardize":
        std = torch.sqrt(torch.clamp(torch.diagonal(cov_x), min=eps))
        inv_std = 1.0 / std
        matrix = torch.diag(inv_std)
    elif args.mode == "zca":
        matrix, projector, spectrum, cov_used, complement_scale = _zca_matrix(
            cov_x,
            eps,
            shrinkage,
            None,
            args.perp_mode,
            args.perp_scale,
            eigvals=eigvals_sorted,
            eigvecs=eigvecs_sorted,
        )
    elif args.mode == "zca_lowrank":
        if rank is None:
            raise ValueError("--rank is required for zca_lowrank mode")
        matrix, projector, spectrum, cov_used, complement_scale = _zca_matrix(
            cov_x,
            eps,
            shrinkage,
            rank,
            args.perp_mode,
            args.perp_scale,
            eigvals=eigvals_sorted,
            eigvecs=eigvecs_sorted,
        )
    elif args.mode == "project_lowrank":
        if rank is None:
            raise ValueError("--rank is required for project_lowrank mode")
        lead = eigvecs_sorted[:, :rank]
        matrix = _projector_from_basis(lead)
        projector = matrix
        spectrum = eigvals_sorted
    elif args.mode == "randrot_white":
        matrix_zca, projector, spectrum, cov_used, complement_scale = _zca_matrix(
            cov_x,
            eps,
            shrinkage,
            None,
            args.perp_mode,
            args.perp_scale,
            eigvals=eigvals_sorted,
            eigvecs=eigvecs_sorted,
        )
        rotation = _random_rotation(dim, args.seed)
        matrix = matrix_zca @ rotation
    elif args.mode == "randrot":
        rotation = _random_rotation(dim, args.seed)
        matrix = rotation
        projector = torch.eye(dim, dtype=torch.float64)
        spectrum = torch.ones(dim, dtype=torch.float64)
    elif args.mode == "pca":
        if rank is None:
            raise ValueError("--rank is required for pca mode")
        lead = eigvecs_sorted[:, :rank]
        if args.whiten:
            vals = eigvals_sorted[:rank]
            matrix = _projector_from_basis(lead, torch.rsqrt(vals + eps))
        else:
            matrix = _projector_from_basis(lead)
        projector = matrix
        spectrum = eigvals_sorted
    elif args.mode == "pls":
        if targets is None:
            raise RuntimeError("Targets are required for PLS mode")
        if rank is None:
            raise ValueError("--rank is required for pls mode")
        matrix, singulars = _pls_matrix(cov_xy, rank, eps, args.whiten)
        projector = matrix
        spectrum = singulars
    elif args.mode == "cca":
        if targets is None:
            raise RuntimeError("Targets are required for CCA mode")
        if rank is None:
            raise ValueError("--rank is required for cca mode")
        matrix, singulars = _cca_matrix(cov_x, cov_y, cov_xy, rank, eps, args.whiten)
        projector = matrix
        spectrum = singulars
    else:
        raise ValueError(f"Unknown mode '{args.mode}'")

    matrix = matrix.to(torch.float64)
    whiten_metrics: Optional[Dict[str, float]] = None
    if cov_used is not None and projector is not None:
        cov_used = cov_used.to(matrix.dtype)
        whiten_metrics = compute_whitening_metrics(
            matrix,
            cov_used,
            projector.to(matrix.dtype),
            complement_scale=complement_scale,
        )
        if whiten_metrics is not None:
            print(
                f"[zca-fit] whitening check: max_abs={whiten_metrics['max_abs']:.3e}, "
                f"rel={whiten_metrics['rel_max_abs']:.3e}"
            )

    zca_metrics: Optional[Dict[str, float]] = None
    if args.mode.startswith("zca"):
        cov_for_report = cov_used if cov_used is not None else _symmetrize(cov_x)
        zca_metrics = compute_zca_diagnostics(flux, mean_x, cov_for_report, matrix, shrinkage)
        if zca_metrics is not None:
            print("[zca-fit] ZCA report:")
            print("sym_err (checking whether cov is symmetric) ~ 0 (e.g. <1e-6)")
            print("white_err_self (self-consistency of whitening) ~ 0.1 (e.g. <1e-1)")
            print("white_err_onX (how well whitened X matches identity) should be close to white_err_self")
            print("cond_Ih (condition number of whitened cov) should be reasonable (e.g. <1e3)")
            print("off_ratio_P (off-diagonal energy in P) should be small (e.g. <1e-1)")
            for key, value in zca_metrics.items():
                print(f"    {key}: {value:.6e}")
        else:
            print("[zca-fit] Skipped ZCA diagnostics (flux data unavailable)")

    bias = _compose_bias(mean_x, matrix)

    matrix32 = matrix.to(torch.float32)
    metadata: Dict[str, Optional[object]] = {
        "mode": args.mode,
        "rank": rank,
        "eps": eps,
        "shrinkage": shrinkage,
        "whiten": bool(args.whiten),
        "seed": int(args.seed),
        "num_samples": n_samples,
        "perp_mode": args.perp_mode,
        "perp_scale": args.perp_scale,
        "cov_stats_source": "computed" if stats_from_dataset else "loaded",
    }
    if cov_stats_path is not None:
        metadata["cov_stats_path"] = str(cov_stats_path)
    data_path = data_cfg.get("file_path")
    if data_path:
        metadata["data_path"] = data_path

    payload = {
        "matrix": matrix32,
        "linear_weight": matrix32,
        "bias": bias.to(torch.float32),
        "mean": mean_x.to(torch.float32),
        "cov": cov_x.to(torch.float32),
        "metadata": metadata,
    }
    if args.mode in {"zca", "zca_lowrank", "project_lowrank", "pca"}:
        payload["eigenvalues"] = spectrum.to(torch.float32)
        payload["projector"] = projector.to(torch.float32)
    if args.mode in {"randrot", "randrot_white"}:
        payload["rotation"] = matrix32
    if cov_used is not None:
        payload["cov_used"] = cov_used.to(torch.float32)
    if whiten_metrics is not None or zca_metrics is not None:
        diagnostics = payload.setdefault("diagnostics", {})
        if whiten_metrics is not None:
            diagnostics["whitening_error"] = whiten_metrics
        if zca_metrics is not None:
            diagnostics["zca_report"] = zca_metrics
    if args.mode in {"pls", "cca"}:
        payload["singular_values"] = spectrum.to(torch.float32)
        payload["cov_xy"] = cov_xy.to(torch.float32)
    if mean_y is not None:
        payload["mean_y"] = mean_y.to(torch.float32)
    if cov_y is not None:
        payload["cov_y"] = cov_y.to(torch.float32)
    if args.mode.startswith("zca"):
        payload["whitening"] = matrix32

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved preprocessor matrix to {out_path} (mode={args.mode}, dim={dim})")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit linear preprocessor statistics")
    parser.add_argument("--config", type=str, default="configs/vit.yaml", help="Config file used to load data")
    parser.add_argument("--mode", type=str, required=True,
                        choices=[
                            "center",
                            "standardize",
                            "zca",
                            "zca_lowrank",
                            "project_lowrank",
                            "randrot",
                            "randrot_white",
                            "pca",
                            "pls",
                            "cca",
                        ])
    parser.add_argument("--output", type=str, required=True, help="Path to the saved statistics (.pt)")
    parser.add_argument("--rank", type=int, default=None, help="Rank for low-rank modes")
    parser.add_argument("--eps", type=float, default=1e-5, help="Numerical stabilisation epsilon")
    parser.add_argument("--shrinkage", type=float, default=0.0, help="Shrinkage coefficient for covariance")
    parser.add_argument("--perp-mode", type=str, default="zero", choices=["zero", "identity", "avg"], help="How to treat the complement subspace in low-rank ZCA")
    parser.add_argument("--perp-scale", type=float, default=None, help="Override complement scale with an explicit value")
    parser.add_argument("--whiten", action="store_true", help="Whiten the selected subspace (PCA/PLS/CCA)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random rotations")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (recompute covariance offline first)")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    fit_preprocessor(args)


if __name__ == "__main__":
    main()
