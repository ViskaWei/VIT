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
        return F.one_hot(labels, num_classes=int(num_labels)).to(torch.float64)
    if labels.dim() == 1:
        labels = labels.unsqueeze(-1)
    return labels.to(torch.float64)


def _covariance_stats(dataset, limit: Optional[int]) -> Tuple[Tensor, Tensor, int]:
    mean = dataset.mean_flux.to(torch.float64)
    cov = dataset.covariance.to(torch.float64)
    num_samples = int(dataset.num_samples)
    if limit is not None and 0 < limit < num_samples:
        raise ValueError(
            "Limiting the dataset requires recomputing covariance; rerun without --limit"
        )
    return mean, cov, num_samples


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


def _zca_matrix(
    cov: Tensor,
    eps: float,
    shrinkage: float,
    rank: Optional[int],
    perp_mode: str,
    perp_scale: Optional[float],
) -> Tuple[Tensor, Tensor, Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if shrinkage > 0.0:
        avg = eigvals.mean()
        eigvals = (1.0 - shrinkage) * eigvals + shrinkage * avg
    inv_sqrt = torch.rsqrt(eigvals + eps)
    if rank is None or rank <= 0 or rank >= eigvals.numel():
        matrix = _scaled_outer(eigvecs, inv_sqrt)
        projector = eigvecs @ eigvecs.t()
        return matrix, projector, eigvals

    r = int(rank)
    lead = eigvecs[:, :r]
    inv_sqrt_r = inv_sqrt[:r]
    matrix = _scaled_outer(lead, inv_sqrt_r)
    projector = lead @ lead.t()

    if perp_scale is not None:
        scale_val = perp_scale
    else:
        mode = (perp_mode or "zero").lower()
        if mode == "identity":
            scale_val = 1.0
        elif mode == "avg" and r < eigvals.numel() - 1:
            resid = eigvals[r:]
            scale_val = float(torch.rsqrt(resid.mean() + eps))
        else:
            scale_val = 0.0

    if scale_val != 0.0:
        eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        matrix = matrix + scale_val * (eye - projector)

    return matrix, projector, eigvals


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
    dataset, task = _select_dataset(config)
    num_labels = int(config.get("model", {}).get("num_labels", 1) or 1)

    mean_x, cov_x, n_samples = _covariance_stats(dataset, args.limit)

    flux = dataset.flux.view(dataset.num_samples, -1)
    flux = _ensure_limit(flux, args.limit).to(torch.float64)

    if args.mode in {"pls", "cca"}:
        targets = _get_label_tensor(dataset, task, num_labels, args.limit)
        mean_y, cov_y, cov_xy = _compute_cross_stats(flux, targets, mean_x)
    else:
        targets = None
        mean_y = cov_y = cov_xy = None

    dim = flux.shape[1]
    eps = float(args.eps)
    shrinkage = float(args.shrinkage)
    rank = args.rank
    if isinstance(rank, int) and rank > dim:
        rank = dim

    if args.mode == "center":
        matrix = torch.eye(dim, dtype=torch.float64)
    elif args.mode == "standardize":
        std = torch.sqrt(torch.clamp(torch.diagonal(cov_x), min=eps))
        inv_std = 1.0 / std
        matrix = torch.diag(inv_std)
    elif args.mode == "zca":
        matrix, projector, spectrum = _zca_matrix(cov_x, eps, shrinkage, None, args.perp_mode, args.perp_scale)
    elif args.mode == "zca_lowrank":
        if rank is None:
            raise ValueError("--rank is required for zca_lowrank mode")
        matrix, projector, spectrum = _zca_matrix(cov_x, eps, shrinkage, rank, args.perp_mode, args.perp_scale)
    elif args.mode == "project_lowrank":
        if rank is None:
            raise ValueError("--rank is required for project_lowrank mode")
        eigvals, eigvecs = torch.linalg.eigh(cov_x)
        idx = torch.argsort(eigvals, descending=True)
        lead = eigvecs[:, idx][:, :rank]
        matrix = _projector_from_basis(lead)
        projector = matrix
        spectrum = eigvals[idx]
    elif args.mode == "randrot_white":
        matrix_zca, projector, spectrum = _zca_matrix(cov_x, eps, shrinkage, None, args.perp_mode, args.perp_scale)
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
        eigvals, eigvecs = torch.linalg.eigh(cov_x)
        idx = torch.argsort(eigvals, descending=True)
        lead = eigvecs[:, idx][:, :rank]
        if args.whiten:
            vals = eigvals[idx][:rank]
            matrix = _projector_from_basis(lead, torch.rsqrt(vals + eps))
        else:
            matrix = _projector_from_basis(lead)
        projector = matrix
        spectrum = eigvals[idx]
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
    bias = _compose_bias(mean_x, matrix)

    matrix32 = matrix.to(torch.float32)
    payload = {
        "matrix": matrix32,
        "linear_weight": matrix32,
        "bias": bias.to(torch.float32),
        "mean": mean_x.to(torch.float32),
        "cov": cov_x.to(torch.float32),
        "metadata": {
            "mode": args.mode,
            "rank": rank,
            "eps": eps,
            "shrinkage": shrinkage,
            "whiten": bool(args.whiten),
            "seed": int(args.seed),
            "num_samples": n_samples,
            "perp_mode": args.perp_mode,
            "perp_scale": args.perp_scale,
        },
    }
    if args.mode in {"zca", "zca_lowrank", "project_lowrank", "pca"}:
        payload["eigenvalues"] = spectrum.to(torch.float32)
        payload["projector"] = projector.to(torch.float32)
    if args.mode in {"randrot", "randrot_white"}:
        payload["rotation"] = matrix32
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
