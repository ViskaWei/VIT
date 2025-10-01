#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.corr_attention import (
    aggregate_heads,
    attention_from_qk,
    cka_similarity,
    compute_pca,
    correlation_from_covariance,
    covariance_series,
    cosine_similarity,
    downsample_matrix,
    find_spectral_lines,
    frobenius_distance,
    load_flux_matrix,
    patch_to_wavelength_mapping,
    project_patch_attention,
    row_softmax,
    rowwise_correlation,
    softmax_series,
    topk_overlap,
)
from src.utils import load_config


def _load_config_from_sources(args: argparse.Namespace) -> dict:
    if args.config:
        return load_config(args.config)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        return ckpt["hyper_parameters"]["config"]
    raise ValueError("Provide either --config or --ckpt for model settings")


def _resolve_model_params(cfg: dict) -> dict:
    model = cfg.get("model", {})
    patch_size = int(model.get("patch_size"))
    stride = model.get("stride_size")
    if stride is None:
        stride = model.get("stride_ratio", 1.0)
        stride = max(1, int(round(float(stride) * patch_size)))
    else:
        stride = int(stride)
    image_size = int(model.get("image_size", 4096))
    heads = int(model.get("num_attention_heads", 1))
    return {
        "patch_size": patch_size,
        "stride": stride,
        "image_size": image_size,
        "num_heads": heads,
    }


def _load_qk(acts: dict, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = f"vit.encoder.layer.{layer}.attention.attention"
    q = acts["activations"][f"{base}.query"].to(torch.float32)
    k = acts["activations"][f"{base}.key"].to(torch.float32)
    return q, k


def _ensure_outdir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_serializable(metrics: dict) -> dict:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().tolist()
        elif isinstance(value, (list, tuple)):
            result[key] = [float(v) for v in value]
        elif isinstance(value, dict):
            result[key] = _to_serializable(value)
        elif isinstance(value, (float, int)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PCA-derived correlations with transformer attention")
    parser.add_argument("--h5", default="data/test_1k/dataset.h5", help="Path to spectrum HDF5")
    parser.add_argument("--activations", required=True, help="Path to saved activations all.pt")
    parser.add_argument("--config", default=None, help="Experiment YAML for model params")
    parser.add_argument("--ckpt", default=None, help="Lightning checkpoint (for config fallback)")
    parser.add_argument("--out", default="results/corr_attention", help="Output directory root")
    parser.add_argument("--limit", type=int, default=512, help="Spectra count used for PCA")
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128], help="PCA ranks (k) to evaluate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Row softmax temperature")
    parser.add_argument("--downsample", type=int, default=None, help="Optional downsample size (e.g., 512)")
    parser.add_argument("--layer", type=int, default=0, help="Transformer encoder layer to extract attention from")
    parser.add_argument("--patch_size", type=int, default=None, help="Override patch size from config")
    parser.add_argument("--stride", type=int, default=None, help="Override stride size from config")
    parser.add_argument("--image_size", type=int, default=None, help="Override image/input length")
    parser.add_argument("--num_heads", type=int, default=None, help="Override number of attention heads")
    parser.add_argument("--save_full", action="store_true", help="Persist full-resolution matrices to disk")
    parser.add_argument("--corr_mode", choices=["gram", "cov", "corr"], default="corr", help="Matrix type derived from PCA")
    parser.add_argument("--zero_diag", action="store_true", help="Zero diagonal before comparisons")
    parser.add_argument("--topk", type=int, nargs="+", default=[10, 25, 50], help="Top-k overlap sizes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config_from_sources(args)
    model_params = _resolve_model_params(cfg)
    if args.patch_size:
        model_params['patch_size'] = int(args.patch_size)
    if args.stride:
        model_params['stride'] = int(args.stride)
    if args.image_size:
        model_params['image_size'] = int(args.image_size)
    if args.num_heads:
        model_params['num_heads'] = int(args.num_heads)
    outdir = _ensure_outdir(args.out)

    # Load spectra and compute PCA basis
    flux, wavelengths = load_flux_matrix(args.h5, limit=args.limit)
    ranks: list[int] = sorted(set(int(r) for r in args.ranks))
    max_rank = max(ranks)
    pca, _ = compute_pca(flux, k_max=max_rank, center=True, scale=(args.corr_mode == "corr"))

    Gram_matrices = covariance_series(
        pca,
        ranks=ranks,
        normalization=("sample" if args.corr_mode in {"cov", "corr"} else "none"),
        zero_diagonal=args.zero_diag,
    )
    if args.corr_mode == "corr":
        matrices = {k: correlation_from_covariance(mat) for k, mat in Gram_matrices.items()}
    else:
        matrices = Gram_matrices

    softmax_mats = softmax_series(
        matrices,
        temperature=args.temperature,
        downsample_to=None,
        zero_diagonal=args.zero_diag,
    )

    # Load attention tensors and project to wavelength grid
    acts = torch.load(args.activations, map_location="cpu")
    query, key = _load_qk(acts, args.layer)
    num_heads = model_params["num_heads"]
    attn = attention_from_qk(query, key, num_heads=num_heads)
    attn = aggregate_heads(attn)

    patch_tokens = attn.shape[-1] - 1  # drop CLS later
    mapping = patch_to_wavelength_mapping(
        signal_length=model_params["image_size"],
        patch_size=model_params["patch_size"],
        stride=model_params["stride"],
        num_patches=patch_tokens,
    )
    attn_wave = project_patch_attention(attn, mapping, drop_cls=True)
    attn_wave = row_softmax(attn_wave)
    if args.zero_diag:
        attn_wave = attn_wave - torch.diag(torch.diagonal(attn_wave))

    if args.downsample:
        attn_down = downsample_matrix(attn_wave, args.downsample)
    else:
        attn_down = None

    # Compute metrics for each PCA rank
    metrics: dict[int, dict[str, object]] = {}
    for k, mat in softmax_mats.items():
        comp = mat
        if comp.shape != attn_wave.shape:
            raise ValueError("Covariance-derived and attention matrices must match shape")
        corr_vec = rowwise_correlation(comp, attn_wave)
        row_stats = {
            "mean": float(corr_vec.mean().item()),
            "median": float(corr_vec.median().item()),
            "std": float(corr_vec.std(unbiased=False).item()),
        }
        overlap_stats = {}
        for topk in args.topk:
            overlap = topk_overlap(comp, attn_wave, k=topk, ignore_diagonal=True)
            overlap_stats[str(topk)] = {
                "mean": float(overlap.mean().item()),
                "median": float(overlap.median().item()),
            }
        metrics[k] = {
            "row_correlation": row_stats,
            "topk_overlap": overlap_stats,
            "cosine": cosine_similarity(comp, attn_wave),
            "frobenius": frobenius_distance(comp, attn_wave),
            "cka": cka_similarity(comp, attn_wave),
        }

    # Persist outputs
    meta = {
        "config": model_params,
        "ranks": ranks,
        "temperature": args.temperature,
        "corr_mode": args.corr_mode,
        "zero_diag": args.zero_diag,
        "limit": args.limit,
        "layer": args.layer,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    (outdir / "metrics.json").write_text(json.dumps(_to_serializable(metrics), indent=2))

    if args.save_full:
        for k, mat in matrices.items():
            torch.save(mat, outdir / f"matrix_{k}.pt")
            torch.save(softmax_mats[k], outdir / f"softmax_{k}.pt")
        torch.save(attn_wave, outdir / "attention_wave.pt")

    if args.downsample:
        torch.save(attn_down, outdir / f"attention_ds{args.downsample}.pt")
        for k, mat in softmax_mats.items():
            torch.save(downsample_matrix(mat, args.downsample), outdir / f"softmax_{k}_ds{args.downsample}.pt")

    # Provide quick references for plotting overlays
    lines = find_spectral_lines(wavelengths)
    (outdir / "spectral_lines.json").write_text(json.dumps(lines, indent=2))


if __name__ == "__main__":
    main()
