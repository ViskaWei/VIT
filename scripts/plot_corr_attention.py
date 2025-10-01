#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.corr_attention import (
    downsample_matrix,
    find_spectral_lines,
    rowwise_correlation,
    topk_overlap,
)

sns.set_theme(style="whitegrid")


def _load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def _locate_matrix_files(eval_dir: Path, prefix: str, ranks: List[int], downsample: int | None) -> Dict[int, Path]:
    files: Dict[int, Path] = {}
    for rank in ranks:
        if downsample:
            candidate = eval_dir / f"{prefix}_{rank}_ds{downsample}.pt"
            if candidate.exists():
                files[rank] = candidate
                continue
        candidate = eval_dir / f"{prefix}_{rank}.pt"
        if candidate.exists():
            files[rank] = candidate
    if len(files) != len(ranks):
        missing = [str(r) for r in ranks if r not in files]
        raise FileNotFoundError(f"Missing matrices for ranks: {', '.join(missing)}")
    return files


def _convert_lines(lines: List[dict], full_size: int, target_size: int) -> List[tuple[str, float]]:
    factor = full_size / target_size
    return [
        (entry["name"], entry["index"] / factor)
        for entry in lines
    ]


def plot_heatmap(mat: torch.Tensor, title: str, out_path: Path, lines: List[tuple[str, float]] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.numpy(), cmap="magma", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Wavelength bins")
    ax.set_ylabel("Wavelength bins")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if lines:
        for name, loc in lines:
            ax.axvline(loc, color="cyan", linestyle="--", linewidth=0.5)
            ax.axhline(loc, color="cyan", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_row_correlation(corr_map: Dict[int, torch.Tensor], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-1.0, 1.0, 60)
    for rank, values in corr_map.items():
        ax.hist(values.numpy(), bins=bins, alpha=0.4, label=f"k={rank}")
    ax.set_xlabel("Row correlation with attention")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_topk_curve(curve: Dict[int, Dict[int, float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    topk_values = sorted({k for curves in curve.values() for k in curves.keys()})
    for rank, scores in curve.items():
        y = [scores.get(k, np.nan) for k in topk_values]
        ax.plot(topk_values, y, marker="o", label=f"k={rank}")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Mean overlap")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCA correlation vs attention diagnostics")
    parser.add_argument("--eval-dir", required=True, help="Directory produced by corr_attention_eval.py")
    parser.add_argument("--out", default=None, help="Destination directory for plots")
    parser.add_argument("--downsample", type=int, default=512, help="Preferred grid size for heatmaps")
    parser.add_argument("--topk", type=int, nargs="+", default=[10, 25, 50, 100], help="Top-k values for curves")
    parser.add_argument("--use-downsample", action="store_true", help="Force plotting from downsampled matrices only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out or (eval_dir / "plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((eval_dir / "meta.json").read_text())
    ranks = [int(r) for r in meta["ranks"]]
    image_size = int(meta["config"]["image_size"])

    # Load spectral feature markers
    lines_path = eval_dir / "spectral_lines.json"
    spectral_lines = json.loads(lines_path.read_text()) if lines_path.exists() else []

    # Load matrices
    attention_path = None
    if args.downsample and (eval_dir / f"attention_ds{args.downsample}.pt").exists():
        attention_path = eval_dir / f"attention_ds{args.downsample}.pt"
        attention = _load_tensor(attention_path)
        attn_for_metrics = attention
        full_dim = args.downsample
    elif (eval_dir / "attention_wave.pt").exists():
        attention = _load_tensor(eval_dir / "attention_wave.pt")
        full_dim = attention.shape[0]
        if args.downsample and not args.use_downsample:
            attention_ds = downsample_matrix(attention, args.downsample)
            attention = attention_ds
        attn_for_metrics = attention if args.use_downsample else _load_tensor(eval_dir / "attention_wave.pt")
    else:
        raise FileNotFoundError("Attention matrix not found in eval directory")

    softmax_files = _locate_matrix_files(eval_dir, "softmax", ranks, args.downsample if args.use_downsample else None)
    matrices = {k: _load_tensor(path) for k, path in softmax_files.items()}

    # Prepare heatmaps
    line_marks = _convert_lines(spectral_lines, image_size, attention.shape[0]) if spectral_lines else None
    plot_heatmap(attention, "Attention (row-normalised)", out_dir / "attention_heatmap.png", line_marks)
    for rank, mat in matrices.items():
        plot_heatmap(mat, f"Softmax(C_k) with k={rank}", out_dir / f"softmax_k{rank}_heatmap.png", line_marks)

    # Compute metrics using full-resolution matrices when available
    if not args.use_downsample and (eval_dir / "attention_wave.pt").exists():
        base_attention = _load_tensor(eval_dir / "attention_wave.pt")
    else:
        base_attention = attn_for_metrics

    corr_vectors: Dict[int, torch.Tensor] = {}
    topk_curves: Dict[int, Dict[int, float]] = {}

    for rank in ranks:
        mat = matrices[rank]
        if mat.shape != base_attention.shape:
            if not args.use_downsample:
                mat = downsample_matrix(mat, base_attention.shape[0])
            else:
                base_attention = downsample_matrix(base_attention, mat.shape[0])
        corr_vectors[rank] = rowwise_correlation(mat, base_attention)
        curve = {}
        for k in args.topk:
            k = min(k, mat.shape[1] - 1)
            overlap = topk_overlap(mat, base_attention, k=k, ignore_diagonal=True)
            curve[k] = float(overlap.mean().item())
        topk_curves[rank] = curve

    plot_row_correlation(corr_vectors, out_dir / "row_correlation_hist.png")
    plot_topk_curve(topk_curves, out_dir / "topk_overlap_curve.png")

    # Save metrics redux as CSV-like JSON for convenience
    summary = {}
    for rank, corr in corr_vectors.items():
        summary[rank] = {
            "corr_mean": float(corr.mean().item()),
            "corr_median": float(corr.median().item()),
            "corr_std": float(corr.std(unbiased=False).item()),
        }
        for k, val in topk_curves[rank].items():
            summary[rank][f"top{int(k)}_mean"] = val
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
