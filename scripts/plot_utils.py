#!/usr/bin/env python3
"""Unified plotting utilities for ViT model analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sns.set_theme(style="whitegrid")


def prepare_tensor_for_heatmap(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape tensor to 2D for heatmap visualization."""
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    # Flatten higher dimensions: (first_dim, *)
    return tensor.reshape(tensor.shape[0], -1)


def plot_weight_heatmap(
    tensor: torch.Tensor,
    title: str,
    output_path: Path,
    cmap: str = "magma",
    dpi: int = 200,
    show_stats: bool = True,
) -> None:
    """Plot weight tensor as heatmap with optional statistics overlay."""
    heatmap = prepare_tensor_for_heatmap(tensor.float())
    array = heatmap.numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(array, cmap=cmap, aspect="auto")
    
    ax.set_title(f"{title}\nshape: {tuple(tensor.shape)}", fontsize=10)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 0")
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value", rotation=270, labelpad=15)
    
    if show_stats:
        stats_text = (
            f"μ={tensor.mean():.3e}\n"
            f"σ={tensor.std():.3e}\n"
            f"min={tensor.min():.3e}\n"
            f"max={tensor.max():.3e}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_heatmap(
    matrix: torch.Tensor,
    title: str,
    output_path: Path,
    spectral_lines: List[Tuple[str, float]] | None = None,
    cmap: str = "magma",
    dpi: int = 200,
) -> None:
    """Plot 2D matrix (e.g., attention, correlation) with optional spectral line markers."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.numpy(), cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Wavelength bins")
    ax.set_ylabel("Wavelength bins")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    if spectral_lines:
        for name, loc in spectral_lines:
            ax.axvline(loc, color="cyan", linestyle="--", linewidth=0.5)
            ax.axhline(loc, color="cyan", linestyle="--", linewidth=0.5)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_distribution_histogram(
    data_dict: Dict[int, torch.Tensor],
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 60,
    xlim: Tuple[float, float] | None = None,
    dpi: int = 200,
) -> None:
    """Plot histogram distributions for multiple datasets (e.g., correlation by rank)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if xlim:
        bins_array = np.linspace(xlim[0], xlim[1], bins)
    else:
        bins_array = bins
    
    for key, values in data_dict.items():
        ax.hist(values.numpy(), bins=bins_array, alpha=0.4, label=f"k={key}")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_metric_curve(
    curve_data: Dict[int, Dict[int, float]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    ylim: Tuple[float, float] | None = None,
    dpi: int = 200,
) -> None:
    """Plot metric curves (e.g., top-k overlap vs k)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    x_values = sorted({x for curves in curve_data.values() for x in curves.keys()})
    
    for rank, scores in curve_data.items():
        y = [scores.get(x, np.nan) for x in x_values]
        ax.plot(x_values, y, marker="o", label=f"k={rank}")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def batch_plot_weights(
    weight_paths: Dict[str, Path],
    output_dir: Path,
    cmap: str = "magma",
    dpi: int = 200,
    show_progress: bool = True,
) -> List[Path]:
    """Generate heatmaps for all weight tensors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    
    if show_progress:
        print(f"🎨 Generating {len(weight_paths)} heatmaps...")
    
    for param_name, tensor_path in weight_paths.items():
        try:
            tensor = torch.load(tensor_path, map_location="cpu")
            safe_name = param_name.replace(".", "_")
            output_path = output_dir / f"{safe_name}.png"
            plot_weight_heatmap(tensor, param_name, output_path, cmap=cmap, dpi=dpi)
            plot_paths.append(output_path)
            if show_progress:
                print(f"  ✓ {param_name}")
        except Exception as e:
            if show_progress:
                print(f"  ✗ {param_name}: {e}")
    
    if show_progress:
        print(f"✓ Generated {len(plot_paths)} heatmaps")
    
    return plot_paths
