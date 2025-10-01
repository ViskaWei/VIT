#!/usr/bin/env python3
"""Plot ViT parameter tensors stored as .pt files."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch


@dataclass
class PlotConfig:
    input_dir: Path
    output_dir: Path
    glob_pattern: str
    show: bool
    cmap: str
    dpi: int


def parse_args(argv: Iterable[str]) -> PlotConfig:
    parser = argparse.ArgumentParser(
        description="Plot ViT weight tensors stored as PyTorch .pt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .pt parameter tensors (e.g. results/inspect/.../params).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs. Defaults to <input_dir>/plots.",
    )
    parser.add_argument(
        "-g",
        "--glob",
        dest="glob_pattern",
        default="*weight.pt",
        help="Glob pattern to select parameter files to plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open matplotlib windows instead of closing the figures immediately.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures.",
    )

    args = parser.parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    return PlotConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        glob_pattern=args.glob_pattern,
        show=args.show,
        cmap=args.cmap,
        dpi=args.dpi,
    )


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        return torch.stack(value)
    if isinstance(value, dict):
        raise ValueError(f"Expected tensor in {path}, got dict keys {list(value.keys())[:5]}")
    raise TypeError(f"Unsupported object type in {path}: {type(value)!r}")


def tensor_for_heatmap(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    first_dim = tensor.shape[0]
    return tensor.reshape(first_dim, -1)


def plot_tensor(path: Path, cfg: PlotConfig) -> None:
    tensor = load_tensor(path).float()
    heatmap_src = tensor_for_heatmap(tensor)
    array = heatmap_src.numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(array, aspect="auto", cmap=cfg.cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(path.stem)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Row Index")
    fig.tight_layout()

    output_path = cfg.output_dir / f"{path.stem}.png"
    fig.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")

    if cfg.show:
        plt.show()
    plt.close(fig)


def main(argv: Iterable[str]) -> None:
    cfg = parse_args(argv)
    paths = sorted(cfg.input_dir.glob(cfg.glob_pattern))
    if not paths:
        raise SystemExit(
            f"No files matching pattern '{cfg.glob_pattern}' in {cfg.input_dir}"
        )

    for path in paths:
        print(f"Plotting {path.name} -> {cfg.output_dir / (path.stem + '.png')}")
        plot_tensor(path, cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
