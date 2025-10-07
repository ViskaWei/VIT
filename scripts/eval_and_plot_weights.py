#!/usr/bin/env python3
"""
Unified script to download W&B model, extract weights, and plot heatmaps.

Usage:
    python scripts/eval_and_plot_weights.py \\
        --config configs/vit.yaml \\
        --wandb-id 3vrlvqj0 \\
        --version v2 \\
        --out results/eval_3vrlvqj0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.vit import ViTLModule, ViTDataModule


class ModelEvaluator:
    """Handle model checkpoint resolution, evaluation, and weight extraction."""

    def __init__(self, config_path: str, output_dir: Path):
        self.config = load_config(config_path)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._get_device()

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def download_wandb_checkpoint(
        self,
        wandb_id: str,
        version: str = "v1",
        entity: str | None = None,
        project: str | None = None,
    ) -> Path:
        """Download checkpoint from W&B or use local cached version."""
        # Check local artifact cache first
        local_art_dir = Path("artifacts") / f"model-{wandb_id}:{version}"
        local_ckpt = local_art_dir / "model.ckpt"

        if local_ckpt.exists():
            print(f"✓ Using cached checkpoint: {local_ckpt}")
            return local_ckpt

        # Download from W&B
        try:
            import wandb
        except ImportError:
            raise RuntimeError(
                "wandb is required to download artifacts. Install with: pip install wandb"
            )

        entity = entity or os.environ.get("WANDB_ENTITY") or "viskawei-johns-hopkins-university"
        project = project or self.config.get("project", "vit-test")

        print(f"⬇ Downloading from W&B: {entity}/{project}/model-{wandb_id}:{version}")
        run = wandb.init(entity=entity, project=project, job_type="download")
        artifact_path = f"{entity}/{project}/model-{wandb_id}:{version}"
        artifact = run.use_artifact(artifact_path, type="model")
        artifact_dir = artifact.download()
        wandb.finish()

        ckpt_path = Path(artifact_dir) / "model.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"model.ckpt not found in artifact: {artifact_dir}")

        print(f"✓ Downloaded to: {ckpt_path}")
        return ckpt_path

    def load_model(self, ckpt_path: Path) -> ViTLModule:
        """Load Lightning module from checkpoint."""
        print(f"⚙ Loading model from: {ckpt_path}")
        
        # Load checkpoint to inspect config
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_config = ckpt.get("hyper_parameters", {}).get("config", {})
        
        # Use checkpoint's config if available, otherwise use provided config
        if ckpt_config:
            print(f"  Using config from checkpoint")
            config_to_use = ckpt_config
        else:
            print(f"  Using provided config")
            config_to_use = self.config
        
        model = ViTLModule.load_from_checkpoint(str(ckpt_path), config=config_to_use)
        model.eval()
        model.to(self.device)
        print(f"✓ Model loaded on: {self.device}")
        return model

    def extract_weights(self, model: ViTLModule, save_dir: Path) -> dict[str, Path]:
        """Extract all model weights and save as .pt files."""
        save_dir.mkdir(parents=True, exist_ok=True)
        weight_paths = {}

        print(f"💾 Extracting weights to: {save_dir}")
        for name, param in model.model.named_parameters():
            safe_name = name.replace(".", "_")
            file_path = save_dir / f"{safe_name}.pt"
            torch.save(param.detach().cpu(), file_path)
            weight_paths[name] = file_path

        print(f"✓ Extracted {len(weight_paths)} weight tensors")
        return weight_paths

    def compute_weight_statistics(self, model: ViTLModule) -> list[dict[str, Any]]:
        """Compute statistics for all model parameters."""
        stats = []
        for name, param in model.model.named_parameters():
            cpu = param.detach().cpu()
            stats.append({
                "name": name,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "numel": int(cpu.numel()),
                "mean": float(cpu.mean().item()) if cpu.numel() > 0 else 0.0,
                "std": float(cpu.std().item()) if cpu.numel() > 1 else 0.0,
                "min": float(cpu.min().item()) if cpu.numel() > 0 else 0.0,
                "max": float(cpu.max().item()) if cpu.numel() > 0 else 0.0,
                "norm": float(cpu.norm().item()) if cpu.numel() > 0 else 0.0,
            })
        return stats

    def evaluate_on_test(
        self,
        model: ViTLModule,
        max_batches: int = 5
    ) -> dict[str, float]:
        """Run evaluation on test set and collect metrics."""
        dm = ViTDataModule.from_config(self.config, test_data=False)
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()

        losses = []
        print(f"🔍 Evaluating on test set (max {max_batches} batches)...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= max_batches:
                    break

                flux, _, labels = batch
                flux = flux.to(self.device)
                labels = labels.to(self.device)

                outputs = model.model(flux, labels=labels, return_dict=True)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    losses.append(float(outputs.loss.item()))

        metrics = {
            "test_loss_mean": float(sum(losses) / len(losses)) if losses else 0.0,
            "test_loss_std": float(torch.tensor(losses).std().item()) if len(losses) > 1 else 0.0,
            "num_batches": len(losses),
        }

        print(f"✓ Evaluation complete: loss={metrics['test_loss_mean']:.4f}")
        return metrics


class WeightPlotter:
    """Generate heatmap visualizations for model weights."""

    def __init__(self, output_dir: Path, cmap: str = "magma", dpi: int = 200):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cmap = cmap
        self.dpi = dpi

    @staticmethod
    def _prepare_for_heatmap(tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor to 2D for heatmap visualization."""
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim == 2:
            return tensor
        # Flatten higher dimensions: (first_dim, *)
        first_dim = tensor.shape[0]
        return tensor.reshape(first_dim, -1)

    def plot_weight(self, tensor_path: Path, param_name: str) -> Path:
        """Create heatmap for a single weight tensor."""
        tensor = torch.load(tensor_path, map_location="cpu").float()
        heatmap = self._prepare_for_heatmap(tensor)
        array = heatmap.numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(array, cmap=self.cmap, aspect="auto")

        # Add title and labels
        ax.set_title(f"{param_name}\nshape: {tuple(tensor.shape)}", fontsize=10)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 0")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Value", rotation=270, labelpad=15)

        # Add statistics text
        stats_text = f"μ={tensor.mean():.3e}\nσ={tensor.std():.3e}\nmin={tensor.min():.3e}\nmax={tensor.max():.3e}"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )

        fig.tight_layout()

        # Save with sanitized filename
        safe_name = param_name.replace(".", "_")
        output_path = self.output_dir / f"{safe_name}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def plot_all_weights(self, weight_paths: dict[str, Path]) -> list[Path]:
        """Generate heatmaps for all weight tensors."""
        print(f"🎨 Generating heatmaps...")
        plot_paths = []

        for param_name, tensor_path in weight_paths.items():
            try:
                output_path = self.plot_weight(tensor_path, param_name)
                plot_paths.append(output_path)
                print(f"  ✓ {param_name}")
            except Exception as e:
                print(f"  ✗ {param_name}: {e}")

        print(f"✓ Generated {len(plot_paths)} heatmaps")
        return plot_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download W&B model, evaluate, and plot weight heatmaps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--wandb-id",
        type=str,
        required=True,
        help="W&B run ID (e.g., 3vrlvqj0)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Artifact version or alias",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (defaults to env WANDB_ENTITY or viskawei-johns-hopkins-university)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project (defaults to config['project'] or vit-test)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (defaults to results/eval_<wandb_id>)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="Number of test batches for evaluation",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap for heatmaps",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved figures",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip test set evaluation, only extract and plot weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.out) if args.out else Path("results") / f"eval_{args.wandb_id}"
    weights_dir = output_dir / "weights"
    plots_dir = output_dir / "plots"

    print("=" * 60)
    print(f"ViT Model Evaluation & Visualization Pipeline")
    print("=" * 60)
    print(f"W&B Model: {args.wandb_id}:{args.version}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Initialize evaluator
    evaluator = ModelEvaluator(args.config, output_dir)

    # Step 1: Download checkpoint
    ckpt_path = evaluator.download_wandb_checkpoint(
        wandb_id=args.wandb_id,
        version=args.version,
        entity=args.entity,
        project=args.project,
    )

    # Step 2: Load model
    model = evaluator.load_model(ckpt_path)

    # Step 3: Extract weights
    weight_paths = evaluator.extract_weights(model, weights_dir)

    # Step 4: Compute statistics
    weight_stats = evaluator.compute_weight_statistics(model)
    stats_file = output_dir / "weight_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(weight_stats, f, indent=2)
    print(f"✓ Saved statistics to: {stats_file}")

    # Step 5: Evaluate on test set (optional)
    if not args.skip_eval:
        eval_metrics = evaluator.evaluate_on_test(model, max_batches=args.max_batches)
        metrics_file = output_dir / "evaluation_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"✓ Saved metrics to: {metrics_file}")

    # Step 6: Generate heatmaps
    plotter = WeightPlotter(plots_dir, cmap=args.cmap, dpi=args.dpi)
    plot_paths = plotter.plot_all_weights(weight_paths)

    # Summary
    print("=" * 60)
    print("✓ Pipeline complete!")
    print(f"  Weights: {weights_dir}")
    print(f"  Plots: {plots_dir}")
    print(f"  Statistics: {stats_file}")
    if not args.skip_eval:
        print(f"  Metrics: {metrics_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
