#!/usr/bin/env python3
"""Download W&B model, extract weights, and generate visualizations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.vit import ViTLModule, ViTDataModule
from scripts.plot_utils import batch_plot_weights


def get_device() -> torch.device:
    """Get available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_wandb_model(
    wandb_id: str,
    version: str = "v1",
    entity: str | None = None,
    project: str | None = None,
    config: dict | None = None,
) -> Path:
    """Download model from W&B or use cached version."""
    # Check cache first
    local_ckpt = Path("artifacts") / f"model-{wandb_id}:{version}" / "model.ckpt"
    if local_ckpt.exists():
        print(f"✓ Using cached: {local_ckpt}")
        return local_ckpt
    
    # Download from W&B
    try:
        import wandb
    except ImportError:
        raise RuntimeError("Install wandb: pip install wandb")
    
    entity = entity or os.environ.get("WANDB_ENTITY") or "viskawei-johns-hopkins-university"
    project = project or (config.get("project") if config else None) or "vit-test"
    
    print(f"⬇ Downloading: {entity}/{project}/model-{wandb_id}:{version}")
    run = wandb.init(entity=entity, project=project, job_type="download")
    artifact = run.use_artifact(f"{entity}/{project}/model-{wandb_id}:{version}", type="model")
    artifact_dir = artifact.download()
    wandb.finish()
    
    ckpt = Path(artifact_dir) / "model.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"model.ckpt not found in {artifact_dir}")
    
    print(f"✓ Downloaded: {ckpt}")
    return ckpt


def load_model(ckpt_path: Path, config: dict) -> ViTLModule:
    """Load Lightning module from checkpoint."""
    print(f"⚙ Loading model: {ckpt_path.name}")
    
    # Use checkpoint's config if available
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_config = ckpt.get("hyper_parameters", {}).get("config")
    use_config = ckpt_config if ckpt_config else config
    
    model = ViTLModule.load_from_checkpoint(str(ckpt_path), config=use_config)
    model.eval()
    device = get_device()
    model.to(device)
    print(f"✓ Loaded on {device}")
    return model


def extract_weights(model: ViTLModule, output_dir: Path) -> dict[str, Path]:
    """Extract all weights and save as .pt files."""
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Extracting weights...")
    weight_paths = {}
    for name, param in model.model.named_parameters():
        file_path = weights_dir / f"{name.replace('.', '_')}.pt"
        torch.save(param.detach().cpu(), file_path)
        weight_paths[name] = file_path
    
    print(f"✓ Extracted {len(weight_paths)} tensors")
    return weight_paths


def compute_statistics(model: ViTLModule) -> list[dict[str, Any]]:
    """Compute statistics for all parameters."""
    stats = []
    for name, param in model.model.named_parameters():
        p = param.detach().cpu()
        stats.append({
            "name": name,
            "shape": list(p.shape),
            "dtype": str(p.dtype),
            "numel": int(p.numel()),
            "mean": float(p.mean()) if p.numel() > 0 else 0.0,
            "std": float(p.std()) if p.numel() > 1 else 0.0,
            "min": float(p.min()) if p.numel() > 0 else 0.0,
            "max": float(p.max()) if p.numel() > 0 else 0.0,
            "norm": float(p.norm()) if p.numel() > 0 else 0.0,
        })
    return stats


def evaluate_model(
    model: ViTLModule,
    config: dict,
    max_batches: int = 5,
) -> dict[str, float]:
    """Evaluate model on test set."""
    dm = ViTDataModule.from_config(config, test_data=False)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    device = next(model.parameters()).device
    
    losses = []
    print(f"🔍 Evaluating ({max_batches} batches)...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            
            flux, _, labels = batch
            outputs = model.model(flux.to(device), labels=labels.to(device), return_dict=True)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                losses.append(float(outputs.loss.item()))
    
    metrics = {
        "test_loss_mean": sum(losses) / len(losses) if losses else 0.0,
        "test_loss_std": float(torch.tensor(losses).std()) if len(losses) > 1 else 0.0,
        "num_batches": len(losses),
    }
    print(f"✓ Loss: {metrics['test_loss_mean']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Download, evaluate, and visualize ViT model")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--wandb-id", required=True, help="W&B run ID")
    parser.add_argument("--version", default="v1", help="Artifact version")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--project", default=None, help="W&B project")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--max-batches", type=int, default=5, help="Test batches")
    parser.add_argument("--cmap", default="magma", help="Colormap")
    parser.add_argument("--dpi", type=int, default=200, help="Plot DPI")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    output_dir = Path(args.out) if args.out else Path("results") / f"eval_{args.wandb_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"ViT Evaluation Pipeline")
    print(f"Model: {args.wandb_id}:{args.version}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Download and load
    ckpt_path = download_wandb_model(
        args.wandb_id, args.version, args.entity, args.project, config
    )
    model = load_model(ckpt_path, config)
    
    # Extract weights
    weight_paths = extract_weights(model, output_dir)
    
    # Compute statistics
    stats = compute_statistics(model)
    stats_file = output_dir / "weight_statistics.json"
    stats_file.write_text(json.dumps(stats, indent=2))
    print(f"✓ Statistics: {stats_file.name}")
    
    # Evaluate (optional)
    if not args.skip_eval:
        try:
            metrics = evaluate_model(model, config, args.max_batches)
            metrics_file = output_dir / "evaluation_metrics.json"
            metrics_file.write_text(json.dumps(metrics, indent=2))
            print(f"✓ Metrics: {metrics_file.name}")
        except Exception as e:
            print(f"⚠ Evaluation failed: {e}")
    
    # Generate plots
    plots_dir = output_dir / "plots"
    batch_plot_weights(weight_paths, plots_dir, cmap=args.cmap, dpi=args.dpi)
    
    print("=" * 60)
    print("✓ Complete!")
    print(f"  Weights: {output_dir / 'weights'}")
    print(f"  Plots: {plots_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
