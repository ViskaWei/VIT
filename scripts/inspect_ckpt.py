import os
import sys
import argparse
import json
from datetime import datetime
from collections import OrderedDict

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from src.vit import ViTLModule, ViTDataModule


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_tensor_to_cpu(x: torch.Tensor) -> torch.Tensor:
    try:
        return x.detach().to("cpu")
    except Exception:
        return x


def collect_param_stats(model: torch.nn.Module) -> list[dict]:
    stats = []
    for name, p in model.named_parameters():
        try:
            cpu = p.detach().to("cpu")
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
        except Exception as e:
            stats.append({"name": name, "error": repr(e)})
    return stats


def register_activation_hooks(model: torch.nn.Module, include_classes=None):
    """Register forward hooks for most leaf modules.

    Returns: (activations: OrderedDict, handles: list)
    """
    activations = OrderedDict()
    handles = []

    def should_hook(mod: torch.nn.Module) -> bool:
        # Skip containers
        from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
        if isinstance(mod, (Sequential, ModuleList, ModuleDict)):
            return False
        # Optionally filter by classes
        if include_classes is not None and not isinstance(mod, include_classes):
            return False
        return True

    def make_hook(name):
        def hook_fn(module, inputs, output):
            try:
                key = name
                # Normalize output to tensors for saving
                if isinstance(output, torch.Tensor):
                    activations[key] = _safe_tensor_to_cpu(output)
                elif isinstance(output, (list, tuple)):
                    activations[key] = tuple(_safe_tensor_to_cpu(x) for x in output if isinstance(x, torch.Tensor))
                elif isinstance(output, dict):
                    out = {k: _safe_tensor_to_cpu(v) for k, v in output.items() if isinstance(v, torch.Tensor)}
                    activations[key] = out
                else:
                    # Fallback: store repr
                    activations[key] = repr(type(output))
            except Exception as e:
                activations[name] = {"error": repr(e)}
        return hook_fn

    for name, module in model.named_modules():
        if name == "":
            continue  # skip root
        if should_hook(module):
            handles.append(module.register_forward_hook(make_hook(name)))

    return activations, handles


def save_activations(acts: OrderedDict, out_dir: str, batch_idx: int, model_out=None):
    """Save all captured artifacts for a batch into a single file.

    Contents include:
    - activations: captured by forward hooks
    - logits, loss: from model outputs (if available)
    - hidden_states, attentions: from model outputs (if available)
    """
    def to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return _safe_tensor_to_cpu(obj)
        if isinstance(obj, (list, tuple)):
            return [to_cpu(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        return obj

    bdir = os.path.join(out_dir, f"batch_{batch_idx}")
    os.makedirs(bdir, exist_ok=True)

    payload = {
        "batch_idx": batch_idx,
        "activations": to_cpu(dict(acts)),
    }

    if model_out is not None:
        try:
            if hasattr(model_out, "logits") and isinstance(model_out.logits, torch.Tensor):
                payload["logits"] = to_cpu(model_out.logits)
            if getattr(model_out, "hidden_states", None) is not None:
                payload["hidden_states"] = to_cpu(list(model_out.hidden_states))
            if getattr(model_out, "attentions", None) is not None:
                payload["attentions"] = to_cpu(list(model_out.attentions))
            if hasattr(model_out, "loss") and model_out.loss is not None:
                try:
                    payload["loss"] = float(model_out.loss.detach().cpu().item())
                except Exception:
                    payload["loss"] = to_cpu(model_out.loss)
        except Exception as e:
            payload["model_out_error"] = repr(e)

    # Single consolidated file per batch
    torch.save(payload, os.path.join(bdir, "all.pt"))

def _resolve_ckpt(args, config):
    """Return a local checkpoint path.

    Priority:
    1) args.ckpt if provided
    2) If args.wandb_id provided, prefer already-downloaded local artifact ./artifacts/model-<id>:<version>/model.ckpt
       Otherwise, download from W&B using args.wandb_id (and entity/project/version)
    """
    if args.ckpt:
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"Specified ckpt not found: {args.ckpt}")
        return args.ckpt

    if args.wandb_id:
        # First try local artifact cache path to avoid re-downloading
        version = args.version or 'v1'
        local_art_dir = os.path.join('artifacts', f"model-{args.wandb_id}:{version}")
        local_ckpt = os.path.join(local_art_dir, 'model.ckpt')
        if os.path.isfile(local_ckpt):
            print(f"✓ Using cached checkpoint: {local_ckpt}")
            return local_ckpt

        try:
            import wandb
        except ImportError as e:
            raise RuntimeError("wandb is required to download artifacts. Install with: pip install wandb") from e

        entity = args.entity or os.environ.get('WANDB_ENTITY') or 'viskawei-johns-hopkins-university'
        project = args.project or (config.get('project') if isinstance(config, dict) else None) or 'vit-test'
        artifact_type = args.artifact_type or 'model'

        print(f"⬇ Downloading from W&B: {entity}/{project}/model-{args.wandb_id}:{version}")
        run = wandb.init(entity=entity, project=project, job_type='download')
        path = f"{entity}/{project}/model-{args.wandb_id}:{version}"
        art = run.use_artifact(path, type=artifact_type)
        a_dir = art.download()
        wandb.finish()
        
        ckpt = os.path.join(a_dir, 'model.ckpt')
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Downloaded artifact missing model.ckpt at: {ckpt}")
        print(f"✓ Downloaded to: {ckpt}")
        return ckpt

    raise ValueError("Must provide either --ckpt or --wandb-id to locate a checkpoint.")


def main():
    parser = argparse.ArgumentParser(description="Load from ckpt, dump params and activations")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to Lightning .ckpt file")
    # If --ckpt is omitted, allow fetching from W&B
    parser.add_argument("--wandb-id", type=str, default=None, help="W&B run short ID used in artifact name (e.g., umaovsqj)")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (defaults to env WANDB_ENTITY or project config)")
    parser.add_argument("--project", type=str, default=None, help="W&B project (defaults to config['project'] or 'vit-test')")
    parser.add_argument("--version", type=str, default='v1', help="Artifact version or alias (default: v1)")
    parser.add_argument("--artifact-type", type=str, default='model', help="Artifact type (default: model)")
    parser.add_argument("--out", type=str, default=None, help="Output dir (default results/inspect/<ckpt_basename>")
    parser.add_argument("--max-batches", type=int, default=1, help="Number of test batches to capture")
    parser.add_argument("--save-full-param-tensors", action="store_true", help="Also save full parameter tensors to disk (can be large)")
    args = parser.parse_args()

    device = _device()

    config = load_config(args.config)
    # Resolve checkpoint path (local or W&B)
    ckpt_path = _resolve_ckpt(args, config)

    # Load Lightning module from checkpoint (rebuilds model from config)
    lm: ViTLModule = ViTLModule.load_from_checkpoint(ckpt_path, config=config)
    lm.eval()
    lm.to(device)

    # Data module and test loader
    dm = ViTDataModule.from_config(config, test_data=False)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    # Output directory
    ckpt_base = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = args.out or os.path.join("results", "inspect", f"{ckpt_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Save parameter stats (JSON) and optionally full tensors
    params_dir = os.path.join(out_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    param_stats = collect_param_stats(lm.model)
    with open(os.path.join(params_dir, "stats.json"), "w") as f:
        json.dump(param_stats, f, indent=2)
    print(f"✓ Saved parameter statistics to: {params_dir}/stats.json")

    if args.save_full_param_tensors:
        print(f"💾 Saving full parameter tensors...")
        saved_count = 0
        for name, p in lm.model.named_parameters():
            try:
                safe_name = name.replace('.', '_')
                path = os.path.join(params_dir, f"{safe_name}.pt")
                torch.save(p.detach().to("cpu"), path)
                saved_count += 1
            except Exception as e:
                print(f"  Warning: Could not save {name}: {e}")
        print(f"✓ Saved {saved_count} parameter tensors")

    # Patch-embedding params will be saved along with all others
    # when --save-full-param-tensors is provided via named_parameters().

    # Register hooks on the underlying model to capture activations
    acts, handles = register_activation_hooks(lm.model)

    # Run on test batches and save activations + model outputs
    print(f"🔍 Capturing activations from {args.max_batches} test batches...")
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= args.max_batches:
                break
            flux, _, labels = batch
            flux = flux.to(device)
            labels = labels.to(device)

            # Clear previous activations
            acts.clear()

            out = lm.model(
                flux,
                labels=labels,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

            # Save consolidated activations + outputs in one file
            save_activations(acts, out_dir, bi, model_out=out)
        print(f"✓ Saved activations for {bi + 1} batches")

    # Remove hooks
    for h in handles:
        h.remove()

    print("=" * 60)
    print(f"✓ Inspection complete!")
    print(f"  Output: {out_dir}")
    print(f"  Parameters: {params_dir}")
    print(f"  Batches: {bi + 1}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# Example usage:
# python scripts/inspect_ckpt.py --config "$CONFIG_DIR/vit.yaml" --ckpt ./checkpoints/epoch=0-val_mae=0.5095.ckpt
