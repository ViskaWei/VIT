"""Estimate NTK spectra for SpectraTransformer under different pre-processing modes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

from src.models import SpectraTransformer, SpectraTransformerConfig
from src.prepca.pipeline import ZCAWhitening


def _load_tensor(path: Path) -> torch.Tensor:
    if path.suffix in {".pt", ".pth"}:
        tensor = torch.load(path)
        if isinstance(tensor, dict):
            raise ValueError(f"File {path} contains a dict; expected a tensor")
        return tensor.float()
    if path.suffix in {".npy", ".npz"}:
        import numpy as np

        array = np.load(path)
        if isinstance(array, np.lib.npyio.NpzFile):
            key = array.files[0]
            array = array[key]
        return torch.as_tensor(array, dtype=torch.float32)
    raise ValueError(f"Unsupported tensor format for {path.suffix}")


def _select_subset(tensor: torch.Tensor, subset: int) -> torch.Tensor:
    if tensor.shape[0] < subset:
        raise ValueError(
            f"Requested subset of {subset} rows but tensor only has {tensor.shape[0]}"
        )
    return tensor[:subset]


def _build_model(args: argparse.Namespace, projector_dim: int | None = None) -> SpectraTransformer:
    config = SpectraTransformerConfig(
        input_dim=args.input_dim,
        num_targets=args.num_targets,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        projector_dim=projector_dim,
    )
    model = SpectraTransformer(config)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _prepare_inputs(args: argparse.Namespace) -> torch.Tensor:
    if args.data is None:
        torch.manual_seed(args.seed)
        return torch.randn(args.subset_size, args.input_dim)
    tensor = _load_tensor(Path(args.data))
    if tensor.dim() > 2:
        tensor = tensor.view(tensor.shape[0], -1)
    if tensor.shape[1] != args.input_dim:
        raise ValueError(
            f"Loaded tensor has feature dimension {tensor.shape[1]}, expected {args.input_dim}"
        )
    return _select_subset(tensor, args.subset_size)


def _last_layer_ntk(model: SpectraTransformer, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        features = model.get_representation(inputs)
    return features @ features.t()


def _full_ntk(model: SpectraTransformer, inputs: torch.Tensor) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    for sample in inputs:
        model.zero_grad(set_to_none=True)
        sample = sample.unsqueeze(0)
        output = model(sample)
        scalar = output.sum()
        grad = torch.autograd.grad(scalar, params, retain_graph=False, create_graph=False)
        grads.append(torch.cat([g.reshape(-1) for g in grad]))
    grad_matrix = torch.stack(grads)
    return grad_matrix @ grad_matrix.t()


def _spectral_stats(gram: torch.Tensor) -> Tuple[float, float, float]:
    gram = gram.cpu()
    eigvals = torch.linalg.eigvalsh(gram)
    lambda_min = float(eigvals.min().item())
    lambda_max = float(eigvals.max().item())
    condition = float("inf") if lambda_min <= 0 else float(lambda_max / lambda_min)
    trace = float(eigvals.sum().item())
    return lambda_min, condition, trace


def run_probe(args: argparse.Namespace) -> Dict[str, Tuple[float, float, float]]:
    inputs = _prepare_inputs(args)
    device = torch.device(args.device)
    inputs = inputs.to(device)

    modes: Dict[str, torch.Tensor] = {}

    if args.mode in {"all", "none"}:
        model = _build_model(args)
        model.to(device)
        gram = _last_layer_ntk(model, inputs) if args.scope == "last" else _full_ntk(model, inputs)
        modes["none"] = gram

    if args.zca is not None and args.mode in {"all", "whitening"}:
        zca = ZCAWhitening.load(args.zca, map_location=device)
        projector_dim = zca.whitening_matrix.shape[0]
        model = _build_model(args, projector_dim=projector_dim)
        model.to(device)
        model.load_preconditioning(zca.whitening_matrix, freeze=True, assume_aligned=True)
        gram = _last_layer_ntk(model, inputs) if args.scope == "last" else _full_ntk(model, inputs)
        modes["whitening"] = gram

    if args.zca is not None and args.mode in {"all", "projector"}:
        zca = ZCAWhitening.load(args.zca, map_location=device)
        projector = zca.projector_matrix
        model = _build_model(args, projector_dim=projector.shape[0])
        model.to(device)
        model.load_preconditioning(projector, freeze=True, assume_aligned=True)
        gram = _last_layer_ntk(model, inputs) if args.scope == "last" else _full_ntk(model, inputs)
        modes["projector"] = gram

    stats = {mode: _spectral_stats(gram) for mode, gram in modes.items()}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None, help="Optional tensor file providing spectra")
    parser.add_argument("--zca", type=str, default=None, help="Path to saved ZCAWhitening state")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint")
    parser.add_argument("--subset-size", type=int, default=512, help="Number of samples used for NTK computation")
    parser.add_argument("--input-dim", type=int, default=4096, help="Spectral dimensionality")
    parser.add_argument("--num-targets", type=int, default=3, help="Number of regression outputs")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for the transformer")
    parser.add_argument("--embed-dim", type=int, default=128, help="Transformer embedding dimension")
    parser.add_argument("--depth", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="Feed-forward hidden size ratio")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the encoder")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    parser.add_argument("--mode", type=str, choices=["all", "none", "whitening", "projector"], default="all")
    parser.add_argument("--scope", type=str, choices=["last", "full"], default="last", help="Compute NTK for last layer or full network")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = run_probe(args)
    for mode, (lambda_min, condition, trace) in stats.items():
        print(f"[{mode}] lambda_min={lambda_min:.6g} condition={condition:.6g} trace={trace:.6g}")


if __name__ == "__main__":
    main()
