"""Batch runner for feature preprocessing ablations.

The script mirrors `run.py` but iterates over a list of predefined (or user
supplied) ablation modes. Each mode corresponds to a statistics artifact
produced by `scripts/fit_preprocessor.py`. For every entry we patch the base
configuration with the appropriate `warmup.stats_path` and launch an
`Experiment`.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Dict, List

from src.utils import load_config
from src.vit import Experiment


DEFAULT_MODES = [
    "center",
    "standardize",
    "zca",
    "zca_lowrank",
    "project_lowrank",
    "randrot_white",
    "randrot",
    "pca",
    "pls",
    "cca",
]


def _parse_keyval(pairs: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected key=value format, got '{item}'")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def _resolve_stats_path(
    name: str,
    stats_root: Path,
    overrides: Dict[str, str],
    suffix_map: Dict[str, str],
) -> Path:
    if name in overrides:
        return Path(overrides[name]).expanduser().resolve()
    suffix = suffix_map.get(name)
    filename = name if suffix is None else f"{name}_{suffix}"
    return (stats_root / f"{filename}.pt").resolve()


def run_ablation(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)
    stats_root = Path(args.stats_root).expanduser().resolve()
    overrides = _parse_keyval(args.stats_override or [])
    suffix_map = _parse_keyval(args.suffix or [])

    modes = args.modes
    if not modes:
        modes = DEFAULT_MODES

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None and args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for idx, name in enumerate(modes, start=1):
        stats_path = _resolve_stats_path(name, stats_root, overrides, suffix_map)
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file for '{name}' not found at {stats_path}")
        print(f"[{idx}/{len(modes)}] Running ablation '{name}' with stats={stats_path}")
        if args.dry_run:
            continue

        cfg = copy.deepcopy(base_config)
        train_cfg = cfg.setdefault("train", {})
        train_cfg["debug"] = int(args.debug)
        if args.gpu is not None:
            train_cfg["gpus"] = int(args.gpu)
        if args.num_workers is not None:
            train_cfg["num_workers"] = int(args.num_workers)
        if args.save:
            train_cfg["save"] = True
        warm_cfg = cfg.setdefault("warmup", {})
        warm_cfg["preprocessor"] = "linear"
        warm_cfg["global"] = False
        warm_cfg["stats_path"] = str(stats_path)
        warm_cfg["ablation"] = name
        warm_cfg["label"] = name

        exp = Experiment(cfg, use_wandb=bool(args.wandb), sweep=False, ckpt_path=args.ckpt)
        exp.run()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a suite of preprocessing ablations")
    parser.add_argument("--config", type=str, default="configs/vit.yaml", help="Base experiment config")
    parser.add_argument("--stats-root", type=str, default="artifacts/preproc", help="Directory containing mode statistics")
    parser.add_argument("--modes", nargs="*", default=None, help="Subset of ablations to run (default: all)")
    parser.add_argument("--stats-override", metavar="name=path", nargs="*", default=None,
                        help="Explicit mapping from ablation name to stats file")
    parser.add_argument("--suffix", metavar="name=tag", nargs="*", default=None,
                        help="Append a suffix to the stats filename before '.pt'")
    parser.add_argument("--gpu", type=int, default=None, help="Number of GPUs to expose to each run")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging for each run")
    parser.add_argument("--save", action="store_true", help="Persist checkpoints/logs as configured")
    parser.add_argument("--debug", action="store_true", help="Enable Lightning fast-dev-run mode")
    parser.add_argument("--dry-run", action="store_true", help="Print runs without executing them")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for warm starts")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
