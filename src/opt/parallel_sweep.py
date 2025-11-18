"""
Parallel Optimizer Sweep Runner
================================

This tool runs optimizer hyperparameter sweeps across multiple GPUs in parallel.
It can search for optimal learning rates, schedulers, and their parameters.

Features:
- Parallel execution across multiple GPUs (default: all 8 available)
- Grid search over learning rates and scheduler types
- Automatic best configuration selection
- Results saved to disk with summary
- Can be used standalone or with wandb

Usage:
    # Basic LR sweep
    python src/opt/parallel_sweep.py configs/exp/att_clp/baseline.yaml --gpus 0,1,2,3,4,5,6,7

    # LR + Scheduler sweep
    python src/opt/parallel_sweep.py configs/exp/att_clp/baseline.yaml \
        --lr 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 \
        --schedulers plateau cosine \
        --gpus 0,1,2,3,4,5,6,7

    # Full sweep with custom parameters
    python src/opt/parallel_sweep.py configs/exp/att_clp/baseline.yaml \
        --lr 1e-4 5e-4 1e-3 5e-3 \
        --schedulers plateau cosine \
        --plateau-factor 0.5 0.7 0.8 \
        --plateau-patience 5 10 15 \
        --gpus 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

# Default search grids
DEFAULT_LR_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
DEFAULT_SCHEDULERS = ['plateau', 'cosine', 'none']
DEFAULT_PLATEAU_FACTORS = [0.5, 0.7, 0.8, 0.9]
DEFAULT_PLATEAU_PATIENCE = [5, 10, 15]


@dataclass
class SweepConfig:
    """Configuration for a single sweep run."""
    idx: int
    lr: float
    scheduler: str | None = None
    factor: float | None = None
    patience: int | None = None
    T_max: int | None = None
    eta_min: float | None = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_opt_config(self) -> Dict[str, Any]:
        """Convert to opt config dict."""
        opt = {'lr': self.lr}
        
        if self.scheduler and self.scheduler != 'none':
            opt['lr_sch'] = self.scheduler
            
            if self.scheduler == 'plateau':
                if self.factor is not None:
                    opt['factor'] = self.factor
                if self.patience is not None:
                    opt['patience'] = self.patience
                    
            elif self.scheduler == 'cosine':
                if self.T_max is not None:
                    opt['T_max'] = self.T_max
                if self.eta_min is not None:
                    opt['eta_min'] = self.eta_min
        
        return opt


@dataclass
class SweepResult:
    """Result from a single sweep run."""
    config: SweepConfig
    metric: float | None
    duration_sec: float
    status: str = "ok"
    message: str | None = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metric": self.metric,
            "duration_sec": self.duration_sec,
            "status": self.status,
            "message": self.message,
        }


def _run_single_trial(
    base_config: Dict[str, Any],
    sweep_cfg: SweepConfig,
    gpu_id: int,
    metric_name: str,
    seed: int,
    skip_test: bool,
) -> SweepResult:
    """Run a single trial on a specific GPU."""
    import lightning as L
    import torch
    
    # Import here to avoid issues with multiprocessing
    from src.utils import load_config
    from src.vit import Experiment
    
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Build config
    config = copy.deepcopy(base_config)
    opt_config = sweep_cfg.to_opt_config()
    
    # Ensure 'opt' key exists
    if 'opt' not in config:
        config['opt'] = {}
    config['opt'].update(opt_config)
    
    # Ensure training settings
    train_cfg = config.setdefault('train', {})
    train_cfg.setdefault('save', False)
    config.setdefault('project', 'vit-opt-sweep')
    
    # Disable viz for speed
    if 'viz' in config:
        config['viz']['enable'] = False
    
    start = time.perf_counter()
    try:
        L.seed_everything(seed + sweep_cfg.idx, workers=True)
        
        experiment = Experiment(
            config,
            use_wandb=False,
            num_gpus=1,  # Each process uses 1 GPU
            sweep=False,
        )
        
        experiment.t.trainer.fit(
            experiment.lightning_module,
            datamodule=experiment.data_module
        )
        
        if not skip_test:
            experiment.t.test_trainer.test(
                experiment.lightning_module,
                datamodule=experiment.data_module
            )
        
        # Get metric
        metric_value = experiment.t.trainer.callback_metrics.get(metric_name)
        if metric_value is not None:
            if hasattr(metric_value, 'item'):
                metric_value = float(metric_value.item())
            else:
                metric_value = float(metric_value)
        
        if metric_value is None or math.isnan(metric_value):
            raise RuntimeError(
                f"Metric '{metric_name}' was not produced. "
                "Check metric name or ensure validation step logs it."
            )
        
        duration = time.perf_counter() - start
        status = "ok"
        message = None
        
    except Exception as exc:
        duration = time.perf_counter() - start
        metric_value = None
        status = "error"
        message = f"{type(exc).__name__}: {exc}"
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return SweepResult(
        config=sweep_cfg,
        metric=metric_value,
        duration_sec=duration,
        status=status,
        message=message,
    )


def _worker_process(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    base_config: Dict[str, Any],
    gpu_id: int,
    metric_name: str,
    seed: int,
    skip_test: bool,
):
    """Worker process that pulls tasks from queue and runs them on a specific GPU."""
    while True:
        try:
            sweep_cfg = task_queue.get(timeout=1)
            if sweep_cfg is None:  # Poison pill
                break
                
            result = _run_single_trial(
                base_config, sweep_cfg, gpu_id, metric_name, seed, skip_test
            )
            result_queue.put(result)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker error on GPU {gpu_id}: {e}")
            break


class ParallelSweepRunner:
    """Runs optimizer sweeps in parallel across multiple GPUs."""
    
    def __init__(
        self,
        base_config_path: str | Path,
        lr_values: Sequence[float] | None = None,
        schedulers: Sequence[str] | None = None,
        plateau_factors: Sequence[float] | None = None,
        plateau_patience: Sequence[int] | None = None,
        cosine_T_max: int | None = None,
        cosine_eta_min: float | None = None,
        *,
        metric_name: str = "val_mae",
        metric_goal: str = "minimize",
        results_dir: str | Path | None = None,
        gpu_ids: Sequence[int] | None = None,
        seed: int = 42,
        skip_test: bool = True,
    ):
        # Load base config
        self.base_config_path = Path(base_config_path).expanduser().resolve()
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {self.base_config_path}")
        
        from src.utils import load_config
        self.base_config = load_config(str(self.base_config_path))
        
        # Search space
        self.lr_values = list(lr_values) if lr_values else DEFAULT_LR_VALUES
        self.schedulers = list(schedulers) if schedulers else ['none']
        self.plateau_factors = list(plateau_factors) if plateau_factors else [0.8]
        self.plateau_patience = list(plateau_patience) if plateau_patience else [10]
        
        # Cosine params
        if cosine_T_max is None:
            cosine_T_max = self.base_config.get('train', {}).get('ep', 50)
        self.cosine_T_max = cosine_T_max
        self.cosine_eta_min = cosine_eta_min if cosine_eta_min is not None else 1e-7
        
        # Metric
        self.metric_name = metric_name
        self.metric_goal = metric_goal
        
        # GPUs
        if gpu_ids is None:
            # Use all 8 GPUs by default
            self.gpu_ids = list(range(8))
        else:
            self.gpu_ids = list(gpu_ids)
        
        if not self.gpu_ids:
            raise ValueError("Must specify at least one GPU")
        
        # Results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_results_dir = Path.cwd() / "opt_runs"
        self.results_root = Path(results_dir).expanduser().resolve() if results_dir else default_results_dir
        self.run_dir = self.results_root / f"parallel_sweep_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.summary_path = self.run_dir / "summary.yaml"
        self.best_config_path = self.run_dir / "best_config.yaml"
        
        # Other
        self.seed = seed
        self.skip_test = skip_test
        
        # Build sweep configs
        self.sweep_configs = self._build_sweep_configs()
    
    def _build_sweep_configs(self) -> List[SweepConfig]:
        """Build all sweep configurations."""
        configs = []
        idx = 0
        
        for lr in self.lr_values:
            for scheduler in self.schedulers:
                if scheduler == 'plateau':
                    # Grid over plateau parameters
                    for factor in self.plateau_factors:
                        for patience in self.plateau_patience:
                            configs.append(SweepConfig(
                                idx=idx,
                                lr=lr,
                                scheduler=scheduler,
                                factor=factor,
                                patience=patience,
                            ))
                            idx += 1
                            
                elif scheduler == 'cosine':
                    # Use cosine parameters
                    configs.append(SweepConfig(
                        idx=idx,
                        lr=lr,
                        scheduler=scheduler,
                        T_max=self.cosine_T_max,
                        eta_min=self.cosine_eta_min,
                    ))
                    idx += 1
                    
                else:  # 'none' or other
                    configs.append(SweepConfig(
                        idx=idx,
                        lr=lr,
                        scheduler=scheduler if scheduler != 'none' else None,
                    ))
                    idx += 1
        
        return configs
    
    def run(self, dry_run: bool = False) -> SweepResult:
        """Run the parallel sweep."""
        self._print_header(dry_run=dry_run)
        
        if dry_run:
            print("\n[Dry run] Would execute the following configurations:")
            for cfg in self.sweep_configs[:10]:  # Show first 10
                print(f"  {cfg.to_dict()}")
            if len(self.sweep_configs) > 10:
                print(f"  ... and {len(self.sweep_configs) - 10} more")
            return None
        
        results = self._run_parallel()
        best = self._select_best(results)
        self._write_summary(results, best)
        self._write_best_config(best)
        self._print_footer(best)
        return best
    
    def _run_parallel(self) -> List[SweepResult]:
        """Run sweep in parallel across GPUs."""
        num_workers = len(self.gpu_ids)
        
        # Create queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Fill task queue
        for cfg in self.sweep_configs:
            task_queue.put(cfg)
        
        # Add poison pills
        for _ in range(num_workers):
            task_queue.put(None)
        
        # Start workers
        workers = []
        for gpu_id in self.gpu_ids:
            p = mp.Process(
                target=_worker_process,
                args=(
                    task_queue,
                    result_queue,
                    self.base_config,
                    gpu_id,
                    self.metric_name,
                    self.seed,
                    self.skip_test,
                )
            )
            p.start()
            workers.append(p)
        
        # Collect results
        results = []
        total = len(self.sweep_configs)
        
        print(f"\n[Running] {total} configurations on {num_workers} GPUs...\n")
        
        for i in range(total):
            result = result_queue.get()
            results.append(result)
            self._print_progress(result, i + 1, total)
        
        # Wait for workers to finish
        for p in workers:
            p.join()
        
        return results
    
    def _print_header(self, *, dry_run: bool) -> None:
        banner = textwrap.dedent(
            f"""
            ==============================================================
            Parallel Optimizer Sweep
            Base config : {self.base_config_path}
            Results dir : {self.run_dir}
            Metric      : {self.metric_name} ({self.metric_goal})
            GPUs        : {', '.join(map(str, self.gpu_ids))} ({len(self.gpu_ids)} parallel workers)
            Configs     : {len(self.sweep_configs)} total
            Dry run     : {"yes" if dry_run else "no"}
            ==============================================================
            
            Search space:
              LR values     : {', '.join(f'{lr:.1e}' for lr in self.lr_values)}
              Schedulers    : {', '.join(self.schedulers)}
            """
        ).strip("\n")
        
        if 'plateau' in self.schedulers:
            banner += f"\n  Plateau factor: {', '.join(map(str, self.plateau_factors))}"
            banner += f"\n  Plateau patience: {', '.join(map(str, self.plateau_patience))}"
        
        if 'cosine' in self.schedulers:
            banner += f"\n  Cosine T_max: {self.cosine_T_max}"
            banner += f"\n  Cosine eta_min: {self.cosine_eta_min}"
        
        print(banner)
    
    def _print_progress(self, result: SweepResult, completed: int, total: int):
        """Print progress for a single result."""
        cfg_str = f"lr={result.config.lr:.1e}"
        if result.config.scheduler:
            cfg_str += f" sch={result.config.scheduler}"
            if result.config.scheduler == 'plateau':
                cfg_str += f" f={result.config.factor} p={result.config.patience}"
        
        metric_str = f"{result.metric:.6f}" if result.metric is not None else "N/A"
        status_icon = "✓" if result.status == "ok" else "✗"
        
        print(
            f"[{completed:3d}/{total:3d}] {status_icon} {cfg_str:40s} | "
            f"{self.metric_name}={metric_str:10s} | {result.duration_sec:5.1f}s"
        )
        
        if result.message:
            print(f"    ↳ {result.message}")
    
    def _print_footer(self, best: SweepResult):
        print("\n" + "=" * 62)
        print("Sweep finished!")
        print(f"\nBest configuration:")
        print(f"  LR          : {best.config.lr:.6f}")
        if best.config.scheduler:
            print(f"  Scheduler   : {best.config.scheduler}")
            if best.config.scheduler == 'plateau':
                print(f"  Factor      : {best.config.factor}")
                print(f"  Patience    : {best.config.patience}")
            elif best.config.scheduler == 'cosine':
                print(f"  T_max       : {best.config.T_max}")
                print(f"  eta_min     : {best.config.eta_min}")
        print(f"  {self.metric_name:11s}: {best.metric:.6f}")
        print(f"\nSummary  : {self.summary_path}")
        print(f"Best cfg : {self.best_config_path}")
        print("=" * 62)
    
    def _select_best(self, results: List[SweepResult]) -> SweepResult:
        """Select the best result."""
        valid = [
            r for r in results
            if r.status == "ok" and r.metric is not None and math.isfinite(r.metric)
        ]
        
        if not valid:
            raise RuntimeError("All trials failed. Check logs above for details.")
        
        reverse = self.metric_goal == "maximize"
        best = sorted(valid, key=lambda r: r.metric, reverse=reverse)[0]
        return best
    
    def _write_summary(self, results: List[SweepResult], best: SweepResult):
        """Write summary to YAML."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "base_config": str(self.base_config_path),
            "metric": {
                "name": self.metric_name,
                "goal": self.metric_goal,
            },
            "search_space": {
                "lr_values": self.lr_values,
                "schedulers": self.schedulers,
                "plateau_factors": self.plateau_factors,
                "plateau_patience": self.plateau_patience,
                "cosine_T_max": self.cosine_T_max,
                "cosine_eta_min": self.cosine_eta_min,
            },
            "gpus": self.gpu_ids,
            "num_configs": len(self.sweep_configs),
            "results": [r.to_dict() for r in results],
            "best": best.to_dict(),
        }
        
        with open(self.summary_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    
    def _write_best_config(self, best: SweepResult):
        """Write best configuration to YAML."""
        best_config = copy.deepcopy(self.base_config)
        
        # Ensure 'opt' key exists
        if 'opt' not in best_config:
            best_config['opt'] = {}
        best_config['opt'].update(best.config.to_opt_config())
        
        best_config.setdefault('_meta', {})
        best_config['_meta'].update({
            'generator': 'src.opt.parallel_sweep',
            'metric': self.metric_name,
            'goal': self.metric_goal,
            'best_metric': best.metric,
            'search_dir': str(self.run_dir),
            'timestamp': datetime.now().isoformat(),
        })
        
        with open(self.best_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best_config, f, sort_keys=False, allow_unicode=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel optimizer sweep across multiple GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Basic LR sweep on 8 GPUs
              python src/opt/parallel_sweep.py configs/exp/att_clp/baseline.yaml
              
              # LR + Scheduler sweep
              python src/opt/parallel_sweep.py configs/exp/att_clp/baseline.yaml \\
                  --lr 1e-4 5e-4 1e-3 5e-3 \\
                  --schedulers plateau cosine \\
                  --gpus 0,1,2,3,4,5,6,7
              
              # Full sweep with custom plateau params
              python src/opt/parallel_sweep.py configs/exp/att_clp/baseline.yaml \\
                  --lr 1e-4 5e-4 1e-3 \\
                  --schedulers plateau \\
                  --plateau-factor 0.5 0.7 0.8 \\
                  --plateau-patience 5 10 15 \\
                  --gpus 0,1,2,3,4,5,6,7
        """)
    )
    
    parser.add_argument(
        "base_config",
        help="Path to the base YAML config (e.g., configs/exp/att_clp/baseline.yaml)",
    )
    
    # Search space
    parser.add_argument(
        "--lr",
        nargs="+",
        type=float,
        help="Learning rate values to search (default: 1e-5, 5e-5, ..., 1e-2)",
    )
    parser.add_argument(
        "--schedulers",
        nargs="+",
        choices=['plateau', 'cosine', 'none'],
        help="Scheduler types to search (default: none)",
    )
    parser.add_argument(
        "--plateau-factor",
        nargs="+",
        type=float,
        help="Plateau reduction factors to search (default: 0.8)",
    )
    parser.add_argument(
        "--plateau-patience",
        nargs="+",
        type=int,
        help="Plateau patience values to search (default: 10)",
    )
    parser.add_argument(
        "--cosine-T-max",
        type=int,
        help="Cosine T_max parameter (default: from config train.ep)",
    )
    parser.add_argument(
        "--cosine-eta-min",
        type=float,
        help="Cosine eta_min parameter (default: 1e-7)",
    )
    
    # Metric
    parser.add_argument(
        "--metric",
        default="val_mae",
        help="Monitored metric name (default: val_mae)",
    )
    parser.add_argument(
        "--goal",
        choices=("minimize", "maximize"),
        default="minimize",
        help="Optimization direction for the metric",
    )
    
    # Execution
    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU IDs to use (default: 0,1,2,3,4,5,6,7)",
    )
    parser.add_argument(
        "--results-dir",
        help="Directory to store sweep results (default: ./opt_runs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Also execute the test loop after each fit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the sweep plan without training",
    )
    
    return parser.parse_args()


def main():
    args = _parse_args()
    
    # Parse GPU IDs
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        gpu_ids = list(range(8))  # Default: all 8 GPUs
    
    runner = ParallelSweepRunner(
        base_config_path=args.base_config,
        lr_values=args.lr,
        schedulers=args.schedulers,
        plateau_factors=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        cosine_T_max=args.cosine_T_max,
        cosine_eta_min=args.cosine_eta_min,
        metric_name=args.metric,
        metric_goal=args.goal,
        results_dir=args.results_dir,
        gpu_ids=gpu_ids,
        seed=args.seed,
        skip_test=not args.run_test,
    )
    
    runner.run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    sys.exit(main())

