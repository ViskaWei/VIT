#!/usr/bin/env python3
"""Test and compare ZCA whitening with and without bias (mean centering).

This script trains two models:
1. With bias (bias=True): Proper mean centering
2. Without bias (bias=False): No mean centering

It compares training dynamics, convergence, and final performance.
"""

import os
import sys
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import time
import json

# Add parent directory to path so we can import from src/
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.vit import Experiment


def create_experiment_config(base_config: dict, use_bias: bool, output_dir: Path, run_suffix: str = "") -> dict:
    """Create a modified config for the experiment.
    
    Args:
        base_config: Base configuration dict
        use_bias: Whether to use bias in preprocessing
        output_dir: Output directory for this experiment
        run_suffix: Suffix to add to wandb run name for uniqueness
        
    Returns:
        Modified config dict
    """
    config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
    
    # Set bias flag
    if 'warmup' not in config:
        config['warmup'] = {}
    config['warmup']['bias'] = use_bias
    
    # Keep the same project name for all experiments (for wandb grouping)
    # The model name will be different based on bias setting (_nobias suffix)
    # This ensures all experiments appear in the same wandb project for easy comparison
    if 'project' not in config:
        config['project'] = 'zca'
    # Otherwise keep the original project name unchanged
    
    # Add a unique suffix to wandb run name to ensure separate runs
    # This helps differentiate the experiments in wandb
    if 'wandb_run_suffix' not in config:
        config['wandb_run_suffix'] = run_suffix
    
    # Reduce epochs for faster comparison (optional)
    # if 'train' in config and 'ep' in config['train']:
    #     config['train']['ep'] = min(config['train']['ep'], 50)
    
    return config


def run_experiment(config: dict, experiment_name: str, output_dir: Path, use_wandb: bool = False, gpu_id: int = 0) -> dict:
    """Run a single training experiment.
    
    Args:
        config: Configuration dict
        experiment_name: Name of the experiment
        output_dir: Output directory for logs and checkpoints
        use_wandb: Whether to use wandb logging
        gpu_id: GPU ID to use for this experiment
        
    Returns:
        Dictionary with experiment results
    """
    # Set GPU visibility for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print("=" * 80)
    print(f"Running experiment: {experiment_name}")
    print(f"Using GPU: {gpu_id}")
    print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_path}")
    
    # Setup checkpoint and log directories
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Override checkpoint directory in config
    os.environ['CKPT_DIR'] = str(checkpoint_dir)
    
    # Enable saving checkpoints for this experiment
    if 'train' not in config:
        config['train'] = {}
    config['train']['save'] = True
    
    # Run experiment using the Experiment class (like vit.py and sweep.py)
    print(f"\nStarting training...")
    exp = Experiment(config, use_wandb=use_wandb, num_gpus=1, test_data=False)
    exp.run()
    
    # Collect results from the trained model
    print(f"\nCollecting results...")
    
    # Get best checkpoint path
    best_checkpoint = None
    if checkpoint_dir.exists():
        ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
        # Filter out 'last.ckpt' and find the best one
        ckpt_files = [f for f in ckpt_files if f.name != 'last.ckpt']
        if ckpt_files:
            # Sort by modification time, take the most recent
            best_checkpoint = str(sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1])
    
    # Get metrics from the logger
    metrics = {}
    try:
        # The trainer's logger should have the metrics
        if hasattr(exp.t, 'trainer') and hasattr(exp.t.trainer, 'logged_metrics'):
            logged = exp.t.trainer.logged_metrics
            metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in logged.items()}
    except Exception as e:
        print(f"Could not retrieve metrics: {e}")
    
    # Find validation loss (try various possible keys)
    val_loss = None
    for key in ['val_mae', 'val/mae', 'val_loss', 'val/loss', 'val_l1_loss']:
        if key in metrics:
            val_loss = metrics[key]
            break
    
    # Collect test results (if available)
    test_results = {}
    for key in metrics:
        if key.startswith('test_') or key.startswith('test/'):
            test_results[key] = metrics[key]
    
    results = {
        'experiment_name': experiment_name,
        'best_val_loss': val_loss,
        'test_results': test_results,
        'best_checkpoint': best_checkpoint,
        'config_path': str(config_path),
        'log_dir': str(output_dir / "logs"),
    }
    
    print(f"\n{experiment_name} Results:")
    if val_loss is not None:
        print(f"  Best val loss: {val_loss:.6f}")
    if test_results:
        for key, value in test_results.items():
            print(f"  {key}: {value:.6f}")
    
    return results


def compare_results(results_with_bias: dict, results_no_bias: dict, output_dir: Path):
    """Compare and visualize results from both experiments.
    
    Args:
        results_with_bias: Results from experiment with bias
        results_no_bias: Results from experiment without bias
        output_dir: Output directory for comparison report
    """
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("# ZCA Bias Comparison Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report_lines.append("## Experiment Setup")
    report_lines.append(f"- Experiment 1: With bias (mean centering)")
    report_lines.append(f"- Experiment 2: Without bias (no mean centering)\n")
    
    # Compare validation loss
    report_lines.append("## Validation Loss")
    val_with = results_with_bias.get('best_val_loss')
    val_without = results_no_bias.get('best_val_loss')
    
    if val_with and val_without:
        improvement = ((val_without - val_with) / val_without) * 100
        report_lines.append(f"- With bias:    {val_with:.6f}")
        report_lines.append(f"- Without bias: {val_without:.6f}")
        report_lines.append(f"- Improvement:  {improvement:+.2f}% {'✓' if improvement > 0 else '✗'}\n")
        
        print("\nValidation Loss:")
        print(f"  With bias:    {val_with:.6f}")
        print(f"  Without bias: {val_without:.6f}")
        print(f"  Improvement:  {improvement:+.2f}% {'✓' if improvement > 0 else '✗'}")
    
    # Compare test results
    report_lines.append("## Test Results")
    test_with = results_with_bias.get('test_results', {})
    test_without = results_no_bias.get('test_results', {})
    
    print("\nTest Results:")
    for key in sorted(set(list(test_with.keys()) + list(test_without.keys()))):
        if key in test_with and key in test_without:
            val_w = test_with[key]
            val_wo = test_without[key]
            improvement = ((val_wo - val_w) / val_wo) * 100 if val_wo != 0 else 0
            
            report_lines.append(f"\n### {key}")
            report_lines.append(f"- With bias:    {val_w:.6f}")
            report_lines.append(f"- Without bias: {val_wo:.6f}")
            report_lines.append(f"- Improvement:  {improvement:+.2f}% {'✓' if improvement > 0 else '✗'}")
            
            print(f"  {key}:")
            print(f"    With bias:    {val_w:.6f}")
            print(f"    Without bias: {val_wo:.6f}")
            print(f"    Improvement:  {improvement:+.2f}% {'✓' if improvement > 0 else '✗'}")
    
    # File paths
    report_lines.append("\n## Output Files")
    report_lines.append(f"- With bias logs:    {results_with_bias.get('log_dir')}")
    report_lines.append(f"- Without bias logs: {results_no_bias.get('log_dir')}")
    report_lines.append(f"- With bias config:    {results_with_bias.get('config_path')}")
    report_lines.append(f"- Without bias config: {results_no_bias.get('config_path')}")
    
    # Conclusion
    report_lines.append("\n## Conclusion")
    if val_with and val_without:
        if val_with < val_without:
            report_lines.append("✓ **WITH BIAS is better** - Mean centering improves performance")
            print("\n✓ CONCLUSION: WITH BIAS is better - Mean centering improves performance")
        elif val_with > val_without:
            report_lines.append("✗ **WITHOUT BIAS is better** - This is unexpected, investigate further")
            print("\n✗ CONCLUSION: WITHOUT BIAS is better - This is unexpected")
        else:
            report_lines.append("~ **No significant difference** - Bias has minimal impact")
            print("\n~ CONCLUSION: No significant difference")
    
    # Save report
    report_path = output_dir / "comparison_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nComparison report saved to: {report_path}")


def run_experiment_wrapper(args_tuple):
    """Wrapper function to run experiment in a separate process.
    
    This is NOT USED in parallel mode (we use subprocess instead).
    Kept for backward compatibility with sequential mode.
    """
    config, experiment_name, output_dir, use_wandb, gpu_id = args_tuple
    return run_experiment(config, experiment_name, output_dir, use_wandb, gpu_id)


def run_experiment_subprocess(config_path: Path, output_dir: Path, gpu_id: int, use_wandb: bool = False) -> subprocess.Popen:
    """Run experiment as a subprocess.
    
    Args:
        config_path: Path to the experiment config YAML file
        output_dir: Output directory for this experiment
        gpu_id: GPU ID to use
        use_wandb: Whether to use wandb logging
        
    Returns:
        subprocess.Popen object
    """
    # Build the command to run
    script_path = Path(__file__)  # This script itself
    
    cmd = [
        sys.executable,  # python3
        str(script_path),
        "--config", str(config_path),
        "--output_dir", str(output_dir),
        "--run_single_experiment",  # Special flag to run just one experiment
        "--gpu_id", str(gpu_id),
    ]
    
    if use_wandb:
        cmd.append("--use_wandb")
    
    # Set environment variable for GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Start the subprocess
    log_file = output_dir / "subprocess.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting subprocess with GPU {gpu_id}, logging to {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT)
        )
    
    return process


def run_single_experiment_from_config(config_path: Path, output_dir: Path, gpu_id: int, use_wandb: bool):
    """Run a single experiment from a config file (used by subprocess).
    
    This is called when --run_single_experiment flag is set.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine experiment name from bias setting
    use_bias = config.get('warmup', {}).get('bias', True)
    experiment_name = "With Bias (Mean Centering)" if use_bias else "Without Bias (No Mean Centering)"
    
    # Run the experiment
    results = run_experiment(config, experiment_name, output_dir, use_wandb, gpu_id)
    
    # Save results to a JSON file so parent process can read them
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        results_serializable = {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in results.items()
        }
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare ZCA whitening with and without bias"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vit.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/bias_comparison",
        help="Output directory for experiment results",
    )
    parser.add_argument(
        "--skip_with_bias",
        action="store_true",
        help="Skip experiment with bias (only run without bias)",
    )
    parser.add_argument(
        "--skip_without_bias",
        action="store_true",
        help="Skip experiment without bias (only run with bias)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel on separate GPUs",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1",
        help="Comma-separated GPU IDs to use (e.g., '0,1' or '4,5')",
    )
    parser.add_argument(
        "--run_single_experiment",
        action="store_true",
        help="Internal flag: run a single experiment (used by subprocess)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID for single experiment (used with --run_single_experiment)",
    )
    
    args = parser.parse_args()
    
    # Special mode: run single experiment (called by subprocess)
    if args.run_single_experiment:
        config_path = Path(args.config).parent / "config.yaml"
        if not config_path.exists():
            # Try using the provided config path directly
            config_path = Path(args.config)
        
        run_single_experiment_from_config(
            config_path=config_path,
            output_dir=Path(args.output_dir),
            gpu_id=args.gpu_id,
            use_wandb=args.use_wandb
        )
        return  # Exit after running single experiment
    
    # Set random seed
    import lightning as L
    L.seed_everything(args.seed)
    
    # Load base config
    print(f"Loading base config from {args.config}")
    base_config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    if len(gpu_ids) < 2 and args.parallel:
        print(f"Warning: Only {len(gpu_ids)} GPU(s) specified, but need 2 for parallel execution")
        print(f"Will run sequentially instead")
        args.parallel = False
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Use wandb: {args.use_wandb}")
    print(f"Parallel mode: {args.parallel}")
    if args.parallel:
        print(f"GPU IDs: {gpu_ids}")
    print()
    
    results = {}
    
    # Prepare experiment configurations
    experiments_to_run = []
    
    if not args.skip_with_bias:
        config_with_bias = create_experiment_config(
            base_config, 
            use_bias=True, 
            output_dir=output_dir / "with_bias",
            run_suffix="_WITH_BIAS"
        )
        experiments_to_run.append((
            config_with_bias,
            "With Bias (Mean Centering)",
            output_dir / "with_bias",
            args.use_wandb,
            gpu_ids[0] if args.parallel else 0,
            'with_bias'
        ))
    
    if not args.skip_without_bias:
        config_no_bias = create_experiment_config(
            base_config, 
            use_bias=False, 
            output_dir=output_dir / "no_bias",
            run_suffix="_NO_BIAS"
        )
        experiments_to_run.append((
            config_no_bias,
            "Without Bias (No Mean Centering)",
            output_dir / "no_bias",
            args.use_wandb,
            gpu_ids[1] if args.parallel and len(gpu_ids) > 1 else 0,
            'no_bias'
        ))
    
    # Run experiments
    if args.parallel and len(experiments_to_run) == 2:
        print("Running experiments in PARALLEL on separate GPUs using subprocesses...")
        print(f"  - Experiment 1 (with bias) on GPU {experiments_to_run[0][4]}")
        print(f"  - Experiment 2 (without bias) on GPU {experiments_to_run[1][4]}")
        print()
        
        # Save configs to temporary files
        processes = []
        temp_config_paths = []
        
        for i, exp in enumerate(experiments_to_run):
            config, name, out_dir, use_wb, gpu_id, key = exp
            
            # Save config to temp file
            temp_config_path = out_dir / "config.yaml"
            temp_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            temp_config_paths.append((temp_config_path, key, out_dir))
            
            print(f"Starting experiment '{name}' on GPU {gpu_id}")
            
            # Start subprocess
            process = run_experiment_subprocess(temp_config_path, out_dir, gpu_id, use_wb)
            processes.append(process)
            
            # Small delay to avoid race conditions
            time.sleep(2)
        
        print(f"\nBoth experiments started! Waiting for completion...")
        print(f"Monitor logs at:")
        for _, _, out_dir in temp_config_paths:
            print(f"  - {out_dir}/subprocess.log")
        print()
        
        # Wait for all processes to complete
        for i, process in enumerate(processes):
            exp_name = "with_bias" if i == 0 else "no_bias"
            print(f"Waiting for {exp_name} experiment to complete...")
            return_code = process.wait()
            
            if return_code != 0:
                print(f"  ✗ {exp_name} failed with return code {return_code}")
            else:
                print(f"  ✓ {exp_name} completed successfully")
        
        # Collect results from JSON files
        print("\nCollecting results...")
        for config_path, key, out_dir in temp_config_paths:
            results_file = out_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results[key] = json.load(f)
                print(f"  ✓ Loaded results for {key}")
            else:
                print(f"  ✗ No results found for {key} at {results_file}")
                results[key] = {}
        
    else:
        print("Running experiments SEQUENTIALLY...")
        print()
        
        # Run experiments sequentially
        for exp in experiments_to_run:
            config, name, out_dir, use_wb, gpu_id, key = exp
            results[key] = run_experiment(config, name, out_dir, use_wb, gpu_id)
    
    # Compare results
    if 'with_bias' in results and 'no_bias' in results:
        compare_results(
            results_with_bias=results['with_bias'],
            results_no_bias=results['no_bias'],
            output_dir=output_dir,
        )
    else:
        print("\nSkipping comparison (only one experiment run)")
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
