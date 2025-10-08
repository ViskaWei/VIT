#!/usr/bin/env python3
"""
Experiment Suite Runner
Runs multiple experiments in parallel across available GPUs.

Usage:
    # Run 4 basic experiments (vit, att, pca, zpca)
    python scripts/run_exp_suite.py --suite basic --gpus 0,1,2,3
    
    # Run basic + variants
    python scripts/run_exp_suite.py --suite extended --gpus 0,1,2,3,4,5,6,7
    
    # Run custom experiment list
    python scripts/run_exp_suite.py --configs vit,att,pca --gpus 0,1,2
    
    # Override parameters for all experiments
    python scripts/run_exp_suite.py --suite basic --gpus 0,1,2,3 --override "train.ep=500,opt.lr=0.02"
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import yaml

# Predefined experiment suites
SUITES = {
    'basic': ['vit', 'att', 'pca', 'zpca'],
    'extended': ['vit', 'att', 'pca', 'zpca', 'zpca_lowrank', 'zpca_adaptive', 'zpca_improved'],
    'minimal': ['vit', 'pca'],
    'warmup': ['att', 'pca', 'zpca'],
}

class ExperimentRunner:
    def __init__(self, config_dir: str = 'configs/exp', gpus: Optional[List[int]] = None, 
                 wandb: bool = True, debug: bool = False, save: bool = False):
        self.config_dir = Path(config_dir)
        self.gpus = gpus or [0]
        self.wandb = wandb
        self.debug = debug
        self.save = save
        self.processes: List[subprocess.Popen] = []
        self.results: Dict[str, dict] = {}
        
    def parse_overrides(self, override_str: Optional[str]) -> Dict:
        """Parse override string like 'train.ep=500,opt.lr=0.02' into nested dict"""
        if not override_str:
            return {}
        
        overrides = {}
        for item in override_str.split(','):
            key, value = item.strip().split('=')
            keys = key.split('.')
            
            # Try to parse value as int, float, or keep as string
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
            
            # Build nested dict
            current = overrides
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        
        return overrides
    
    def load_config_with_base(self, config_path: Path) -> Dict:
        """Load config and merge with base.yaml if it exists"""
        base_path = config_path.parent / 'base.yaml'
        
        # Load base config if exists
        if base_path.exists():
            with open(base_path) as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Load experiment-specific config
        with open(config_path) as f:
            exp_config = yaml.safe_load(f)
        
        # Deep merge exp_config into base config
        def deep_update(base, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        if exp_config:
            deep_update(config, exp_config)
        
        return config
    
    def apply_overrides(self, config_path: Path, overrides: Dict, output_path: Path):
        """Load config (with base.yaml merge) and apply overrides, save to output path"""
        # Load config with base.yaml inheritance
        config = self.load_config_with_base(config_path)
        
        # Apply overrides if any
        if overrides:
            def deep_update(base, updates):
                for key, value in updates.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        deep_update(base[key], value)
                    else:
                        base[key] = value
            
            deep_update(config, overrides)
        
        # Save merged config to temporary file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return output_path
    
    def run_experiment(self, exp_name: str, gpu_id: int, overrides: Optional[Dict] = None) -> subprocess.Popen:
        """Launch a single experiment on specified GPU"""
        config_path = self.config_dir / f"{exp_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        # Always create temp config with base.yaml merged + overrides applied
        temp_config_path = Path(f'/tmp/exp_suite_{exp_name}_{gpu_id}.yaml')
        config_path = self.apply_overrides(config_path, overrides, temp_config_path)
        
        # Build command
        cmd = [
            'python', 'scripts/run.py',
            '-f', str(config_path),
            '-g', '1',  # Each process gets 1 GPU
            '-w', '1' if self.wandb else '0',
            '--debug', '1' if self.debug else '0',
        ]
        
        if self.save:
            cmd.append('--save')
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        print(f"[Suite] Launching {exp_name} on GPU {gpu_id}")
        print(f"[Suite] Command: {' '.join(cmd)}")
        print(f"[Suite] Config: {config_path}")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.results[exp_name] = {
            'gpu': gpu_id,
            'process': process,
            'status': 'running',
            'start_time': time.time()
        }
        
        return process
    
    def monitor_processes(self):
        """Monitor running processes and report status"""
        running = True
        while running:
            running = False
            for exp_name, info in self.results.items():
                process = info['process']
                if process.poll() is None:
                    running = True
                elif info['status'] == 'running':
                    # Process just finished
                    returncode = process.returncode
                    elapsed = time.time() - info['start_time']
                    info['status'] = 'completed' if returncode == 0 else 'failed'
                    info['elapsed'] = elapsed
                    
                    status_msg = f"✓ {exp_name} completed" if returncode == 0 else f"✗ {exp_name} failed (exit code {returncode})"
                    print(f"\n[Suite] {status_msg} in {elapsed/60:.1f}m on GPU {info['gpu']}")
            
            if running:
                time.sleep(5)  # Check every 5 seconds
    
    def run_suite(self, experiments: List[str], overrides: Optional[Dict] = None):
        """Run a suite of experiments across available GPUs"""
        print(f"\n{'='*60}")
        print(f"Experiment Suite Runner")
        print(f"{'='*60}")
        print(f"Experiments: {', '.join(experiments)}")
        print(f"GPUs: {self.gpus}")
        print(f"WandB: {self.wandb}, Debug: {self.debug}, Save: {self.save}")
        if overrides:
            print(f"Overrides: {overrides}")
        print(f"{'='*60}\n")
        
        # Assign experiments to GPUs (round-robin)
        for i, exp_name in enumerate(experiments):
            gpu_id = self.gpus[i % len(self.gpus)]
            try:
                self.run_experiment(exp_name, gpu_id, overrides)
                # Small delay to avoid race conditions
                time.sleep(2)
            except Exception as e:
                print(f"[Suite] Failed to launch {exp_name}: {e}")
                self.results[exp_name] = {'status': 'launch_failed', 'error': str(e)}
        
        print(f"\n[Suite] All experiments launched. Monitoring...\n")
        
        # Monitor until all complete
        self.monitor_processes()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print experiment summary"""
        print(f"\n{'='*60}")
        print(f"Experiment Suite Summary")
        print(f"{'='*60}")
        
        completed = [name for name, info in self.results.items() if info['status'] == 'completed']
        failed = [name for name, info in self.results.items() if info['status'] == 'failed']
        launch_failed = [name for name, info in self.results.items() if info['status'] == 'launch_failed']
        
        print(f"✓ Completed: {len(completed)}/{len(self.results)}")
        if completed:
            for name in completed:
                elapsed = self.results[name].get('elapsed', 0)
                print(f"  - {name}: {elapsed/60:.1f}m on GPU {self.results[name]['gpu']}")
        
        if failed:
            print(f"\n✗ Failed: {len(failed)}")
            for name in failed:
                elapsed = self.results[name].get('elapsed', 0)
                print(f"  - {name}: {elapsed/60:.1f}m on GPU {self.results[name]['gpu']}")
        
        if launch_failed:
            print(f"\n✗ Launch Failed: {len(launch_failed)}")
            for name in launch_failed:
                print(f"  - {name}: {self.results[name].get('error', 'Unknown error')}")
        
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run experiment suite in parallel across GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Experiment selection
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument('--suite', choices=list(SUITES.keys()),
                          help='Predefined experiment suite')
    exp_group.add_argument('--configs', type=str,
                          help='Comma-separated list of config names (e.g., vit,att,pca)')
    
    # GPU configuration
    parser.add_argument('--gpus', type=str, required=True,
                       help='Comma-separated GPU IDs (e.g., 0,1,2,3)')
    
    # Experiment options
    parser.add_argument('--config-dir', type=str, default='configs/exp',
                       help='Directory containing experiment configs')
    parser.add_argument('--wandb', type=int, default=1, choices=[0, 1],
                       help='Enable WandB logging (default: 1)')
    parser.add_argument('--debug', type=int, default=0, choices=[0, 1],
                       help='Debug mode (default: 0)')
    parser.add_argument('--save', action='store_true',
                       help='Save checkpoints and logs')
    
    # Override parameters
    parser.add_argument('--override', type=str,
                       help='Override config parameters (e.g., "train.ep=500,opt.lr=0.02")')
    
    args = parser.parse_args()
    
    # Parse GPUs
    gpus = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Get experiment list
    if args.suite:
        experiments = SUITES[args.suite]
    else:
        experiments = [x.strip() for x in args.configs.split(',')]
    
    # Parse overrides
    overrides = None
    if args.override:
        runner = ExperimentRunner()
        overrides = runner.parse_overrides(args.override)
    
    # Run suite
    runner = ExperimentRunner(
        config_dir=args.config_dir,
        gpus=gpus,
        wandb=bool(args.wandb),
        debug=bool(args.debug),
        save=args.save
    )
    
    runner.run_suite(experiments, overrides)


if __name__ == '__main__':
    main()
