import os
import argparse
import wandb
from src.utils import load_config
from src.vit import Experiment

os.environ['WANDB_ENTITY'] = 'viskawei-johns-hopkins-university'
os.environ['VIT_PROJECT'] = 'vit-test'

def train_fn(args=None):
    # Load base config (prefer wandb.config override, fallback to env)
    arg_cfg = getattr(args, 'vit_config', None) if args is not None else None
    cfg_path = (
        wandb.config.get('vit_config', None)
        or arg_cfg
        or os.environ.get('VIT_CONFIG', 'configs/vit.yaml')
    )
    config = load_config(cfg_path)

    # Ensure warmup section exists
    if 'warmup' not in config:
        config['warmup'] = {}

    # Pick r from sweep param or fallback to base config or 32
    r_arg = getattr(args, 'r', None) if args is not None else None
    r_cfg = wandb.config.get('r', None)
    if r_cfg is None:
        r_cfg = r_arg if r_arg is not None else config.get('warmup', {}).get('r', 32)
    config['warmup']['r'] = int(r_cfg)

    # Run one training with W&B logging in sweep mode
    exp = Experiment(config, use_wandb=True, sweep=True, num_gpus=1, test_data=False)
    exp.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, default=None, help='Run as W&B agent for an existing sweep ID (accepts short ID or entity/project/ID)')
    parser.add_argument('--entity', type=str, default=os.environ.get('WANDB_ENTITY'), help='W&B entity (org/user)')
    parser.add_argument('--project', type=str, default=os.environ.get('WANDB_PROJECT', 'vit-test'), help='W&B project name')
    # Accept sweep-injected overrides to avoid argparse errors
    parser.add_argument('--r', type=int, default=None)
    parser.add_argument('--vit_config', type=str, default=None)
    args, _unknown = parser.parse_known_args()

    if args.sweep_id:
        # Allow short ID by supplying entity/project explicitly
        if '/' in args.sweep_id:
            wandb.agent(args.sweep_id, function=lambda: train_fn(args))
        else:
            if not args.entity or not args.project:
                raise SystemExit("Please provide --entity and --project when using a short sweep_id.")
            wandb.agent(args.sweep_id, function=lambda: train_fn(args), entity=args.entity, project=args.project)
    else:
        # Direct run under a sweep-like context (set default config)
        wandb.init(project=args.project or 'vit-test', config={'r': args.r or 32, 'vit_config': args.vit_config or os.environ.get('VIT_CONFIG', 'configs/vit.yaml')})
        train_fn(args)
