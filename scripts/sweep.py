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

    # Keep dataloader workers modest under multi-agent sweeps
    train = config.setdefault('train', {})
    if 'num_workers' not in train:
        # Map legacy key 'workers' if present; otherwise default low for stability
        train['num_workers'] = int(train.get('workers', 2))
    # Cap workers to avoid oversubscription when many agents run in parallel
    try:
        cpu_cnt = os.cpu_count() or 8
        # Reserve CPUs roughly evenly if multiple GPUs/agents; conservative cap
        train['num_workers'] = max(0, min(int(train['num_workers']), max(1, cpu_cnt // 8)))
    except Exception:
        pass

    # Run one training with W&B logging in sweep mode
    exp = Experiment(config, use_wandb=True, sweep=True, num_gpus=1, test_data=False)
    # Ensure the W&B run has a readable, deterministic name
    try:
        run = wandb.run
        if run is None:
            # Fallback: try Lightning's WandbLogger experiment handle
            trainer = getattr(getattr(exp, 't', None), 'trainer', None)
            if trainer and getattr(trainer, 'logger', None) and hasattr(trainer.logger, 'experiment'):
                run = trainer.logger.experiment
        if run is not None and hasattr(exp, 'lightning_module') and hasattr(exp.lightning_module, 'model'):
            run.name = exp.lightning_module.model.name
            # Persist the change to the UI immediately
            try:
                run.save()
            except Exception:
                pass
    except Exception:
        pass
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
