import os
import sys
import argparse
import torch

from src.utils import load_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['WANDB_ENTITY'] = 'viskawei-johns-hopkins-university'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from src.vit import Experiment

def parse_args():
    parser = argparse.ArgumentParser(description='ViT experiment runner')
    parser.add_argument('-f', '--config', type=str, help='config file', default='configs/vit.yaml')
    parser.add_argument('-w', '--wandb', type=int, help='use wandb: 0=off, 1=on', default=0)
    parser.add_argument('--save', action='store_true', help='save checkpoints (local if -w 0, wandb if -w 1)')
    parser.add_argument('-g', '--gpu', type=int, help='gpu number', default=None)
    parser.add_argument('--debug', type=int, help='debug mode', default=0)
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file', default=None)
    parser.add_argument('--seed', type=int, help='random seed for reproducibility', default=42)
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    # Set random seed before anything else
    import lightning as L
    L.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    config = load_config(args.config)
    
    # GPU setup
    if args.gpu is None:
        args.gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    config['train']['gpus'] = args.gpu
    config['train']['debug'] = args.debug
    
    # Saving logic: only when --save is explicitly set
    config['train']['save'] = args.save
    
    # Wandb usage
    use_wandb = bool(args.wandb)
    
    print(f"[Setup] Random seed: {args.seed}")
    print(f"[Setup] Deterministic mode: ON")
    
    Experiment(config, use_wandb=use_wandb, sweep=False, ckpt_path=args.ckpt, test_data=False).run()

if __name__ == "__main__":
    args = parse_args()
    main(args)
