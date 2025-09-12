import os
import sys
import argparse
import torch
import wandb

from src.utils import load_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['WANDB_ENTITY'] = 'viskawei-johns-hopkins-university'
# os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from src.vit import Experiment

ckpt=None
# ckpt = '/home/swei20/SirenSpec/src/artifacts/model-ypx736v3:v0/model.ckpt'

def parse_args():
    parser = argparse.ArgumentParser(description='blindspot experiment')
    parser.add_argument('-f', '--config', type=str, help='config file')
    parser.add_argument('-w', '--wandb', type=int, help='use wandb logging', default=0)
    parser.add_argument('--save', action='store_true', help='only when set, save checkpoints and log to W&B')
    parser.add_argument('-g', '--gpu', type=int, help='gpu number', default=torch.cuda.device_count())
    parser.add_argument('--debug', type=int, help='debug mode', default=0)
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file', default=ckpt)
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    config = load_config(args.config or 'configs/vit.yaml')
    config['train']['gpus'] = args.gpu
    config['train']['debug'] = args.debug
    use_wandb = bool(args.wandb)
    # Respect config 'train.save' and allow CLI to turn it on
    config['train']['save'] = bool(config.get('train', {}).get('save', False) or args.save)
    Experiment(config, use_wandb=use_wandb, sweep=False, ckpt_path=args.ckpt, test_data=False).run()

if __name__ == "__main__":
    args = parse_args()
    main(args)
