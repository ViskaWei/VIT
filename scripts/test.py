import argparse
import os
import sys

import lightning as L
import torch

from src.utils import load_config
from src.vit import Experiment

os.environ.setdefault('WANDB_ENTITY', 'viskawei-johns-hopkins-university')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def parse_args():
    parser = argparse.ArgumentParser(description='ViT evaluation runner')
    parser.add_argument('-f', '--config', type=str, default='configs/exp/att_clp/baseline.yaml', help='config file')
    parser.add_argument('-w', '--wandb', type=int, default=0, help='use wandb: 0=off, 1=on')
    parser.add_argument('-g', '--gpu', type=int, default=None, help='number of gpus (optional)')
    parser.add_argument('--debug', type=int, default=0, help='debug flag propagated to config')
    parser.add_argument('--ckpt', type=str, default='best', help="checkpoint path or 'best'/'last'")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = load_config(args.config)

    gpu_count = args.gpu
    if gpu_count is None:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    train_cfg = config.setdefault('train', {})
    train_cfg['gpus'] = gpu_count
    train_cfg['debug'] = args.debug
    # Never save during pure evaluation
    train_cfg['save'] = False

    use_wandb = bool(args.wandb)
    exp = Experiment(config, use_wandb=use_wandb, sweep=False, num_gpus=gpu_count or None)

    ckpt_path = args.ckpt if args.ckpt not in (None, '', 'none', 'None') else None
    print(f"[test] config={args.config} wandb={use_wandb} ckpt={ckpt_path or 'current'}")
    exp.t.test_trainer.test(exp.lightning_module, datamodule=exp.data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    sys.exit(main(parse_args()))
