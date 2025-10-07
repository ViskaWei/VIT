#!/bin/bash
# Evaluate W&B model 3vrlvqj0:v2
set -e

VENV_PYTHON="${VIRTUAL_ENV:-/Users/viskawei/Desktop/VIT/.venv}/bin/python"

"$VENV_PYTHON" scripts/eval_model.py \
    --config configs/vit.yaml \
    --wandb-id 3vrlvqj0 \
    --version v2 \
    --out results/eval_3vrlvqj0_v2 \
    --skip-eval \
    --cmap magma \
    --dpi 200
