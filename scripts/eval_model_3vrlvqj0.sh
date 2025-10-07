#!/bin/bash
# Quick evaluation script for W&B model 3vrlvqj0:v2

set -e

WANDB_ID="3vrlvqj0"
VERSION="v2"
CONFIG="configs/vit.yaml"
OUTPUT_DIR="results/eval_${WANDB_ID}_${VERSION}"

echo "Evaluating W&B model: ${WANDB_ID}:${VERSION}"
echo "Output directory: ${OUTPUT_DIR}"

# Use the virtual environment python
PYTHON="${VIRTUAL_ENV:-/Users/viskawei/Desktop/VIT/.venv}/bin/python"

"$PYTHON" scripts/eval_and_plot_weights.py \
    --config "$CONFIG" \
    --wandb-id "$WANDB_ID" \
    --version "$VERSION" \
    --out "$OUTPUT_DIR" \
    --skip-eval \
    --cmap magma \
    --dpi 200

echo ""
echo "✅ Done! Results saved to: ${OUTPUT_DIR}"
echo "  - Weights: ${OUTPUT_DIR}/weights/"
echo "  - Plots: ${OUTPUT_DIR}/plots/"
echo "  - Statistics: ${OUTPUT_DIR}/weight_statistics.json"
