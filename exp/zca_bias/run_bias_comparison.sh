#!/bin/bash
# Quick start script for bias comparison experiment

set -e

echo "=========================================="
echo "Bias Comparison Experiment"
echo "=========================================="
echo ""

# Check environment
if [ -z "$PCA_DIR" ]; then
    echo "ERROR: PCA_DIR not set. Please run: source init.sh"
    exit 1
fi

# Configuration
CONFIG=${1:-"zca_bias_comparison.yaml"}
OUTPUT_DIR=${2:-"./results_zca_bias_comparison_$(date +%Y%m%d_%H%M%S)"}
USE_WANDB=${3:-"--use_wandb"}  # Default: enable wandb

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Wandb:       $USE_WANDB"
echo ""

# Ask for confirmation
read -p "Start experiment? This may take several hours. (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run experiment
echo ""
echo "Starting experiment..."
echo ""

# Create output directory first
mkdir -p "$OUTPUT_DIR"
echo "Created output directory: $OUTPUT_DIR"
echo "Logs will be saved to: ${OUTPUT_DIR}/experiment.log"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python "${SCRIPT_DIR}/zca_bias_comparison.py" \
    --config "${SCRIPT_DIR}/${CONFIG}" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    $USE_WANDB \
    2>&1 | tee "${OUTPUT_DIR}/experiment.log"

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
echo ""
echo "View results:"
echo "  cat ${OUTPUT_DIR}/comparison_report.md"
echo ""
