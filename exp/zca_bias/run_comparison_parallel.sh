#!/bin/bash
# Run ZCA bias comparison experiments in parallel on 2 GPUs

# Default values
CONFIG="exp/zca_bias/zca_bias_comparison.yaml"
OUTPUT_DIR="results_zca_bias_comparison_$(date +%Y%m%d_%H%M%S)"
GPU_IDS="0,1"
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH       Config file (default: exp/zca_bias/zca_bias_comparison.yaml)"
            echo "  --output_dir PATH   Output directory (default: results_zca_bias_comparison_TIMESTAMP)"
            echo "  -g, --gpus IDS      Comma-separated GPU IDs (default: 0,1)"
            echo "  --seed NUM          Random seed (default: 42)"
            echo "  --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -g 4,5 --seed 123"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "ZCA Bias Comparison - PARALLEL MODE"
echo "=================================="
echo "Config:      $CONFIG"
echo "Output dir:  $OUTPUT_DIR"
echo "GPU IDs:     $GPU_IDS"
echo "Seed:        $SEED"
echo "=================================="
echo ""

# Run the comparison script in parallel mode
python exp/zca_bias/zca_bias_comparison.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --gpu_ids "$GPU_IDS" \
    --seed "$SEED" \
    --parallel \
    --use_wandb \
    2>&1 | tee "$OUTPUT_DIR/experiment.log"

echo ""
echo "=================================="
echo "Experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================="
