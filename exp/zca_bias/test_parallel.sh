#!/bin/bash
# Quick test of parallel execution with minimal epochs

echo "==================================="
echo "Testing Parallel Execution"
echo "==================================="
echo ""
echo "This will run a quick test with 3 epochs on GPUs 4,5"
echo ""

# Create a test output directory
OUTPUT_DIR="results_parallel_test_$(date +%Y%m%d_%H%M%S)"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the test
python3 exp/zca_bias/zca_bias_comparison.py \
    --config exp/zca_bias/zca_bias_comparison.yaml \
    --output_dir "$OUTPUT_DIR" \
    --parallel \
    --gpu_ids 4,5 \
    --seed 42 \
    --use_wandb

EXIT_CODE=$?

echo ""
echo "==================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully!"
else
    echo "✗ Test failed with exit code: $EXIT_CODE"
fi
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check logs:"
echo "  - $OUTPUT_DIR/with_bias/subprocess.log"
echo "  - $OUTPUT_DIR/no_bias/subprocess.log"
echo ""
echo "Check WandB for two separate runs:"
echo "  - ZCA_fzperm_ViT_..._WITH_BIAS"
echo "  - ZCA_fzperm_ViT_..._nobias_NO_BIAS"
echo "==================================="

exit $EXIT_CODE
