#!/bin/bash
# Pre-flight check for ZCA bias comparison experiment

set -e

echo "=========================================="
echo "ZCA Bias Comparison - Pre-flight Check"
echo "=========================================="
echo ""

ERRORS=0

# Check 1: Current directory
echo "1. Checking current directory..."
if [[ $(basename $(pwd)) != "VIT" ]]; then
    echo "   ✗ Not in VIT directory"
    echo "   → Run: cd /home/swei20/VIT"
    ERRORS=$((ERRORS + 1))
else
    echo "   ✓ In VIT directory"
fi

# Check 2: Environment variables
echo ""
echo "2. Checking environment variables..."
if [ -z "$PCA_DIR" ]; then
    echo "   ✗ PCA_DIR not set"
    echo "   → Run: source init.sh"
    ERRORS=$((ERRORS + 1))
else
    echo "   ✓ PCA_DIR = $PCA_DIR"
fi

if [ -z "$TRAIN_DIR" ]; then
    echo "   ✗ TRAIN_DIR not set"
    ERRORS=$((ERRORS + 1))
else
    echo "   ✓ TRAIN_DIR = $TRAIN_DIR"
fi

if [ -z "$VAL_DIR" ]; then
    echo "   ✗ VAL_DIR not set"
    ERRORS=$((ERRORS + 1))
else
    echo "   ✓ VAL_DIR = $VAL_DIR"
fi

if [ -z "$TEST_DIR" ]; then
    echo "   ✗ TEST_DIR not set"
    ERRORS=$((ERRORS + 1))
else
    echo "   ✓ TEST_DIR = $TEST_DIR"
fi

# Check 3: Required files
echo ""
echo "3. Checking required files..."

if [ ! -f "exp/zca_bias/zca_bias_comparison.py" ]; then
    echo "   ✗ Missing: exp/zca_bias/zca_bias_comparison.py"
    ERRORS=$((ERRORS + 1))
else
    echo "   ✓ Found: exp/zca_bias/zca_bias_comparison.py"
fi

if [ ! -f "exp/zca_bias/zca_bias_comparison.yaml" ]; then
    echo "   ✗ Missing: exp/zca_bias/zca_bias_comparison.yaml"
    ERRORS=$((ERRORS + 1))
else
    echo "   ✓ Found: exp/zca_bias/zca_bias_comparison.yaml"
fi

if [ ! -f "exp/zca_bias/run_bias_comparison.sh" ]; then
    echo "   ✗ Missing: exp/zca_bias/run_bias_comparison.sh"
    ERRORS=$((ERRORS + 1))
else
    echo "   ✓ Found: exp/zca_bias/run_bias_comparison.sh"
fi

# Check 4: Covariance file
echo ""
echo "4. Checking covariance file..."
if [ -n "$PCA_DIR" ]; then
    if [ ! -f "$PCA_DIR/cov.pt" ]; then
        echo "   ✗ Missing: $PCA_DIR/cov.pt"
        echo "   → Generate with: python scripts/prepro/calculate_cov.py"
        ERRORS=$((ERRORS + 1))
    else
        echo "   ✓ Found: $PCA_DIR/cov.pt"
        
        # Check if cov.pt contains 'mean' key
        python -c "
import torch
cov = torch.load('$PCA_DIR/cov.pt')
if 'mean' in cov:
    print('   ✓ cov.pt contains mean field')
else:
    print('   ✗ cov.pt missing mean field')
    print('   → Regenerate with: python scripts/prepro/calculate_cov.py')
    exit(1)
" || ERRORS=$((ERRORS + 1))
    fi
fi

# Check 5: Data files
echo ""
echo "5. Checking data files..."
if [ -n "$TRAIN_DIR" ]; then
    if [ ! -f "$TRAIN_DIR/dataset.h5" ]; then
        echo "   ✗ Missing: $TRAIN_DIR/dataset.h5"
        ERRORS=$((ERRORS + 1))
    else
        echo "   ✓ Found: $TRAIN_DIR/dataset.h5"
    fi
fi

if [ -n "$VAL_DIR" ]; then
    if [ ! -f "$VAL_DIR/dataset.h5" ]; then
        echo "   ✗ Missing: $VAL_DIR/dataset.h5"
        ERRORS=$((ERRORS + 1))
    else
        echo "   ✓ Found: $VAL_DIR/dataset.h5"
    fi
fi

if [ -n "$TEST_DIR" ]; then
    if [ ! -f "$TEST_DIR/dataset.h5" ]; then
        echo "   ✗ Missing: $TEST_DIR/dataset.h5"
        ERRORS=$((ERRORS + 1))
    else
        echo "   ✓ Found: $TEST_DIR/dataset.h5"
    fi
fi

# Check 6: Python imports
echo ""
echo "6. Checking Python imports..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')

try:
    from src.utils import load_config
    print('   ✓ Can import src.utils')
except Exception as e:
    print(f'   ✗ Cannot import src.utils: {e}')
    sys.exit(1)

try:
    from src.models.builder import get_model
    print('   ✓ Can import src.models.builder')
except Exception as e:
    print(f'   ✗ Cannot import src.models.builder: {e}')
    sys.exit(1)

try:
    from src.vit import ViTDataModule
    print('   ✓ Can import src.vit.ViTDataModule')
except Exception as e:
    print(f'   ✗ Cannot import src.vit.ViTDataModule: {e}')
    sys.exit(1)

try:
    import pytorch_lightning as pl
    print('   ✓ Can import pytorch_lightning')
except Exception as e:
    print(f'   ✗ Cannot import pytorch_lightning: {e}')
    sys.exit(1)
" || ERRORS=$((ERRORS + 1))

# Check 7: GPU availability
echo ""
echo "7. Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'   ✓ CUDA available: {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        print(f'   ✓ GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('   ✗ CUDA not available (will use CPU - very slow)')
"

# Summary
echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed!"
    echo "=========================================="
    echo ""
    echo "Ready to run experiment:"
    echo "  bash exp/zca_bias/run_bias_comparison.sh"
    echo ""
    exit 0
else
    echo "✗ Found $ERRORS error(s)"
    echo "=========================================="
    echo ""
    echo "Please fix the errors above before running the experiment."
    echo ""
    exit 1
fi
