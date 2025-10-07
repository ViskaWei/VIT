#!/bin/bash
# ZCA Report Generation Script

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (two levels up from scripts/prepro/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Use the virtual environment Python
PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Run the Python script
$PYTHON "$PROJECT_ROOT/scripts/prepro/zca_report.py" \
  --cov_path "$PROJECT_ROOT/data/cov.pt" \
  --out_dir ./zca_report \
  --rank auto \
  --energy_target 0.99 \
  --r_cap 16 \
  --gamma 0.10 \
  --eps_rel 1e-6 \
  --viz_dim 256 \
  --tail_stat median \
  --floor_rel 1e-3