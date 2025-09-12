#!/usr/bin/env bash
set -Eeuo pipefail

# Load project environment variables if present
if [ -f ./.env ]; then
  # Export variables defined in .env automatically
  set -a
  . ./.env
  set +a
fi

# Determine virtualenv activate script
if [ -z "${VENV_PATH:-}" ]; then
  if [ -f ".venv/bin/activate" ]; then
    VENV_PATH=".venv/bin/activate"
  elif [ -f "venv/bin/activate" ]; then
    VENV_PATH="venv/bin/activate"
  else
    echo "No virtualenv found. Create one with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
  fi
fi

# Activate virtual environment
# shellcheck disable=SC1090
source "$VENV_PATH"

# Ensure PYTHONPATH includes this repo root
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Ensure required data root is set
if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is not set. Define it in .env or your shell profile." >&2
  exit 1
fi

start_time=$(date +%s)
# Enable W&B and saving by default; disable fast_dev_run for real training
python ./scripts/run.py -f "$CONFIG_DIR/vit.yaml" -w 1 --save --debug 0

end_time=$(date +%s)
elapsed_time_1=$((end_time - start_time))
