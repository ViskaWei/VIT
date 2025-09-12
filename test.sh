#!/usr/bin/env bash
set -Eeuo pipefail
# Load .env if present
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi

source "$VENV_PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# Ensure required data root is set
if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is not set. Define it in .env or your shell profile." >&2
  exit 1
fi

start_time=$(date +%s)
# Enable W&B and saving by default; disable fast_dev_run for real training
python ./scripts/test.py -f "$CONFIG_DIR/test.yaml" -w 1 --save --debug 0

end_time=$(date +%s)
elapsed_time_1=$((end_time - start_time))

# exec "$SHELL" -i
