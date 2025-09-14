#!/usr/bin/env bash
set -Eeuo pipefail

# Simple unified launcher for local vs 8-GPU server.
# Usage:
#   ./launch.sh [run|test] [-g N] [--save] [--wandb 0|1] [--debug 0|1]
# Defaults: mode=run, -g 1, --wandb 1 (enabled), --debug 0

MODE="run"
GPU_COUNT=1
GPU_SET=0
WANDB=1
DEBUG=0
EXTRA_ARGS=()

if [[ $# -gt 0 ]]; then
  case "$1" in
    run|test)
      MODE="$1"; shift ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)
      GPU_COUNT="$2"; GPU_SET=1; shift 2 ;;
    --save)
      EXTRA_ARGS+=("--save"); shift ;;
    -w|--wandb)
      WANDB="$2"; shift 2 ;;
    -d|--debug)
      DEBUG="$2"; shift 2 ;;
    --server8)
      MACHINE="server8"; shift ;;
    --local)
      MACHINE="local"; shift ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Load .env if present (local defaults)
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi

ROOT_LOCAL="/Users/viskawei/Desktop/VIT"
ROOT_SERVER8="/home/swei20/VIT"

# Auto-detect machine when not explicitly set
if [ -z "${MACHINE:-}" ]; then
  if [ -f "/srv/local/tmp/swei20/miniconda3/etc/profile.d/conda.sh" ]; then
    MACHINE="server8"
  else
    MACHINE="local"
  fi
fi

if [ "$MACHINE" = "server8" ]; then
  # 8-GPU server: use system Conda env and expose all 8 GPUs
  # shellcheck disable=SC1091
  source /srv/local/tmp/swei20/miniconda3/etc/profile.d/conda.sh
  conda activate viska-torch-3
  export ROOT="$ROOT_SERVER8"
  # If user explicitly passed -g on server, treat it as GPU ID selector
  if [ "$GPU_SET" = "1" ]; then
    # Only set when not already constrained by caller
    if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
      export CUDA_VISIBLE_DEVICES="$GPU_COUNT"
    fi
    GPU_COUNT=1
  else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  fi
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
else
  # Local (or other) machine: use project venv if available
  export ROOT="${ROOT:-$ROOT_LOCAL}"
  # Determine venv activate path if not provided
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
  # shellcheck disable=SC1090
  source "$VENV_PATH"
  export PYTHONPATH="$PWD:${PYTHONPATH:-}"
fi

# Ensure data root is available if required by configs
if [ -z "${DATA_ROOT:-}" ]; then
  echo "Warning: DATA_ROOT is not set; configs may reference it." >&2
fi

CONFIG_DIR_DEFAULT="${CONFIG_DIR:-$ROOT/configs/anyon}"

if [ "$MODE" = "test" ]; then
  CONFIG_FILE="$CONFIG_DIR_DEFAULT/test.yaml"
  PY="./scripts/test.py"
  WANDB=0
  DEBUG=1
else
  CONFIG_FILE="$CONFIG_DIR_DEFAULT/run.yaml"
  PY="./scripts/run.py"
fi

echo "[launch] MACHINE=$MACHINE MODE=$MODE ROOT=$ROOT GPU_COUNT=$GPU_COUNT WANDB=$WANDB DEBUG=$DEBUG CVD=${CUDA_VISIBLE_DEVICES:-unset}"

# Bash -u can complain on empty arrays; only expand when non-empty
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  python "$PY" -f "$CONFIG_FILE" -w "$WANDB" -g "$GPU_COUNT" --debug "$DEBUG" "${EXTRA_ARGS[@]}"
else
  python "$PY" -f "$CONFIG_FILE" -w "$WANDB" -g "$GPU_COUNT" --debug "$DEBUG"
fi
