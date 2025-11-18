#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal launcher that keeps only four entry points:
#   run   - standard training (scripts/run.py)
#   test  - evaluation only (scripts/test.py)
#   lr    - learning-rate + scheduler sweep (src/opt/parallel_sweep.py)
#   sweep - W&B sweep helper (wraps wandb CLI + scripts/sweep.py)
#
# Examples:
#   ./launch.sh run -c configs/exp/att_clp/baseline.yaml --wandb 1 --save
#   ./launch.sh test --ckpt checkpoints/best.ckpt -c my_config.yaml
#   ./launch.sh lr -c configs/config.yaml -g 0,1,2,3
#   ./launch.sh sweep -c configs/sweep.yaml -e myorg -p vit -g 0,1 --count 5

MODE="run"
WANDB=1
DEBUG=0
SAVE=0
DRY_RUN=0
CUSTOM_CONFIG=""
CKPT_PATH=""
GPU_COUNT=1
GPU_SET=0
GPU_OVERRIDE=""
SWEEP_CONFIG=""
SWEEP_ENTITY="${WANDB_ENTITY:-}"
SWEEP_PROJECT="${WANDB_PROJECT:-vit-test}"
SWEEP_COUNT=""
EXTRA_ARGS=()

if [[ $# -gt 0 ]]; then
  case "$1" in
    run|test|lr|sweep)
      MODE="$1"; shift ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CUSTOM_CONFIG="$2"; shift 2 ;;
    --sweep-config)
      SWEEP_CONFIG="$2"; shift 2 ;;
    -g|--gpu)
      GPU_COUNT="$2"; GPU_SET=1; GPU_OVERRIDE="$2"; shift 2 ;;
    -w|--wandb)
      WANDB="$2"; shift 2 ;;
    -d|--debug)
      DEBUG="$2"; shift 2 ;;
    --save)
      SAVE=1; shift ;;
    --ckpt)
      CKPT_PATH="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -e|--entity)
      SWEEP_ENTITY="$2"; shift 2 ;;
    -p|--project)
      SWEEP_PROJECT="$2"; shift 2 ;;
    --count)
      SWEEP_COUNT="$2"; shift 2 ;;
    --server8)
      MACHINE="server8"; shift ;;
    --local)
      MACHINE="local"; shift ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./launch.sh [run|test|lr|sweep] [options]
  -c, --config PATH        Config file (YAML). Required for sweep.
  -g, --gpu VALUE          run/test: number (or single id on server)
                           lr/sweep: comma-separated GPU ids
  -w, --wandb {0,1}        Enable/disable W&B logging (default: 1)
  -d, --debug INT          Debug flag forwarded to scripts (default: 0)
      --save               Save checkpoints during run
      --ckpt PATH          Checkpoint path for test mode
      --dry-run            Preview lr sweep without launching jobs
  -e, --entity NAME        W&B entity (sweep mode)
  -p, --project NAME       W&B project (sweep mode, default: $SWEEP_PROJECT)
      --count N            Runs per agent for sweep mode
USAGE
      exit 0 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Load environment overrides
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi

ROOT_LOCAL="${ROOT_LOCAL:-$PWD}"
ROOT_SERVER8="/home/swei20/VIT"

if [ -z "${MACHINE:-}" ]; then
  if [ -f "/srv/local/tmp/swei20/miniconda3/etc/profile.d/conda.sh" ]; then
    MACHINE="server8"
  else
    MACHINE="local"
  fi
fi

if [ "$MACHINE" = "server8" ]; then
  # shellcheck disable=SC1091
  source /srv/local/tmp/swei20/miniconda3/etc/profile.d/conda.sh
  conda activate viska-torch-3
  export ROOT="$ROOT_SERVER8"
  if [ "$GPU_SET" = "1" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_OVERRIDE"
    GPU_COUNT=1
  else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  fi
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
else
  export ROOT="${ROOT:-$ROOT_LOCAL}"
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

DEFAULT_CONFIG="$ROOT/configs/exp/att_clp/baseline.yaml"
LR_DEFAULT_CONFIG="$ROOT/configs/config.yaml"
DEFAULT_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
GPU_LIST_OVERRIDE="${GPU_OVERRIDE:-${GPUS:-$DEFAULT_GPU_LIST}}"

if [ "$MODE" = "lr" ]; then
  CONFIG_FILE="${CUSTOM_CONFIG:-$LR_DEFAULT_CONFIG}"
  GPU_LIST="$GPU_LIST_OVERRIDE"
  if [ -z "$GPU_LIST" ]; then
    GPU_LIST="0"
  fi
  echo "[launch] MODE=lr CONFIG=$CONFIG_FILE GPUS=$GPU_LIST DRY_RUN=$DRY_RUN"
  mkdir -p "$ROOT/opt_runs/sweep"
  if [ "$DRY_RUN" = "1" ]; then
    python "$ROOT/src/opt/parallel_sweep.py" \
      "$CONFIG_FILE" \
      --lr 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 \
      --schedulers plateau cosine none \
      --gpus "$GPU_LIST" \
      --dry-run
    exit 0
  fi

  python "$ROOT/src/opt/parallel_sweep.py" \
    "$CONFIG_FILE" \
    --lr 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 \
    --schedulers none \
    --gpus "$GPU_LIST" \
    --results-dir "$ROOT/opt_runs/sweep"

  LATEST_DIR=$(ls -td "$ROOT/opt_runs/sweep/parallel_sweep_"* 2>/dev/null | head -1)
  if [ -z "$LATEST_DIR" ]; then
    echo "Error: sweep results missing" >&2
    exit 1
  fi
  BEST_LR=$(python -c "import yaml,sys; data=yaml.safe_load(open('${LATEST_DIR}/summary.yaml')); print(data['best']['config']['lr'])" 2>/dev/null || true)
  if [ -z "$BEST_LR" ]; then
    echo "Error: unable to read best LR" >&2
    exit 1
  fi
  echo "[launch] Best LR from sweep: $BEST_LR"

  python "$ROOT/src/opt/parallel_sweep.py" \
    "$CONFIG_FILE" \
    --lr "$BEST_LR" \
    --schedulers plateau cosine none \
    --plateau-factor 0.8 \
    --plateau-patience 10 \
    --gpus "$GPU_LIST" \
    --results-dir "$ROOT/opt_runs/sweep"

  LATEST_DIR=$(ls -td "$ROOT/opt_runs/sweep/parallel_sweep_"* 2>/dev/null | head -1)
  python - <<PY
import yaml
from pathlib import Path
summary = Path('${LATEST_DIR}') / 'summary.yaml'
if summary.exists():
    data = yaml.safe_load(summary.read_text())
    best = data.get('best', {})
    config = best.get('config', {})
    print('──────── Sweep Summary ────────')
    print(f"LR        : {config.get('lr')}")
    print(f"Scheduler : {config.get('scheduler', 'none')}")
    if config.get('scheduler') == 'plateau':
        if 'factor' in config: print(f"Factor    : {config['factor']}")
        if 'patience' in config: print(f"Patience  : {config['patience']}")
    metric = best.get('metric')
    if metric is not None:
        print(f"val_mae   : {metric:.6f}")
    print(f"Summary   : {summary}")
    bc = Path('${LATEST_DIR}') / 'best_config.yaml'
    if bc.exists():
        print(f"Best cfg  : {bc}")
    print('────────────────────────────────')
else:
    print('summary.yaml missing in latest sweep directory')
PY
  exit 0
fi

if [ "$MODE" = "sweep" ]; then
  SWEEP_FILE="${SWEEP_CONFIG:-$CUSTOM_CONFIG}"
  if [ -z "$SWEEP_FILE" ]; then
    echo "Please provide -c/--config with a sweep YAML." >&2
    exit 1
  fi
  if [ ! -f "$SWEEP_FILE" ]; then
    echo "Sweep config not found: $SWEEP_FILE" >&2
    exit 1
  fi
  if ! command -v wandb >/dev/null 2>&1; then
    echo "wandb CLI not found. Activate your environment and install wandb." >&2
    exit 1
  fi
  if [ -z "$SWEEP_ENTITY" ]; then
    echo "Set --entity or WANDB_ENTITY for sweep mode." >&2
    exit 1
  fi

  GPU_LIST="$GPU_LIST_OVERRIDE"
  if [ -z "$GPU_LIST" ]; then
    GPU_LIST="0"
  fi

  echo "[launch] Creating sweep: config=$SWEEP_FILE entity=$SWEEP_ENTITY project=$SWEEP_PROJECT"
  CREATE_OUT=$(wandb sweep -e "$SWEEP_ENTITY" -p "$SWEEP_PROJECT" "$SWEEP_FILE" 2>&1 | tee /dev/stderr)
  SWEEP_ID=$(echo "$CREATE_OUT" | grep -oE 'Creating sweep with ID: [A-Za-z0-9]+' | awk '{print $NF}' | tail -n1)
  if [ -z "$SWEEP_ID" ]; then
    echo "Failed to parse sweep ID." >&2
    exit 1
  fi
  FULL_ID="$SWEEP_ENTITY/$SWEEP_PROJECT/$SWEEP_ID"
  echo "[launch] Sweep ID: $FULL_ID"

  IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
  if [ "${#GPU_ARR[@]}" -eq 0 ]; then
    echo "No GPUs specified for sweep agents." >&2
    exit 1
  fi

  echo "[launch] Starting ${#GPU_ARR[@]} agent(s) on GPUs: $GPU_LIST"
  PIDS=()
  for GPU in "${GPU_ARR[@]}"; do
    GPU=$(echo "$GPU" | xargs)
    if [ -z "$GPU" ]; then
      continue
    fi
    if [ -n "$SWEEP_COUNT" ]; then
      CUDA_VISIBLE_DEVICES="$GPU" wandb agent --count "$SWEEP_COUNT" "$FULL_ID" &
    else
      CUDA_VISIBLE_DEVICES="$GPU" wandb agent "$FULL_ID" &
    fi
    PIDS+=($!)
    echo "  → GPU $GPU PID ${PIDS[-1]}"
  done

  trap 'echo; echo "Stopping sweep agents..."; kill ${PIDS[@]} 2>/dev/null || true; exit 0' INT TERM
  wait
  exit 0
fi

CONFIG_FILE="${CUSTOM_CONFIG:-$DEFAULT_CONFIG}"
PYTHON_BIN=python

if [ "$MODE" = "test" ]; then
  PY="./scripts/test.py"
  echo "[launch] MODE=test CONFIG=$CONFIG_FILE MACHINE=$MACHINE WANDB=$WANDB DEBUG=$DEBUG"
  CMD=("$PYTHON_BIN" "$PY" -f "$CONFIG_FILE" -w "$WANDB" --debug "$DEBUG")
  if [ -n "$CKPT_PATH" ]; then
    CMD+=(--ckpt "$CKPT_PATH")
  fi
  if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi
  printf '[launch] CMD=%s ' "${CMD[@]}"; echo
  "${CMD[@]}"
  exit 0
fi

# MODE=run
PY="./scripts/run.py"
echo "[launch] MODE=run CONFIG=$CONFIG_FILE MACHINE=$MACHINE WANDB=$WANDB DEBUG=$DEBUG SAVE=$SAVE"
CMD=("$PYTHON_BIN" "$PY" -f "$CONFIG_FILE" -w "$WANDB" --debug "$DEBUG")
if [ "$SAVE" -eq 1 ]; then
  CMD+=(--save)
fi
if [ -n "$CKPT_PATH" ]; then
  CMD+=(--ckpt "$CKPT_PATH")
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi
printf '[launch] CMD=%s ' "${CMD[@]}"; echo
"${CMD[@]}"
