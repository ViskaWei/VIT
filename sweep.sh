#!/usr/bin/env bash
set -euo pipefail

# Simple helper to create a W&B sweep and launch multi-GPU agents.

# Load .env if present so WANDB_* and others are available
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi

# Defaults (can be overridden by flags or env)
ENTITY_DEFAULT="${WANDB_ENTITY:-}"
PROJECT_DEFAULT="${WANDB_PROJECT:-vit-test}"
SWEEP_YAML_DEFAULT="configs/sweeps/vit_r_sweep.yaml"
GPUS_DEFAULT="${GPUS:-0}"  # comma-separated list, e.g. "0,1,2,3,4,5,6,7"
COUNT_DEFAULT=""           # number of runs per agent; empty = unlimited

usage() {
  echo "Usage: $0 [-e entity] [-p project] [-c sweep_yaml] [-g gpu_list] [--count N]" >&2
  echo "  -e, --entity       W&B entity (org/user). Defaults to WANDB_ENTITY or required."
  echo "  -p, --project      W&B project. Defaults to '${PROJECT_DEFAULT}'."
  echo "  -c, --config       Sweep YAML path. Defaults to '${SWEEP_YAML_DEFAULT}'."
  echo "  -g, --gpus         Comma-separated GPU IDs. Defaults to '${GPUS_DEFAULT}'."
  echo "      --count        Optional: number of runs per agent."
  echo "Examples:"
  echo "  $0 -e viskawei-johns-hopkins-university -p vit-test -g 0,1,2,3"
  echo "  GPUS=0,1,2,3 WANDB_ENTITY=myorg $0"
}

ENTITY="$ENTITY_DEFAULT"
PROJECT="$PROJECT_DEFAULT"
SWEEP_YAML="$SWEEP_YAML_DEFAULT"
GPUS="$GPUS_DEFAULT"
COUNT="$COUNT_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--entity)   ENTITY="$2"; shift 2 ;;
    -p|--project)  PROJECT="$2"; shift 2 ;;
    -c|--config)   SWEEP_YAML="$2"; shift 2 ;;
    -g|--gpus)     GPUS="$2"; shift 2 ;;
    --count)       COUNT="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v wandb >/dev/null 2>&1; then
  echo "wandb CLI not found. Activate your venv and run 'pip install wandb'." >&2
  exit 1
fi

if [[ -z "${ENTITY}" ]]; then
  echo "Missing --entity (or set WANDB_ENTITY)." >&2
  usage
  exit 1
fi

echo "Creating sweep: config=${SWEEP_YAML}, entity=${ENTITY}, project=${PROJECT}"
CREATE_OUT=$(wandb sweep -e "${ENTITY}" -p "${PROJECT}" "${SWEEP_YAML}" 2>&1 | tee /dev/stderr)

# Extract short sweep id using grep (more portable than sed)
SWEEP_ID=$(echo "${CREATE_OUT}" | grep -oE 'Creating sweep with ID: [a-zA-Z0-9]+' | grep -oE '[a-zA-Z0-9]+$' | tail -n1)
if [[ -z "${SWEEP_ID}" ]]; then
  echo "Failed to parse sweep ID from wandb output." >&2
  exit 1
fi
FULL_ID="${ENTITY}/${PROJECT}/${SWEEP_ID}"
echo "Sweep ID: ${FULL_ID}"

# Launch one agent per GPU
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
echo "Launching ${#GPU_ARR[@]} agent(s) on GPUs: ${GPUS}"

PIDS=()
for GPU in "${GPU_ARR[@]}"; do
  if [[ -n "${COUNT}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" wandb agent --count "${COUNT}" "${FULL_ID}" &
  else
    CUDA_VISIBLE_DEVICES="${GPU}" wandb agent "${FULL_ID}" &
  fi
  PIDS+=($!)
  echo "Started agent on GPU ${GPU} with PID $!"
done

echo "All agents started. Waiting... (Ctrl-C to stop)"
trap 'echo "Stopping agents..."; kill ${PIDS[@]} 2>/dev/null || true; exit 0' INT TERM
wait
