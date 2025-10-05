#!/usr/bin/env bash

# Launch each ablation mode on its own GPU (up to 8 concurrent jobs).
#
# Environment overrides:
#   CONFIG        - training config passed to scripts/ablation.py (default: $CONFIG_DIR/run.yaml)
#   STATS_ROOT    - directory of fitted preprocessing stats (default: $PCA_DIR)
#   GPU_LIST      - space/comma separated GPU ids (default: 0 1 2 3 4 5 6 7)
#   MODES         - whitespace separated list of modes to run (default: center standardize ... cca)
#   COMMON_FLAGS  - extra flags forwarded to scripts/ablation.py (e.g., "--wandb --save")

set -euo pipefail

CONFIG=${CONFIG:-${CONFIG_DIR:-configs/anyon}/run.yaml}
STATS_ROOT=${STATS_ROOT:-${PCA_DIR:-artifacts/preproc}}

# shellcheck disable=SC2206
GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})

DEFAULT_MODES=(
  center
  standardize
  zca
  zca_lowrank
  project_lowrank
  randrot_white
  randrot
  pca
  pls
  cca
)

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES=(${MODES})
else
  MODES=(${DEFAULT_MODES[@]})
fi

if (( ${#GPU_LIST[@]} == 0 )); then
  echo "[error] GPU_LIST is empty" >&2
  exit 1
fi

echo "Config     : $CONFIG"
echo "Stats root : $STATS_ROOT"
echo "GPUs       : ${GPU_LIST[*]}"
echo "Modes      : ${MODES[*]}"
echo

launch() {
  local gpu_id="$1"
  local mode="$2"
  shift 2
  echo "[launch] GPU=$gpu_id mode=$mode"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    python3 scripts/ablation.py \
      --config "$CONFIG" \
      --stats-root "$STATS_ROOT" \
      --gpu 1 \
      --modes "$mode" \
      ${COMMON_FLAGS:-} \
      "$@" &
}

max_jobs=${#GPU_LIST[@]}
for ((i = 0; i < ${#MODES[@]}; ++i)); do
  gpu_index=$((i % max_jobs))
  gpu_id=${GPU_LIST[$gpu_index]}
  launch "$gpu_id" "${MODES[$i]}"
  if (((i + 1) % max_jobs == 0)); then
    wait
  fi
done

wait
echo "All ablation jobs finished."

