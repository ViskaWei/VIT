#!/usr/bin/env bash

# Launch the ZCA ablation trio (zca, zca_lowrank, randrot) using up to 8 GPUs.
# Each mode runs as its own `scripts/ablation.py` process pinned to an
# individual GPU so the jobs execute in parallel.

set -euo pipefail

CONFIG=${CONFIG:-${CONFIG_DIR:-configs/volta}/run.yaml}
STATS_ROOT=${STATS_ROOT:-${PCA_DIR:-artifacts/preproc}}

# Default to all 8 GPUs; override by exporting GPU_LIST="0 2 4" etc.
# shellcheck disable=SC2206
GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})

MODES=(zca zca_lowrank randrot)

COMMON_FLAGS_DEFAULT=(--wandb --save)
if [[ -n "${COMMON_FLAGS:-}" ]]; then
  # Tokenise COMMON_FLAGS respecting whitespace
  read -r -a COMMON_FLAGS_DEFAULT <<< "${COMMON_FLAGS}"
fi

if (( ${#GPU_LIST[@]} == 0 )); then
  echo "[error] GPU_LIST is empty" >&2
  exit 1
fi

echo "Config     : $CONFIG"
echo "Stats root : $STATS_ROOT"
echo "GPUs       : ${GPU_LIST[*]}"
echo "Modes      : ${MODES[*]}"
echo "Flags      : ${COMMON_FLAGS_DEFAULT[*]}"
echo

launch() {
  local gpu_id="$1"
  local mode="$2"
  echo "[launch] GPU=$gpu_id mode=$mode"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    python3 scripts/ablation.py \
      --config "$CONFIG" \
      --stats-root "$STATS_ROOT" \
      --gpu 1 \
      --modes "$mode" \
      "${COMMON_FLAGS_DEFAULT[@]}" &
}

max_jobs=${#GPU_LIST[@]}
for ((i = 0; i < ${#MODES[@]}; ++i)); do
  gpu_index=$((i % max_jobs))
  launch "${GPU_LIST[$gpu_index]}" "${MODES[$i]}"
done

wait
echo "All ZCA ablation jobs finished."

