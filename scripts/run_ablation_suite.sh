#!/usr/bin/env bash

# Generate preprocessing statistics for all ablation modes, then print the
# command needed to launch the corresponding training sweep.

set -euo pipefail

CONFIG_DIR=${CONFIG_DIR:-configs/anyon}
PCA_DIR=${PCA_DIR:-artifacts/preproc}

FIT_CONFIG=${FIT_CONFIG:-"$CONFIG_DIR/zca.yaml"}
TRAIN_CONFIG=${TRAIN_CONFIG:-"$CONFIG_DIR/run.yaml"}
STATS_ROOT=${STATS_ROOT:-"$PCA_DIR"}

mkdir -p "$STATS_ROOT"

RANK=${RANK:-64}
EPS=${EPS:-1e-5}
SHRINKAGE=${SHRINKAGE:-0.0}
SEED=${SEED:-0}

run_fit() {
  local mode="$1"
  shift
  local tag="$mode"
  if [[ $# -gt 0 && "$1" != --* ]]; then
    tag="$1"
    shift
  fi
  local out_path="$STATS_ROOT/${tag}.pt"
  echo "[fit] mode=$mode tag=$tag -> $out_path"
  python3 scripts/fit_preprocessor.py \
    --config "$FIT_CONFIG" \
    --mode "$mode" \
    --output "$out_path" \
    "$@"
}

# run_fit center
# run_fit standardize
# run_fit zca zca --eps "$EPS" --shrinkage "$SHRINKAGE"
run_fit zca_lowrank zca_lowrank --rank "$RANK" --eps "$EPS" --shrinkage "$SHRINKAGE" --perp-mode avg
# run_fit project_lowrank project_lowrank --rank "$RANK"
# run_fit randrot_white randrot_white --seed "$SEED" --eps "$EPS" --shrinkage "$SHRINKAGE"
# run_fit randrot randrot --seed "$SEED"
# run_fit pca pca --rank "$RANK"
# run_fit pls pls --rank "$RANK"
# run_fit cca cca --rank "$RANK"

echo
echo "Ablation stats ready under $STATS_ROOT"

# RUN_ABLATION=${RUN_ABLATION:-1}
# ABLATION_FLAGS=${ABLATION_FLAGS:-}

# CMD=(
#   python3 scripts/ablation.py
#   --config "$TRAIN_CONFIG"
#   --stats-root "$STATS_ROOT"
# )

# if (( RUN_ABLATION )); then
#   echo "Launching ablation sweep: ${CMD[*]} ${ABLATION_FLAGS}"
#   ${CMD[@]} ${ABLATION_FLAGS}
# else
#   echo "Launch the sweep with:"
#   echo "  ${CMD[*]} ${ABLATION_FLAGS}"
# fi

