#!/usr/bin/env bash
# Run code_full notebooks on the GPU box: a TRAIN pass, then a LOAD pass for clean outputs.
#
#   source .venv/bin/activate                  # from the repo root
#   bash day2_neural/code_full/run_gpu.sh 00_recap_full 01_gfm_recipe
#   bash day2_neural/code_full/run_gpu.sh all  # every notebook, in chain order
#
# Stops at the first failure. Wall-clock minutes per pass are appended to
# results/day2_full/gpu_timings.csv. Nothing is committed; do that by hand after checking.
# Dry run on a laptop: GD_SMOKE=1 BOOTCAMP_ACCELERATOR=cpu bash run_gpu.sh 00_recap_full
# (writes to results/day2_full_smoke instead and skips the CUDA check).
set -euo pipefail

ORDER=(00_recap_full 01_gfm_recipe 02_hpo_round1 03_calendar_hierarchy 04_price_response
       05_hpo_round2_pareto 05b_synthetic_discounts 06_cluster_gfms 07_foundation
       08_hierarchy_reconcile 09_uncertainty 10_selection_standings)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$ROOT/results/day2_full"
[ "${GD_SMOKE:-}" = "1" ] && OUT="$ROOT/results/day2_full_smoke"
LOGDIR="$OUT/gpu_logs"
TIMINGS="$OUT/gpu_timings.csv"
mkdir -p "$LOGDIR"
[ -f "$TIMINGS" ] || echo "notebook,pass,minutes,finished" > "$TIMINGS"

export BOOTCAMP_ACCELERATOR="${BOOTCAMP_ACCELERATOR:-gpu}"

if [ "$#" -eq 0 ]; then
    echo "usage: run_gpu.sh <notebook> [...] | all" >&2
    exit 1
fi
if [ "$1" = "all" ]; then
    NOTEBOOKS=("${ORDER[@]}")
else
    NOTEBOOKS=("$@")
fi

if [ "$BOOTCAMP_ACCELERATOR" = "gpu" ]; then
    python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available')"
fi

cd "$HERE"
for nb in "${NOTEBOOKS[@]}"; do
    nb="${nb%.py}"
    [ -f "$nb.py" ] || { echo "no such notebook: $nb.py" >&2; exit 1; }
    jupytext --to ipynb --update "$nb.py"

    echo "=== $nb: train pass ($(date '+%H:%M')) ==="
    start=$(date +%s)
    GD_TRAIN=1 jupyter nbconvert --to notebook --execute "$nb.ipynb" \
        --output-dir "$LOGDIR" --output "${nb}_train" --ExecutePreprocessor.timeout=-1
    mins=$(( ($(date +%s) - start) / 60 ))
    echo "$nb,train,$mins,$(date -Iseconds)" >> "$TIMINGS"

    echo "=== $nb: load pass ($(date '+%H:%M')) ==="
    start=$(date +%s)
    jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" \
        --ExecutePreprocessor.timeout=-1
    mins=$(( ($(date +%s) - start) / 60 ))
    echo "$nb,load,$mins,$(date -Iseconds)" >> "$TIMINGS"
    du -h "$nb.ipynb"
done
echo "done: ${NOTEBOOKS[*]}"
