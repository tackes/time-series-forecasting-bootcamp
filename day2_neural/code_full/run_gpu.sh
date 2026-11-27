#!/usr/bin/env bash
# Run code_full notebooks on the GPU box: a TRAIN pass, then a LOAD pass for clean outputs.
#
#   source .venv/bin/activate                  # from the repo root
#   bash day2_neural/code_full/run_gpu.sh 00_recap_full 01_gfm_recipe
#   bash day2_neural/code_full/run_gpu.sh all  # every notebook, in chain order
#
# Stops at the first failure. A TRAIN pass that dies from GPU or host memory (CUDA out of
# memory, a killed kernel) is retried up to GD_RETRIES more times (default 2): every finished
# fit is cached, so a retry resumes rather than starting again. Any other error stops at once.
# Wall-clock minutes per pass are appended to results/day2_full/gpu_timings.csv, and RAM and
# GPU memory once a minute to gpu_logs/memory.csv. Nothing is committed; do that by hand.
# Dry run on a laptop: GD_SMOKE=1 BOOTCAMP_ACCELERATOR=cpu bash run_gpu.sh 00_recap_full
# (writes to results/day2_full_smoke instead and skips the CUDA check).
set -euo pipefail

ORDER=(00_recap_full 01_gfm_recipe 02_hpo_round1 03_calendar_hierarchy 04_price_response
       05_hpo_round2_pareto 05b_synthetic_discounts 05c_scaler_floor 06_cluster_gfms
       07_foundation 07b_new_foundation 08_hierarchy_reconcile 09_uncertainty
       10_selection_standings 11_all_series 12_price_plan)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$ROOT/results/day2_full"
[ "${GD_SMOKE:-}" = "1" ] && OUT="$ROOT/results/day2_full_smoke"
LOGDIR="$OUT/gpu_logs"
TIMINGS="$OUT/gpu_timings.csv"
RETRIES="${GD_RETRIES:-2}"
mkdir -p "$LOGDIR"
[ -f "$TIMINGS" ] || echo "notebook,pass,minutes,finished" > "$TIMINGS"

export BOOTCAMP_ACCELERATOR="${BOOTCAMP_ACCELERATOR:-gpu}"
# One GPU per run. Lightning would otherwise train on every visible card, which hangs inside
# a notebook on a multi-GPU box. Pick another card with CUDA_VISIBLE_DEVICES=1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

# Memory log: kernel RAM (GB) and GPU memory (MiB) once a minute, stopped when this script ends.
log_memory() {
    local file="$LOGDIR/memory.csv"
    [ -f "$file" ] || echo "time,notebook,kernel_rss_gb,gpu_mib" > "$file"
    while true; do
        local pid rss gpu
        pid=$(pgrep -n -f "ipykernel_launcher -f /tmp" || true)   # newest kernel: this pass
        rss=$( [ -n "$pid" ] && ps -o rss= -p "$pid" | awk '{printf "%.2f", $1/1048576}' || true)
        gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
              -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null || echo NA)
        echo "$(date -Iseconds),${CURRENT_NB:-},${rss:-NA},${gpu}" >> "$file"
        sleep 60
    done
}

# Retry only failures that a fresh process can fix: memory, not code.
retryable() {
    grep -qE "OutOfMemoryError|CUDA out of memory|DeadKernelError|Kernel died|Killed" "$1"
}

cd "$HERE"
for nb in "${NOTEBOOKS[@]}"; do
    nb="${nb%.py}"
    [ -f "$nb.py" ] || { echo "no such notebook: $nb.py" >&2; exit 1; }
    jupytext --to ipynb --update "$nb.py"
    export CURRENT_NB="$nb"
    log_memory & LOGGER=$!
    trap 'kill $LOGGER 2>/dev/null || true' EXIT

    echo "=== $nb: train pass ($(date '+%H:%M')) ==="
    start=$(date +%s)
    attempt=0
    until GD_TRAIN=1 jupyter nbconvert --to notebook --execute "$nb.ipynb" \
            --output-dir "$LOGDIR" --output "${nb}_train" --ExecutePreprocessor.timeout=-1 \
            2> >(tee "$LOGDIR/${nb}_train.err" >&2); do
        sleep 2                                   # let the stderr copy finish
        attempt=$((attempt + 1))
        cp "$LOGDIR/${nb}_train.err" "$LOGDIR/${nb}_train_failed${attempt}.err"
        if [ "$attempt" -gt "$RETRIES" ] || ! retryable "$LOGDIR/${nb}_train.err"; then
            echo "=== $nb: train pass FAILED (attempt $attempt; see gpu_logs/${nb}_train_failed${attempt}.err) ===" >&2
            exit 1
        fi
        echo "=== $nb: train pass died (memory or a killed kernel), retrying from the cache ($attempt of $RETRIES) ==="
    done
    mins=$(( ($(date +%s) - start) / 60 ))
    echo "$nb,train,$mins,$(date -Iseconds)" >> "$TIMINGS"

    echo "=== $nb: load pass ($(date '+%H:%M')) ==="
    start=$(date +%s)
    jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" \
        --ExecutePreprocessor.timeout=-1
    mins=$(( ($(date +%s) - start) / 60 ))
    echo "$nb,load,$mins,$(date -Iseconds)" >> "$TIMINGS"
    du -h "$nb.ipynb"
    kill "$LOGGER" 2>/dev/null || true
done
echo "done: ${NOTEBOOKS[*]}"
