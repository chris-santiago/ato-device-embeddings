#!/usr/bin/env bash
# Calibration-window ablation: full 5x5 sweep -> aggregate -> plot.
# Runs the 5 seeds in parallel (each process is single-threaded for FastText
# determinism); each seed process runs all 5 calibration sizes serially.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(42 123 456 789 2024)
mkdir -p "$HERE/logs"

echo "[1/3] Running 5x5 sweep (5 seeds in parallel) ..."
for s in "${SEEDS[@]}"; do
  uv run "$HERE/calib_sweep.py" --seed "$s" --out "$HERE/results" \
    > "$HERE/logs/seed_${s}.log" 2>&1 &
done
wait

echo "[2/3] Aggregating across seeds ..."
uv run "$HERE/aggregate_calib.py"

echo "[3/3] Plotting ..."
uv run "$HERE/plot_calib_sweep.py"

echo "Done. See $HERE/SUMMARY.md"
