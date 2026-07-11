#!/usr/bin/env bash
# Phase-1 critique ablations (plan/12-CRITIQUE_ABLATIONS.md): A1 + A2a + A3p
# across the standard 5 seeds, then deterministic aggregation.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

SEEDS=(42 123 456 789 2024)

for s in "${SEEDS[@]}"; do
  echo "=== baseline_controls seed ${s} ==="
  uv run baseline_controls.py --seed "${s}" 2>&1 | tee "logs/baseline_controls_seed_${s}.log"
done

for s in "${SEEDS[@]}"; do
  echo "=== h6_perevent_collapse seed ${s} ==="
  uv run h6_perevent_collapse.py --seed "${s}" 2>&1 | tee "logs/h6_perevent_seed_${s}.log"
done

for s in "${SEEDS[@]}"; do
  echo "=== h6_likelihood_incumbent seed ${s} ==="
  uv run h6_likelihood_incumbent.py --seed "${s}" 2>&1 | tee "logs/h6_likelihood_seed_${s}.log"
done

for s in "${SEEDS[@]}"; do
  echo "=== a4_nosub seed ${s} ==="
  uv run a4_nosub.py --seed "${s}" 2>&1 | tee "logs/a4_nosub_seed_${s}.log"
done

uv run aggregate_ablations.py
