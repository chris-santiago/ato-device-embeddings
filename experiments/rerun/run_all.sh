#!/usr/bin/env bash
# run_all.sh — Orchestrate 5-seed reproducibility rerun for H2, H6, and RBA.
#
# Runs each (seed × phase) combination, skipping already-complete results.
# After all experiments complete, runs consistency checks and aggregation.
#
# Usage:
#   ./experiments/rerun/run_all.sh
#   ./experiments/rerun/run_all.sh --seeds "42 123"
#   ./experiments/rerun/run_all.sh --phases "h2 rba"
#   ./experiments/rerun/run_all.sh --smoke
#   ./experiments/rerun/run_all.sh --force      # re-run even if results.json exists
#   ./experiments/rerun/run_all.sh --dry-run    # print commands without executing
#
# Prerequisites:
#   - uv installed and in PATH
#   - data/rba/rba.parquet present (run scripts/rba/data_prep.py first for RBA)
#
# Logs: seeds/seed_<N>/<phase>/logs/run_<timestamp>.log

set -euo pipefail

RERUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$RERUN_ROOT/scripts"
SEEDS_ROOT="$RERUN_ROOT/seeds"

DEFAULT_SEEDS=(42 123 456 789 2024)
DEFAULT_PHASES=(h2 h6 rba)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SEEDS=()
PHASES=()
FORCE=0
DRY_RUN=0
SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            IFS=' ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --phases)
            IFS=' ' read -r -a PHASES <<< "$2"; shift 2 ;;
        --force)
            FORCE=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --smoke)
            SMOKE=1; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--seeds '42 123'] [--phases 'h2 h6 rba'] [--force] [--dry-run] [--smoke]" >&2
            exit 1 ;;
    esac
done

[[ ${#SEEDS[@]}  -eq 0 ]] && SEEDS=("${DEFAULT_SEEDS[@]}")
[[ ${#PHASES[@]} -eq 0 ]] && PHASES=("${DEFAULT_PHASES[@]}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_cmd() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

is_complete() {
    local results_json="$1"
    [[ -f "$results_json" ]] || return 1
    # Verify the file is valid non-empty JSON containing "seed"
    python3 -c "
import json, sys
try:
    d = json.load(open('$results_json'))
    sys.exit(0 if 'seed' in d else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

timestamp() {
    date -u +"%Y%m%dT%H%M%SZ"
}

# ---------------------------------------------------------------------------
# Smoke mode — chain individual --smoke passes for each script
# ---------------------------------------------------------------------------

if [[ $SMOKE -eq 1 ]]; then
    echo "========================================"
    echo "  SMOKE MODE — per-script verification"
    echo "========================================"
    echo ""

    FAIL=0

    for phase in "${PHASES[@]}"; do
        echo "── $(echo "$phase" | tr '[:lower:]' '[:upper:]') smoke ──"

        if [[ "$phase" == "rba" ]]; then
            # RBA --smoke requires parquet; skip gracefully if absent
            parquet="$RERUN_ROOT/../../data/rba/rba.parquet"
            if [[ ! -f "$parquet" ]]; then
                echo "  [SKIP] RBA smoke requires data/rba/rba.parquet (run data_prep.py first)"
                echo ""
                continue
            fi
            # RBA exits 1 when h2_replicated=False — expected on tiny smoke data.
            # Treat as PASS if results.json was written (pipeline completed).
            run_cmd uv run "$SCRIPTS/rba/rba_rerun.py" --seed 42 --smoke || true
            rba_smoke_result="$SEEDS_ROOT/seed_42/rba/results.json"
            if [[ $DRY_RUN -eq 0 && -f "$rba_smoke_result" ]]; then
                echo "  [PASS] rba_rerun.py --smoke (pipeline OK; non-replication expected on 5k-user smoke data)"
                rm -rf "$SEEDS_ROOT/seed_42/rba"
            elif [[ $DRY_RUN -eq 1 ]]; then
                echo "  [DRY-RUN] rba smoke would run and clean up"
            else
                echo "  [FAIL] rba_rerun.py --smoke (results.json not written)" >&2
                FAIL=1
            fi
        else
            # All other rerun scripts accept --seed; use 42 for smoke
            if run_cmd uv run "$SCRIPTS/$phase/${phase}_rerun.py" --seed 42 --smoke; then
                echo "  [PASS] ${phase}_rerun.py --smoke"
            else
                echo "  [FAIL] ${phase}_rerun.py --smoke" >&2
                FAIL=1
            fi

            # H6 smoke writes real results to seeds/seed_42/h6/ — clean up after
            if [[ "$phase" == "h6" && $DRY_RUN -eq 0 ]]; then
                rm -f "$SEEDS_ROOT/seed_42/h6/results.json" \
                      "$SEEDS_ROOT/seed_42/h6/scores.npz"
            fi
        fi

        echo ""
    done

    echo "── H6 consistency checker smoke ──"
    if run_cmd uv run "$SCRIPTS/h6/check_consistency.py" --smoke; then
        echo "  [PASS] h6/check_consistency.py --smoke"
    else
        echo "  [FAIL] h6/check_consistency.py --smoke" >&2
        FAIL=1
    fi
    echo ""

    echo "── Cross-phase consistency checker smoke ──"
    if run_cmd uv run "$SCRIPTS/check_consistency.py" --smoke; then
        echo "  [PASS] check_consistency.py --smoke"
    else
        echo "  [FAIL] check_consistency.py --smoke" >&2
        FAIL=1
    fi
    echo ""

    echo "── Aggregator smoke ──"
    if run_cmd uv run "$SCRIPTS/aggregate.py" --smoke; then
        echo "  [PASS] aggregate.py --smoke"
    else
        echo "  [FAIL] aggregate.py --smoke" >&2
        FAIL=1
    fi
    echo ""

    if [[ $FAIL -eq 0 ]]; then
        echo "========================================"
        echo "  SMOKE PASS — all scripts verified"
        echo "========================================"
        exit 0
    else
        echo "========================================"
        echo "  SMOKE FAIL — see errors above"
        echo "========================================"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Main seed × phase loop
# ---------------------------------------------------------------------------

echo "========================================"
echo "  Rerun: seeds=${SEEDS[*]}"
echo "         phases=${PHASES[*]}"
[[ $FORCE -eq 1 ]]   && echo "         --force (re-running existing)"
[[ $DRY_RUN -eq 1 ]] && echo "         --dry-run"
echo "========================================"
echo ""

N_RUN=0
N_SKIP=0
N_FAIL=0

for seed in "${SEEDS[@]}"; do
    for phase in "${PHASES[@]}"; do
        seed_dir="$SEEDS_ROOT/seed_${seed}/${phase}"
        results_json="$seed_dir/results.json"
        logs_dir="$seed_dir/logs"
        log_file="$logs_dir/run_$(timestamp).log"

        echo "── seed=${seed} phase=${phase} ──"

        # Skip if complete and not forcing
        if [[ $FORCE -eq 0 ]] && is_complete "$results_json"; then
            echo "  [SKIP] results.json exists and is valid — use --force to re-run"
            N_SKIP=$((N_SKIP + 1))
            continue
        fi

        # Require parquet for RBA
        if [[ "$phase" == "rba" ]]; then
            parquet="$RERUN_ROOT/../../data/rba/rba.parquet"
            if [[ ! -f "$parquet" ]]; then
                echo "  [SKIP] RBA requires data/rba/rba.parquet — run scripts/rba/data_prep.py first" >&2
                N_SKIP=$((N_SKIP + 1))
                continue
            fi
        fi

        mkdir -p "$logs_dir"

        echo "  Running... (log: $log_file)"
        set +e
        run_cmd uv run "$SCRIPTS/$phase/${phase}_rerun.py" --seed "$seed" \
            2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}
        set -e

        if [[ $exit_code -eq 0 ]]; then
            echo "  [DONE] seed=${seed} phase=${phase}"
            N_RUN=$((N_RUN + 1))
        else
            echo "  [FAIL] seed=${seed} phase=${phase} — exit code ${exit_code}" >&2
            N_FAIL=$((N_FAIL + 1))
        fi
        echo ""
    done
done

# ---------------------------------------------------------------------------
# Post-run: consistency checks and aggregation
# ---------------------------------------------------------------------------

echo "========================================"
echo "  Post-run checks"
echo "========================================"
echo ""

echo "[1/6] H6 detailed consistency checks ..."
run_cmd uv run "$SCRIPTS/h6/check_consistency.py" \
    --seeds-root "$SEEDS_ROOT" || {
    echo "  WARNING: H6 consistency check failed — review before aggregating" >&2
}
echo ""

echo "[2/6] Cross-phase consistency checks ..."
run_cmd uv run "$SCRIPTS/check_consistency.py" \
    --seeds-root "$SEEDS_ROOT" || {
    echo "  WARNING: Cross-phase consistency check failed — review before aggregating" >&2
}
echo ""

echo "[3/6] Aggregating results ..."
run_cmd uv run "$SCRIPTS/aggregate.py" \
    --seeds-root "$SEEDS_ROOT"
echo ""

echo "[4/6] Generating H2 figures ..."
run_cmd uv run "$SCRIPTS/h2/h2_figures.py"
echo ""

echo "[5/6] Generating H6 figures ..."
run_cmd uv run "$SCRIPTS/h6/h6_figures.py"
echo ""

echo "[6/6] Generating RBA figures ..."
run_cmd uv run "$SCRIPTS/rba/rba_figures.py"
echo ""
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "========================================"
echo "  Summary"
echo "    Ran:     $N_RUN"
echo "    Skipped: $N_SKIP"
echo "    Failed:  $N_FAIL"
echo "========================================"

if [[ $N_FAIL -gt 0 ]]; then
    echo "WARN: $N_FAIL experiment(s) failed — check logs under seeds/" >&2
    exit 1
fi

exit 0
