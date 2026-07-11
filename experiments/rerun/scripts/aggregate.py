# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
# ]
# ///
#
# Seed Aggregation — All Phases (H2, H6, RBA)
# ============================================
# Reads results.json from each completed seed for H2, H6, and RBA.
# Produces aggregate/{h2,h6,rba}_aggregate.csv and aggregate/aggregate.json.
#
# CSV format (one row per metric):
#   phase, metric, n_seeds, mean, std, min, max
#
# Boolean metrics are cast to float (0.0/1.0); mean = proportion True.
# Any metric with 0 valid values across all seeds is flagged as a WARNING.
#
# Usage:
#   uv run experiments/rerun/scripts/aggregate.py
#   uv run experiments/rerun/scripts/aggregate.py --smoke

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

RERUN_ROOT = Path(__file__).resolve().parent.parent
SEEDS_ROOT = RERUN_ROOT / "seeds"
AGG_DIR    = RERUN_ROOT / "aggregate"
ALL_SEEDS  = [42, 123, 456, 789, 2024]

# ---------------------------------------------------------------------------
# Metric definitions — (label, key_path_list)
# ---------------------------------------------------------------------------

H2_METRICS = [
    # AUC
    ("auc.mean_pool.spoof",          ["auc", "mean_pool", "spoof"]),
    ("auc.mean_pool.novel",          ["auc", "mean_pool", "novel"]),
    ("auc.mean_pool.fleet",          ["auc", "mean_pool", "fleet"]),
    ("auc.concat_w1.spoof",          ["auc", "concat_w1", "spoof"]),
    ("auc.concat_w1.novel",          ["auc", "concat_w1", "novel"]),
    ("auc.concat_w1.fleet",          ["auc", "concat_w1", "fleet"]),
    ("auc.trivial.spoof",            ["auc", "trivial", "spoof"]),
    ("auc.trivial.novel",            ["auc", "trivial", "novel"]),
    ("auc.trivial.fleet",            ["auc", "trivial", "fleet"]),
    # PR-AUC
    ("pr_auc.mean_pool.spoof",       ["pr_auc", "mean_pool", "spoof"]),
    ("pr_auc.mean_pool.novel",       ["pr_auc", "mean_pool", "novel"]),
    ("pr_auc.mean_pool.fleet",       ["pr_auc", "mean_pool", "fleet"]),
    ("pr_auc.concat_w1.spoof",       ["pr_auc", "concat_w1", "spoof"]),
    ("pr_auc.concat_w1.novel",       ["pr_auc", "concat_w1", "novel"]),
    ("pr_auc.concat_w1.fleet",       ["pr_auc", "concat_w1", "fleet"]),
    ("pr_auc.trivial.spoof",         ["pr_auc", "trivial", "spoof"]),
    ("pr_auc.trivial.novel",         ["pr_auc", "trivial", "novel"]),
    ("pr_auc.trivial.fleet",         ["pr_auc", "trivial", "fleet"]),
    # PR-AUC Delta CIs
    ("ci.pr_spoof_delta.estimate",   ["bootstrap_ci_95", "pr_auc_deltas", "spoof_delta", "estimate"]),
    ("ci.pr_spoof_delta.lower",      ["bootstrap_ci_95", "pr_auc_deltas", "spoof_delta", "lower"]),
    ("ci.pr_spoof_delta.upper",      ["bootstrap_ci_95", "pr_auc_deltas", "spoof_delta", "upper"]),
    ("ci.pr_novel_delta.estimate",   ["bootstrap_ci_95", "pr_auc_deltas", "novel_delta", "estimate"]),
    ("ci.pr_novel_delta.lower",      ["bootstrap_ci_95", "pr_auc_deltas", "novel_delta", "lower"]),
    ("ci.pr_novel_delta.upper",      ["bootstrap_ci_95", "pr_auc_deltas", "novel_delta", "upper"]),
    ("ci.pr_fleet_delta.estimate",   ["bootstrap_ci_95", "pr_auc_deltas", "fleet_delta", "estimate"]),
    ("ci.pr_fleet_delta.lower",      ["bootstrap_ci_95", "pr_auc_deltas", "fleet_delta", "lower"]),
    ("ci.pr_fleet_delta.upper",      ["bootstrap_ci_95", "pr_auc_deltas", "fleet_delta", "upper"]),
    # Silhouette
    ("silhouette.mean_pool",         ["silhouette", "mean_pool"]),
    ("silhouette.concat",            ["silhouette", "concat"]),
    # Compactness
    ("compactness.mean_pool.mean",   ["compactness", "mean_pool", "mean"]),
    ("compactness.concat.mean",      ["compactness", "concat", "mean"]),
    # Delta CIs
    ("ci.spoof_delta.estimate",      ["bootstrap_ci_95", "deltas", "spoof_delta", "estimate"]),
    ("ci.spoof_delta.lower",         ["bootstrap_ci_95", "deltas", "spoof_delta", "lower"]),
    ("ci.spoof_delta.upper",         ["bootstrap_ci_95", "deltas", "spoof_delta", "upper"]),
    ("ci.novel_delta.estimate",      ["bootstrap_ci_95", "deltas", "novel_delta", "estimate"]),
    ("ci.novel_delta.lower",         ["bootstrap_ci_95", "deltas", "novel_delta", "lower"]),
    ("ci.novel_delta.upper",         ["bootstrap_ci_95", "deltas", "novel_delta", "upper"]),
    ("ci.fleet_delta.estimate",      ["bootstrap_ci_95", "deltas", "fleet_delta", "estimate"]),
    ("ci.fleet_delta.lower",         ["bootstrap_ci_95", "deltas", "fleet_delta", "lower"]),
    ("ci.fleet_delta.upper",         ["bootstrap_ci_95", "deltas", "fleet_delta", "upper"]),
    ("ci.silhouette_delta.estimate", ["bootstrap_ci_95", "deltas", "silhouette_delta", "estimate"]),
    ("ci.silhouette_delta.lower",    ["bootstrap_ci_95", "deltas", "silhouette_delta", "lower"]),
    ("ci.silhouette_delta.upper",    ["bootstrap_ci_95", "deltas", "silhouette_delta", "upper"]),
    # T2 window sweep
    ("t2.concat_w1.spoof_auc",       ["t2_window_sweep", "concat_w1", "spoof_auc"]),
    ("t2.concat_w3.spoof_auc",       ["t2_window_sweep", "concat_w3", "spoof_auc"]),
    ("t2.concat_w6.spoof_auc",       ["t2_window_sweep", "concat_w6", "spoof_auc"]),
    ("t2.mean_pool_spoof_auc",       ["t2_window_sweep", "mean_pool_spoof_auc"]),
    # T7 trivial margins
    ("t7.mean_pool_spoof_margin",    ["t7_trivial_baseline", "mean_pool_spoof_margin"]),
    ("t7.concat_w1_spoof_margin",    ["t7_trivial_baseline", "concat_w1_spoof_margin"]),
    # T8 robust config
    ("t8.robust.within_feature_mean", ["t8_token_similarity", "robust_config", "within_feature_mean"]),
    ("t8.robust.cross_feature_mean",  ["t8_token_similarity", "robust_config", "cross_feature_mean"]),
    ("t8.robust.within_cross_ratio",  ["t8_token_similarity", "robust_config", "within_cross_ratio"]),
    ("t8.robust.collapse_detected",   ["t8_token_similarity", "robust_config", "collapse_detected"]),
    # T8 degenerate config
    ("t8.degen.within_feature_mean",  ["t8_token_similarity", "degenerate_config", "within_feature_mean"]),
    ("t8.degen.cross_feature_mean",   ["t8_token_similarity", "degenerate_config", "cross_feature_mean"]),
    ("t8.degen.within_cross_ratio",   ["t8_token_similarity", "degenerate_config", "within_cross_ratio"]),
    ("t8.degen.collapse_detected",    ["t8_token_similarity", "degenerate_config", "collapse_detected"]),
    # Enrollment diagnostics
    ("enrollment.mean_pool_dist",     ["enrollment_diagnostics", "mean_pool_enrollment_dist"]),
    ("enrollment.concat_dist",        ["enrollment_diagnostics", "concat_enrollment_dist"]),
]

H6_METRICS = [
    ("spoof_k1.mp_raw.roc_auc",        ["spoof_k1", "mp_raw", "roc_auc"]),
    ("spoof_k1.mp_raw.pr_auc",         ["spoof_k1", "mp_raw", "pr_auc"]),
    ("spoof_k1.mp_rank_norm.roc_auc",  ["spoof_k1", "mp_rank_norm", "roc_auc"]),
    ("spoof_k1.mp_rank_norm.pr_auc",   ["spoof_k1", "mp_rank_norm", "pr_auc"]),
    ("spoof_k1.trivial.roc_auc",       ["spoof_k1", "trivial", "roc_auc"]),
    ("spoof_k1.trivial.pr_auc",        ["spoof_k1", "trivial", "pr_auc"]),
    ("spoof_k2.mp_raw.roc_auc",        ["spoof_k2", "mp_raw", "roc_auc"]),
    ("spoof_k2.mp_raw.pr_auc",         ["spoof_k2", "mp_raw", "pr_auc"]),
    ("spoof_k3.mp_raw.roc_auc",        ["spoof_k3", "mp_raw", "roc_auc"]),
    ("spoof_k3.mp_raw.pr_auc",         ["spoof_k3", "mp_raw", "pr_auc"]),
    ("novel.mp_raw.roc_auc",           ["novel", "mp_raw", "roc_auc"]),
    ("novel.mp_raw.pr_auc",            ["novel", "mp_raw", "pr_auc"]),
    ("fleet_aggregate.mp_raw.roc_auc",    ["fleet_aggregate", "mp_raw", "roc_auc"]),
    ("fleet_aggregate.mp_raw.pr_auc",     ["fleet_aggregate", "mp_raw", "pr_auc"]),
    ("fleet_aggregate.two_stage.roc_auc", ["fleet_aggregate", "two_stage", "roc_auc"]),
    ("fleet_aggregate.trivial.roc_auc",   ["fleet_aggregate", "trivial", "roc_auc"]),
    ("fleet_aggregate.c3_delta",          ["fleet_aggregate", "two_stage_vs_trivial_roc_delta"]),
    ("fleet_residual.mp_raw.roc_auc",     ["fleet_residual", "mp_raw", "roc_auc"]),
    ("fleet_residual.mp_raw.pr_auc",      ["fleet_residual", "mp_raw", "pr_auc"]),
    ("fleet_residual.two_stage.roc_auc",  ["fleet_residual", "two_stage", "roc_auc"]),
    ("fleet_residual.trivial.roc_auc",    ["fleet_residual", "trivial", "roc_auc"]),
    ("fleet_residual.c3_delta",           ["fleet_residual", "two_stage_vs_trivial_roc_delta"]),
    ("t8.within_feature_mean",   ["t8_token_similarity", "within_feature_mean"]),
    ("t8.cross_feature_mean",    ["t8_token_similarity", "cross_feature_mean"]),
    ("t8.within_cross_ratio",    ["t8_token_similarity", "within_cross_ratio"]),
    ("t8.collapse_detected",     ["t8_token_similarity", "collapse_detected"]),
    ("verdicts.primary_criterion_confirmed",  ["verdicts", "primary_criterion_confirmed"]),
    ("verdicts.rank_norm_collapse_confirmed", ["verdicts", "rank_norm_collapse_confirmed"]),
    ("verdicts.gate_blinds_fleet_confirmed",  ["verdicts", "gate_blinds_fleet_confirmed"]),
]

RBA_METRICS = [
    ("auc.mean_pool.roc_auc",  ["auc", "mean_pool", "roc_auc"]),
    ("auc.mean_pool.pr_auc",   ["auc", "mean_pool", "pr_auc"]),
    ("auc.concat.roc_auc",     ["auc", "concat",    "roc_auc"]),
    ("auc.concat.pr_auc",      ["auc", "concat",    "pr_auc"]),
    ("auc.trivial.roc_auc",    ["auc", "trivial",   "roc_auc"]),
    ("auc.trivial.pr_auc",     ["auc", "trivial",   "pr_auc"]),
    ("t6.mean_pool",           ["t6_compactness", "mean_pool"]),
    ("t6.concat",              ["t6_compactness", "concat"]),
    ("t8.within_feature_mean", ["t8_token_similarity", "within_feature_mean"]),
    ("t8.cross_feature_mean",  ["t8_token_similarity", "cross_feature_mean"]),
    ("t8.within_cross_ratio",  ["t8_token_similarity", "within_cross_ratio"]),
    ("t8.collapse_detected",   ["t8_token_similarity", "collapse_detected"]),
    ("h2_replicated",          ["h2_replicated"]),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(d: dict, *keys, default=None):
    node = d
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def load_phase_results(seeds_root: Path, phase: str) -> list[dict]:
    results = []
    for seed in ALL_SEEDS:
        path = seeds_root / f"seed_{seed}" / phase / "results.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if "seed" not in data:
                print(f"  WARNING: {path} missing 'seed' key — skipped")
                continue
            results.append(data)
        except json.JSONDecodeError as e:
            print(f"  WARNING: {path} is invalid JSON ({e}) — skipped")
    return results


def agg_metric(label: str, key_path: list, results: list[dict]) -> dict | None:
    vals = []
    for res in results:
        v = _get(res, *key_path)
        if v is None:
            continue
        if isinstance(v, dict):
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    a = np.array(vals, dtype=float)
    return {
        "metric":   label,
        "n_seeds":  len(vals),
        "mean":     float(np.mean(a)),
        "std":      float(np.std(a, ddof=0)),
        "min":      float(np.min(a)),
        "max":      float(np.max(a)),
    }


def aggregate_phase(metrics: list, results: list[dict]) -> list[dict]:
    rows = []
    empty = []
    for label, key_path in metrics:
        row = agg_metric(label, key_path, results)
        if row is None:
            empty.append(label)
        else:
            rows.append(row)
    if empty:
        print(f"  WARNING: {len(empty)} metric(s) returned no values: {empty[:5]}"
              + (" ..." if len(empty) > 5 else ""))
    return rows


def write_csv(phase: str, rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{phase}_aggregate.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "n_seeds", "mean", "std", "min", "max"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote: {path}  ({len(rows)} metrics, n_seeds up to {max(r['n_seeds'] for r in rows)})")
    return path


def write_aggregate_json(all_agg: dict, out_dir: Path) -> Path:
    path = out_dir / "aggregate.json"
    with open(path, "w") as f:
        json.dump(all_agg, f, indent=2)
    print(f"  Wrote: {path}")
    return path


# ---------------------------------------------------------------------------
# Smoke — synthetic results matching real schema
# ---------------------------------------------------------------------------

def _ci(est, lo, hi):
    return {"estimate": est, "lower": lo, "upper": hi}


def make_smoke_h2(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    def j(v, w=0.005): return float(v + rng.uniform(-w, w))
    return {
        "seed": seed, "timestamp": "2026-04-21T00:00:00Z",
        "config": {"sg": 1, "epochs": 20, "corpus": "per_account"},
        "auc": {
            "mean_pool": {"spoof": j(0.985), "novel": j(0.990), "fleet": j(0.975)},
            "concat_w1": {"spoof": j(0.920), "novel": j(0.930), "fleet": j(0.910)},
            "trivial":   {"spoof": j(0.500), "novel": j(0.500), "fleet": j(0.500)},
        },
        "pr_auc": {
            "mean_pool": {"spoof": j(0.960), "novel": j(0.985), "fleet": j(0.950)},
            "concat_w1": {"spoof": j(0.850), "novel": j(0.900), "fleet": j(0.840)},
            "trivial":   {"spoof": j(0.500), "novel": j(0.500), "fleet": j(0.500)},
        },
        "silhouette": {"mean_pool": j(0.42), "concat": j(0.28)},
        "compactness": {
            "mean_pool": {"mean": j(0.036), "ci_lower": j(0.030), "ci_upper": j(0.042)},
            "concat":    {"mean": j(0.047), "ci_lower": j(0.040), "ci_upper": j(0.054)},
        },
        "bootstrap_ci_95": {
            "per_model": {
                "mean_pool": {"spoof": _ci(j(0.985), j(0.978), j(0.992))},
                "concat_w1": {"spoof": _ci(j(0.920), j(0.912), j(0.928))},
            },
            "deltas": {
                "spoof_delta":      _ci(j(0.065), j(0.055), j(0.075)),
                "novel_delta":      _ci(j(0.060), j(0.050), j(0.070)),
                "fleet_delta":      _ci(j(0.065), j(0.055), j(0.075)),
                "silhouette_delta": _ci(j(0.140), j(0.120), j(0.160)),
            },
            "pr_auc_deltas": {
                "spoof_delta": _ci(j(0.110), j(0.090), j(0.130)),
                "novel_delta": _ci(j(0.085), j(0.070), j(0.100)),
                "fleet_delta": _ci(j(0.110), j(0.090), j(0.130)),
            },
        },
        "t2_window_sweep": {
            "concat_w1": {"spoof_auc": j(0.920)},
            "concat_w3": {"spoof_auc": j(0.935)},
            "concat_w6": {"spoof_auc": j(0.945)},
            "mean_pool_spoof_auc": j(0.985),
        },
        "t3_prefixed_concat": {
            "mean_pool_silhouette": j(0.42),
            "prefixed_concat_silhouette": j(0.35),
            "gap": j(0.07),
        },
        "t4_tz_attribution": {"mean_pool": j(0.12), "concat": j(0.08)},
        "t5_tz_permutation": {
            "mean_pool_spoof_auc": j(0.985),
            "position_1": {"concat_spoof_auc": j(0.900)},
            "position_2": {"concat_spoof_auc": j(0.905)},
            "position_3": {"concat_spoof_auc": j(0.910)},
            "position_4": {"concat_spoof_auc": j(0.915)},
            "position_5": {"concat_spoof_auc": j(0.912)},
            "position_6": {"concat_spoof_auc": j(0.920)},
        },
        "t7_trivial_baseline": {
            "mean_pool_spoof_margin": j(0.485),
            "concat_w1_spoof_margin": j(0.420),
        },
        "t8_token_similarity": {
            "robust_config": {
                "within_feature_mean": j(0.563),
                "cross_feature_mean":  j(0.490),
                "within_cross_ratio":  j(1.149),
                "collapse_detected":   False,
            },
            "degenerate_config": {
                "within_feature_mean": j(0.950),
                "cross_feature_mean":  j(0.940),
                "within_cross_ratio":  j(1.011),
                "collapse_detected":   True,
            },
        },
        "enrollment_diagnostics": {
            "mean_pool_enrollment_dist": j(0.18),
            "concat_enrollment_dist":    j(0.24),
        },
    }


def make_smoke_h6(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    def j(v, w=0.005): return float(v + rng.uniform(-w, w))
    def sb(roc=0.95, pr=0.85): return {"roc_auc": j(roc), "pr_auc": j(pr), "ci_lower": j(pr - 0.05), "ci_upper": j(pr + 0.05)}
    return {
        "seed": seed, "timestamp": "2026-04-21T00:00:00Z",
        "config": {"n_accounts": 5000, "n_fleet_accounts": 250, "n_pre_lag": 100, "n_post_lag": 150},
        "spoof_k1": {
            "mp_raw":       sb(0.995, 0.810),
            "mp_rank_norm": sb(0.972, 0.200),
            "trivial":      sb(0.500, 0.010),
        },
        "spoof_k2": {"mp_raw": sb(0.997, 0.900), "trivial": sb(0.500, 0.010)},
        "spoof_k3": {"mp_raw": sb(0.999, 0.950), "trivial": sb(0.500, 0.010)},
        "novel":    {"mp_raw": sb(0.999, 0.965), "trivial": sb(0.500, 0.010)},
        "fleet_aggregate": {
            "mp_raw":    sb(0.994, 0.918),
            "two_stage": sb(0.459, 0.010),
            "trivial":   sb(0.459, 0.010),
            "two_stage_vs_trivial_roc_delta": 0.0,
        },
        "fleet_residual": {
            "mp_raw":    sb(0.997, 0.947),
            "two_stage": sb(0.457, 0.010),
            "trivial":   sb(0.457, 0.010),
            "two_stage_vs_trivial_roc_delta": 0.0,
            "top1pct_tp": 10, "top1pct_prec": 0.667,
        },
        "t8_token_similarity": {
            "within_feature_mean": j(0.392),
            "cross_feature_mean":  j(0.050),
            "within_cross_ratio":  j(7.84),
            "collapse_detected":   False,
        },
        "verdicts": {
            "primary_criterion_confirmed":  True,
            "rank_norm_collapse_confirmed": True,
            "gate_blinds_fleet_confirmed":  True,
        },
    }


def make_smoke_rba(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    def j(v, w=0.005): return float(v + rng.uniform(-w, w))
    return {
        "seed": seed, "timestamp": "2026-04-21T00:00:00Z",
        "split_percentile": 50, "n_ato_test_events": 500,
        "auc": {
            "mean_pool": {"roc_auc": j(0.850), "pr_auc": j(0.032), "ci_lower": j(0.025), "ci_upper": j(0.040)},
            "concat":    {"roc_auc": j(0.820), "pr_auc": j(0.025), "ci_lower": j(0.018), "ci_upper": j(0.032)},
            "trivial":   {"roc_auc": j(0.780), "pr_auc": j(0.003)},
        },
        "t6_compactness": {"mean_pool": j(0.036), "concat": j(0.047)},
        "t8_token_similarity": {
            "within_feature_mean": j(0.563),
            "cross_feature_mean":  j(0.490),
            "within_cross_ratio":  j(1.149),
            "collapse_detected":   False,
        },
        "h2_replicated": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate seed results for H2, H6, and RBA")
    parser.add_argument("--seeds-root", type=Path, default=SEEDS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=AGG_DIR)
    parser.add_argument("--smoke", action="store_true",
                        help="Run on 3 synthetic seeds; assert all CSVs produced; clean up")
    args = parser.parse_args()

    print("=" * 60)
    print("Seed Aggregation — H2 / H6 / RBA")
    print("=" * 60)

    import tempfile, shutil

    if args.smoke:
        smoke_seeds = [42, 123, 456]
        h2_results  = [make_smoke_h2(s)  for s in smoke_seeds]
        h6_results  = [make_smoke_h6(s)  for s in smoke_seeds]
        rba_results = [make_smoke_rba(s) for s in smoke_seeds]
        tmp_root = Path(tempfile.mkdtemp())
        out_dir  = tmp_root / "aggregate_smoke"
        print(f"\n[SMOKE] Synthetic data for seeds {smoke_seeds}")
    else:
        tmp_root = None
        out_dir  = args.out_dir
        print("\n[1/3] Loading H2 results ...")
        h2_results = load_phase_results(args.seeds_root, "h2")
        print(f"  {len(h2_results)} seed(s): {[r['seed'] for r in h2_results]}")
        print("\n[2/3] Loading H6 results ...")
        h6_results = load_phase_results(args.seeds_root, "h6")
        print(f"  {len(h6_results)} seed(s): {[r['seed'] for r in h6_results]}")
        print("\n[3/3] Loading RBA results ...")
        rba_results = load_phase_results(args.seeds_root, "rba")
        print(f"  {len(rba_results)} seed(s): {[r['seed'] for r in rba_results]}")

    all_agg: dict = {}
    expected_csvs = []

    for phase, metrics, results in [
        ("h2",  H2_METRICS,  h2_results),
        ("h6",  H6_METRICS,  h6_results),
        ("rba", RBA_METRICS, rba_results),
    ]:
        print(f"\n── {phase.upper()} ({len(results)} seed(s)) ──")
        if not results:
            print("  No results — skipping")
            continue
        rows = aggregate_phase(metrics, results)
        write_csv(phase, rows, out_dir)
        all_agg[phase] = {r["metric"]: r for r in rows}
        expected_csvs.append(out_dir / f"{phase}_aggregate.csv")

    if all_agg:
        print()
        write_aggregate_json(all_agg, out_dir)

    if args.smoke:
        missing = [p for p in expected_csvs if not p.exists()]
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)
        if missing:
            for m in missing:
                print(f"ASSERTION FAILED: {m.name} not created", file=sys.stderr)
            sys.exit(1)
        print("\nSMOKE PASS — all CSVs and aggregate.json produced.")
    else:
        n = max((len(h2_results), len(h6_results), len(rba_results)), default=0)
        print(f"\nDone. Up to {n}/5 seeds aggregated.")
        if n < 5:
            print(f"  NOTE: rerun after all 5 seeds complete for final numbers.")


if __name__ == "__main__":
    main()
