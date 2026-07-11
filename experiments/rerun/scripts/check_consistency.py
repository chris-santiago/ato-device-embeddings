# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
# ]
# ///
#
# Cross-Phase Consistency Checker
# ================================
# Covers H2, H6 (high-level), and RBA across all completed seeds.
# Detailed H6 per-seed checks live in scripts/h6/check_consistency.py;
# run both for complete coverage.
#
# H2 checks (per seed):
#   H2-S1  Required top-level keys present
#   H2-S2  T8 robust config: collapse_detected == False
#   H2-S3  T8 degenerate config: collapse_detected == True
#   H2-S4  mean_pool spoof AUC > concat_w1 spoof AUC (primary H2 claim)
#   H2-S5  mean_pool spoof AUC > trivial spoof AUC
#   H2-S6  spoof delta CI lower bound > 0 (mean_pool beats concat_w1 on spoof)
#
# H2 cross-seed (≥2 seeds):
#   H2-X1  mean_pool spoof AUC std < 0.02
#   H2-X2  T8 degenerate within_feature_mean std < 0.01 (mechanistic determinism)
#
# H6 checks (per seed, high-level):
#   H6-S1  T8 no collapse in robust config
#   H6-S2  All 3 verdicts present and True
#   H6-S3  C3 delta < 1e-4 on fleet_aggregate and fleet_residual
#
# H6 cross-seed (≥2 seeds):
#   H6-X1  Verdict stability: all 3 verdicts agree across every seed
#
# RBA checks (per seed):
#   RBA-S1  Required keys present
#   RBA-S2  T8 no collapse
#   RBA-S3  h2_replicated == True
#   (RBA-S4 removed: mp > concat point-estimate ordering is underpowered at n_ato=9)
#
# RBA cross-seed (≥2 seeds):
#   RBA-X1  h2_replicated stability (all seeds agree)
#
# Cross-phase (when all 3 phases have ≥1 result):
#   XP-1  Same seed IDs represented in all three phases
#   XP-2  No T8 collapse in any phase (summary)
#
# Exit code: 0 if all checks pass, 1 if any fail.
# With only partial seed data, prints a WARNING but still runs available checks.
#
# Usage:
#   uv run experiments/rerun/scripts/check_consistency.py
#   uv run experiments/rerun/scripts/check_consistency.py --smoke

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

RERUN_ROOT = Path(__file__).resolve().parent.parent
SEEDS_ROOT = RERUN_ROOT / "seeds"
ALL_SEEDS  = [42, 123, 456, 789, 2024]

H2_REQUIRED_KEYS = [
    "seed", "timestamp", "config", "auc", "silhouette", "compactness",
    "bootstrap_ci_95", "t2_window_sweep", "t5_tz_permutation",
    "t7_trivial_baseline", "t8_token_similarity", "enrollment_diagnostics",
]
H6_VERDICT_KEYS = [
    "primary_criterion_confirmed",
    "rank_norm_collapse_confirmed",
    "gate_blinds_fleet_confirmed",
]
RBA_REQUIRED_KEYS = [
    "seed", "timestamp", "auc", "t6_compactness", "t8_token_similarity", "h2_replicated",
]
C3_DELTA_THRESHOLD = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(d: dict, *keys, default: Any = None) -> Any:
    node: Any = d
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


class CheckResult:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def record(self, label: str, passed: bool, detail: str = "") -> bool:
        self.rows.append((label, passed, detail))
        status = "PASS" if passed else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"  {label:8s}  [{status}] {suffix}")
        return passed

    @property
    def n_pass(self) -> int: return sum(1 for _, p, _ in self.rows if p)
    @property
    def n_fail(self) -> int: return sum(1 for _, p, _ in self.rows if not p)


# ---------------------------------------------------------------------------
# H2 checks
# ---------------------------------------------------------------------------

def check_h2_seed(res: dict, seed: int, cr: CheckResult) -> None:
    tag = f"seed {seed}"

    missing = [k for k in H2_REQUIRED_KEYS if k not in res]
    cr.record("H2-S1", not missing,
              f"{tag}: missing {missing}" if missing else f"{tag}: schema OK")

    collapse_robust = _get(res, "t8_token_similarity", "robust_config", "collapse_detected", default=True)
    cr.record("H2-S2", not collapse_robust,
              f"{tag}: robust collapse_detected={collapse_robust}")

    collapse_degen = _get(res, "t8_token_similarity", "degenerate_config", "collapse_detected", default=False)
    cr.record("H2-S3", bool(collapse_degen),
              f"{tag}: degenerate collapse_detected={collapse_degen}")

    mp   = float(_get(res, "auc", "mean_pool", "spoof", default=float("nan")))
    cat  = float(_get(res, "auc", "concat_w1", "spoof", default=float("nan")))
    triv = float(_get(res, "auc", "trivial",   "spoof", default=float("nan")))
    cr.record("H2-S4", not _is_nan(mp) and not _is_nan(cat) and mp > cat,
              f"{tag}: mp={mp:.4f} vs cat={cat:.4f}")
    cr.record("H2-S5", not _is_nan(mp) and not _is_nan(triv) and mp > triv,
              f"{tag}: mp={mp:.4f} vs trivial={triv:.4f}")

    lo_raw = _get(res, "bootstrap_ci_95", "deltas", "spoof_delta", "lower", default=None)
    lo = None if lo_raw is None else float(lo_raw)
    cr.record("H2-S6", lo is not None and lo > 0,
              f"{tag}: spoof_delta CI lower={lo}")


def check_h2_cross(all_results: dict, cr: CheckResult) -> None:
    seeds = sorted(all_results)
    mp_aucs = [float(_get(all_results[s], "auc", "mean_pool", "spoof", default=float("nan")))
               for s in seeds]
    valid = [v for v in mp_aucs if not _is_nan(v)]
    if len(valid) >= 2:
        std = float(np.std(np.array(valid, dtype=float), ddof=0))
        cr.record("H2-X1", std < 0.02,
                  f"mp spoof AUC std={std:.4f} ({'OK' if std < 0.02 else 'EXCEEDS 0.02'})")

    degen_wf = [float(_get(all_results[s], "t8_token_similarity", "degenerate_config",
                           "within_feature_mean", default=float("nan"))) for s in seeds]
    valid_d = [v for v in degen_wf if not _is_nan(v)]
    if len(valid_d) >= 2:
        std_d = float(np.std(np.array(valid_d, dtype=float), ddof=0))
        cr.record("H2-X2", std_d < 0.01,
                  f"degen within_feature_mean std={std_d:.4f} "
                  f"({'OK' if std_d < 0.01 else 'EXCEEDS 0.01 — mechanism may not be fully deterministic'})")


# ---------------------------------------------------------------------------
# H6 checks (high-level; detailed checks in h6/check_consistency.py)
# ---------------------------------------------------------------------------

def check_h6_seed(res: dict, seed: int, cr: CheckResult) -> None:
    tag = f"seed {seed}"

    collapse = _get(res, "t8_token_similarity", "collapse_detected", default=True)
    wf = _get(res, "t8_token_similarity", "within_feature_mean", default=float("nan"))
    cr.record("H6-S1", not collapse,
              f"{tag}: within_feature_mean={wf:.4f} collapse={collapse}")

    verdicts = res.get("verdicts", {})
    all_present = all(k in verdicts for k in H6_VERDICT_KEYS)
    all_true    = all_present and all(verdicts[k] is True for k in H6_VERDICT_KEYS)
    cr.record("H6-S2", all_true,
              f"{tag}: verdicts={{{', '.join(f'{k}={verdicts.get(k)}' for k in H6_VERDICT_KEYS)}}}")

    d_agg = _get(res, "fleet_aggregate", "two_stage_vs_trivial_roc_delta", default=None)
    d_res = _get(res, "fleet_residual",  "two_stage_vs_trivial_roc_delta", default=None)
    ok = (d_agg is not None and d_agg < C3_DELTA_THRESHOLD and
          d_res is not None and d_res < C3_DELTA_THRESHOLD)
    cr.record("H6-S3", ok,
              f"{tag}: fleet_agg_delta={d_agg} fleet_res_delta={d_res}")


def check_h6_cross(all_results: dict, cr: CheckResult) -> None:
    seeds = sorted(all_results)
    stable = True
    detail_parts = []
    for key in H6_VERDICT_KEYS:
        vals = [_get(all_results[s], "verdicts", key) for s in seeds]
        unique = set(vals)
        if len(unique) > 1:
            stable = False
            detail_parts.append(f"{key}: INCONSISTENT {dict(zip(seeds, vals))}")
        else:
            detail_parts.append(f"{key}={vals[0]} ({len(seeds)}/{len(seeds)})")
    cr.record("H6-X1", stable,
              "; ".join(detail_parts) if not stable else f"all 3 verdicts stable across {len(seeds)} seeds")


# ---------------------------------------------------------------------------
# RBA checks
# ---------------------------------------------------------------------------

def check_rba_seed(res: dict, seed: int, cr: CheckResult) -> None:
    tag = f"seed {seed}"

    missing = [k for k in RBA_REQUIRED_KEYS if k not in res]
    cr.record("RBA-S1", not missing,
              f"{tag}: missing {missing}" if missing else f"{tag}: schema OK")

    collapse = _get(res, "t8_token_similarity", "collapse_detected", default=True)
    cr.record("RBA-S2", not collapse,
              f"{tag}: collapse_detected={collapse}")

    replicated = _get(res, "h2_replicated", default=None)
    cr.record("RBA-S3", replicated is True,
              f"{tag}: h2_replicated={replicated}")

    # RBA-S4 (mp > concat ROC-AUC) removed: n_ato_test_events=9 makes point-estimate
    # ordering underpowered (CI width ~0.30). Pre-registered claim is h2_replicated.


def check_rba_cross(all_results: dict, cr: CheckResult) -> None:
    seeds = sorted(all_results)
    vals = [_get(all_results[s], "h2_replicated") for s in seeds]
    unique = set(vals)
    cr.record("RBA-X1", len(unique) == 1,
              f"h2_replicated: {dict(zip(seeds, vals))}" if len(unique) > 1
              else f"h2_replicated={vals[0]} ({len(seeds)}/{len(seeds)} agree)")


# ---------------------------------------------------------------------------
# Cross-phase checks
# ---------------------------------------------------------------------------

def check_cross_phase(h2: dict, h6: dict, rba: dict, cr: CheckResult) -> None:
    s_h2  = set(h2)
    s_h6  = set(h6)
    s_rba = set(rba)
    all_same = s_h2 == s_h6 == s_rba
    cr.record("XP-1", all_same,
              f"H2={sorted(s_h2)} H6={sorted(s_h6)} RBA={sorted(s_rba)}" if not all_same
              else f"All phases: seeds {sorted(s_h2)}")

    any_collapse = False
    collapse_where = []
    for seed in sorted(s_h2 | s_h6 | s_rba):
        if h2.get(seed) and _get(h2[seed], "t8_token_similarity", "robust_config", "collapse_detected"):
            collapse_where.append(f"H2/seed_{seed}")
            any_collapse = True
        if h6.get(seed) and _get(h6[seed], "t8_token_similarity", "collapse_detected"):
            collapse_where.append(f"H6/seed_{seed}")
            any_collapse = True
        if rba.get(seed) and _get(rba[seed], "t8_token_similarity", "collapse_detected"):
            collapse_where.append(f"RBA/seed_{seed}")
            any_collapse = True
    cr.record("XP-2", not any_collapse,
              f"collapse in: {collapse_where}" if any_collapse else "no T8 collapse in any phase or seed")


# ---------------------------------------------------------------------------
# Smoke data
# ---------------------------------------------------------------------------

def _make_h2_smoke(seed: int) -> dict:
    return {
        "seed": seed, "timestamp": "2026-04-21T00:00:00Z",
        "config": {}, "silhouette": {}, "compactness": {}, "t2_window_sweep": {},
        "t5_tz_permutation": {}, "enrollment_diagnostics": {},
        "auc": {
            "mean_pool": {"spoof": 0.985, "novel": 0.990, "fleet": 0.975},
            "concat_w1": {"spoof": 0.920, "novel": 0.930, "fleet": 0.910},
            "trivial":   {"spoof": 0.500, "novel": 0.500, "fleet": 0.500},
        },
        "bootstrap_ci_95": {
            "deltas": {"spoof_delta": {"estimate": 0.065, "lower": 0.055, "upper": 0.075}},
        },
        "t7_trivial_baseline": {"mean_pool_spoof_margin": 0.485, "concat_w1_spoof_margin": 0.420},
        "t8_token_similarity": {
            "robust_config":    {"within_feature_mean": 0.563, "cross_feature_mean": 0.490,
                                  "within_cross_ratio": 1.149, "collapse_detected": False},
            "degenerate_config": {"within_feature_mean": 0.950, "cross_feature_mean": 0.940,
                                   "within_cross_ratio": 1.011, "collapse_detected": True},
        },
    }


def _make_h6_smoke(seed: int) -> dict:
    def sb(roc=0.95, pr=0.85): return {"roc_auc": roc, "pr_auc": pr}
    return {
        "seed": seed, "timestamp": "2026-04-21T00:00:00Z",
        "config": {}, "spoof_k1": {}, "spoof_k2": {}, "spoof_k3": {}, "novel": {},
        "fleet_aggregate": {
            **{k: sb(0.459) for k in ("two_stage", "trivial")}, "mp_raw": sb(0.994, 0.918),
            "two_stage_vs_trivial_roc_delta": 0.0,
        },
        "fleet_residual": {
            **{k: sb(0.457) for k in ("two_stage", "trivial")}, "mp_raw": sb(0.997, 0.947),
            "two_stage_vs_trivial_roc_delta": 0.0,
        },
        "t8_token_similarity": {
            "within_feature_mean": 0.392, "cross_feature_mean": 0.050,
            "within_cross_ratio": 7.84, "collapse_detected": False,
        },
        "verdicts": {
            "primary_criterion_confirmed":  True,
            "rank_norm_collapse_confirmed": True,
            "gate_blinds_fleet_confirmed":  True,
        },
    }


def _make_rba_smoke(seed: int) -> dict:
    return {
        "seed": seed, "timestamp": "2026-04-21T00:00:00Z",
        "auc": {
            "mean_pool": {"roc_auc": 0.850, "pr_auc": 0.032},
            "concat":    {"roc_auc": 0.820, "pr_auc": 0.025},
            "trivial":   {"roc_auc": 0.780, "pr_auc": 0.003},
        },
        "t6_compactness": {"mean_pool": 0.036, "concat": 0.047},
        "t8_token_similarity": {
            "within_feature_mean": 0.563, "cross_feature_mean": 0.490,
            "within_cross_ratio": 1.149, "collapse_detected": False,
        },
        "h2_replicated": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load(seeds_root: Path, phase: str) -> dict[int, dict]:
    out = {}
    for seed in ALL_SEEDS:
        path = seeds_root / f"seed_{seed}" / phase / "results.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if "seed" not in data:
                print(f"  WARNING: {path} missing 'seed' — skipped")
                continue
            out[seed] = data
        except json.JSONDecodeError as e:
            print(f"  WARNING: {path} invalid JSON ({e}) — skipped")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-phase consistency checks for H2, H6, and RBA rerun")
    parser.add_argument("--seeds-root", type=Path, default=SEEDS_ROOT)
    parser.add_argument("--smoke", action="store_true",
                        help="Run on synthetic data; exit 0 if all checks pass")
    args = parser.parse_args()

    print("=" * 60)
    print("Cross-Phase Consistency Checker")
    print("=" * 60)

    if args.smoke:
        smoke_seeds = [42, 123, 456]
        h2  = {s: _make_h2_smoke(s)  for s in smoke_seeds}
        h6  = {s: _make_h6_smoke(s)  for s in smoke_seeds}
        rba = {s: _make_rba_smoke(s) for s in smoke_seeds}
        print(f"\n[SMOKE] Synthetic seeds: {smoke_seeds}\n")
    else:
        print("\nLoading results ...")
        h2  = _load(args.seeds_root, "h2")
        h6  = _load(args.seeds_root, "h6")
        rba = _load(args.seeds_root, "rba")
        print(f"  H2:  {sorted(h2)} ({len(h2)} seeds)")
        print(f"  H6:  {sorted(h6)} ({len(h6)} seeds)")
        print(f"  RBA: {sorted(rba)} ({len(rba)} seeds)")

    if not any([h2, h6, rba]):
        print("\nNo seed results found — nothing to check yet.")
        sys.exit(0)

    max_n = max(len(h2), len(h6), len(rba))
    if max_n < 5 and not args.smoke:
        print(f"\nWARNING: only {max_n}/5 seeds available. Cross-seed stats have limited power.\n")

    cr = CheckResult()

    # H2
    if h2:
        print("\n── H2 ──")
        for seed, res in sorted(h2.items()):
            check_h2_seed(res, seed, cr)
        if len(h2) >= 2:
            check_h2_cross(h2, cr)

    # H6
    if h6:
        print("\n── H6 (high-level; run h6/check_consistency.py for full detail) ──")
        for seed, res in sorted(h6.items()):
            check_h6_seed(res, seed, cr)
        if len(h6) >= 2:
            check_h6_cross(h6, cr)

    # RBA
    if rba:
        print("\n── RBA ──")
        for seed, res in sorted(rba.items()):
            check_rba_seed(res, seed, cr)
        if len(rba) >= 2:
            check_rba_cross(rba, cr)

    # Cross-phase
    if h2 and h6 and rba:
        print("\n── Cross-phase ──")
        check_cross_phase(h2, h6, rba, cr)

    total = cr.n_pass + cr.n_fail
    print(f"\n{'='*60}")
    print(f"  {cr.n_pass}/{total} checks passed"
          + (f"  ({cr.n_fail} FAILED)" if cr.n_fail else "  — all OK"))
    if not args.smoke and max_n < 5:
        print("  NOTE: rerun with all 5 seeds for full cross-seed coverage")
    print("=" * 60)

    sys.exit(0 if cr.n_fail == 0 else 1)


if __name__ == "__main__":
    main()
