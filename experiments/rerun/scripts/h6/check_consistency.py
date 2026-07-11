# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
# ]
# ///
#
# H6 Consistency Checker
# ======================
# Reads all available seed results for H6 and runs per-seed schema/sanity
# checks plus cross-seed consistency checks. Intended to run after all 5
# seeds complete; works with any number of available seeds (1–5).
#
# Per-seed checks:
#   S1  Required top-level schema keys present
#   S2  Required verdict keys present
#   S3  fleet_aggregate and fleet_residual have two_stage_vs_trivial_roc_delta
#   S4  No NaN in mp_raw roc_auc / pr_auc for primary attack types
#   S5  T8 collapse_detected == False (robust config must not collapse)
#   S6  C3 delta: two_stage_vs_trivial_roc_delta < 1e-4 on both fleet blocks
#   S7  Baseline: mp_raw pr_auc > trivial pr_auc on spoof_k1
#   S8  Rank-norm visible: mp_raw pr_auc > mp_rank_norm pr_auc on spoof_k1
#
# Cross-seed checks (requires ≥2 seeds):
#   X1  Verdict stability: all 3 verdicts agree across every seed
#   X2  mp_raw spoof_k1 pr_auc cross-seed std < 0.05
#   X3  k=1 is hardest per seed: k1 < min(k2, k3) (mp_raw pr_auc; k2 vs k3 not claimed)
#
# Exit code: 0 if all checks pass (or no data yet), 1 if any check fails.
# Prints a WARNING (not failure) when fewer than 5 seeds are available.
#
# Usage:
#   uv run experiments/rerun/scripts/h6/check_consistency.py
#   uv run experiments/rerun/scripts/h6/check_consistency.py --smoke

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RERUN_ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS_ROOT = RERUN_ROOT / "seeds"
ALL_SEEDS  = [42, 123, 456, 789, 2024]

PRIMARY_ATTACKS   = ["spoof_k1", "novel", "fleet_aggregate", "fleet_residual"]
REQUIRED_TOP_KEYS = [
    "seed", "timestamp", "config", "t8_token_similarity", "verdicts",
    "spoof_k1", "spoof_k2", "spoof_k3", "novel", "fleet_aggregate", "fleet_residual",
]
REQUIRED_VERDICT_KEYS = [
    "primary_criterion_confirmed",
    "rank_norm_collapse_confirmed",
    "gate_blinds_fleet_confirmed",
]
C3_DELTA_THRESHOLD = 1e-4


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


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


class CheckResult:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def record(self, label: str, passed: bool, detail: str = "") -> bool:
        self.rows.append((label, passed, detail))
        status = "PASS" if passed else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"  {label:4s}  [{status}] {suffix}")
        return passed

    @property
    def n_pass(self) -> int:
        return sum(1 for _, p, _ in self.rows if p)

    @property
    def n_fail(self) -> int:
        return sum(1 for _, p, _ in self.rows if not p)


# ---------------------------------------------------------------------------
# Synthetic result for smoke testing
# ---------------------------------------------------------------------------

def _make_synthetic_result(seed: int = 42) -> dict:
    def sb(roc=0.95, pr=0.85, lo=0.80, hi=0.90):
        return {"roc_auc": roc, "pr_auc": pr, "ci_lower": lo, "ci_upper": hi}

    def frb(roc=0.97, pr=0.90, lo=0.85, hi=0.95):
        return {"roc_auc": roc, "pr_auc": pr, "ci_lower": lo, "ci_upper": hi,
                "top1pct_tp": 10, "top1pct_prec": 0.667}

    return {
        "seed": seed,
        "timestamp": "2026-04-21T00:00:00Z",
        "config": {"n_accounts": 5000, "n_fleet_accounts": 250,
                   "n_pre_lag": 100, "n_post_lag": 150},
        "spoof_k1": {
            "mp_raw":       sb(0.995, 0.810, 0.766, 0.854),
            "mp_rank_norm": sb(0.972, 0.200, 0.170, 0.230),
            "trivial":      sb(0.500, 0.010, 0.005, 0.015),
        },
        "spoof_k2": {
            "mp_raw":  sb(0.997, 0.900), "trivial": sb(0.500, 0.010),
        },
        "spoof_k3": {
            "mp_raw":  sb(0.999, 0.950), "trivial": sb(0.500, 0.010),
        },
        "novel": {
            "mp_raw":  sb(0.999, 0.965), "trivial": sb(0.500, 0.010),
        },
        "fleet_aggregate": {
            "mp_raw":    sb(0.994, 0.918),
            "two_stage": sb(0.459, 0.010),
            "trivial":   sb(0.459, 0.010),
            "two_stage_vs_trivial_roc_delta": 0.0,
        },
        "fleet_residual": {
            "mp_raw":    frb(0.997, 0.947),
            "two_stage": frb(0.457, 0.010),
            "trivial":   frb(0.457, 0.010),
            "two_stage_vs_trivial_roc_delta": 0.0,
        },
        "t8_token_similarity": {
            "within_feature_mean": 0.392,
            "cross_feature_mean":  0.050,
            "within_cross_ratio":  7.84,
            "collapse_detected":   False,
        },
        "verdicts": {
            "primary_criterion_confirmed":  True,
            "rank_norm_collapse_confirmed": True,
            "gate_blinds_fleet_confirmed":  True,
        },
    }


# ---------------------------------------------------------------------------
# Per-seed checks
# ---------------------------------------------------------------------------

def check_seed(res: dict, seed: int, cr: CheckResult) -> None:
    prefix = f"seed {seed}"

    # S1 — required top-level keys
    missing = [k for k in REQUIRED_TOP_KEYS if k not in res]
    cr.record("S1", not missing,
              f"{prefix}: missing keys {missing}" if missing else f"{prefix}: schema OK")

    # S2 — required verdict keys
    verdicts = res.get("verdicts", {})
    missing_v = [k for k in REQUIRED_VERDICT_KEYS if k not in verdicts]
    cr.record("S2", not missing_v,
              f"{prefix}: missing verdict keys {missing_v}" if missing_v
              else f"{prefix}: verdicts schema OK")

    # S3 — C3 delta field present in both fleet blocks
    has_agg = "two_stage_vs_trivial_roc_delta" in res.get("fleet_aggregate", {})
    has_res = "two_stage_vs_trivial_roc_delta" in res.get("fleet_residual", {})
    cr.record("S3", has_agg and has_res,
              f"{prefix}: delta field present fleet_agg={has_agg} fleet_res={has_res}")

    # S4 — no NaN in mp_raw roc_auc / pr_auc for primary attacks
    nan_fields = []
    for atk in PRIMARY_ATTACKS:
        for field in ("roc_auc", "pr_auc"):
            v = _get(res, atk, "mp_raw", field)
            if v is None or _is_nan(v):
                nan_fields.append(f"{atk}.mp_raw.{field}")
    cr.record("S4", not nan_fields,
              f"{prefix}: NaN in {nan_fields}" if nan_fields else f"{prefix}: no NaN")

    # S5 — T8 no collapse
    collapse = _get(res, "t8_token_similarity", "collapse_detected", default=True)
    wf_mean  = _get(res, "t8_token_similarity", "within_feature_mean", default=float("nan"))
    cr.record("S5", not collapse,
              f"{prefix}: within_feature_mean={wf_mean:.4f} collapse={collapse}")

    # S6 — C3 delta < 1e-4
    d_agg = _get(res, "fleet_aggregate", "two_stage_vs_trivial_roc_delta", default=None)
    d_res = _get(res, "fleet_residual",  "two_stage_vs_trivial_roc_delta", default=None)
    agg_ok = isinstance(d_agg, (int, float)) and float(d_agg) < C3_DELTA_THRESHOLD
    res_ok = isinstance(d_res, (int, float)) and float(d_res) < C3_DELTA_THRESHOLD
    cr.record("S6", agg_ok and res_ok,
              f"{prefix}: fleet_agg_delta={d_agg} fleet_res_delta={d_res} "
              f"(threshold={C3_DELTA_THRESHOLD})")

    # S7 — baseline: mp_raw pr_auc > trivial pr_auc on spoof_k1
    mp_pr   = _get(res, "spoof_k1", "mp_raw",  "pr_auc", default=float("nan"))
    triv_pr = _get(res, "spoof_k1", "trivial", "pr_auc", default=float("nan"))
    baseline_ok = (isinstance(mp_pr, (int, float)) and isinstance(triv_pr, (int, float))
                   and not _is_nan(mp_pr) and not _is_nan(triv_pr)
                   and float(mp_pr) > float(triv_pr))
    cr.record("S7", baseline_ok,
              f"{prefix}: spoof_k1 mp_raw PR={mp_pr:.4f} > trivial PR={triv_pr:.4f}")

    # S8 — rank-norm visible: mp_raw pr_auc > mp_rank_norm pr_auc on spoof_k1
    rn_pr = _get(res, "spoof_k1", "mp_rank_norm", "pr_auc", default=None)
    if rn_pr is None:
        cr.record("S8", False, f"{prefix}: mp_rank_norm missing from spoof_k1")
    else:
        rn_ok = (isinstance(mp_pr, (int, float)) and isinstance(rn_pr, (int, float))
                 and not _is_nan(mp_pr) and not _is_nan(rn_pr)
                 and float(mp_pr) > float(rn_pr))
        cr.record("S8", rn_ok,
                  f"{prefix}: mp_raw PR={mp_pr:.4f} > mp_rank_norm PR={rn_pr:.4f}")

    # X3 (per-seed component) — k=1 is the hardest attack (k1 < min(k2, k3))
    # k2 vs k3 ordering is not claimed: both saturate above 0.94 PR-AUC and
    # the inversion is within bootstrap noise at that level.
    k1_pr = _get(res, "spoof_k1", "mp_raw", "pr_auc", default=float("nan"))
    k2_pr = _get(res, "spoof_k2", "mp_raw", "pr_auc", default=float("nan"))
    k3_pr = _get(res, "spoof_k3", "mp_raw", "pr_auc", default=float("nan"))
    ks_typed = [float(v) for v in (k1_pr, k2_pr, k3_pr)
                if isinstance(v, (int, float)) and not _is_nan(v)]
    mono = len(ks_typed) == 3 and ks_typed[0] < min(ks_typed[1], ks_typed[2])
    cr.record("X3", mono,
              f"{prefix}: spoof k gradient k1={k1_pr:.4f} k2={k2_pr:.4f} k3={k3_pr:.4f}")


# ---------------------------------------------------------------------------
# Cross-seed checks
# ---------------------------------------------------------------------------

def check_cross_seed(all_results: dict, cr: CheckResult) -> None:
    seeds = sorted(all_results)
    n = len(seeds)

    # X1 — verdict stability
    stability_ok = True
    stability_detail = []
    for key in REQUIRED_VERDICT_KEYS:
        vals = [_get(all_results[s], "verdicts", key) for s in seeds]
        unique = set(vals)
        if len(unique) > 1:
            stability_ok = False
            stability_detail.append(f"{key}: {dict(zip(seeds, vals))}")
        else:
            stability_detail.append(f"{key}: {vals[0]} ({n}/{n} agree)")
    cr.record("X1", stability_ok,
              "; ".join(stability_detail) if not stability_ok
              else f"all 3 verdicts consistent across {n} seeds")

    # X2 — mp_raw spoof_k1 pr_auc std < 0.05
    k1_prs = [_get(all_results[s], "spoof_k1", "mp_raw", "pr_auc", default=float("nan"))
              for s in seeds]
    valid  = [float(v) for v in k1_prs
              if isinstance(v, (int, float)) and not _is_nan(v)]
    if len(valid) >= 2:
        std = float(np.std(valid, ddof=0))
        mean = float(np.mean(valid))
        std_ok = std < 0.05
        cr.record("X2", std_ok,
                  f"spoof_k1 mp_raw pr_auc: mean={mean:.4f} std={std:.4f} "
                  f"({'OK' if std_ok else 'EXCEEDS 0.05 threshold'})")
    else:
        cr.record("X2", True, f"only {len(valid)} valid values — skipped (need ≥2)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H6 cross-seed consistency checks")
    p.add_argument("--seeds-root", type=Path, default=SEEDS_ROOT)
    p.add_argument("--smoke", action="store_true",
                   help="Run against synthetic data; exit 0 if all checks pass")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("H6 Consistency Checker")
    print("=" * 60)

    # Load results
    if args.smoke:
        all_results = {42: _make_synthetic_result(42)}
        print("\n[SMOKE] Using synthetic seed-42 result\n")
    else:
        all_results = {}
        for seed in ALL_SEEDS:
            p = args.seeds_root / f"seed_{seed}" / "h6" / "results.json"
            if p.exists():
                with open(p) as f:
                    all_results[seed] = json.load(f)

    if not all_results:
        print("\nNo seed results found — nothing to check yet.")
        print(f"Expected: {args.seeds_root}/seed_{{42,...}}/h6/results.json")
        sys.exit(0)

    n = len(all_results)
    if n < 5 and not args.smoke:
        print(f"\nWARNING: Only {n}/5 seeds available. "
              f"Cross-seed stats will have limited power.\n")

    cr = CheckResult()

    # Per-seed checks
    for seed, res in sorted(all_results.items()):
        print(f"\n── Seed {seed} ──")
        check_seed(res, seed, cr)

    # Cross-seed checks
    if len(all_results) >= 2:
        print("\n── Cross-seed ──")
        check_cross_seed(all_results, cr)
    else:
        print("\n── Cross-seed ── (skipped — need ≥2 seeds)")

    # Summary
    total = cr.n_pass + cr.n_fail
    print(f"\n{'='*60}")
    print(f"  {cr.n_pass}/{total} checks passed"
          + (f"  ({cr.n_fail} FAILED)" if cr.n_fail else "  — all OK"))
    if n < 5 and not args.smoke:
        print(f"  NOTE: rerun with all 5 seeds for full verdict stability report")
    print("=" * 60)

    sys.exit(0 if cr.n_fail == 0 else 1)


if __name__ == "__main__":
    main()
