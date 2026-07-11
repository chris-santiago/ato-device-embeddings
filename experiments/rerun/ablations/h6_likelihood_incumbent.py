# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim==4.4.0",
#   "numpy==2.2.6",
#   "scikit-learn==1.7.2",
#   "matplotlib==3.10.8",
# ]
# ///
#
# A2b — Likelihood Incumbent at 1:100 (H6 open-vocab, C3 comparison)
# ==================================================================
# Pre-specified in plan/12-CRITIQUE_ABLATIONS.md (A2b section). Tests whether the
# Freeman-style smoothed per-feature likelihood scorer (a) holds its phase-1
# advantage over the embedding on the realistic spoof gradient at 1:100, and
# (b) detects fleet_residual contamination that the binary gate suppresses.
#
# Parity rule: likelihood counts use centroid_events (train[:40]) so both the
# centroid and the likelihood tables see fleet contamination iff the injected
# event lands in the first 40 training events.
#
# Reuse policy: drives scripts/h6/h6_rerun.py as a library — marginals, account
# generation, corpus/training, mp/trivial/two-stage scoring, top-1% metrics,
# and bootstrap CI are its unmodified functions.
#
# Usage:
#   uv run experiments/rerun/ablations/h6_likelihood_incumbent.py --smoke
#   uv run experiments/rerun/ablations/h6_likelihood_incumbent.py --seed 42

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

H6_DIR = Path(__file__).resolve().parent.parent / "scripts" / "h6"
sys.path.insert(0, str(H6_DIR))
import h6_rerun as H6  # noqa: E402

LIK_ALPHA = 1.0
LIK_LAMBDAS = [1.0, 0.9, 0.5]
NEG_RATIO = 100
POPULATIONS = ["spoof_k1", "spoof_k2", "spoof_k3", "novel", "fleet_residual"]
OUT_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Likelihood scorer
# ---------------------------------------------------------------------------
def build_global_tables(accounts):
    """Pooled counts and per-feature vocab sizes over all centroid windows."""
    counts = {f: Counter() for f in H6.FEATURE_ORDER}
    n = 0
    for acc in accounts:
        for e in acc["centroid_events"]:
            for f in H6.FEATURE_ORDER:
                counts[f][e.get(f, "unknown")] += 1
            n += 1
    vocab_sizes = {f: len(counts[f]) for f in H6.FEATURE_ORDER}
    return counts, n, vocab_sizes


def account_tables(acc):
    counts = {
        f: Counter(e.get(f, "unknown") for e in acc["centroid_events"])
        for f in H6.FEATURE_ORDER
    }
    return counts, len(acc["centroid_events"])


def score_lik(event, acct_counts, n_acct, lam, glob_counts, glob_n, vocab_sizes):
    nll = 0.0
    for f in H6.FEATURE_ORDER:
        v = event.get(f, "unknown")
        v_size = vocab_sizes[f]
        p_acct = (acct_counts[f].get(v, 0) + LIK_ALPHA) / (n_acct + LIK_ALPHA * v_size)
        p_glob = (glob_counts[f].get(v, 0) + LIK_ALPHA) / (glob_n + LIK_ALPHA * v_size)
        nll += -math.log(lam * p_acct + (1.0 - lam) * p_glob)
    return nll


# ---------------------------------------------------------------------------
# Score collection (mirrors h6_rerun.collect_* population definitions)
# ---------------------------------------------------------------------------
def collect_population(accounts, centroids, model, population, glob):
    glob_counts, glob_n, vocab_sizes = glob
    scorers = (["mp_raw", "trivial", "two_stage"]
               + [f"lik_lam{lam}" for lam in LIK_LAMBDAS]
               + [f"lik_lam{lam}_two_stage" for lam in LIK_LAMBDAS])
    scores: dict = {s: ([], []) for s in scorers}
    for acc, centroid in zip(accounts, centroids):
        if population == "fleet_residual":
            if not acc["has_fleet"] or acc.get("fleet_blocklist_active", False):
                continue
            atk_events = acc["fleet_atk"]
        elif population == "novel":
            atk_events = acc["novel_atk"]
        else:
            atk_events = acc[population]
        kd = acc["known_devices"]
        acct_counts, n_acct = account_tables(acc)
        pairs = [(e, 0) for e in acc["enroll_neg"]] + [(e, 1) for e in atk_events]
        for e, lbl in pairs:
            gated = H6.device_key(e) in kd
            scores["mp_raw"][0].append(H6.score_single_stage(e, centroid, model))
            scores["trivial"][0].append(H6.score_trivial(e, kd))
            scores["two_stage"][0].append(H6.score_two_stage(e, centroid, kd, model))
            for lam in LIK_LAMBDAS:
                nll = score_lik(e, acct_counts, n_acct, lam, glob_counts, glob_n, vocab_sizes)
                scores[f"lik_lam{lam}"][0].append(nll)
                scores[f"lik_lam{lam}_two_stage"][0].append(0.0 if gated else nll)
            for s in scorers:
                scores[s][1].append(lbl)
    return {s: (np.array(v[0]), np.array(v[1])) for s, v in scores.items()}


# ---------------------------------------------------------------------------
# Paired bootstrap delta on score arrays (identical resample indices)
# ---------------------------------------------------------------------------
def paired_delta(scores_a, scores_b, labels, n_boot, rng):
    point_roc = roc_auc_score(labels, scores_a) - roc_auc_score(labels, scores_b)
    point_pr = (average_precision_score(labels, scores_a)
                - average_precision_score(labels, scores_b))
    idx = np.arange(len(labels))
    roc_d, pr_d = [], []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        sl = labels[sample]
        if sl.sum() == 0 or sl.sum() == len(sl):
            continue
        sa, sb = scores_a[sample], scores_b[sample]
        roc_d.append(roc_auc_score(sl, sa) - roc_auc_score(sl, sb))
        pr_d.append(average_precision_score(sl, sa) - average_precision_score(sl, sb))
    return {
        "roc": [float(point_roc), float(np.percentile(roc_d, 2.5)), float(np.percentile(roc_d, 97.5))],
        "pr": [float(point_pr), float(np.percentile(pr_d, 2.5)), float(np.percentile(pr_d, 97.5))],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(seed: int, smoke: bool) -> dict:
    n_accounts = H6.N_ACCOUNTS_SMOKE if smoke else H6.N_ACCOUNTS_FULL
    n_boot = 200 if smoke else H6.N_BOOTSTRAP
    n_enroll_neg = H6.N_ENROLL_NEG * NEG_RATIO

    print(f"[1/4] Generating H6 accounts ({n_accounts}, 1:{NEG_RATIO}, seed={seed})...")
    marginals = H6.Marginals(H6.MARGINALS_PATH)
    rng = np.random.default_rng(seed)
    accounts, fleet_info = H6.generate_accounts(marginals, n_accounts, rng,
                                                n_enroll_neg=n_enroll_neg)
    print(f"  fleet accounts: {fleet_info['n_fleet_accounts']} "
          f"(pre-lag {fleet_info['n_pre_lag']}, post-lag {fleet_info['n_post_lag']})")

    print(f"[2/4] Training mp model + likelihood tables (seed={seed})...")
    model = H6.train_model(H6.build_corpus(accounts), seed)
    centroids = H6.compute_centroids(accounts, model)
    glob = build_global_tables(accounts)

    print("[3/4] Scoring populations...")
    metrics, raw_scores = {}, {}
    boot_rng = np.random.default_rng(seed)
    for pop in POPULATIONS:
        collected = collect_population(accounts, centroids, model, pop, glob)
        raw_scores[pop] = collected
        metrics[pop] = {}
        for name, (sc, lb) in collected.items():
            if lb.sum() == 0:
                metrics[pop][name] = None
                continue
            roc, roc_lo, roc_hi, pr, pr_lo, pr_hi = H6.bootstrap_auc(sc, lb, n_boot, boot_rng)
            top1 = H6.top_pct_metrics(sc, lb, 0.01, seed)
            metrics[pop][name] = {
                "roc_auc": float(roc), "roc_ci": [float(roc_lo), float(roc_hi)],
                "pr_auc": float(pr), "pr_ci": [float(pr_lo), float(pr_hi)],
                "top1pct_tp": top1["tp"], "top1pct_precision": top1["precision"],
            }
        shown = ["mp_raw", "trivial", "two_stage"] + [f"lik_lam{lam}" for lam in LIK_LAMBDAS]
        line = "  ".join(
            f"{n}={metrics[pop][n]['roc_auc']:.3f}" for n in shown if metrics[pop].get(n)
        )
        print(f"  {pop:<15} {line}")

    # Adverse rule (plan 12 A2b): best variant per seed by spoof_k1 ROC point.
    best_lam = max(LIK_LAMBDAS,
                   key=lambda lam: metrics["spoof_k1"][f"lik_lam{lam}"]["roc_auc"])
    best_name = f"lik_lam{best_lam}"
    print(f"  best likelihood variant by spoof_k1 ROC: {best_name}")

    print(f"[4/4] Paired delta mp_raw − {best_name} on spoof_k1 (n_boot={n_boot})...")
    sk1 = raw_scores["spoof_k1"]
    delta = paired_delta(sk1["mp_raw"][0], sk1[best_name][0], sk1["mp_raw"][1],
                         n_boot, np.random.default_rng(seed))
    fr_best = metrics["fleet_residual"][best_name]
    verdicts = {
        "a2b_lik_detects_fleet_residual": bool(fr_best["top1pct_tp"] > 0),
        "a2b_mp_beats_lik_spoofk1_roc": bool(delta["roc"][1] > 0),
        "a2b_mp_beats_lik_spoofk1_pr": bool(delta["pr"][1] > 0),
    }

    print("\n--- fleet_residual top-1% ---")
    for name in ["mp_raw", "two_stage", "trivial", best_name, f"{best_name}_two_stage"]:
        m = metrics["fleet_residual"][name]
        print(f"  {name:<22} TP={m['top1pct_tp']:>4}  prec={m['top1pct_precision']:.3f}  "
              f"ROC={m['roc_auc']:.3f}")
    print(f"  spoof_k1 delta (mp−{best_name}): "
          f"dROC={delta['roc'][0]:+.4f} [{delta['roc'][1]:+.4f}, {delta['roc'][2]:+.4f}]  "
          f"dPR={delta['pr'][0]:+.4f} [{delta['pr'][1]:+.4f}, {delta['pr'][2]:+.4f}]")
    print(f"  verdicts: {verdicts}")

    return {
        "schema_version": 1,
        "script": "h6_likelihood_incumbent.py",
        "plan": "plan/12-CRITIQUE_ABLATIONS.md (A2b)",
        "seed": seed,
        "smoke": smoke,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_accounts": n_accounts,
        "n_bootstrap": n_boot,
        "neg_ratio": NEG_RATIO,
        "fleet_info": {k: fleet_info[k] for k in
                       ["n_fleet_accounts", "n_pre_lag", "n_post_lag"]},
        "config": {"lik_alpha": LIK_ALPHA, "lik_lambdas": LIK_LAMBDAS,
                   "count_window": "centroid_events (train[:40])"},
        "metrics": metrics,
        "spoof_k1_delta_mp_minus_lik_best": delta,
        "best_likelihood_variant": best_name,
        "verdicts": verdicts,
    }


def main():
    p = argparse.ArgumentParser(description="A2b likelihood incumbent at 1:100 (plan 12)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="50 accounts, 200 bootstraps, schema assertions, no file written")
    args = p.parse_args()

    results = run(args.seed, args.smoke)

    if args.smoke:
        assert set(results["metrics"]) == set(POPULATIONS), "smoke: population keys"
        for pop in POPULATIONS:
            m = results["metrics"][pop]["mp_raw"]
            assert m and not math.isnan(m["roc_auc"]), f"smoke: NaN mp_raw ROC {pop}"
        assert len(results["verdicts"]) == 3
        assert all(not math.isnan(x) for x in
                   results["spoof_k1_delta_mp_minus_lik_best"]["roc"]), "smoke: NaN delta"
        print("\nSMOKE PASS")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"h6_likelihood_seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
