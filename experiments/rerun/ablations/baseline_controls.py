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
# A1 + A2a — Frozen-Random-Embedding Control and Likelihood Incumbent (C1)
# ========================================================================
# Pre-specified in plan/12-CRITIQUE_ABLATIONS.md. Two existential controls for
# the C1 claim, sharing one dataset and one trained model per seed:
#
#   A1  mp_random:  frozen iid N(0,1) per-token vectors, no training, same
#       mean-pool + centroid-cosine scorer. Tests whether trained FastText adds
#       anything beyond set-overlap geometry.
#   A2a lik_*:      smoothed per-account per-feature categorical likelihood
#       (Freeman-style incumbent), lambda in {1.0, 0.9, 0.5}. The comparison
#       uses the BEST variant per seed by spoof ROC (adverse to the embedding).
#
# Reuse policy: drives scripts/h2/h2_rerun.py as a library — data generation,
# robust-config training, scoring, bootstrap CIs, and paired deltas are all its
# unmodified functions.
#
# Usage:
#   uv run experiments/rerun/ablations/baseline_controls.py --smoke
#   uv run experiments/rerun/ablations/baseline_controls.py --seed 42

import argparse
import json
import math
import sys
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

H2_DIR = Path(__file__).resolve().parent.parent / "scripts" / "h2"
sys.path.insert(0, str(H2_DIR))
import h2_rerun as H  # noqa: E402

VECTOR_DIM = 64
LIK_ALPHA = 1.0
LIK_LAMBDAS = [1.0, 0.9, 0.5]
ATTACKS = ["novel", "fleet", "spoof"]
OUT_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# A1 — frozen random embeddings
# ---------------------------------------------------------------------------
def build_random_token_table(seed: int) -> dict:
    """One frozen N(0,1) vector per closed-vocab feature token.

    Tokens are sorted so the (token -> vector) assignment depends only on the
    seed, not on dict iteration order.
    """
    tokens = sorted(H.make_token(f, v) for f in H.FEATURE_ORDER for v in H.FEATURES[f])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xA1]))
    return {t: rng.standard_normal(VECTOR_DIM) for t in tokens}


def embed_mp_random(event, table):
    return np.mean([table[H.make_token(f, event[f])] for f in H.FEATURE_ORDER], axis=0)


class RandomConcatTable:
    """Frozen random vector per concatenated device string (random device-hash
    control). Deterministic across runs: derived from crc32 of the string, not
    Python's salted hash()."""

    def __init__(self, seed: int):
        self.seed = seed
        self._cache: dict = {}

    def vector(self, s: str) -> np.ndarray:
        if s not in self._cache:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, 0xC47, zlib.crc32(s.encode())])
            )
            self._cache[s] = rng.standard_normal(VECTOR_DIM)
        return self._cache[s]


def embed_cat_random(event, table: RandomConcatTable):
    return table.vector(H.device_to_concat(event))


# ---------------------------------------------------------------------------
# A2a — per-feature likelihood incumbent
# ---------------------------------------------------------------------------
def global_feature_counts(accounts):
    counts = {f: Counter() for f in H.FEATURE_ORDER}
    n = 0
    for acc in accounts:
        for e in acc["train_events"]:
            for f in H.FEATURE_ORDER:
                counts[f][e[f]] += 1
            n += 1
    return counts, n


def score_all_likelihood(accounts, lam, alpha, glob_counts, glob_n):
    """NLL scorer in the same result-dict shape as h2_rerun.score_all, so the
    paired bootstrap delta functions apply unchanged."""
    result = {k: {"scores": [], "labels": []} for k in ["novel", "fleet", "spoof", "neg", "known"]}
    for acc in accounts:
        acct_counts = {f: Counter(e[f] for e in acc["train_events"]) for f in H.FEATURE_ORDER}
        n_acct = len(acc["train_events"])
        for at, (ev, lbl) in acc["test"].items():
            nll = 0.0
            for f in H.FEATURE_ORDER:
                v_size = len(H.FEATURES[f])
                p_acct = (acct_counts[f].get(ev[f], 0) + alpha) / (n_acct + alpha * v_size)
                p_glob = (glob_counts[f].get(ev[f], 0) + alpha) / (glob_n + alpha * v_size)
                nll += -math.log(lam * p_acct + (1.0 - lam) * p_glob)
            result[at]["scores"].append(nll)
            result[at]["labels"].append(lbl)
    return result


# ---------------------------------------------------------------------------
# Metrics assembly
# ---------------------------------------------------------------------------
def scorer_metrics(result, n_boot, seed):
    out = {}
    for at in ATTACKS:
        m, lo, hi = H.bootstrap_auc(result, at, n_boot, seed)
        pm, plo, phi = H.bootstrap_pr_auc(result, at, n_boot, seed)
        out[at] = {
            "roc_auc": H.compute_auc(result, at),
            "pr_auc": H.compute_pr_auc(result, at),
            "roc_ci": [m, lo, hi],
            "pr_ci": [pm, plo, phi],
        }
    return out


def paired_deltas(base_result, other_result, n_boot, seed):
    out = {}
    for at in ATTACKS:
        r_pt, r_lo, r_hi = H.bootstrap_auc_delta(base_result, other_result, at, n_boot, seed)
        p_pt, p_lo, p_hi = H.bootstrap_pr_auc_delta(base_result, other_result, at, n_boot, seed)
        out[at] = {"roc": [r_pt, r_lo, r_hi], "pr": [p_pt, p_lo, p_hi]}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(seed: int, smoke: bool) -> dict:
    n_accounts = H.N_ACCOUNTS_SMOKE if smoke else H.N_ACCOUNTS
    n_boot = H.N_BOOTSTRAP_SMOKE if smoke else H.N_BOOTSTRAP

    print(f"[1/5] Generating data ({n_accounts} accounts, seed={seed})...")
    accounts = H.generate_dataset(seed, n_accounts=n_accounts)
    trivial_roc, trivial_pr = H.evaluate_trivial(accounts)

    print(f"[2/5] Training robust mean-pool model (seed={seed})...")
    mp_model = H.train_robust_mp(accounts, seed)
    mp_result = H.score_all(accounts, H.embed_mp, mp_model)

    print("[3/5] Scoring frozen-random controls (A1)...")
    rand_table = build_random_token_table(seed)
    rand_result = H.score_all(accounts, embed_mp_random, rand_table)
    cat_rand_result = H.score_all(accounts, embed_cat_random, RandomConcatTable(seed))

    print("[4/5] Scoring likelihood incumbent (A2a)...")
    glob_counts, glob_n = global_feature_counts(accounts)
    lik_results = {
        f"lik_lam{lam}": score_all_likelihood(accounts, lam, LIK_ALPHA, glob_counts, glob_n)
        for lam in LIK_LAMBDAS
    }
    # Adverse selection rule (plan 12): best variant per seed by spoof ROC point.
    best_name = max(lik_results, key=lambda k: H.compute_auc(lik_results[k], "spoof"))
    print(f"  best likelihood variant by spoof ROC: {best_name}")

    print(f"[5/5] Bootstrap CIs and paired deltas (n_boot={n_boot})...")
    scorers = {
        "mp_trained": scorer_metrics(mp_result, n_boot, seed),
        "mp_random": scorer_metrics(rand_result, n_boot, seed),
        "cat_random": scorer_metrics(cat_rand_result, n_boot, seed),
        **{name: scorer_metrics(res, n_boot, seed) for name, res in lik_results.items()},
        "trivial": {
            at: {"roc_auc": trivial_roc[at], "pr_auc": trivial_pr[at]} for at in ATTACKS
        },
    }
    deltas = {
        "mp_minus_random": paired_deltas(mp_result, rand_result, n_boot, seed),
        "mp_minus_lik_best": paired_deltas(mp_result, lik_results[best_name], n_boot, seed),
    }

    verdicts = {
        "a1_trained_beats_random_roc": bool(deltas["mp_minus_random"]["spoof"]["roc"][1] > 0),
        "a1_trained_beats_random_pr": bool(deltas["mp_minus_random"]["spoof"]["pr"][1] > 0),
        "a2_mp_beats_likelihood_roc": bool(deltas["mp_minus_lik_best"]["spoof"]["roc"][1] > 0),
        "a2_mp_beats_likelihood_pr": bool(deltas["mp_minus_lik_best"]["spoof"]["pr"][1] > 0),
    }
    descriptive = {
        "random_minus_trivial_spoof_roc": scorers["mp_random"]["spoof"]["roc_auc"] - trivial_roc["spoof"],
        "best_likelihood_variant": best_name,
    }

    results = {
        "schema_version": 1,
        "script": "baseline_controls.py",
        "plan": "plan/12-CRITIQUE_ABLATIONS.md",
        "seed": seed,
        "smoke": smoke,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_accounts": n_accounts,
        "n_bootstrap": n_boot,
        "config": {
            "vector_dim": VECTOR_DIM,
            "lik_alpha": LIK_ALPHA,
            "lik_lambdas": LIK_LAMBDAS,
        },
        "scorers": scorers,
        "deltas": deltas,
        "verdicts": verdicts,
        "descriptive": descriptive,
    }

    print("\n--- Spoof summary ---")
    for name in ["mp_trained", "mp_random", "cat_random", best_name, "trivial"]:
        s = scorers[name]["spoof"]
        print(f"  {name:<12} ROC={s['roc_auc']:.4f}  PR={s['pr_auc']:.4f}")
    for dname, d in deltas.items():
        r = d["spoof"]["roc"]
        p = d["spoof"]["pr"]
        print(f"  {dname:<22} dROC={r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]  "
              f"dPR={p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}]")
    print(f"  verdicts: {verdicts}")
    return results


def main():
    p = argparse.ArgumentParser(description="A1 + A2a baseline controls (plan 12)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="50 accounts, 200 bootstraps, schema assertions, no file written")
    args = p.parse_args()

    results = run(args.seed, args.smoke)

    if args.smoke:
        for name, m in results["scorers"].items():
            for at in ATTACKS:
                assert not math.isnan(m[at]["roc_auc"]), f"smoke: NaN ROC for {name}/{at}"
                assert not math.isnan(m[at]["pr_auc"]), f"smoke: NaN PR for {name}/{at}"
        for dname, d in results["deltas"].items():
            for at in ATTACKS:
                assert all(not math.isnan(x) for x in d[at]["roc"]), f"smoke: NaN delta {dname}/{at}"
        assert set(results["verdicts"]) == {
            "a1_trained_beats_random_roc", "a1_trained_beats_random_pr",
            "a2_mp_beats_likelihood_roc", "a2_mp_beats_likelihood_pr",
        }
        print("\nSMOKE PASS")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"baseline_controls_seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
