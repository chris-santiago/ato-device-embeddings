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
# A4 — No-Subword Cell (C1 mechanism isolation)
# =============================================
# Pre-specified in plan/12-CRITIQUE_ABLATIONS.md (A4 section). Retrains the
# mean-pool and concat encoders with max_n=0 (subwords disabled; identical
# FastText class, one knob changed) to isolate what character n-grams
# contribute to C1:
#   - mp_nosub vs mp_trained: do subwords help or hurt per-feature mean-pool?
#   - mp_nosub vs mp_random:  does training alone (no subword confound) beat
#     frozen random vectors?
#   - cat_nosub: OOV rate per population — concat cannot score unseen device
#     strings without subwords (OOV scores distance 1.0, recorded).
#
# Reuse policy: drives scripts/h2/h2_rerun.py as a library; random table
# construction is imported from baseline_controls.py (A1) so the mp_random
# scorer is bit-identical across ablations.
#
# Usage:
#   uv run experiments/rerun/ablations/a4_nosub.py --smoke
#   uv run experiments/rerun/ablations/a4_nosub.py --seed 42

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gensim.models import FastText

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "scripts" / "h2"))
sys.path.insert(0, str(_HERE))
import h2_rerun as H  # noqa: E402
from baseline_controls import (  # noqa: E402
    ATTACKS,
    build_random_token_table,
    embed_mp_random,
    paired_deltas,
    scorer_metrics,
)

OUT_DIR = _HERE / "results"


def _nosub_kwargs(seed):
    kwargs = H._robust_kwargs(seed)
    kwargs["min_n"] = 1
    kwargs["max_n"] = 0  # max_n < min_n disables the n-gram bucket entirely
    return kwargs


def train_mp_nosub(accounts, seed):
    return FastText(sentences=H.build_mp_corpus_per_account(accounts), **_nosub_kwargs(seed))


def train_cat_nosub(accounts, seed):
    return FastText(sentences=H.build_cat_corpus_per_account(accounts), **_nosub_kwargs(seed))


class CatNosubScorer:
    """Concat embedding without subwords: OOV device strings cannot be embedded
    and score distance 1.0 (plan 12 A4). Tracks the OOV rate."""

    def __init__(self, model):
        self.model = model
        self.n_oov = 0
        self.n_total = 0

    def embed(self, event):
        key = H.device_to_concat(event)
        self.n_total += 1
        if key in self.model.wv.key_to_index:
            return self.model.wv[key]
        self.n_oov += 1
        return None


def score_all_cat_nosub(accounts, scorer):
    """score_all equivalent with OOV-aware concat scoring. The centroid uses
    training events only (all in-vocab by construction)."""
    result = {k: {"scores": [], "labels": []} for k in ["novel", "fleet", "spoof", "neg", "known"]}
    for acc in accounts:
        centroid = np.mean([scorer.embed(e) for e in acc["train_events"]], axis=0)
        for at, (ev, lbl) in acc["test"].items():
            vec = scorer.embed(ev)
            dist = 1.0 if vec is None else H.cosine_dist(vec, centroid)
            result[at]["scores"].append(dist)
            result[at]["labels"].append(lbl)
    return result


def oov_rates(accounts, model):
    """Per-population fraction of test events whose concat string is OOV."""
    rates = {}
    for at in ["novel", "fleet", "spoof", "neg", "known"]:
        n_oov, n = 0, 0
        for acc in accounts:
            ev, _ = acc["test"][at]
            n += 1
            if H.device_to_concat(ev) not in model.wv.key_to_index:
                n_oov += 1
        rates[at] = n_oov / n
    return rates


def run(seed: int, smoke: bool) -> dict:
    n_accounts = H.N_ACCOUNTS_SMOKE if smoke else H.N_ACCOUNTS
    n_boot = H.N_BOOTSTRAP_SMOKE if smoke else H.N_BOOTSTRAP

    print(f"[1/4] Generating data ({n_accounts} accounts, seed={seed})...")
    accounts = H.generate_dataset(seed, n_accounts=n_accounts)
    trivial_roc, trivial_pr = H.evaluate_trivial(accounts)

    print(f"[2/4] Training mp_trained, mp_nosub, cat_nosub (seed={seed})...")
    mp_model = H.train_robust_mp(accounts, seed)
    mp_nosub_model = train_mp_nosub(accounts, seed)
    cat_nosub_model = train_cat_nosub(accounts, seed)

    print("[3/4] Scoring...")
    mp_result = H.score_all(accounts, H.embed_mp, mp_model)
    mp_nosub_result = H.score_all(accounts, H.embed_mp, mp_nosub_model)
    rand_result = H.score_all(accounts, embed_mp_random, build_random_token_table(seed))
    cat_scorer = CatNosubScorer(cat_nosub_model)
    cat_nosub_result = score_all_cat_nosub(accounts, cat_scorer)
    cat_oov = oov_rates(accounts, cat_nosub_model)

    within_trained, _ = H.token_similarity_analysis(mp_model)
    within_nosub, _ = H.token_similarity_analysis(mp_nosub_model)

    print(f"[4/4] Bootstrap CIs and paired deltas (n_boot={n_boot})...")
    scorers = {
        "mp_trained": scorer_metrics(mp_result, n_boot, seed),
        "mp_nosub": scorer_metrics(mp_nosub_result, n_boot, seed),
        "mp_random": scorer_metrics(rand_result, n_boot, seed),
        "cat_nosub": scorer_metrics(cat_nosub_result, n_boot, seed),
        "trivial": {at: {"roc_auc": trivial_roc[at], "pr_auc": trivial_pr[at]} for at in ATTACKS},
    }
    deltas = {
        "mp_trained_minus_nosub": paired_deltas(mp_result, mp_nosub_result, n_boot, seed),
        "mp_nosub_minus_random": paired_deltas(mp_nosub_result, rand_result, n_boot, seed),
    }
    verdicts = {
        "a4_subwords_help_mp_roc": bool(deltas["mp_trained_minus_nosub"]["spoof"]["roc"][1] > 0),
        "a4_subwords_hurt_mp_roc": bool(deltas["mp_trained_minus_nosub"]["spoof"]["roc"][2] < 0),
        "a4_training_helps_nosub_roc": bool(deltas["mp_nosub_minus_random"]["spoof"]["roc"][1] > 0),
        "a4_training_helps_nosub_pr": bool(deltas["mp_nosub_minus_random"]["spoof"]["pr"][1] > 0),
    }
    descriptive = {
        "cat_nosub_oov_rate": cat_oov,
        "within_feature_cos_mp_trained": float(within_trained.mean()),
        "within_feature_cos_mp_nosub": float(within_nosub.mean()),
    }

    print("\n--- Spoof summary ---")
    for name in ["mp_trained", "mp_nosub", "mp_random", "cat_nosub", "trivial"]:
        s = scorers[name]["spoof"]
        print(f"  {name:<12} ROC={s['roc_auc']:.4f}  PR={s['pr_auc']:.4f}")
    for dname, d in deltas.items():
        r, p = d["spoof"]["roc"], d["spoof"]["pr"]
        print(f"  {dname:<24} dROC={r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]  "
              f"dPR={p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}]")
    print("  cat_nosub OOV rates: " + "  ".join(f"{k}={v:.2f}" for k, v in cat_oov.items()))
    print(f"  within-feature cos: trained={descriptive['within_feature_cos_mp_trained']:.4f}  "
          f"nosub={descriptive['within_feature_cos_mp_nosub']:.4f}")
    print(f"  verdicts: {verdicts}")

    return {
        "schema_version": 1,
        "script": "a4_nosub.py",
        "plan": "plan/12-CRITIQUE_ABLATIONS.md (A4)",
        "seed": seed,
        "smoke": smoke,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_accounts": n_accounts,
        "n_bootstrap": n_boot,
        "scorers": scorers,
        "deltas": deltas,
        "verdicts": verdicts,
        "descriptive": descriptive,
    }


def main():
    p = argparse.ArgumentParser(description="A4 no-subword cell (plan 12)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="50 accounts, 200 bootstraps, schema assertions, no file written")
    args = p.parse_args()

    results = run(args.seed, args.smoke)

    if args.smoke:
        for name, m in results["scorers"].items():
            for at in ATTACKS:
                assert not math.isnan(m[at]["roc_auc"]), f"smoke: NaN ROC for {name}/{at}"
        for dname, d in results["deltas"].items():
            for at in ATTACKS:
                assert all(not math.isnan(x) for x in d[at]["roc"]), f"smoke: NaN delta {dname}/{at}"
        assert len(results["verdicts"]) == 4
        print("\nSMOKE PASS")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"a4_nosub_seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
