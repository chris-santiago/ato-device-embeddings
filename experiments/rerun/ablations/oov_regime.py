# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim==4.4.0",
#   "numpy==2.2.6",
#   "scikit-learn==1.7.2",
# ]
# ///
#
# OOV Regime — Likelihood Incumbent vs FastText under deliberate out-of-vocabulary drift
# ======================================================================================
# Investigates how much discrimination each model loses as we inject genuinely
# out-of-vocabulary tokens into evaluation events (both benign and attack) at rising
# levels. The training corpus and likelihood tables stay OOV-free by construction, so
# the Freeman-style categorical incumbent floors every novel value to its smoothing
# constant while FastText composes an OOV vector from character n-grams.
#
# Two arms decompose FastText's OOV advantage:
#   - morphological: version-like suffix on os/browser (browser_chrome -> browser_chrome_v577).
#     Shares n-grams with a trained token -> FastText's subword composition is on trial.
#   - arbitrary: random string on high-cardinality geo/network (region, asn_bucket).
#     Shares only the "feature_" prefix -> FastText edge = graceful degradation only.
# gap(morphological) - gap(arbitrary) at a matched level isolates the subword benefit.
#
# Reuse: drives scripts/h6/h6_rerun.py (account generation, corpus/training, mp/trivial/
# two-stage scorers, bootstrap, top-1%) and imports the likelihood scorer + smoothing
# tables from the sibling h6_likelihood_incumbent.py as libraries. Nothing is modified.
#
# Parity: likelihood counts use centroid_events (train[:40]); FastText trains on train;
# OOV is injected only into eval events, so injected tokens are absent from both.
#
# Usage:
#   uv run experiments/rerun/ablations/oov_regime.py --smoke
#   uv run experiments/rerun/ablations/oov_regime.py --seed 42

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gensim.models import FastText
from sklearn.metrics import average_precision_score, roc_auc_score

ABLATIONS_DIR = Path(__file__).resolve().parent
H6_DIR = ABLATIONS_DIR.parent / "scripts" / "h6"
sys.path.insert(0, str(H6_DIR))
sys.path.insert(0, str(ABLATIONS_DIR))
import h6_rerun as H6  # noqa: E402
import h6_likelihood_incumbent as LIK  # noqa: E402

# --- OOV sweep configuration -------------------------------------------------
LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
POPULATIONS = ["spoof_k1", "novel"]
ARMS = {
    # Cross-feature arms (realistic scenarios): morphological drift on stable low-card
    # os/browser vs arbitrary novelty on high-card geo/network.
    "morphological": {"features": ["os", "browser"], "kind": "morph"},
    "arbitrary": {"features": ["region", "asn_bucket"], "kind": "rand"},
    # Same-feature disambiguation arms: both target region (high-card, discriminative) so
    # morphology-vs-floor-avoidance is isolated from which feature carries the signal.
    "region_morph": {"features": ["region"], "kind": "morph"},
    "region_arb": {"features": ["region"], "kind": "rand"},
}
ARM_ORDER = ["morphological", "arbitrary", "region_morph", "region_arb"]
POP_ORDER = {p: i for i, p in enumerate(POPULATIONS)}
ALPHABET = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))

BASE_SCORERS = ["mp_raw", "mp_nosub", "trivial", "two_stage"]
LIK_SCORERS = [f"lik_lam{lam}" for lam in LIK.LIK_LAMBDAS]
ALL_SCORERS = BASE_SCORERS + LIK_SCORERS

# Per-cell metrics are point estimates (cheap single pass); cross-seed std across the 5
# seeds is the headline uncertainty. A bootstrap is kept ONLY for the spoof_k1 delta —
# the one inferential claim — bounded to these resample counts (12 arm*level cells each).
N_DELTA_BOOT_FULL = 100
N_DELTA_BOOT_SMOKE = 50
SMOKE_LEVELS = [0.0, 0.5, 1.0]
# Cap negatives scored per cell so imbalanced (1:100) runs stay tractable. Balanced runs
# (neg_ratio=1 => 5 neg/acct) keep the full 400 accounts; 1:100 (500 neg/acct) shrinks to
# ~100 accounts — same ~1% base rate, ~500 positives, far less per-event scoring.
TARGET_NEG_PER_CELL = 50_000
N_ACCOUNTS_MIN = 100
DEFAULT_NEG_RATIO = 100  # negatives per positive; 1:100 mirrors the A2b incumbent so PR-AUC
#                          is measured under realistic ATO imbalance (not the 5:5 default)
OUT_DIR = ABLATIONS_DIR / "results"


# ---------------------------------------------------------------------------
# OOV injection
# ---------------------------------------------------------------------------
def build_banned(accounts) -> dict:
    """Per-feature set of every value seen in any training event.

    Injected OOV values are checked against this so they are provably absent from
    both the FastText corpus (train) and the likelihood tables (centroid_events subset
    of train) — i.e. genuinely out-of-vocabulary.
    """
    banned = {f: set() for f in H6.FEATURE_ORDER}
    for acc in accounts:
        for e in acc["train"]:
            for f in H6.FEATURE_ORDER:
                banned[f].add(str(e.get(f, "unknown")))
    return banned


def make_oov_value(base: str, kind: str, banned: set, rng) -> str:
    val = base
    for _ in range(50):
        if kind == "morph":
            val = f"{base}_v{int(rng.integers(100, 10000))}"
        else:
            val = "".join(rng.choice(ALPHABET, size=8))
        if val not in banned:
            break
    return val  # last attempt returned if all 50 collided (negligible probability)


def inject_oov(event: dict, arm_name: str, p: float, banned: dict, rng) -> dict:
    """With per-event probability p, replace one of the arm's target features with a
    novel token. Returns the original event untouched when not selected."""
    if p <= 0.0 or rng.random() >= p:
        return event
    spec = ARMS[arm_name]
    feats = spec["features"]
    f = feats[int(rng.integers(len(feats)))]
    e = dict(event)
    base = str(e.get(f, "unknown"))
    e[f] = make_oov_value(base, spec["kind"], banned[f], rng)
    return e


# ---------------------------------------------------------------------------
# No-subword mean-pool with per-feature fallback — the README's recommended OOV
# remedy (max_n=0 plain-token embeddings; an unseen value falls back to the mean of
# that feature's trained token vectors). Contrasts subword composition, which recovers
# the specific novel value via its stem, against a fallback that collapses every unseen
# value of a feature to one point.
# ---------------------------------------------------------------------------
def train_nosub_model(corpus, seed):
    # min_n=1, max_n=0 disables the n-gram bucket entirely (A4 convention).
    return FastText(sentences=corpus, **{**H6.ROBUST_KWARGS, "min_n": 1, "max_n": 0, "seed": seed})


def build_nosub_fallback(model) -> dict:
    """Per-feature fallback = mean of that feature's trained token vectors."""
    fb = {}
    for f in H6.FEATURE_ORDER:
        pref = f + "_"
        vecs = [model.wv[t] for t in model.wv.key_to_index if t.startswith(pref)]
        fb[f] = np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size, dtype=np.float32)
    return fb


def embed_mp_nosub(event, model, fallback) -> np.ndarray:
    vecs = []
    for f in H6.FEATURE_ORDER:
        tok = H6.make_token(f, event.get(f, "unknown"))
        vecs.append(model.wv[tok] if tok in model.wv.key_to_index else fallback[f])
    return np.mean(vecs, axis=0)


def compute_nosub_centroids(accounts, model, fallback) -> list:
    return [np.mean([embed_mp_nosub(e, model, fallback) for e in acc["centroid_events"]], axis=0)
            for acc in accounts]


# ---------------------------------------------------------------------------
# Score collection under injection (both benign and attack events drift)
# ---------------------------------------------------------------------------
def collect_population(accounts, centroids, model, population, glob, arm, p, banned, inj_rng, nosub):
    glob_counts, glob_n, vocab_sizes = glob
    ns_model, ns_cents, ns_fallback = nosub
    scores = {s: [] for s in ALL_SCORERS}
    labels = []
    n_inject, n_total = 0, 0
    for acc, centroid, nsc in zip(accounts, centroids, ns_cents):
        atk_events = acc["novel_atk"] if population == "novel" else acc[population]
        kd = acc["known_devices"]
        acct_counts, n_acct = LIK.account_tables(acc)  # clean tables
        pairs = [(e, 0) for e in acc["enroll_neg"]] + [(e, 1) for e in atk_events]
        for e, lbl in pairs:
            e2 = inject_oov(e, arm, p, banned, inj_rng)
            n_total += 1
            if e2 is not e:
                n_inject += 1
            # Embed once and reuse for mp_raw + two_stage (FastText OOV composition is the
            # hot path); identical to score_single_stage / score_two_stage / score_trivial.
            emb = H6.embed_mp(e2, model)
            mp = H6.cosine_dist(emb, centroid)
            gated = H6.device_key(e2) in kd
            scores["mp_raw"].append(mp)
            scores["mp_nosub"].append(H6.cosine_dist(embed_mp_nosub(e2, ns_model, ns_fallback), nsc))
            scores["trivial"].append(0.0 if gated else 1.0)
            scores["two_stage"].append(0.0 if gated else mp)
            for lam in LIK.LIK_LAMBDAS:
                scores[f"lik_lam{lam}"].append(
                    LIK.score_lik(e2, acct_counts, n_acct, lam, glob_counts, glob_n, vocab_sizes)
                )
            labels.append(lbl)
    lab = np.array(labels)
    obs_rate = n_inject / n_total if n_total else 0.0
    return {s: (np.array(v), lab) for s, v in scores.items()}, obs_rate


def cell_metrics(collected, seed) -> dict:
    """Point ROC-AUC / PR-AUC / top-1% per scorer. No per-cell bootstrap — at 1:100 that
    dominated runtime, and cross-seed std is the uncertainty we actually report."""
    out = {}
    for name, (sc, lb) in collected.items():
        if lb.sum() == 0 or lb.sum() == len(lb):
            out[name] = None
            continue
        top1 = H6.top_pct_metrics(sc, lb, 0.01, seed)
        out[name] = {
            "roc_auc": float(roc_auc_score(lb, sc)),
            "pr_auc": float(average_precision_score(lb, sc)),
            "top1pct_tp": top1["tp"], "top1pct_precision": top1["precision"],
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(seed: int, smoke: bool, neg_ratio: int) -> dict:
    n_delta_boot = N_DELTA_BOOT_SMOKE if smoke else N_DELTA_BOOT_FULL
    levels = SMOKE_LEVELS if smoke else LEVELS
    n_enroll_neg = H6.N_ENROLL_NEG * neg_ratio
    if smoke:
        n_accounts = H6.N_ACCOUNTS_SMOKE
    else:
        n_accounts = min(H6.N_ACCOUNTS_FULL,
                         max(N_ACCOUNTS_MIN, TARGET_NEG_PER_CELL // n_enroll_neg))

    print(f"[1/3] Generating H6 accounts ({n_accounts}, seed={seed}, "
          f"1:{neg_ratio} => {n_enroll_neg} benign/acct, "
          f"~{n_accounts * n_enroll_neg} neg/cell)...")
    marginals = H6.Marginals(H6.MARGINALS_PATH)
    accounts, _ = H6.generate_accounts(marginals, n_accounts, np.random.default_rng(seed),
                                       n_enroll_neg=n_enroll_neg)

    print(f"[2/3] Training FastText (subword + no-subword) + likelihood tables (seed={seed})...")
    corpus = H6.build_corpus(accounts)
    model = H6.train_model(corpus, seed)
    centroids = H6.compute_centroids(accounts, model)
    ns_model = train_nosub_model(corpus, seed)
    ns_fallback = build_nosub_fallback(ns_model)
    ns_cents = compute_nosub_centroids(accounts, ns_model, ns_fallback)
    nosub = (ns_model, ns_cents, ns_fallback)
    glob = LIK.build_global_tables(accounts)
    banned = build_banned(accounts)
    banned_sizes = {f: len(banned[f]) for f in H6.FEATURE_ORDER}

    # Best incumbent variant by clean (p=0) spoof_k1 ROC — the adverse-selection rule
    # from A2b, giving the incumbent its best foot forward before OOV pressure.
    clean_rng = np.random.default_rng([seed, 999, 0])
    clean_sk1, _ = collect_population(accounts, centroids, model, "spoof_k1", glob,
                                      ARM_ORDER[0], 0.0, banned, clean_rng, nosub)
    clean_metrics = cell_metrics(clean_sk1, seed)
    best_lam = max(LIK.LIK_LAMBDAS,
                   key=lambda lam: clean_metrics[f"lik_lam{lam}"]["roc_auc"])
    best_name = f"lik_lam{best_lam}"
    print(f"  best incumbent variant (clean spoof_k1 ROC): {best_name}")

    print(f"[3/3] Sweeping {len(ARM_ORDER)} arms x {len(levels)} levels x "
          f"{len(POPULATIONS)} populations (delta n_boot={n_delta_boot})...")
    results = {}
    for arm_id, arm in enumerate(ARM_ORDER):
        results[arm] = {}
        for p in levels:
            key = f"{p:.2f}"
            pop_metrics, obs_rate, sk1_scores = {}, {}, None
            for pop in POPULATIONS:
                inj_rng = np.random.default_rng([seed, arm_id, int(round(p * 100)),
                                                 POP_ORDER[pop]])
                collected, rate = collect_population(accounts, centroids, model, pop,
                                                     glob, arm, p, banned, inj_rng, nosub)
                pop_metrics[pop] = cell_metrics(collected, seed)
                obs_rate[pop] = rate
                if pop == "spoof_k1":
                    sk1_scores = collected
            assert sk1_scores is not None, "spoof_k1 must be in POPULATIONS"
            delta_rng = np.random.default_rng([seed, arm_id, int(round(p * 100)), 99])
            delta = LIK.paired_delta(sk1_scores["mp_raw"][0], sk1_scores[best_name][0],
                                     sk1_scores["mp_raw"][1], n_delta_boot, delta_rng)
            # Subword vs the README's fallback strategy — the claim under test.
            delta_ns_rng = np.random.default_rng([seed, arm_id, int(round(p * 100)), 98])
            delta_ns = LIK.paired_delta(sk1_scores["mp_raw"][0], sk1_scores["mp_nosub"][0],
                                        sk1_scores["mp_raw"][1], n_delta_boot, delta_ns_rng)
            results[arm][key] = {
                "obs_oov_rate": obs_rate,
                "metrics": pop_metrics,
                "delta_mp_minus_likbest_spoof_k1": delta,
                "delta_mp_minus_nosub_spoof_k1": delta_ns,
            }
            m = pop_metrics["spoof_k1"]
            print(f"  {arm:<14} p={p:<4} spoof_k1 PR  mp={m['mp_raw']['pr_auc']:.3f}  "
                  f"nosub={m['mp_nosub']['pr_auc']:.3f}  {best_name}={m[best_name]['pr_auc']:.3f}  "
                  f"dPR(mp-nosub)={delta_ns['pr'][0]:+.3f}"
                  f"[{delta_ns['pr'][1]:+.3f},{delta_ns['pr'][2]:+.3f}]  obs={obs_rate['spoof_k1']:.2f}")

    return {
        "schema_version": 1,
        "script": "oov_regime.py",
        "seed": seed,
        "smoke": smoke,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_accounts": n_accounts,
        "n_delta_bootstrap": n_delta_boot,
        "config": {
            "levels": levels, "arms": ARM_ORDER, "populations": POPULATIONS,
            "lik_lambdas": LIK.LIK_LAMBDAS, "arm_targets": {a: ARMS[a] for a in ARM_ORDER},
            "neg_ratio": neg_ratio, "n_enroll_neg": n_enroll_neg,
        },
        "banned_vocab_sizes": banned_sizes,
        "best_likelihood_variant": best_name,
        "results": results,
    }


def main():
    p = argparse.ArgumentParser(description="OOV regime: incumbent vs FastText degradation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg-ratio", type=int, default=DEFAULT_NEG_RATIO,
                   help=f"negatives per positive (default {DEFAULT_NEG_RATIO}, i.e. 1:100 imbalance)")
    p.add_argument("--smoke", action="store_true",
                   help="50 accounts, reduced grid, schema assertions, no file written")
    args = p.parse_args()

    results = run(args.seed, args.smoke, args.neg_ratio)

    if args.smoke:
        assert set(results["results"]) == set(ARM_ORDER), "smoke: arm keys"
        for arm in ARM_ORDER:
            assert set(results["results"][arm]) == {f"{p:.2f}" for p in SMOKE_LEVELS}, \
                f"smoke: level keys for {arm}"
            for key, cell in results["results"][arm].items():
                for pop in POPULATIONS:
                    m = cell["metrics"][pop]["mp_raw"]
                    assert m and not np.isnan(m["roc_auc"]), f"smoke: NaN mp_raw {arm} {key} {pop}"
                assert len(cell["delta_mp_minus_likbest_spoof_k1"]["roc"]) == 3
        # p=0 must reproduce a clean (near-zero) observed OOV rate; p=1 near full.
        for arm in ARM_ORDER:
            assert results["results"][arm]["0.00"]["obs_oov_rate"]["spoof_k1"] == 0.0, \
                "smoke: p=0 must inject nothing"
            assert results["results"][arm]["1.00"]["obs_oov_rate"]["spoof_k1"] > 0.9, \
                "smoke: p=1 must inject nearly all"
        print("\nSMOKE PASS")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"oov_regime_seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
