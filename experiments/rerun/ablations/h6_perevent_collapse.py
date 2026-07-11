# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim==4.4.0",
#   "numpy==2.2.6",
#   "scikit-learn==1.7.2",
#   "matplotlib==3.10.8",
#   "scipy==1.15.3",
# ]
# ///
#
# A3p — Correlated-Marginals Collapse Precursor (C2)
# ==================================================
# Pre-specified in plan/12-CRITIQUE_ABLATIONS.md. Tests whether the C2 per-event
# collapse depends on the closed-vocab generator's feature independence, using
# the H6 generator's real coupling from RBA marginals (os::device_type joint,
# browser|os, region|country, asn|country) with rtt_bucket as the built-in
# independence control.
#
# Per seed: 4 cells {SG, CBOW} x {per-event, per-account} on identical events.
# Metrics per feature: within-feature cosine (T8 wv convention + subword-free
# vectors_vocab sensitivity) and within-feature context-distribution JSD
# (h2_cooccurrence methodology, open vocab).
#
# Reuse policy: drives scripts/h6/h6_rerun.py as a library — marginals, account
# generation, corpus building, tokenization, and training kwargs are its
# unmodified definitions.
#
# Usage:
#   uv run experiments/rerun/ablations/h6_perevent_collapse.py --smoke
#   uv run experiments/rerun/ablations/h6_perevent_collapse.py --seed 42

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from gensim.models import FastText
from scipy.spatial.distance import jensenshannon

H6_DIR = Path(__file__).resolve().parent.parent / "scripts" / "h6"
sys.path.insert(0, str(H6_DIR))
import h6_rerun as H6  # noqa: E402

MIN_TOKEN_COUNT = 5          # pre-specified filter (plan 12)
COLLAPSE_THRESHOLD = 0.9     # decision a2b73375
WINDOW = 6                   # matches ROBUST_KWARGS window
CONDITIONED = ["region", "asn_bucket", "browser", "device_type"]
OUT_DIR = Path(__file__).resolve().parent / "results"

# Longest-first so e.g. "asn_bucket_..." never matches a shorter feature prefix.
_PREFIXES = sorted(H6.FEATURE_ORDER, key=len, reverse=True)


def token_feature(token: str) -> str | None:
    for f in _PREFIXES:
        if token.startswith(f + "_"):
            return f
    return None


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def build_per_event_corpus(accounts):
    """Degenerate shape: one 7-token sentence per training event."""
    return [H6.device_to_tokens(e) for acc in accounts for e in acc["train"]]


# ---------------------------------------------------------------------------
# Embedding-space metric: within-feature cosine
# ---------------------------------------------------------------------------
def _pairwise_mean_cosine(vecs: np.ndarray) -> float:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / (norms + 1e-10)
    sim = unit @ unit.T
    n = len(vecs)
    iu = np.triu_indices(n, k=1)
    return float(sim[iu].mean())


def within_feature_cosine(model, tokens_by_feature):
    """Per-feature mean pairwise cosine, on composed wv vectors (T8 convention)
    and on subword-free input vectors (vectors_vocab sensitivity)."""
    wv_means, vocab_means = {}, {}
    pooled_wv = []
    for f, tokens in tokens_by_feature.items():
        present = [t for t in tokens if t in model.wv.key_to_index]
        if len(present) < 2:
            wv_means[f] = float("nan")
            vocab_means[f] = float("nan")
            continue
        wv_vecs = np.array([model.wv[t] for t in present])
        idx = [model.wv.key_to_index[t] for t in present]
        vocab_vecs = model.wv.vectors_vocab[idx]
        wv_means[f] = _pairwise_mean_cosine(wv_vecs)
        vocab_means[f] = _pairwise_mean_cosine(vocab_vecs)
        norms = np.linalg.norm(wv_vecs, axis=1, keepdims=True)
        unit = wv_vecs / (norms + 1e-10)
        sim = unit @ unit.T
        iu = np.triu_indices(len(present), k=1)
        pooled_wv.extend(sim[iu].tolist())
    pooled = float(np.mean(pooled_wv)) if pooled_wv else float("nan")
    return wv_means, vocab_means, pooled


# ---------------------------------------------------------------------------
# Corpus-statistics metric: within-feature context JSD
# ---------------------------------------------------------------------------
def cooccurrence_matrix(corpus, token_index, window=WINDOW):
    size = len(token_index)
    matrix = np.zeros((size, size), dtype=np.float64)
    for sentence in corpus:
        indices = [token_index[t] for t in sentence if t in token_index]
        for pos, center in enumerate(indices):
            lo = max(0, pos - window)
            hi = min(len(indices), pos + window + 1)
            for ctx in range(lo, hi):
                if ctx != pos:
                    matrix[center, indices[ctx]] += 1.0
    return matrix


def within_feature_jsd(matrix, token_index, tokens_by_feature):
    means = {}
    for f, tokens in tokens_by_feature.items():
        rows = []
        for t in tokens:
            row = matrix[token_index[t]]
            total = row.sum()
            if total > 0:
                rows.append(row / total)
        jsds = [float(jensenshannon(a, b)) for a, b in combinations(rows, 2)]
        means[f] = float(np.mean(jsds)) if jsds else float("nan")
    return means


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(seed: int, smoke: bool) -> dict:
    n_accounts = H6.N_ACCOUNTS_SMOKE if smoke else H6.N_ACCOUNTS_FULL

    print(f"[1/4] Generating H6 accounts ({n_accounts}, seed={seed})...")
    marginals = H6.Marginals(H6.MARGINALS_PATH)
    rng = np.random.default_rng(seed)
    accounts, _fleet = H6.generate_accounts(marginals, n_accounts, rng)

    corpora = {
        "per_event": build_per_event_corpus(accounts),
        "per_account": H6.build_corpus(accounts),
    }
    token_counts = Counter(t for sent in corpora["per_event"] for t in sent)

    tokens_by_feature, token_totals = {}, {}
    for f in H6.FEATURE_ORDER:
        all_toks = [t for t in token_counts if token_feature(t) == f]
        kept = sorted(t for t in all_toks if token_counts[t] >= MIN_TOKEN_COUNT)
        tokens_by_feature[f] = kept
        token_totals[f] = {"total": len(all_toks), "filtered": len(kept)}
        print(f"  {f:<12} tokens: {len(all_toks):>4} total, {len(kept):>4} kept (count>={MIN_TOKEN_COUNT})")

    print("[2/4] Context-distribution JSD per corpus shape...")
    filtered_tokens = sorted(t for toks in tokens_by_feature.values() for t in toks)
    token_index = {t: i for i, t in enumerate(filtered_tokens)}
    jsd = {}
    for shape, corpus in corpora.items():
        matrix = cooccurrence_matrix(corpus, token_index)
        jsd[shape] = within_feature_jsd(matrix, token_index, tokens_by_feature)
        print(f"  {shape:<12} " + "  ".join(f"{f}={jsd[shape][f]:.3f}" for f in H6.FEATURE_ORDER))

    print("[3/4] Training 2x2 cells (SG/CBOW x per-event/per-account)...")
    cells = {}
    for sg, obj in [(1, "sg"), (0, "cbow")]:
        for shape, corpus in corpora.items():
            name = f"{obj}_{shape}"
            model = FastText(sentences=corpus, **{**H6.ROBUST_KWARGS, "sg": sg, "seed": seed})
            wv_means, vocab_means, pooled = within_feature_cosine(model, tokens_by_feature)
            cells[name] = {
                "within_cos_wv": wv_means,
                "within_cos_vocab": vocab_means,
                "pooled_within_cos_wv": pooled,
            }
            print(f"  {name:<18} pooled_within={pooled:.4f}  " +
                  "  ".join(f"{f}={wv_means[f]:.3f}" for f in H6.FEATURE_ORDER))

    print("[4/4] Verdicts (per-event SG cell, wv metric)...")
    pe_sg = cells["sg_per_event"]["within_cos_wv"]
    pe_jsd = jsd["per_event"]
    valid = {f: v for f, v in pe_sg.items() if not math.isnan(v)}
    verdicts = {
        "a3p_rtt_max_cosine": bool(max(valid, key=valid.get) == "rtt_bucket"),
        "a3p_rtt_min_jsd": bool(min(pe_jsd, key=pe_jsd.get) == "rtt_bucket"),
        "a3p_conditioned_below_threshold": bool(
            all(pe_sg[f] < COLLAPSE_THRESHOLD for f in CONDITIONED)
        ),
        "a3p_collapse_all": bool(cells["sg_per_event"]["pooled_within_cos_wv"] > COLLAPSE_THRESHOLD),
    }
    print(f"  {verdicts}")

    return {
        "schema_version": 1,
        "script": "h6_perevent_collapse.py",
        "plan": "plan/12-CRITIQUE_ABLATIONS.md",
        "seed": seed,
        "smoke": smoke,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_accounts": n_accounts,
        "config": {
            "min_token_count": MIN_TOKEN_COUNT,
            "collapse_threshold": COLLAPSE_THRESHOLD,
            "window": WINDOW,
            "conditioned_features": CONDITIONED,
        },
        "token_totals": token_totals,
        "jsd": jsd,
        "cells": cells,
        "verdicts": verdicts,
    }


def main():
    p = argparse.ArgumentParser(description="A3p correlated-marginals collapse precursor (plan 12)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="50 accounts, schema assertions, no file written")
    args = p.parse_args()

    results = run(args.seed, args.smoke)

    if args.smoke:
        for shape in ["per_event", "per_account"]:
            assert set(results["jsd"][shape]) == set(H6.FEATURE_ORDER), f"smoke: jsd keys {shape}"
        assert set(results["cells"]) == {"sg_per_event", "sg_per_account",
                                         "cbow_per_event", "cbow_per_account"}
        for cell in results["cells"].values():
            assert not math.isnan(cell["pooled_within_cos_wv"]), "smoke: NaN pooled cosine"
        assert len(results["verdicts"]) == 4
        print("\nSMOKE PASS")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"h6_perevent_seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
