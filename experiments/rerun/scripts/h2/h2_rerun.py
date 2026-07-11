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
# H2 Rerun — Seed-Parameterized Experiment Runner
# ================================================
# Phase 1 of the 5-seed reproducibility rerun for H2 (mechanism).
#
# Covers: T1 (AUC + bootstrap CIs), T2 (window sweep), T3 (prefixed-concat),
# T4 (tz-counterfactual), T5 (tz-permutation), T6 (compactness),
# T8 (token similarity, robust + degenerate). Delta CIs: fbfd0775. Results JSON: dd56db0b.
#
# Usage:
#   uv run h2_rerun.py --seed 42
#   uv run h2_rerun.py --seed 123 --smoke   # fast verification (50 accts, 200 boot)

import argparse
import json
import sys
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import FastText
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score

N_ACCOUNTS = 400
N_TRAIN = 60
FLEET_FRAC = 0.25
N_BOOTSTRAP = 1000

N_ACCOUNTS_SMOKE = 50
N_BOOTSTRAP_SMOKE = 200

FEATURES = {
    "os":      ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz":      ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang":    ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "network": ["wifi", "lte", "5g", "broadband"],
    "screen":  ["small", "medium", "large", "xlarge"],
}
FEATURE_ORDER = ["os", "browser", "tz", "lang", "network", "screen"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_token(f, v): return f"{f}_{v}"
def device_key(d): return tuple(d[f] for f in FEATURE_ORDER)
def device_to_tokens(d): return [make_token(f, d[f]) for f in FEATURE_ORDER]
def device_to_concat(d): return "_".join(d[f] for f in FEATURE_ORDER)

def zipf_weights(n, s=1.5):
    w = np.array([1.0 / k**s for k in range(1, n+1)])
    return w / w.sum()

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_centroid(events, embed_fn, model):
    return np.mean([embed_fn(e, model) for e in events], axis=0)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_dataset(seed, n_accounts=N_ACCOUNTS, n_train=N_TRAIN):
    rng = np.random.default_rng(seed)
    fleet_device = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
    accounts = []
    for acct_id in range(n_accounts):
        n_dev = int(rng.integers(2, 5))
        known, seen = [], set()
        while len(known) < n_dev:
            d = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
            k = device_key(d)
            if k not in seen:
                seen.add(k); known.append(d)
        primary = known[0]
        weights = zipf_weights(n_dev)
        train_events = []
        for _ in range(n_train):
            idx = rng.choice(n_dev, p=weights)
            train_events.append(dict(known[idx]))
        is_fleet = rng.random() < FLEET_FRAC
        if is_fleet:
            replace_idx = int(rng.integers(0, len(train_events)))
            train_events[replace_idx] = dict(fleet_device)
        # Novel: foreign OS, tz, language
        novel = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
        attempts = 0
        while (novel["os"] == primary["os"] or novel["tz"] == primary["tz"]
               or novel["lang"] == primary["lang"]) and attempts < 1000:
            novel = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
            attempts += 1
        # Spoof: 3-field — tz changed + randomised network/screen (issue ce3a6063)
        spoof = dict(primary)
        spoof["tz"]      = rng.choice([t for t in FEATURES["tz"] if t != primary["tz"]])
        spoof["network"] = rng.choice(FEATURES["network"])
        spoof["screen"]  = rng.choice(FEATURES["screen"])
        # Negative: same OS/browser/tz/lang, different network/screen
        neg = dict(primary)
        neg["network"] = rng.choice([n for n in FEATURES["network"] if n != primary["network"]])
        neg["screen"]  = rng.choice([s for s in FEATURES["screen"]  if s != primary["screen"]])
        accounts.append({
            "id": acct_id, "known_devices": known, "primary": primary,
            "train_events": train_events, "is_fleet": is_fleet,
            "fleet_device": fleet_device,
            "test": {
                "novel": (novel, 1), "fleet": (dict(fleet_device), 1),
                "spoof": (spoof, 1), "neg": (neg, 0), "known": (dict(primary), 0),
            },
        })
    return accounts

# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------
def build_mp_corpus_per_account(accounts):
    sentences = []
    for acc in accounts:
        flat = []
        for e in acc["train_events"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences

def build_cat_corpus_per_account(accounts):
    sentences = []
    for acc in accounts:
        sent = [device_to_concat(e) for e in acc["train_events"]]
        if sent:
            sentences.append(sent)
    return sentences

def build_per_event_corpus(accounts):
    """Degenerate config: one 6-token sentence per login event (ml-lab PoC style)."""
    sentences = []
    for acc in accounts:
        for e in acc["train_events"]:
            sentences.append(device_to_tokens(e))
    return sentences

# ---------------------------------------------------------------------------
# Model training — robust config (sg=1, per-account, epochs=20)
# ---------------------------------------------------------------------------
def _robust_kwargs(seed):
    return dict(
        vector_size=64, window=6, sg=1, negative=10,
        min_count=1, epochs=20, min_n=3, max_n=6,
        seed=seed, workers=1,
    )

def train_robust_mp(accounts, seed):
    return FastText(sentences=build_mp_corpus_per_account(accounts), **_robust_kwargs(seed))

def train_robust_cat(accounts, seed):
    return FastText(sentences=build_cat_corpus_per_account(accounts), **_robust_kwargs(seed))

# ---------------------------------------------------------------------------
# Model training — degenerate config (sg=0/CBOW, per-event, epochs=10)
# ---------------------------------------------------------------------------
def _degenerate_kwargs(seed):
    return dict(
        vector_size=64, window=6, sg=0,
        min_count=1, epochs=10, min_n=3, max_n=6,
        seed=seed, workers=1,
    )

def train_degenerate(accounts, seed):
    """CBOW + per-event corpus — the config that produces T8 within-feature collapse."""
    return FastText(sentences=build_per_event_corpus(accounts), **_degenerate_kwargs(seed))

# ---------------------------------------------------------------------------
# Model training — 2×2 factorial (issue e2156012)
# Base: robust params (epochs=20, negative=10); only sg and corpus vary.
# Diagonal SG+per-account cell reuses mp_model from train_robust_mp.
# ---------------------------------------------------------------------------
def _factorial_kwargs(seed, sg):
    return dict(
        vector_size=64, window=6, sg=sg, negative=10,
        min_count=1, epochs=20, min_n=3, max_n=6,
        seed=seed, workers=1,
    )

def train_factorial_cbow_per_event(accounts, seed):
    """Controlled diagonal: CBOW + per-event at epochs=20 (stronger than PoC epochs=10)."""
    return FastText(sentences=build_per_event_corpus(accounts), **_factorial_kwargs(seed, sg=0))

def train_factorial_cbow_per_account(accounts, seed):
    """Off-diagonal: CBOW + per-account corpus."""
    return FastText(sentences=build_mp_corpus_per_account(accounts), **_factorial_kwargs(seed, sg=0))

def train_factorial_sg_per_event(accounts, seed):
    """Off-diagonal: SG + per-event corpus."""
    return FastText(sentences=build_per_event_corpus(accounts), **_factorial_kwargs(seed, sg=1))

# ---------------------------------------------------------------------------
# T2/T3/T5: additional corpus builders and training helpers
# ---------------------------------------------------------------------------
def device_to_prefixed(d):
    """T3: key:val|key:val|... format (preserves feature identity without n-gram bleed)."""
    return "|".join(f"{f}:{d[f]}" for f in FEATURE_ORDER)

def device_to_concat_ordered(d, order):
    """T5: concat string with features in permuted order."""
    return "_".join(d[f] for f in order)

def build_prefixed_corpus(accounts):
    sentences = []
    for acc in accounts:
        sent = [device_to_prefixed(e) for e in acc["train_events"]]
        if sent:
            sentences.append(sent)
    return sentences

def build_cat_corpus_ordered(accounts, order):
    sentences = []
    for acc in accounts:
        sent = [device_to_concat_ordered(e, order) for e in acc["train_events"]]
        if sent:
            sentences.append(sent)
    return sentences

def train_cat_window(accounts, seed, window):
    """T2: concat model at window ∈ {1,3,6}."""
    kwargs = dict(_robust_kwargs(seed))
    kwargs["window"] = window
    return FastText(sentences=build_cat_corpus_per_account(accounts), **kwargs)

def train_prefixed_cat(accounts, seed):
    """T3: prefixed-concat model, window=1 (n-gram contamination structurally absent)."""
    kwargs = dict(_robust_kwargs(seed))
    kwargs["window"] = 1
    return FastText(sentences=build_prefixed_corpus(accounts), **kwargs)

def train_cat_ordered(accounts, seed, order):
    """T5: concat model with permuted feature order, window=1."""
    kwargs = dict(_robust_kwargs(seed))
    kwargs["window"] = 1
    return FastText(sentences=build_cat_corpus_ordered(accounts, order), **kwargs)

def embed_prefixed(event, model):
    return model.wv[device_to_prefixed(event)]

def embed_cat_ordered(event, model, order):
    return model.wv[device_to_concat_ordered(event, order)]

def compute_silhouette(accounts, embed_fn, model):
    """Per-account device-cluster silhouette using known_devices as cluster members."""
    vecs, lbl = [], []
    for i, acc in enumerate(accounts):
        for d in acc["known_devices"]:
            vecs.append(embed_fn(d, model))
            lbl.append(i)
    vecs_arr = np.array(vecs)
    lbl_arr  = np.array(lbl)
    uniq, cnts = np.unique(lbl_arr, return_counts=True)
    valid = uniq[cnts >= 2]
    if len(valid) < 2:
        return float("nan")
    mask = np.isin(lbl_arr, valid)
    return float(silhouette_score(vecs_arr[mask], lbl_arr[mask], metric="cosine"))

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_mp(event, model):
    return np.mean([model.wv[make_token(f, event[f])] for f in FEATURE_ORDER], axis=0)

def embed_cat(event, model):
    return model.wv[device_to_concat(event)]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_all(accounts, embed_fn, model):
    result = {k: {"scores": [], "labels": []} for k in ["novel", "fleet", "spoof", "neg", "known"]}
    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_fn, model)
        for at, (ev, lbl) in acc["test"].items():
            result[at]["scores"].append(cosine_dist(embed_fn(ev, model), centroid))
            result[at]["labels"].append(lbl)
    return result

def compute_auc(result, attack_type):
    neg_s = result["neg"]["scores"] + result["known"]["scores"]
    neg_l = result["neg"]["labels"] + result["known"]["labels"]
    comb_s = result[attack_type]["scores"] + neg_s
    comb_l = result[attack_type]["labels"] + neg_l
    try: return roc_auc_score(comb_l, comb_s)
    except: return float("nan")

def compute_pr_auc(result, attack_type):
    neg_s = result["neg"]["scores"] + result["known"]["scores"]
    neg_l = result["neg"]["labels"] + result["known"]["labels"]
    comb_s = result[attack_type]["scores"] + neg_s
    comb_l = result[attack_type]["labels"] + neg_l
    try: return average_precision_score(comb_l, comb_s)
    except: return float("nan")

def bootstrap_auc(result, attack_type, n_boot, seed):
    rng = np.random.default_rng(seed)
    neg_s = np.array(result["neg"]["scores"] + result["known"]["scores"])
    neg_l = np.array(result["neg"]["labels"] + result["known"]["labels"])
    att_s = np.array(result[attack_type]["scores"])
    att_l = np.array(result[attack_type]["labels"])
    comb_s = np.concatenate([att_s, neg_s])
    comb_l = np.concatenate([att_l, neg_l])
    n = len(comb_s)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(comb_l[idx])) < 2: continue
        aucs.append(roc_auc_score(comb_l[idx], comb_s[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def bootstrap_pr_auc(result, attack_type, n_boot, seed):
    rng = np.random.default_rng(seed)
    neg_s = np.array(result["neg"]["scores"] + result["known"]["scores"])
    neg_l = np.array(result["neg"]["labels"] + result["known"]["labels"])
    att_s = np.array(result[attack_type]["scores"])
    att_l = np.array(result[attack_type]["labels"])
    comb_s = np.concatenate([att_s, neg_s])
    comb_l = np.concatenate([att_l, neg_l])
    n = len(comb_s)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(comb_l[idx])) < 2: continue
        vals.append(average_precision_score(comb_l[idx], comb_s[idx]))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def bootstrap_pr_auc_delta(mp_result, cat_result, attack_type, n_boot, seed):
    """Bootstrap CI for PR-AUC(mp) − PR-AUC(cat) on a given attack type."""
    rng = np.random.default_rng(seed)
    neg_mp  = np.array(mp_result["neg"]["scores"]  + mp_result["known"]["scores"])
    neg_cat = np.array(cat_result["neg"]["scores"] + cat_result["known"]["scores"])
    neg_lbl = np.array(mp_result["neg"]["labels"]  + mp_result["known"]["labels"])
    att_mp  = np.array(mp_result[attack_type]["scores"])
    att_cat = np.array(cat_result[attack_type]["scores"])
    att_lbl = np.array(mp_result[attack_type]["labels"])
    yt  = np.concatenate([att_lbl, neg_lbl])
    ymp = np.concatenate([att_mp,  neg_mp])
    ycc = np.concatenate([att_cat, neg_cat])
    if len(np.unique(yt)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(average_precision_score(yt, ymp) - average_precision_score(yt, ycc))
    n = len(yt)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(yt[idx])) < 2:
            continue
        deltas.append(float(average_precision_score(yt[idx], ymp[idx]) - average_precision_score(yt[idx], ycc[idx])))
    if not deltas:
        return point, float("nan"), float("nan")
    deltas = np.array(deltas)
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

def bootstrap_auc_delta(mp_result, cat_result, attack_type, n_boot, seed):
    """Bootstrap CI for AUC(mp) − AUC(cat) on a given attack type."""
    rng = np.random.default_rng(seed)
    neg_mp  = np.array(mp_result["neg"]["scores"]  + mp_result["known"]["scores"])
    neg_cat = np.array(cat_result["neg"]["scores"] + cat_result["known"]["scores"])
    neg_lbl = np.array(mp_result["neg"]["labels"]  + mp_result["known"]["labels"])
    att_mp  = np.array(mp_result[attack_type]["scores"])
    att_cat = np.array(cat_result[attack_type]["scores"])
    att_lbl = np.array(mp_result[attack_type]["labels"])
    yt  = np.concatenate([att_lbl, neg_lbl])
    ymp = np.concatenate([att_mp,  neg_mp])
    ycc = np.concatenate([att_cat, neg_cat])
    if len(np.unique(yt)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(roc_auc_score(yt, ymp) - roc_auc_score(yt, ycc))
    n = len(yt)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(yt[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(yt[idx], ymp[idx]) - roc_auc_score(yt[idx], ycc[idx])))
    if not deltas:
        return point, float("nan"), float("nan")
    deltas = np.array(deltas)
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

def bootstrap_sil_delta(accounts, mp_model, cat_w1_model, n_boot, seed):
    """Bootstrap CI for silhouette(mp) − silhouette(cat_w1) over known_devices clusters."""
    rng = np.random.default_rng(seed)
    mp_vecs, cat_vecs, lbl = [], [], []
    for i, acc in enumerate(accounts):
        for d in acc["known_devices"]:
            mp_vecs.append(embed_mp(d, mp_model))
            cat_vecs.append(embed_cat(d, cat_w1_model))
            lbl.append(i)
    mp_arr  = np.array(mp_vecs)
    cat_arr = np.array(cat_vecs)
    lbl_arr = np.array(lbl)
    uniq, cnts = np.unique(lbl_arr, return_counts=True)
    valid = uniq[cnts >= 2]
    if len(valid) < 2:
        return float("nan"), float("nan"), float("nan")
    mask    = np.isin(lbl_arr, valid)
    mp_arr  = mp_arr[mask]
    cat_arr = cat_arr[mask]
    lbl_arr = lbl_arr[mask]
    point = float(silhouette_score(mp_arr, lbl_arr, metric="cosine") -
                  silhouette_score(cat_arr, lbl_arr, metric="cosine"))
    n = len(lbl_arr)
    deltas = []
    for _ in range(n_boot):
        idx     = rng.integers(0, n, size=n)
        sub_lbl = lbl_arr[idx]
        u2, c2  = np.unique(sub_lbl, return_counts=True)
        v2      = u2[c2 >= 2]
        if len(v2) < 2:
            continue
        m2 = np.isin(sub_lbl, v2)
        try:
            deltas.append(float(
                silhouette_score(mp_arr[idx][m2], sub_lbl[m2], metric="cosine") -
                silhouette_score(cat_arr[idx][m2], sub_lbl[m2], metric="cosine")
            ))
        except Exception:
            continue
    if not deltas:
        return point, float("nan"), float("nan")
    deltas = np.array(deltas)
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

def evaluate_trivial(accounts):
    neg_s, neg_l = [], []
    for acc in accounts:
        for at in ["neg", "known"]:
            ev, lbl = acc["test"][at]
            score = 0.0 if any(device_key(d) == device_key(ev) for d in acc["known_devices"]) else 1.0
            neg_s.append(score); neg_l.append(lbl)
    roc_aucs, pr_aucs = {}, {}
    for at in ["novel", "fleet", "spoof"]:
        a_s, a_l = [], []
        for acc in accounts:
            ev, lbl = acc["test"][at]
            score = 0.0 if any(device_key(d) == device_key(ev) for d in acc["known_devices"]) else 1.0
            a_s.append(score); a_l.append(lbl)
        try: roc_aucs[at] = roc_auc_score(a_l + neg_l, a_s + neg_s)
        except: roc_aucs[at] = float("nan")
        try: pr_aucs[at] = average_precision_score(a_l + neg_l, a_s + neg_s)
        except: pr_aucs[at] = float("nan")
    return roc_aucs, pr_aucs

# ---------------------------------------------------------------------------
# T4: Tz-counterfactual
# ---------------------------------------------------------------------------
def tz_counterfactual(accounts, embed_fn, model):
    act, cf, attr = [], [], []
    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_fn, model)
        spoof_ev, _ = acc["test"]["spoof"]
        d_act = cosine_dist(embed_fn(spoof_ev, model), centroid)
        cf_ev = dict(spoof_ev); cf_ev["tz"] = acc["primary"]["tz"]
        d_cf  = cosine_dist(embed_fn(cf_ev, model), centroid)
        act.append(d_act); cf.append(d_cf); attr.append(d_act - d_cf)
    return np.array(act), np.array(cf), np.array(attr)

# ---------------------------------------------------------------------------
# T6: Per-account centroid compactness
# ---------------------------------------------------------------------------
def per_account_compactness(accounts, embed_fn, model):
    mean_dists = []
    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_fn, model)
        dists = [cosine_dist(embed_fn(e, model), centroid) for e in acc["train_events"]]
        mean_dists.append(np.mean(dists))
    return np.array(mean_dists)

def bootstrap_mean_compactness(accounts, embed_fn, model, n_boot, seed):
    dists = per_account_compactness(accounts, embed_fn, model)
    rng = np.random.default_rng(seed)
    n = len(dists)
    means = [np.mean(dists[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    return float(np.mean(dists)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def enrollment_diagnostics(accounts, mp_model, cat_model):
    """Mean cosine distance from primary device embedding to per-account centroid."""
    mp_dists, cat_dists = [], []
    for acc in accounts:
        c_mp  = compute_centroid(acc["train_events"], embed_mp,  mp_model)
        c_cat = compute_centroid(acc["train_events"], embed_cat, cat_model)
        mp_dists.append(cosine_dist(embed_mp( acc["primary"], mp_model),  c_mp))
        cat_dists.append(cosine_dist(embed_cat(acc["primary"], cat_model), c_cat))
    return float(np.mean(mp_dists)), float(np.mean(cat_dists))

# ---------------------------------------------------------------------------
# T8: Token similarity
# ---------------------------------------------------------------------------
def token_similarity_analysis(model):
    tokens, feat_types = [], []
    for f in FEATURE_ORDER:
        for v in FEATURES[f]:
            tokens.append(make_token(f, v)); feat_types.append(f)
    vecs = np.array([model.wv[t] for t in tokens])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / (norms + 1e-10)
    sim_mat = vecs_n @ vecs_n.T
    feat_types = np.array(feat_types)
    n = len(tokens)
    within, cross = [], []
    for i in range(n):
        for j in range(i+1, n):
            if feat_types[i] == feat_types[j]: within.append(sim_mat[i, j])
            else: cross.append(sim_mat[i, j])
    return np.array(within), np.array(cross)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name, figures_dir):
    out = figures_dir / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")

def plot_summary_auc(mp_aucs, mp_cis, cat_aucs, cat_cis, trivial_aucs, figures_dir):
    attacks = ["novel", "fleet", "spoof"]
    x = np.arange(len(attacks))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"mean-pool": "#2196F3", "concat": "#FF5722", "trivial": "#9E9E9E"}
    for i, (label, aucs, cis, color) in enumerate([
        ("mean-pool", mp_aucs, mp_cis, colors["mean-pool"]),
        ("concat",    cat_aucs, cat_cis, colors["concat"]),
    ]):
        offset = (i - 0.5) * w
        ax.bar(x + offset, [aucs[a] for a in attacks], w, label=label, color=color, alpha=0.85)
        for xi, at in enumerate(attacks):
            lo, hi = cis[at][1], cis[at][2]
            m = aucs[at]
            ax.errorbar(x[xi] + offset, m, yerr=[[m-lo],[hi-m]], fmt="none",
                        color="black", capsize=3, linewidth=1.2)
            ax.text(x[xi] + offset, hi + 0.008, f"{m:.3f}", ha="center", fontsize=7.5)
    ax.bar(x + w, [trivial_aucs[a] for a in attacks], w,
           label="trivial baseline", color=colors["trivial"], alpha=0.7)
    for xi, at in enumerate(attacks):
        ax.text(x[xi] + w, trivial_aucs[at] + 0.008, f"{trivial_aucs[at]:.3f}",
                ha="center", fontsize=7.5)
    ax.set_xticks(x + w/3); ax.set_xticklabels(attacks)
    ax.set_ylim(0.35, 1.12); ax.set_ylabel("ROC-AUC")
    ax.set_title("H2 Rerun: AUC by attack type\n(sg=1, per-account corpus, epochs=20)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    save_fig(fig, "h2_summary_auc.png", figures_dir)

def plot_t4_counterfactual(mp_attr, cat_attr, figures_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mp_attr,  bins=30, color="#2196F3", alpha=0.7, label=f"mean-pool (μ={mp_attr.mean():.4f})")
    ax.hist(cat_attr, bins=30, color="#FF5722", alpha=0.7, label=f"concat    (μ={cat_attr.mean():.4f})")
    ax.axvline(mp_attr.mean(),  color="#2196F3", linestyle="--", linewidth=1.5)
    ax.axvline(cat_attr.mean(), color="#FF5722", linestyle="--", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Tz-attributable cosine distance (actual − counterfactual)")
    ax.set_ylabel("Count")
    ax.set_title("T4: Tz-counterfactual attribution\n(robust config, sg=1, per-account)")
    ax.legend(fontsize=8)
    save_fig(fig, "h2_t4_tz_counterfactual.png", figures_dir)

def plot_t6_compactness(mp_m, mp_lo, mp_hi, cat_m, cat_lo, cat_hi, figures_dir):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    x = np.array([0, 1])
    ax.bar(x, [mp_m, cat_m], color=["#2196F3", "#FF5722"], alpha=0.85)
    ax.errorbar(0, mp_m, yerr=[[mp_m-mp_lo],[mp_hi-mp_m]], fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.errorbar(1, cat_m, yerr=[[cat_m-cat_lo],[cat_hi-cat_m]], fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.text(0, mp_hi + 0.002, f"{mp_m:.4f}", ha="center", fontsize=9)
    ax.text(1, cat_hi + 0.002, f"{cat_m:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["mean-pool", "concat"])
    ax.set_ylabel("Per-account centroid compactness\n(lower = tighter cluster)")
    ax.set_title("T6: Per-account compactness\n(robust config)")
    save_fig(fig, "h2_t6_compactness.png", figures_dir)

def plot_t8_comparison(within_robust, cross_robust,
                        within_f_cbow_pe, cross_f_cbow_pe,
                        within_f_cbow_pa, cross_f_cbow_pa,
                        within_f_sg_pe,   cross_f_sg_pe,
                        figures_dir):
    # 2×2 factorial grid: rows=corpus, cols=objective
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cells = [
        # (within, cross, title, ax, border_color)
        (within_f_cbow_pe, cross_f_cbow_pe, "CBOW + per-event\n(collapse expected)", axes[0, 0], "#FF5722"),
        (within_robust,    cross_robust,    "SG + per-account\n(no collapse expected)", axes[0, 1], "#2196F3"),
        (within_f_cbow_pa, cross_f_cbow_pa, "CBOW + per-account\n(off-diagonal)",       axes[1, 0], "#FF9800"),
        (within_f_sg_pe,   cross_f_sg_pe,   "SG + per-event\n(off-diagonal)",            axes[1, 1], "#9C27B0"),
    ]
    for within, cross, title, ax, color in cells:
        ax.hist(within, bins=30, color=color,    alpha=0.75,
                label=f"Within-feature (μ={within.mean():.4f})")
        ax.hist(cross,  bins=30, color="#9E9E9E", alpha=0.50,
                label=f"Cross-feature  (μ={cross.mean():.4f})")
        ax.axvline(within.mean(), color=color,    linestyle="--", linewidth=1.5)
        ax.axvline(cross.mean(),  color="#9E9E9E", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Count")
        ax.set_xlim(-1.1, 1.1)
        ax.set_title(f"T8: {title}", fontsize=9)
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("per-event corpus\nCount", fontsize=8)
    axes[1, 0].set_ylabel("per-account corpus\nCount", fontsize=8)
    for ax in axes[0]: ax.set_xlabel("")
    fig.text(0.30, 0.97, "CBOW (sg=0)", ha="center", fontsize=9, fontweight="bold")
    fig.text(0.72, 0.97, "Skip-gram (sg=1)", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("T8: Within-Feature Embedding Collapse — 2×2 Factorial\n"
                 "(epochs=20, negative=10 standardized; degenerate_config PoC at epochs=10 separate)",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "h2_t8_token_similarity.png", figures_dir)

def plot_t2_window_sweep(mp_spoof_auc, mp_sil, window_results, figures_dir):
    """window_results: {window_int: {"spoof_auc": float, "sil": float}}"""
    windows = sorted(window_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("T2: Concat Window Sweep vs. Mean-Pool", fontsize=11, fontweight="bold")
    colors = {1: "#FF5722", 3: "#FF9800", 6: "#FFC107"}

    ax = axes[0]
    ax.axhline(mp_spoof_auc, color="#2196F3", linestyle="--", linewidth=1.5,
               label=f"mean-pool ({mp_spoof_auc:.4f})")
    for win in windows:
        ax.bar(win, window_results[win]["spoof_auc"], color=colors[win], alpha=0.85,
               label=f"concat w={win} ({window_results[win]['spoof_auc']:.4f})")
    ax.set_xticks(windows); ax.set_xticklabels([f"w={w}" for w in windows])
    ax.set_ylabel("Spoof ROC-AUC"); ax.set_title("Spoof AUC by concat window")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = ["mean-pool"] + [f"concat\nw={w}" for w in windows]
    svals = [mp_sil] + [window_results[w]["sil"] for w in windows]
    cols  = ["#2196F3"] + [colors[w] for w in windows]
    bars  = ax2.bar(names, svals, color=cols, alpha=0.88)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Silhouette (cosine)"); ax2.set_title("Per-account cluster separation")
    for bar, val in zip(bars, svals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "h2_t2_window_sweep.png", figures_dir)

def plot_t3_prefixed_concat(mp_spoof_auc, mp_sil, pf_spoof_auc, pf_sil,
                             cat_spoof_auc, cat_sil, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle("T3: Prefixed-Concat vs. Plain Concat vs. Mean-Pool",
                 fontsize=11, fontweight="bold")
    names  = ["mean-pool", "prefixed\nconcat", "plain\nconcat"]
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    axes[0].bar(names, [mp_spoof_auc, pf_spoof_auc, cat_spoof_auc], color=colors, alpha=0.88)
    axes[0].set_ylabel("Spoof ROC-AUC"); axes[0].set_title("Spoof AUC")
    for i, v in enumerate([mp_spoof_auc, pf_spoof_auc, cat_spoof_auc]):
        axes[0].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)

    bars = axes[1].bar(names, [mp_sil, pf_sil, cat_sil], color=colors, alpha=0.88)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Silhouette (cosine)"); axes[1].set_title("Per-account cluster separation")
    for bar, val in zip(bars, [mp_sil, pf_sil, cat_sil]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "h2_t3_prefixed_concat.png", figures_dir)

def plot_t5_tz_permutation(mp_spoof_auc, cat_w1_spoof_auc, perm_results, figures_dir):
    """perm_results: {tz_position: spoof_auc}"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("T5: Tz-Position Permutation — Spoof AUC", fontsize=11, fontweight="bold")
    positions = sorted(perm_results.keys())
    aucs = [perm_results[p] for p in positions]
    delta = mp_spoof_auc - cat_w1_spoof_auc
    recovery_50 = cat_w1_spoof_auc + 0.5 * delta

    ax.plot(positions, aucs, "o-", color="#FF9800", linewidth=2, markersize=7,
            label="Concat (tz at pos i)")
    ax.axhline(mp_spoof_auc,      color="#2196F3", linestyle="--", linewidth=1.5,
               label=f"mean-pool ({mp_spoof_auc:.4f})")
    ax.axhline(cat_w1_spoof_auc,  color="#FF5722", linestyle=":",  linewidth=1.5,
               label=f"concat w=1 baseline ({cat_w1_spoof_auc:.4f})")
    ax.axhline(recovery_50,       color="#9C27B0", linestyle="-.", linewidth=1,
               label=f"50% recovery ({recovery_50:.4f})")
    ax.set_xlabel("Tz feature position in concat string (0=first)")
    ax.set_ylabel("Spoof ROC-AUC")
    ax.set_title("Does moving tz earlier recover the spoof AUC gap?")
    ax.set_xticks(positions); ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, "h2_t5_tz_permutation.png", figures_dir)

# ---------------------------------------------------------------------------
# Results JSON
# ---------------------------------------------------------------------------
def _ci_block(m, lo, hi):
    return {"estimate": float(m), "lower": float(lo), "upper": float(hi)}

def write_results_json(results, figures_dir):
    out_path = figures_dir.parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="H2 rerun — seed-parameterized experiment runner")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 50 accounts, 200 bootstraps, assert schema, exit 0 on pass")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    seed = args.seed
    n_accounts = N_ACCOUNTS_SMOKE if args.smoke else N_ACCOUNTS
    n_boot     = N_BOOTSTRAP_SMOKE if args.smoke else N_BOOTSTRAP

    rerun_root  = Path(__file__).resolve().parent.parent.parent
    figures_dir = rerun_root / "seeds" / f"seed_{seed}" / "h2" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"H2 Rerun  seed={seed}  {'[SMOKE]' if args.smoke else ''}")
    print("sg=1, per-account corpus, epochs=20, negative=10")
    print("=" * 65)

    print(f"\n[1/7] Generating data ({n_accounts} accounts, seed={seed})...")
    accounts = generate_dataset(seed, n_accounts=n_accounts)
    trivial_aucs, trivial_pr_aucs = evaluate_trivial(accounts)

    print("\n[2/7] Building corpora (per-account)...")
    mp_corp  = build_mp_corpus_per_account(accounts)
    cat_corp = build_cat_corpus_per_account(accounts)
    print(f"  mp  corpus: {len(mp_corp)} sentences, "
          f"{sum(len(s) for s in mp_corp):,} tokens")
    print(f"  cat corpus: {len(cat_corp)} sentences, "
          f"{sum(len(s) for s in cat_corp):,} tokens")

    print(f"\n[3/7] Training models (robust config, seed={seed})...")
    mp_model  = train_robust_mp(accounts, seed)
    cat_model = train_robust_cat(accounts, seed)

    print(f"\n[4/7] Training degenerate model (CBOW, per-event, seed={seed}) + 2×2 factorial...")
    degen_model      = train_degenerate(accounts, seed)
    fact_cbow_pe     = train_factorial_cbow_per_event(accounts, seed)
    fact_cbow_pa     = train_factorial_cbow_per_account(accounts, seed)
    fact_sg_pe       = train_factorial_sg_per_event(accounts, seed)

    print(f"\n[5/7] Scoring and bootstrap CIs (n_boot={n_boot})...")
    mp_result  = score_all(accounts, embed_mp,  mp_model)
    cat_result = score_all(accounts, embed_cat, cat_model)

    attacks = ["novel", "fleet", "spoof"]
    mp_aucs, mp_cis   = {}, {}
    cat_aucs, cat_cis = {}, {}
    mp_pr_aucs, mp_pr_cis   = {}, {}
    cat_pr_aucs, cat_pr_cis = {}, {}
    for at in attacks:
        mp_m,  mp_lo,  mp_hi  = bootstrap_auc(mp_result,  at, n_boot, seed)
        cat_m, cat_lo, cat_hi = bootstrap_auc(cat_result, at, n_boot, seed)
        mp_aucs[at]  = mp_m;  mp_cis[at]  = (mp_m,  mp_lo,  mp_hi)
        cat_aucs[at] = cat_m; cat_cis[at] = (cat_m, cat_lo, cat_hi)
        mp_pr_m,  mp_pr_lo,  mp_pr_hi  = bootstrap_pr_auc(mp_result,  at, n_boot, seed)
        cat_pr_m, cat_pr_lo, cat_pr_hi = bootstrap_pr_auc(cat_result, at, n_boot, seed)
        mp_pr_aucs[at]  = mp_pr_m;  mp_pr_cis[at]  = (mp_pr_m,  mp_pr_lo,  mp_pr_hi)
        cat_pr_aucs[at] = cat_pr_m; cat_pr_cis[at] = (cat_pr_m, cat_pr_lo, cat_pr_hi)

    print("\n--- ROC-AUC Results (w=6 robust config) ---")
    print(f"  {'Attack':<8} {'mean-pool':>12} {'[95% CI]':>20}  {'concat_w6':>12} {'[95% CI]':>20}  {'trivial':>10}")
    print("  " + "-" * 86)
    for at in attacks:
        mp_m,  mp_lo,  mp_hi  = mp_cis[at]
        cat_m, cat_lo, cat_hi = cat_cis[at]
        print(f"  {at:<8} {mp_m:>12.4f} [{mp_lo:.4f}, {mp_hi:.4f}]  "
              f"{cat_m:>12.4f} [{cat_lo:.4f}, {cat_hi:.4f}]  {trivial_aucs[at]:>10.4f}")

    print("\n--- PR-AUC Results (w=6 robust config) ---")
    print(f"  {'Attack':<8} {'mean-pool':>12} {'[95% CI]':>20}  {'concat_w6':>12} {'[95% CI]':>20}  {'trivial':>10}")
    print("  " + "-" * 86)
    for at in attacks:
        mp_m,  mp_lo,  mp_hi  = mp_pr_cis[at]
        cat_m, cat_lo, cat_hi = cat_pr_cis[at]
        print(f"  {at:<8} {mp_m:>12.4f} [{mp_lo:.4f}, {mp_hi:.4f}]  "
              f"{cat_m:>12.4f} [{cat_lo:.4f}, {cat_hi:.4f}]  {trivial_pr_aucs[at]:>10.4f}")

    print(f"\n  Training cat w=1 for primary comparison (issues fbfd0775, dd56db0b)...")
    cat_w1_model  = train_cat_window(accounts, seed, 1)
    cat_w1_result = score_all(accounts, embed_cat, cat_w1_model)
    cat_w1_aucs, cat_w1_cis = {}, {}
    cat_w1_pr_aucs, cat_w1_pr_cis = {}, {}
    for at in attacks:
        m, lo, hi         = bootstrap_auc(cat_w1_result, at, n_boot, seed)
        cat_w1_aucs[at]   = m
        cat_w1_cis[at]    = (m, lo, hi)
        pm, plo, phi      = bootstrap_pr_auc(cat_w1_result, at, n_boot, seed)
        cat_w1_pr_aucs[at] = pm
        cat_w1_pr_cis[at]  = (pm, plo, phi)

    print("\n--- ROC-AUC Delta CIs (mp − cat_w1) ---")
    delta_cis = {}
    for at in attacks:
        pt, lo, hi   = bootstrap_auc_delta(mp_result, cat_w1_result, at, n_boot, seed)
        delta_cis[at] = (pt, lo, hi)
        print(f"  {at:<6} Δ={pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    sil_delta_ci = bootstrap_sil_delta(accounts, mp_model, cat_w1_model, n_boot, seed)
    print(f"  sil    Δ={sil_delta_ci[0]:+.4f}  95% CI [{sil_delta_ci[1]:+.4f}, {sil_delta_ci[2]:+.4f}]")

    print("\n--- PR-AUC Delta CIs (mp − cat_w1) ---")
    pr_delta_cis = {}
    for at in attacks:
        pt, lo, hi      = bootstrap_pr_auc_delta(mp_result, cat_w1_result, at, n_boot, seed)
        pr_delta_cis[at] = (pt, lo, hi)
        print(f"  {at:<6} Δ={pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    print("\n[6/7] Running T4, T6, T8...")

    # T4
    _, _, mp_attr  = tz_counterfactual(accounts, embed_mp,  mp_model)
    _, _, cat_attr = tz_counterfactual(accounts, embed_cat, cat_model)
    print(f"\n  T4 (tz-counterfactual):")
    print(f"    mean-pool tz-attr mean: {mp_attr.mean():.4f}")
    print(f"    concat    tz-attr mean: {cat_attr.mean():.4f}")

    # T6
    mp_c,  mp_c_lo,  mp_c_hi  = bootstrap_mean_compactness(accounts, embed_mp,  mp_model, n_boot, seed)
    cat_c, cat_c_lo, cat_c_hi = bootstrap_mean_compactness(accounts, embed_cat, cat_model, n_boot, seed)
    print(f"\n  T6 (centroid compactness — lower = tighter):")
    print(f"    mean-pool: {mp_c:.4f} [{mp_c_lo:.4f}, {mp_c_hi:.4f}]")
    print(f"    concat:    {cat_c:.4f} [{cat_c_lo:.4f}, {cat_c_hi:.4f}]")

    # T8 — robust, degenerate (historical PoC), and 2×2 factorial (issue e2156012)
    within_robust,    cross_robust    = token_similarity_analysis(mp_model)
    within_degen,     cross_degen     = token_similarity_analysis(degen_model)
    within_f_cbow_pe, cross_f_cbow_pe = token_similarity_analysis(fact_cbow_pe)
    within_f_cbow_pa, cross_f_cbow_pa = token_similarity_analysis(fact_cbow_pa)
    within_f_sg_pe,   cross_f_sg_pe   = token_similarity_analysis(fact_sg_pe)
    collapse_robust    = bool(within_robust.mean()    > 0.9)  # threshold: decision a2b73375
    collapse_degen     = bool(within_degen.mean()     > 0.9)
    collapse_f_cbow_pe = bool(within_f_cbow_pe.mean() > 0.9)
    collapse_f_cbow_pa = bool(within_f_cbow_pa.mean() > 0.9)
    collapse_f_sg_pe   = bool(within_f_sg_pe.mean()   > 0.9)
    print(f"\n  T8 (token similarity):")
    print(f"    robust         within={within_robust.mean():.4f}  cross={cross_robust.mean():.4f}"
          f"  ratio={within_robust.mean() / (cross_robust.mean() + 1e-10):.4f}"
          f"  collapse={collapse_robust}")
    print(f"    degen (PoC)    within={within_degen.mean():.4f}  cross={cross_degen.mean():.4f}"
          f"  ratio={within_degen.mean() / (cross_degen.mean() + 1e-10):.4f}"
          f"  collapse={collapse_degen}")
    print(f"\n  T8 2×2 factorial (epochs=20 standardized):")
    print(f"    CBOW+per-event within={within_f_cbow_pe.mean():.4f}  collapse={collapse_f_cbow_pe}")
    print(f"    CBOW+per-acct  within={within_f_cbow_pa.mean():.4f}  collapse={collapse_f_cbow_pa}")
    print(f"    SG+per-event   within={within_f_sg_pe.mean():.4f}  collapse={collapse_f_sg_pe}")
    print(f"    SG+per-acct    within={within_robust.mean():.4f}  collapse={collapse_robust}  (reuses robust)")

    if collapse_robust:
        print("\n  ABORT: T8 collapse detected in robust config (within > 0.9). Debug corpus.")
        sys.exit(2)

    if not args.smoke:
        print("\n  Generating figures...")
        plot_summary_auc(mp_aucs, mp_cis, cat_aucs, cat_cis, trivial_aucs, figures_dir)
        plot_t4_counterfactual(mp_attr, cat_attr, figures_dir)
        plot_t6_compactness(mp_c, mp_c_lo, mp_c_hi, cat_c, cat_c_lo, cat_c_hi, figures_dir)
        plot_t8_comparison(within_robust, cross_robust,
                           within_f_cbow_pe, cross_f_cbow_pe,
                           within_f_cbow_pa, cross_f_cbow_pa,
                           within_f_sg_pe,   cross_f_sg_pe,
                           figures_dir)

    # T2/T3/T5
    print("\n[7/7] Running T2 (window sweep), T3 (prefixed-concat), T5 (tz-permutation)...")

    # T2: concat window sweep (w=1,3,6)
    mp_sil = compute_silhouette(accounts, embed_mp, mp_model)
    window_results = {}
    for win in [1, 3, 6]:
        if win == 1:
            cat_w = cat_w1_model   # reuse model trained in [5/7] — avoids double training
        elif win == 6:
            cat_w = cat_model
        else:
            cat_w = train_cat_window(accounts, seed, win)
        spoof_auc_w = compute_auc(score_all(accounts, embed_cat, cat_w), "spoof")
        sil_w = compute_silhouette(accounts, embed_cat, cat_w)
        window_results[win] = {"spoof_auc": spoof_auc_w, "sil": sil_w}
        print(f"  T2 concat w={win}: spoof_auc={spoof_auc_w:.4f}  sil={sil_w:.4f}")
    print(f"  T2 mean-pool:        spoof_auc={mp_aucs['spoof']:.4f}  sil={mp_sil:.4f}")

    # T3: prefixed-concat silhouette comparison
    pf_model     = train_prefixed_cat(accounts, seed)
    pf_spoof_auc = compute_auc(score_all(accounts, embed_prefixed, pf_model), "spoof")
    pf_sil       = compute_silhouette(accounts, embed_prefixed, pf_model)
    cat_w1_sil   = window_results[1]["sil"]
    sil_gap      = mp_sil - pf_sil
    print(f"\n  T3 prefixed: spoof_auc={pf_spoof_auc:.4f}  sil={pf_sil:.4f}  "
          f"gap_vs_mp={sil_gap:+.4f}")

    # T5: tz-position permutation (window=1)
    remaining = [f for f in FEATURE_ORDER if f != "tz"]
    cat_w1_spoof = window_results[1]["spoof_auc"]
    delta_target = mp_aucs["spoof"] - cat_w1_spoof
    perm_results = {}
    for tz_pos in range(6):
        order = remaining[:tz_pos] + ["tz"] + remaining[tz_pos:]
        perm_model  = train_cat_ordered(accounts, seed, order)
        perm_scored = score_all(accounts, lambda e, m: embed_cat_ordered(e, m, order), perm_model)
        perm_results[tz_pos] = compute_auc(perm_scored, "spoof")
        pct = (perm_results[tz_pos] - cat_w1_spoof) / delta_target * 100 if delta_target > 0 else 0
        print(f"  T5 tz@pos{tz_pos}: spoof_auc={perm_results[tz_pos]:.4f}  "
              f"({pct:.1f}% recovery)")

    best_perm = max(perm_results.values())
    best_pct  = (best_perm - cat_w1_spoof) / delta_target * 100 if delta_target > 0 else 0
    print(f"\n  T5 best permutation: {best_perm:.4f} ({best_pct:.1f}% recovery)")
    print(f"  T5 verdict: {'DEFENSE wins' if best_perm < mp_aucs['spoof'] else 'CRITIQUE wins'} "
          f"(best perm {best_perm:.4f} vs mp {mp_aucs['spoof']:.4f})")

    enroll_mp, enroll_cat = enrollment_diagnostics(accounts, mp_model, cat_model)
    print(f"\n  Enrollment diagnostics: mean-pool={enroll_mp:.4f}  concat={enroll_cat:.4f}")

    if not args.smoke:
        plot_t2_window_sweep(mp_aucs["spoof"], mp_sil, window_results, figures_dir)
        plot_t3_prefixed_concat(mp_aucs["spoof"], mp_sil, pf_spoof_auc, pf_sil,
                                 window_results[1]["spoof_auc"], cat_w1_sil, figures_dir)
        plot_t5_tz_permutation(mp_aucs["spoof"], cat_w1_spoof, perm_results, figures_dir)

        t5_block: dict = {f"position_{pos+1}": {"concat_spoof_auc": float(auc)}
                          for pos, auc in sorted(perm_results.items())}
        t5_block["mean_pool_spoof_auc"] = float(mp_aucs["spoof"])
        results_payload = {
            "seed": seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "sg": 1, "epochs": 20, "negative": 10, "min_n": 3, "max_n": 6,
                "window": 6, "vector_size": 64, "corpus": "per_account",
            },
            "auc": {
                "mean_pool": {at: float(mp_aucs[at])      for at in ["novel", "fleet", "spoof"]},
                "concat_w1": {at: float(cat_w1_aucs[at])  for at in ["novel", "fleet", "spoof"]},
                "trivial":   {at: float(trivial_aucs[at]) for at in ["novel", "fleet", "spoof"]},
            },
            "pr_auc": {
                "mean_pool": {at: float(mp_pr_aucs[at])      for at in ["novel", "fleet", "spoof"]},
                "concat_w1": {at: float(cat_w1_pr_aucs[at])  for at in ["novel", "fleet", "spoof"]},
                "trivial":   {at: float(trivial_pr_aucs[at]) for at in ["novel", "fleet", "spoof"]},
            },
            "bootstrap_ci_95": {
                "per_model": {
                    "mean_pool": {at: _ci_block(*mp_cis[at])     for at in ["novel", "fleet", "spoof"]},
                    "concat_w1": {at: _ci_block(*cat_w1_cis[at]) for at in ["novel", "fleet", "spoof"]},
                },
                "pr_auc_per_model": {
                    "mean_pool": {at: _ci_block(*mp_pr_cis[at])     for at in ["novel", "fleet", "spoof"]},
                    "concat_w1": {at: _ci_block(*cat_w1_pr_cis[at]) for at in ["novel", "fleet", "spoof"]},
                },
                "deltas": {
                    "spoof_delta":      _ci_block(*delta_cis["spoof"]),
                    "novel_delta":      _ci_block(*delta_cis["novel"]),
                    "fleet_delta":      _ci_block(*delta_cis["fleet"]),
                    "silhouette_delta": _ci_block(*sil_delta_ci),
                },
                "pr_auc_deltas": {
                    "spoof_delta": _ci_block(*pr_delta_cis["spoof"]),
                    "novel_delta": _ci_block(*pr_delta_cis["novel"]),
                    "fleet_delta": _ci_block(*pr_delta_cis["fleet"]),
                },
            },
            "silhouette": {
                "mean_pool": float(mp_sil),
                "concat":    float(cat_w1_sil),
            },
            "compactness": {
                "mean_pool": {"mean": float(mp_c),  "ci_lower": float(mp_c_lo),  "ci_upper": float(mp_c_hi)},
                "concat":    {"mean": float(cat_c), "ci_lower": float(cat_c_lo), "ci_upper": float(cat_c_hi)},
            },
            "t2_window_sweep": {
                "concat_w1":           {"spoof_auc": float(window_results[1]["spoof_auc"])},
                "concat_w3":           {"spoof_auc": float(window_results[3]["spoof_auc"])},
                "concat_w6":           {"spoof_auc": float(window_results[6]["spoof_auc"])},
                "mean_pool_spoof_auc": float(mp_aucs["spoof"]),
            },
            "t3_prefixed_concat": {
                "mean_pool_silhouette":       float(mp_sil),
                "prefixed_concat_silhouette": float(pf_sil),
                "gap":                        float(mp_sil - pf_sil),
            },
            "t4_tz_attribution": {
                "mean_pool": float(mp_attr.mean()),
                "concat":    float(cat_attr.mean()),
            },
            "t5_tz_permutation": t5_block,
            "t7_trivial_baseline": {
                "mean_pool_spoof_margin": float(mp_aucs["spoof"]     - trivial_aucs["spoof"]),
                "concat_w1_spoof_margin": float(cat_w1_aucs["spoof"] - trivial_aucs["spoof"]),
            },
            "t8_token_similarity": {
                "robust_config": {
                    "within_feature_mean": float(within_robust.mean()),
                    "cross_feature_mean":  float(cross_robust.mean()),
                    "within_cross_ratio":  float(within_robust.mean() / (cross_robust.mean() + 1e-10)),
                    "collapse_detected":   collapse_robust,
                },
                "degenerate_config": {
                    "within_feature_mean": float(within_degen.mean()),
                    "cross_feature_mean":  float(cross_degen.mean()),
                    "within_cross_ratio":  float(within_degen.mean() / (cross_degen.mean() + 1e-10)),
                    "collapse_detected":   collapse_degen,
                },
                "factorial_2x2": {
                    "params": "epochs=20, negative=10, window=6 (standardized; only sg and corpus vary)",
                    "sg_per_account": {
                        "within_feature_mean": float(within_robust.mean()),
                        "cross_feature_mean":  float(cross_robust.mean()),
                        "within_cross_ratio":  float(within_robust.mean() / (cross_robust.mean() + 1e-10)),
                        "collapse_detected":   collapse_robust,
                        "note": "reuses robust_config model",
                    },
                    "cbow_per_event": {
                        "within_feature_mean": float(within_f_cbow_pe.mean()),
                        "cross_feature_mean":  float(cross_f_cbow_pe.mean()),
                        "within_cross_ratio":  float(within_f_cbow_pe.mean() / (cross_f_cbow_pe.mean() + 1e-10)),
                        "collapse_detected":   collapse_f_cbow_pe,
                    },
                    "cbow_per_account": {
                        "within_feature_mean": float(within_f_cbow_pa.mean()),
                        "cross_feature_mean":  float(cross_f_cbow_pa.mean()),
                        "within_cross_ratio":  float(within_f_cbow_pa.mean() / (cross_f_cbow_pa.mean() + 1e-10)),
                        "collapse_detected":   collapse_f_cbow_pa,
                    },
                    "sg_per_event": {
                        "within_feature_mean": float(within_f_sg_pe.mean()),
                        "cross_feature_mean":  float(cross_f_sg_pe.mean()),
                        "within_cross_ratio":  float(within_f_sg_pe.mean() / (cross_f_sg_pe.mean() + 1e-10)),
                        "collapse_detected":   collapse_f_sg_pe,
                    },
                },
            },
            "enrollment_diagnostics": {
                "mean_pool_enrollment_dist": enroll_mp,
                "concat_enrollment_dist":    enroll_cat,
            },
        }
        out_path = write_results_json(results_payload, figures_dir)
        print(f"\n  results.json: {out_path}")

    # Smoke assertions
    if args.smoke:
        assert not collapse_robust, "smoke: T8 collapse unexpected in robust config"
        assert collapse_degen, "smoke: T8 collapse expected in degenerate config but not detected"
        assert not np.isnan(within_f_cbow_pe.mean()), "smoke: NaN in T8 factorial cbow_per_event"
        assert not np.isnan(within_f_cbow_pa.mean()), "smoke: NaN in T8 factorial cbow_per_account"
        assert not np.isnan(within_f_sg_pe.mean()),   "smoke: NaN in T8 factorial sg_per_event"
        assert mp_aucs["spoof"] > 0.5, f"smoke: mp spoof AUC too low: {mp_aucs['spoof']:.4f}"
        assert cat_aucs["spoof"] > 0.5, f"smoke: cat spoof AUC too low: {cat_aucs['spoof']:.4f}"
        assert mp_aucs["spoof"] > cat_aucs["spoof"], \
            f"smoke: mp spoof ({mp_aucs['spoof']:.4f}) should exceed cat ({cat_aucs['spoof']:.4f})"
        for at in attacks:
            assert not np.isnan(mp_aucs[at]),  f"smoke: NaN in mp {at} AUC"
            assert not np.isnan(cat_aucs[at]), f"smoke: NaN in cat {at} AUC"
        assert not np.isnan(mp_sil), "smoke: NaN in mp silhouette"
        for win in [1, 3, 6]:
            assert not np.isnan(window_results[win]["spoof_auc"]), f"smoke: NaN in T2 w={win} spoof AUC"
        assert not np.isnan(pf_spoof_auc), "smoke: NaN in T3 prefixed spoof AUC"
        assert len(perm_results) == 6, "smoke: T5 must have 6 permutation results"
        assert all(not np.isnan(v) for v in perm_results.values()), "smoke: NaN in T5 permutation results"
        for at in attacks:
            assert not np.isnan(cat_w1_aucs[at]), f"smoke: NaN in cat_w1 {at} AUC"
        for at in attacks:
            pt, lo, hi = delta_cis[at]
            assert not np.isnan(pt), f"smoke: NaN in {at} delta CI estimate"
        assert not np.isnan(sil_delta_ci[0]), "smoke: NaN in silhouette delta CI"
        assert not np.isnan(enroll_mp),  "smoke: NaN in mp enrollment distance"
        assert not np.isnan(enroll_cat), "smoke: NaN in cat enrollment distance"
        for at in attacks:
            assert not np.isnan(mp_pr_aucs[at]),     f"smoke: NaN in mp {at} PR-AUC"
            assert not np.isnan(cat_w1_pr_aucs[at]), f"smoke: NaN in cat_w1 {at} PR-AUC"
        for at in attacks:
            pt, lo, hi = pr_delta_cis[at]
            assert not np.isnan(pt), f"smoke: NaN in {at} PR-AUC delta CI estimate"
        print("\nSMOKE PASS")
        return

    print("\n--- SUMMARY ---")
    for at in attacks:
        m, lo, hi = mp_cis[at]
        print(f"mp_auc_{at}={m:.4f}  lo={lo:.4f}  hi={hi:.4f}")
    for at in attacks:
        m, lo, hi = cat_cis[at]
        print(f"cat_w6_auc_{at}={m:.4f}  lo={lo:.4f}  hi={hi:.4f}")
    for at in attacks:
        m, lo, hi = cat_w1_cis[at]
        print(f"cat_w1_auc_{at}={m:.4f}  lo={lo:.4f}  hi={hi:.4f}")
    for at in attacks:
        pt, lo, hi = delta_cis[at]
        print(f"delta_{at}={pt:+.4f}  lo={lo:+.4f}  hi={hi:+.4f}")
    pt, lo, hi = sil_delta_ci
    print(f"delta_sil={pt:+.4f}  lo={lo:+.4f}  hi={hi:+.4f}")
    for at in attacks:
        m, lo, hi = mp_pr_cis[at]
        print(f"mp_pr_auc_{at}={m:.4f}  lo={lo:.4f}  hi={hi:.4f}")
    for at in attacks:
        m, lo, hi = cat_w1_pr_cis[at]
        print(f"cat_w1_pr_auc_{at}={m:.4f}  lo={lo:.4f}  hi={hi:.4f}")
    for at in attacks:
        pt, lo, hi = pr_delta_cis[at]
        print(f"pr_delta_{at}={pt:+.4f}  lo={lo:+.4f}  hi={hi:+.4f}")
    print(f"mp_t4_attr_mean={mp_attr.mean():.4f}")
    print(f"cat_t4_attr_mean={cat_attr.mean():.4f}")
    print(f"mp_compactness={mp_c:.4f}  lo={mp_c_lo:.4f}  hi={mp_c_hi:.4f}")
    print(f"cat_compactness={cat_c:.4f}  lo={cat_c_lo:.4f}  hi={cat_c_hi:.4f}")
    print(f"t8_robust_within={within_robust.mean():.4f}  cross={cross_robust.mean():.4f}  collapse={collapse_robust}")
    print(f"t8_degen_within={within_degen.mean():.4f}   cross={cross_degen.mean():.4f}   collapse={collapse_degen}")
    print(f"t2_mp_sil={mp_sil:.4f}")
    for win in [1, 3, 6]:
        print(f"t2_cat_w{win}_spoof={window_results[win]['spoof_auc']:.4f}  sil={window_results[win]['sil']:.4f}")
    print(f"t3_prefixed_spoof={pf_spoof_auc:.4f}  sil={pf_sil:.4f}  sil_gap={sil_gap:.4f}")
    for pos in sorted(perm_results.keys()):
        print(f"t5_tz_pos{pos}_spoof={perm_results[pos]:.4f}")
    print(f"enroll_mp={enroll_mp:.4f}  enroll_cat={enroll_cat:.4f}")
    print("Done.")

if __name__ == "__main__":
    main()
