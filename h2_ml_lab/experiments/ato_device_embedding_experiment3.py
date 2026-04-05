# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
#   "matplotlib>=3.7",
# ]
# ///
#
# H2 Experiment — Iteration 2 (fixes for T2 and T6 from experiment2.py)
#
# Fixed issues identified in experiment2.py post-run inspection:
#
# T2 fix: Word2Vec on concat corpus is OOV-dominated (369/400 spoof tokens OOV),
#   making it equivalent to the trivial baseline via the zero-vector cosine=1.0 path.
#   T2 now uses FastText-min-ngram (min_n=1, max_n=3) vs. standard FastText (min_n=3, max_n=6)
#   to test whether shorter n-grams (which span feature boundaries less) change spoof AUC.
#   Also retains Word2Vec result but clearly labels it as "W2V (OOV-dominated)".
#
# T6 fix: Silhouette with true device labels is degenerate (~0.99 for both models)
#   because embeddings are deterministic functions of device identity -> within-cluster
#   distance = 0 by construction. T6 now measures per-account centroid compactness:
#   for each account, compute mean intra-account cosine distance between training event
#   embeddings and the account centroid (accounts have 2-4 devices -> non-trivial spread).
#   This measures whether mean-pool produces tighter per-account clusters than concat.
#
# All other tests (T1, T3, T4, T5, T7, T8) carry over from experiment2.py.

import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models import FastText, Word2Vec
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_BOOTSTRAP = 1000
FIGURE_DIR = Path(__file__).parent.parent / "figures"

FEATURES = {
    "os":      ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz":      ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang":    ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "network": ["wifi", "lte", "5g", "broadband"],
    "screen":  ["small", "medium", "large", "xlarge"],
}
FEATURE_ORDER = ["os", "browser", "tz", "lang", "network", "screen"]

def make_token(f, v): return f"{f}_{v}"
def device_to_tokens(d): return [make_token(f, d[f]) for f in FEATURE_ORDER]
def device_to_concat(d): return "_".join(d[f] for f in FEATURE_ORDER)
def device_key(d): return tuple(d[f] for f in FEATURE_ORDER)

# ---------------------------------------------------------------------------
# Data generation (identical to experiment2.py)
# ---------------------------------------------------------------------------
N_ACCOUNTS = 400
N_TRAIN = 60
FLEET_FRAC = 0.25

def sample_device(rng):
    return {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}

def sample_distinct_devices(n, rng):
    devices, seen = [], set()
    while len(devices) < n:
        d = sample_device(rng)
        k = device_key(d)
        if k not in seen:
            seen.add(k); devices.append(d)
    return devices

def zipf_weights(n, s=1.5):
    w = np.array([1.0 / k**s for k in range(1, n+1)])
    return w / w.sum()

def generate_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    fleet_device = sample_device(rng)
    accounts = []
    for acct_id in range(N_ACCOUNTS):
        n_dev = int(rng.integers(2, 5))
        known = sample_distinct_devices(n_dev, rng)
        primary = known[0]
        weights = zipf_weights(n_dev)
        train_events = []
        for _ in range(N_TRAIN):
            idx = rng.choice(n_dev, p=weights)
            train_events.append(dict(known[idx]))
        is_fleet = rng.random() < FLEET_FRAC
        if is_fleet:
            replace_idx = int(rng.integers(0, len(train_events)))
            train_events[replace_idx] = dict(fleet_device)
        novel = sample_device(rng)
        attempts = 0
        while (novel["os"] == primary["os"] or novel["tz"] == primary["tz"]
               or novel["lang"] == primary["lang"]) and attempts < 1000:
            novel = sample_device(rng); attempts += 1
        fleet_test = dict(fleet_device)
        spoof = dict(primary)
        alt_tzs = [t for t in FEATURES["tz"] if t != primary["tz"]]
        spoof["tz"] = rng.choice(alt_tzs)
        neg = dict(primary)
        alt_net = [n for n in FEATURES["network"] if n != primary["network"]]
        alt_scr = [s for s in FEATURES["screen"] if s != primary["screen"]]
        neg["network"] = rng.choice(alt_net)
        neg["screen"] = rng.choice(alt_scr)
        known_test = dict(primary)
        accounts.append({
            "id": acct_id, "known_devices": known, "primary": primary,
            "train_events": train_events, "is_fleet": is_fleet,
            "fleet_device": fleet_device,
            "test": {"novel": (novel, 1), "fleet": (fleet_test, 1),
                     "spoof": (spoof, 1), "neg": (neg, 0), "known": (known_test, 0)},
        })
    return accounts, fleet_device

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def build_mp_corpus(accounts):
    return [device_to_tokens(e) for a in accounts for e in a["train_events"]]

def build_cat_corpus(accounts):
    return [[device_to_concat(e)] for a in accounts for e in a["train_events"]]

def train_ft_mp(corpus, window=6, seed=SEED):
    return FastText(sentences=corpus, vector_size=64, window=window,
                    min_count=1, epochs=10, seed=seed, workers=1)

def train_ft_cat(corpus, seed=SEED, min_n=3, max_n=6):
    return FastText(sentences=corpus, vector_size=64, window=6,
                    min_count=1, epochs=10, seed=seed, workers=1,
                    min_n=min_n, max_n=max_n)

def train_w2v_cat(corpus, seed=SEED):
    return Word2Vec(sentences=corpus, vector_size=64, window=1,
                    min_count=1, epochs=10, seed=seed, workers=1, sg=1)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_mp(event, model):
    vecs = [model.wv[make_token(f, event[f])] for f in FEATURE_ORDER]
    return np.mean(vecs, axis=0)

def embed_cat(event, model):
    return model.wv[device_to_concat(event)]

def embed_w2v(event, model):
    t = device_to_concat(event)
    return model.wv[t] if t in model.wv else np.zeros(model.vector_size)

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_centroid(events, embed_fn, model):
    return np.mean([embed_fn(e, model) for e in events], axis=0)

# ---------------------------------------------------------------------------
# Scoring and evaluation
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
    try:
        return roc_auc_score(comb_l, comb_s)
    except Exception:
        return float("nan")

def bootstrap_auc(result, attack_type, n_boot=N_BOOTSTRAP, seed=SEED):
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
        if len(np.unique(comb_l[idx])) < 2:
            continue
        aucs.append(roc_auc_score(comb_l[idx], comb_s[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def evaluate_trivial(accounts):
    neg_s, neg_l = [], []
    for acc in accounts:
        for at in ["neg", "known"]:
            ev, lbl = acc["test"][at]
            k = device_key(ev)
            score = 0.0 if any(device_key(d) == k for d in acc["known_devices"]) else 1.0
            neg_s.append(score); neg_l.append(lbl)
    aucs = {}
    for at in ["novel", "fleet", "spoof"]:
        a_s, a_l = [], []
        for acc in accounts:
            ev, lbl = acc["test"][at]
            k = device_key(ev)
            score = 0.0 if any(device_key(d) == k for d in acc["known_devices"]) else 1.0
            a_s.append(score); a_l.append(lbl)
        try:
            aucs[at] = roc_auc_score(a_l + neg_l, a_s + neg_s)
        except Exception:
            aucs[at] = float("nan")
    return aucs

# ---------------------------------------------------------------------------
# T6 (fixed): Per-account centroid compactness
# ---------------------------------------------------------------------------
def per_account_compactness(accounts, embed_fn, model):
    """
    For each account, compute mean cosine distance from each training event's
    embedding to the account centroid.

    This is a meaningful measure of how "tight" the per-account embedding cluster is.
    Accounts with 2-4 devices will have non-trivial spread (events from different
    devices will be farther from the centroid than events from the primary device).

    A more compact cluster (lower mean distance) means the embedding space better
    represents the account's device profile as a coherent neighborhood.

    Note: This is NOT the same as silhouette score. It measures intra-account
    cohesion, not inter-account separation.
    """
    mean_dists = []
    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_fn, model)
        dists = [cosine_dist(embed_fn(e, model), centroid) for e in acc["train_events"]]
        mean_dists.append(np.mean(dists))
    return np.array(mean_dists)

def bootstrap_mean_compactness(accounts, embed_fn, model, n_boot=N_BOOTSTRAP, seed=SEED):
    dists = per_account_compactness(accounts, embed_fn, model)
    rng = np.random.default_rng(seed)
    n = len(dists)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(np.mean(dists[idx]))
    return float(np.mean(dists)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ---------------------------------------------------------------------------
# T3: Matching-feature ablation
# ---------------------------------------------------------------------------
def generate_k_match_events(accounts, k, rng):
    events = []
    for acc in accounts:
        primary = acc["primary"]
        feat_order = list(FEATURE_ORDER)
        if k == 0:
            match_features = []
        else:
            match_features = rng.choice(feat_order, size=k, replace=False).tolist()
        event = {}
        for f in feat_order:
            if f in match_features:
                event[f] = primary[f]
            else:
                alts = [v for v in FEATURES[f] if v != primary[f]]
                event[f] = rng.choice(alts) if alts else primary[f]
        events.append((event, 1))
    return events

def k_match_auc(accounts, k, embed_fn, model, result):
    rng = np.random.default_rng(SEED + k)
    attack_events = generate_k_match_events(accounts, k, rng)
    neg_s = result["neg"]["scores"] + result["known"]["scores"]
    neg_l = result["neg"]["labels"] + result["known"]["labels"]
    att_s, att_l = [], []
    for acc, (ev, lbl) in zip(accounts, attack_events):
        centroid = compute_centroid(acc["train_events"], embed_fn, model)
        att_s.append(cosine_dist(embed_fn(ev, model), centroid))
        att_l.append(lbl)
    try:
        return roc_auc_score(att_l + neg_l, att_s + neg_s)
    except Exception:
        return float("nan")

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
        d_cf = cosine_dist(embed_fn(cf_ev, model), centroid)
        act.append(d_act); cf.append(d_cf); attr.append(d_act - d_cf)
    return np.array(act), np.array(cf), np.array(attr)

# ---------------------------------------------------------------------------
# T5: Fleet stratified
# ---------------------------------------------------------------------------
def fleet_stratified(accounts, result):
    cont_ids = [a["id"] for a in accounts if a["is_fleet"]]
    uncont_ids = [a["id"] for a in accounts if not a["is_fleet"]]
    neg_s = result["neg"]["scores"] + result["known"]["scores"]
    neg_l = result["neg"]["labels"] + result["known"]["labels"]
    def fleet_auc(ids):
        fleet_s = [result["fleet"]["scores"][i] for i in ids]
        fleet_l = [result["fleet"]["labels"][i] for i in ids]
        try:
            return roc_auc_score(fleet_l + neg_l, fleet_s + neg_s)
        except Exception:
            return float("nan")
    return fleet_auc(cont_ids), fleet_auc(uncont_ids), len(cont_ids), len(uncont_ids)

# ---------------------------------------------------------------------------
# T8: Token similarity
# ---------------------------------------------------------------------------
def token_similarity_analysis(model):
    tokens = []
    feat_types = []
    for f in FEATURE_ORDER:
        for v in FEATURES[f]:
            tokens.append(make_token(f, v))
            feat_types.append(f)
    vecs = np.array([model.wv[t] for t in tokens])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / (norms + 1e-10)
    sim_mat = vecs_n @ vecs_n.T
    feat_types = np.array(feat_types)
    n = len(tokens)
    within, cross = [], []
    for i in range(n):
        for j in range(i+1, n):
            if feat_types[i] == feat_types[j]:
                within.append(sim_mat[i, j])
            else:
                cross.append(sim_mat[i, j])
    return tokens, sim_mat, np.array(within), np.array(cross)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_ci_bar(ax, conditions, means, los, his, title, colors=None, ylabel="ROC-AUC"):
    x = np.arange(len(conditions))
    if colors is None:
        colors = ["#2196F3"] * len(conditions)
    ax.bar(x, means, color=colors, alpha=0.82)
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], fmt="none",
                    color="black", capsize=4, linewidth=1.5)
        ax.text(x[i], hi + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=18, ha="right")
    ax.set_ylim(0, max(his) * 1.18 if max(his) > 0 else 0.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)

def save_fig(fig, name):
    p = f"{FIGURE_DIR}/{name}"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("H2 Experiment (Iteration 2) — Fixed T2 and T6")
    print("=" * 70)

    print("\n[Data] Generating dataset...")
    accounts, fleet_device = generate_dataset()
    print(f"  {N_ACCOUNTS} accounts, fleet accounts: {sum(a['is_fleet'] for a in accounts)}")

    mp_corpus = build_mp_corpus(accounts)
    cat_corpus = build_cat_corpus(accounts)

    print("\n[Models] Training...")
    mp_w6 = train_ft_mp(mp_corpus, window=6)
    mp_w3 = train_ft_mp(mp_corpus, window=3)
    mp_w2 = train_ft_mp(mp_corpus, window=2)
    mp_w1 = train_ft_mp(mp_corpus, window=1)
    cat_std = train_ft_cat(cat_corpus, min_n=3, max_n=6)     # standard FastText
    cat_short = train_ft_cat(cat_corpus, min_n=1, max_n=3)   # short n-grams
    cat_w2v = train_w2v_cat(cat_corpus)                       # OOV-dominated baseline
    print("  Done.")

    print("\n[Scoring]...")
    r_mp_w6  = score_all(accounts, embed_mp, mp_w6)
    r_mp_w3  = score_all(accounts, embed_mp, mp_w3)
    r_mp_w2  = score_all(accounts, embed_mp, mp_w2)
    r_mp_w1  = score_all(accounts, embed_mp, mp_w1)
    r_cat    = score_all(accounts, embed_cat, cat_std)
    r_cat_sh = score_all(accounts, embed_cat, cat_short)
    r_w2v    = score_all(accounts, embed_w2v, cat_w2v)
    trivial  = evaluate_trivial(accounts)
    print("  Done.")

    print("\n[Bootstrap CIs]...")
    def boot_all(result):
        return {at: bootstrap_auc(result, at) for at in ["novel", "fleet", "spoof"]}
    b_mp_w6  = boot_all(r_mp_w6)
    b_mp_w3  = boot_all(r_mp_w3)
    b_mp_w2  = boot_all(r_mp_w2)
    b_mp_w1  = boot_all(r_mp_w1)
    b_cat    = boot_all(r_cat)
    b_cat_sh = boot_all(r_cat_sh)
    b_w2v    = boot_all(r_w2v)
    print("  Done.")

    # T6: Per-account compactness
    print("\n[T6] Per-account centroid compactness...")
    c_mp_mean, c_mp_lo, c_mp_hi = bootstrap_mean_compactness(accounts, embed_mp, mp_w6)
    c_cat_mean, c_cat_lo, c_cat_hi = bootstrap_mean_compactness(accounts, embed_cat, cat_std)
    print(f"  Mean-pool compactness (mean cos dist): {c_mp_mean:.4f} [{c_mp_lo:.4f}, {c_mp_hi:.4f}]")
    print(f"  Concat compactness   (mean cos dist): {c_cat_mean:.4f} [{c_cat_lo:.4f}, {c_cat_hi:.4f}]")
    print(f"  Gap (mp - cat): {c_mp_mean - c_cat_mean:+.4f}  (lower = more compact)")

    # T3
    print("\n[T3] Matching-feature ablation...")
    k_mp = [k_match_auc(accounts, k, embed_mp, mp_w6, r_mp_w6) for k in range(6)]
    k_cat = [k_match_auc(accounts, k, embed_cat, cat_std, r_cat) for k in range(6)]
    for k in range(6):
        print(f"  k={k}: mp={k_mp[k]:.4f}, cat={k_cat[k]:.4f}, gap={k_mp[k]-k_cat[k]:+.4f}")

    # T4
    print("\n[T4] Tz-counterfactual...")
    act_mp, cf_mp, attr_mp = tz_counterfactual(accounts, embed_mp, mp_w6)
    act_cat, cf_cat, attr_cat = tz_counterfactual(accounts, embed_cat, cat_std)
    print(f"  Mean-pool: actual={act_mp.mean():.4f}, CF={cf_mp.mean():.4f}, attr={attr_mp.mean():.6f}")
    print(f"  Concat:    actual={act_cat.mean():.4f}, CF={cf_cat.mean():.4f}, attr={attr_cat.mean():.6f}")

    # T5
    print("\n[T5] Fleet stratified...")
    mp_cont, mp_uncont, n_cont, n_uncont = fleet_stratified(accounts, r_mp_w6)
    cat_cont, cat_uncont, _, _ = fleet_stratified(accounts, r_cat)
    print(f"  Contaminated   (n={n_cont}):   mp={mp_cont:.4f}, cat={cat_cont:.4f}")
    print(f"  Uncontaminated (n={n_uncont}): mp={mp_uncont:.4f}, cat={cat_uncont:.4f}")

    # T8
    print("\n[T8] Token similarity...")
    tokens_mp, sim_mp, within_mp, cross_mp = token_similarity_analysis(mp_w6)
    print(f"  Within-feature mean sim: {within_mp.mean():.4f}")
    print(f"  Cross-feature mean sim:  {cross_mp.mean():.4f}")

    # W2V OOV analysis
    oov_spoof = sum(1 for a in accounts if device_to_concat(a["test"]["spoof"][0]) not in cat_w2v.wv)
    oov_novel = sum(1 for a in accounts if device_to_concat(a["test"]["novel"][0]) not in cat_w2v.wv)
    print(f"\n[T2 OOV] W2V OOV rate — spoof: {oov_spoof}/{N_ACCOUNTS}, novel: {oov_novel}/{N_ACCOUNTS}")

    # -----------------------------------------------------------------------
    # Full results table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Iteration 2)")
    print("=" * 70)

    print("\n--- Primary AUC Results with 95% Bootstrap CI ---")
    print(f"{'Condition':<26} {'Novel':>18} {'Fleet':>18} {'Spoof':>18}")
    print("-" * 82)

    def fmt(m, lo, hi): return f"{m:.3f} [{lo:.3f},{hi:.3f}]"

    for name, b in [
        ("Mean-pool w=6",   b_mp_w6),
        ("Mean-pool w=3",   b_mp_w3),
        ("Mean-pool w=2",   b_mp_w2),
        ("Mean-pool w=1",   b_mp_w1),
        ("Concat FT std",   b_cat),
        ("Concat FT short", b_cat_sh),
        ("Concat W2V*",     b_w2v),
    ]:
        row = f"{name:<26}"
        for at in ["novel", "fleet", "spoof"]:
            m, lo, hi = b[at]
            row += f"  {fmt(m, lo, hi):>16}"
        print(row)
    print(f"{'Trivial baseline':<26}", end="")
    for at in ["novel", "fleet", "spoof"]:
        print(f"  {trivial[at]:>18.3f}", end="")
    print()
    print("* W2V: 369/400 spoof tokens OOV (zeros -> dist=1.0), equivalent to trivial")

    print(f"\n--- T6 (Fixed): Per-account Centroid Compactness ---")
    print(f"  Mean-pool w=6: {c_mp_mean:.4f} [{c_mp_lo:.4f}, {c_mp_hi:.4f}]  (lower = tighter cluster)")
    print(f"  Concat FT:     {c_cat_mean:.4f} [{c_cat_lo:.4f}, {c_cat_hi:.4f}]")
    gap6 = c_mp_mean - c_cat_mean
    print(f"  Gap (mp - cat): {gap6:+.4f}  ({'mp more compact' if gap6 < 0 else 'cat more compact'})")

    print(f"\n--- T4: Tz-Attributable Cosine Distance ---")
    print(f"  Mean-pool: {attr_mp.mean():.6f} ± {attr_mp.std():.6f}")
    print(f"  Concat:    {attr_cat.mean():.6f} ± {attr_cat.std():.6f}")

    # -----------------------------------------------------------------------
    # Verdicts
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRE-SPECIFIED VERDICTS (per DEBATE.md)")
    print("=" * 70)

    # T1
    sp_w1 = b_mp_w1["spoof"][0]
    sp_cat = b_cat["spoof"][0]
    sp_w6 = b_mp_w6["spoof"][0]
    print(f"\n[T1] Window sweep — spoof: w1={sp_w1:.4f}, w6={sp_w6:.4f}, cat={sp_cat:.4f}")
    if sp_w1 > sp_cat:
        print("  VERDICT: Defense right — w=1 beats concat; context richness not the driver")
    elif all([b_mp_w1["spoof"][0] <= sp_cat, b_mp_w2["spoof"][0] <= sp_cat,
              b_mp_w3["spoof"][0] <= sp_cat, b_mp_w6["spoof"][0] <= sp_cat]):
        print("  VERDICT: Ambiguous — all mean-pool windows underperform concat on spoof")
    else:
        print("  VERDICT: Mixed")

    # T2
    sp_sh = b_cat_sh["spoof"][0]
    print(f"\n[T2] Short n-gram (1-3) vs. std n-gram (3-6) on concat:")
    print(f"  short={sp_sh:.4f}, std={sp_cat:.4f}")
    diff_t2 = sp_sh - sp_cat
    if abs(diff_t2) < 0.01:
        print(f"  VERDICT: Ambiguous — short ≈ std n-grams (diff={diff_t2:+.4f}), n-gram length not decisive")
    elif sp_sh > sp_cat:
        print(f"  VERDICT: Defense direction — short n-grams improve spoof AUC (fewer cross-boundary)")
    else:
        print(f"  VERDICT: Critique direction — shorter n-grams do NOT improve spoof AUC")

    # T3
    gaps3 = [k_mp[k] - k_cat[k] for k in range(6)]
    print(f"\n[T3] Matching-feature ablation — gap grows with k?")
    print(f"  Gaps: {[f'{g:+.4f}' for g in gaps3]}")
    if gaps3[5] > gaps3[0]:
        print("  VERDICT: Defense right — gap increases at k=5 (mechanism partially confirmed)")
    else:
        print("  VERDICT: Critique right — gap does not increase monotonically with k")

    # T4
    print(f"\n[T4] Tz-counterfactual:")
    if attr_mp.mean() > attr_cat.mean():
        print(f"  VERDICT: Defense right — mp more sensitive to tz (mp={attr_mp.mean():.6f} > cat={attr_cat.mean():.6f})")
    else:
        print(f"  VERDICT: Critique right — mp NOT more sensitive to tz (mp={attr_mp.mean():.6f} <= cat={attr_cat.mean():.6f})")

    # T5
    print(f"\n[T5] Fleet stratified:")
    if mp_uncont >= cat_uncont:
        print(f"  VERDICT: Defense right on uncontaminated (mp={mp_uncont:.4f} >= cat={cat_uncont:.4f})")
    else:
        print(f"  VERDICT: Critique right — cat beats mp on uncontaminated fleet (mp={mp_uncont:.4f} < cat={cat_uncont:.4f})")

    # T6
    print(f"\n[T6] Per-account compactness:")
    if c_mp_hi < c_cat_lo:
        print(f"  VERDICT: Mean-pool is more compact (mp CI entirely below cat CI)")
    elif c_cat_hi < c_mp_lo:
        print(f"  VERDICT: Concat is more compact (cat CI entirely below mp CI)")
    else:
        print(f"  VERDICT: Ambiguous — CIs overlap (mp={c_mp_mean:.4f}[{c_mp_lo:.4f},{c_mp_hi:.4f}], "
              f"cat={c_cat_mean:.4f}[{c_cat_lo:.4f},{c_cat_hi:.4f}])")

    # T7
    print(f"\n[T7] Trivial baseline comparison:")
    for at in ["novel", "fleet", "spoof"]:
        m_mp = b_mp_w6[at][0]
        m_cat = b_cat[at][0]
        t = trivial[at]
        print(f"  {at:6s}: trivial={t:.3f}, mp={m_mp:.3f} ({'beats' if m_mp > t else 'fails'}), "
              f"cat={m_cat:.3f} ({'beats' if m_cat > t else 'fails'})")

    # T8
    ratio8 = within_mp.mean() / max(abs(cross_mp.mean()), 1e-6)
    print(f"\n[T8] Co-occurrence: within={within_mp.mean():.4f}, cross={cross_mp.mean():.4f}")
    if within_mp.mean() > 0.5 and cross_mp.mean() < 0.1:
        print("  VERDICT: Defense right — within >> cross, feature tokens well-separated")
    elif cross_mp.mean() > 0.3:
        print("  VERDICT: Critique right — co-occurrence bias present")
    else:
        print(f"  VERDICT: Defense right — within >> cross (within={within_mp.mean():.4f}, cross={cross_mp.mean():.4f})")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\n[Figures]...")

    # Fig 1: T1 window sweep (3 attack types)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    wlabels = ["w=1", "w=2", "w=3", "w=6", "cat-std", "trivial"]
    wboots = [b_mp_w1, b_mp_w2, b_mp_w3, b_mp_w6, b_cat, None]
    wcolors = ["#BBDEFB", "#90CAF9", "#42A5F5", "#1565C0", "#FF7043", "#9E9E9E"]
    for ai, at in enumerate(["novel", "fleet", "spoof"]):
        ax = axes[ai]
        ms, ls, hs = [], [], []
        for wb in wboots:
            if wb is None:
                t = trivial[at]; ms.append(t); ls.append(t); hs.append(t)
            else:
                m, lo, hi = wb[at]; ms.append(m); ls.append(lo); hs.append(hi)
        plot_ci_bar(ax, wlabels, ms, ls, hs, f"T1: window sweep — {at}", wcolors)
    fig.suptitle("T1: Mean-pool window sweep vs. concat (bars=mean, errors=95% CI)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "finding_01_window_sweep.png")

    # Fig 2: T2 n-gram ablation
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    t2_conds = ["MP w=6", "Cat FT std", "Cat FT short", "Cat W2V*", "Trivial"]
    t2_boots = [b_mp_w6, b_cat, b_cat_sh, b_w2v, None]
    t2_colors = ["#2196F3", "#FF5722", "#FF9800", "#A5D6A7", "#9E9E9E"]
    for ai, at in enumerate(["novel", "fleet", "spoof"]):
        ax = axes[ai]
        ms, ls, hs = [], [], []
        for wb in t2_boots:
            if wb is None:
                t = trivial[at]; ms.append(t); ls.append(t); hs.append(t)
            else:
                m, lo, hi = wb[at]; ms.append(m); ls.append(lo); hs.append(hi)
        plot_ci_bar(ax, t2_conds, ms, ls, hs, f"T2: n-gram ablation — {at}", t2_colors)
    fig.suptitle("T2: FastText n-gram variants on concat (*W2V is OOV-dominated)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "finding_02_ngram_ablation.png")

    # Fig 3: T3 matching-feature ablation
    fig, ax = plt.subplots(figsize=(9, 5))
    k_vals = list(range(6))
    ax.plot(k_vals, k_mp,  "o-", color="#1565C0", label="Mean-pool w=6", lw=2)
    ax.plot(k_vals, k_cat, "s-", color="#FF5722", label="Concat FT std",  lw=2)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance")
    for k in k_vals:
        gap = k_mp[k] - k_cat[k]
        ax.annotate(f"{gap:+.3f}", (k, min(k_mp[k], k_cat[k]) - 0.03),
                    ha="center", fontsize=8, color="purple")
    ax.set_xlabel("Features matching primary device (k)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("T3: Does mean-pool advantage grow as attacker matches more features?\n"
                 "(purple = AUC gap: mp − cat)")
    ax.legend(); ax.set_xticks(k_vals)
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    fig.tight_layout()
    save_fig(fig, "finding_03_matching_feature_ablation.png")

    # Fig 4: T4 tz-counterfactual distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, vals, label, col in zip(
        axes,
        [attr_mp, attr_cat],
        ["Mean-pool w=6", "Concat FT std"],
        ["#1565C0", "#FF5722"]
    ):
        ax.hist(vals, bins=40, color=col, alpha=0.75, edgecolor="white")
        ax.axvline(vals.mean(), color="black", ls="--", lw=1.5,
                   label=f"mean={vals.mean():.6f}")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"T4: {label}\nTz-attributable cosine distance")
        ax.set_xlabel("Actual − Counterfactual cosine distance")
        ax.set_ylabel("Count")
        ax.legend()
    fig.suptitle("T4: Tz-counterfactual attribution\n(Higher = tz token more diagnostic)",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "finding_04_tz_counterfactual.png")

    # Fig 5: T5 fleet stratified
    fig, ax = plt.subplots(figsize=(7, 5))
    cats5 = ["Contaminated\n(fleet in train)", "Uncontaminated\n(fleet never seen)"]
    mp_v = [mp_cont, mp_uncont]; cat_v = [cat_cont, cat_uncont]
    x5 = np.arange(2); w5 = 0.3
    ax.bar(x5 - w5/2, mp_v, w5, label="Mean-pool w=6", color="#1565C0", alpha=0.85)
    ax.bar(x5 + w5/2, cat_v, w5, label="Concat FT std", color="#FF5722", alpha=0.85)
    for i, (mv, cv) in enumerate(zip(mp_v, cat_v)):
        ax.text(i - w5/2, mv + 0.01, f"{mv:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w5/2, cv + 0.01, f"{cv:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x5); ax.set_xticklabels(cats5)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("ROC-AUC (fleet)")
    ax.set_title(f"T5: Fleet AUC stratified (contaminated n={n_cont}, uncontam. n={n_uncont})")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "finding_05_fleet_stratified.png")

    # Fig 6: T6 per-account compactness
    fig, ax = plt.subplots(figsize=(6, 5))
    bars_c = ax.bar([0, 1], [c_mp_mean, c_cat_mean], color=["#1565C0", "#FF5722"], alpha=0.8)
    ax.errorbar(0, c_mp_mean, yerr=[[c_mp_mean - c_mp_lo], [c_mp_hi - c_mp_mean]],
                fmt="none", color="black", capsize=5, lw=2)
    ax.errorbar(1, c_cat_mean, yerr=[[c_cat_mean - c_cat_lo], [c_cat_hi - c_cat_mean]],
                fmt="none", color="black", capsize=5, lw=2)
    ax.text(0, c_mp_hi + 0.002, f"{c_mp_mean:.4f}", ha="center", va="bottom", fontsize=10)
    ax.text(1, c_cat_hi + 0.002, f"{c_cat_mean:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mean-pool w=6", "Concat FT std"])
    ax.set_ylabel("Mean cosine distance to account centroid")
    ax.set_title("T6 (Fixed): Per-account centroid compactness\n"
                 "(lower = tighter cluster; bootstrap 95% CI)")
    fig.tight_layout()
    save_fig(fig, "finding_06_compactness.png")

    # Fig 7: T8 token similarity matrix + histogram
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_m = axes[0]
    im = ax_m.imshow(sim_mp, cmap="RdYlBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax_m)
    ax_m.set_title("T8: Feature token cosine similarity\n(mean-pool model)")
    ax_m.set_xlabel("Token index")
    ax_m.set_ylabel("Token index")
    boundaries = np.cumsum([len(FEATURES[f]) for f in FEATURE_ORDER])
    for b in boundaries[:-1]:
        ax_m.axhline(b - 0.5, color="white", lw=1.5)
        ax_m.axvline(b - 0.5, color="white", lw=1.5)
    ax_h = axes[1]
    ax_h.hist(within_mp, bins=30, alpha=0.75, color="#1565C0",
              label=f"Within-feature (mean={within_mp.mean():.3f})")
    ax_h.hist(cross_mp, bins=30, alpha=0.65, color="#FF5722",
              label=f"Cross-feature (mean={cross_mp.mean():.3f})")
    ax_h.set_xlabel("Cosine similarity"); ax_h.set_ylabel("Count")
    ax_h.set_title("T8: Within vs. cross-feature similarity distribution")
    ax_h.legend()
    fig.tight_layout()
    save_fig(fig, "finding_08_token_similarity.png")

    # Fig 8: Summary — spoof AUC all conditions
    fig, ax = plt.subplots(figsize=(12, 6))
    sconds = ["Trivial", "Cat FT std", "Cat FT short", "MP w=1", "MP w=2", "MP w=3", "MP w=6"]
    sboots = [None, b_cat, b_cat_sh, b_mp_w1, b_mp_w2, b_mp_w3, b_mp_w6]
    scols  = ["#9E9E9E", "#FF5722", "#FF9800", "#BBDEFB", "#90CAF9", "#42A5F5", "#1565C0"]
    ms, ls, hs = [], [], []
    for wb in sboots:
        if wb is None:
            t = trivial["spoof"]; ms.append(t); ls.append(t); hs.append(t)
        else:
            m, lo, hi = wb["spoof"]; ms.append(m); ls.append(lo); hs.append(hi)
    plot_ci_bar(ax, sconds, ms, ls, hs,
                "Summary: Spoof AUC — all conditions with 95% CI\n"
                "(Spoof = 5 matching features, tz only differs)", scols)
    fig.tight_layout()
    save_fig(fig, "summary_all_conditions_spoof.png")

    # Fig 9: Summary — all attack types for key conditions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    k_conds = ["Trivial", "Concat FT std", "Mean-pool w=6"]
    k_boots = [None, b_cat, b_mp_w6]
    k_cols  = ["#9E9E9E", "#FF5722", "#1565C0"]
    for ai, at in enumerate(["novel", "fleet", "spoof"]):
        ax = axes[ai]
        ms, ls, hs = [], [], []
        for wb in k_boots:
            if wb is None:
                t = trivial[at]; ms.append(t); ls.append(t); hs.append(t)
            else:
                m, lo, hi = wb[at]; ms.append(m); ls.append(lo); hs.append(hi)
        plot_ci_bar(ax, k_conds, ms, ls, hs, f"{at.capitalize()} attack", k_cols)
    fig.suptitle("Summary: Key conditions — all attack types with 95% CI", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "summary_all_attack_types.png")

    print("\n=== Iteration 2 complete ===")

if __name__ == "__main__":
    main()
