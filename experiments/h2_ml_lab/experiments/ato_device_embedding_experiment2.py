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
# H2 Experiment — Mean-pool vs. Concat FastText for ATO Device Embedding
#
# Implements all 8 empirical tests from DEBATE.md:
#   T1 — Window sweep (window=1,2,3,6 for mean-pool)
#   T2 — N-gram ablation via Word2Vec (no subword n-grams on concat corpus)
#   T3 — Matching-feature ablation (k=0..5 features matching primary)
#   T4 — Tz-counterfactual attribution
#   T5 — Fleet AUC stratified (contaminated vs. uncontaminated)
#   T6 — Corrected silhouette (true device labels)
#   T7 — Trivial baseline (exact 6/6 set-membership)
#   T8 — Co-occurrence bias in mean-pool token space
#
# Pre-specified verdicts per DEBATE.md.
# Bootstrap CIs: N=1000, percentile method.

import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gensim.models import FastText, Word2Vec
from sklearn.metrics import roc_auc_score
from sklearn.metrics import silhouette_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_BOOTSTRAP = 1000
FIGURE_DIR = Path(__file__).parent.parent / "figures"

# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------
FEATURES = {
    "os":      ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz":      ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang":    ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "network": ["wifi", "lte", "5g", "broadband"],
    "screen":  ["small", "medium", "large", "xlarge"],
}
FEATURE_ORDER = ["os", "browser", "tz", "lang", "network", "screen"]

def make_token(feature, value):
    return f"{feature}_{value}"

def device_to_tokens(device):
    return [make_token(f, device[f]) for f in FEATURE_ORDER]

def device_to_concat_string(device):
    return "_".join(device[f] for f in FEATURE_ORDER)

def device_key(device):
    return tuple(device[f] for f in FEATURE_ORDER)

# ---------------------------------------------------------------------------
# Data generation
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
            seen.add(k)
            devices.append(d)
    return devices

def zipf_weights(n, s=1.5):
    w = np.array([1.0 / k**s for k in range(1, n + 1)])
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

        # Novel: OS, tz, and lang all differ from primary
        novel = sample_device(rng)
        attempts = 0
        while (novel["os"] == primary["os"] or novel["tz"] == primary["tz"]
               or novel["lang"] == primary["lang"]) and attempts < 1000:
            novel = sample_device(rng)
            attempts += 1

        # Fleet test: the shared fleet device
        fleet_test = dict(fleet_device)

        # Spoof: OS/browser/lang match primary, tz differs
        spoof = dict(primary)
        alt_tzs = [t for t in FEATURES["tz"] if t != primary["tz"]]
        spoof["tz"] = rng.choice(alt_tzs)

        # Neg (enrollment): same OS/browser/tz/lang, different network/screen
        neg = dict(primary)
        alt_net = [n for n in FEATURES["network"] if n != primary["network"]]
        alt_scr = [s for s in FEATURES["screen"] if s != primary["screen"]]
        neg["network"] = rng.choice(alt_net)
        neg["screen"] = rng.choice(alt_scr)

        # Known: exact match to primary
        known_test = dict(primary)

        accounts.append({
            "id": acct_id,
            "known_devices": known,
            "primary": primary,
            "train_events": train_events,
            "is_fleet": is_fleet,
            "fleet_device": fleet_device,
            "test": {
                "novel": (novel, 1),
                "fleet": (fleet_test, 1),
                "spoof": (spoof, 1),
                "neg": (neg, 0),
                "known": (known_test, 0),
            },
        })

    return accounts, fleet_device

# ---------------------------------------------------------------------------
# FastText training
# ---------------------------------------------------------------------------
def build_mean_pool_corpus(accounts):
    return [device_to_tokens(e) for a in accounts for e in a["train_events"]]

def build_concat_corpus(accounts):
    return [[device_to_concat_string(e)] for a in accounts for e in a["train_events"]]

def train_fasttext_mp(corpus, window=6, seed=SEED):
    return FastText(
        sentences=corpus, vector_size=64, window=window,
        min_count=1, epochs=10, seed=seed, workers=1,
    )

def train_fasttext_cat(corpus, seed=SEED):
    return FastText(
        sentences=corpus, vector_size=64, window=6,
        min_count=1, epochs=10, seed=seed, workers=1,
    )

def train_word2vec_cat(corpus, seed=SEED):
    """Word2Vec on concat tokens — no character n-grams."""
    return Word2Vec(
        sentences=corpus, vector_size=64, window=1,
        min_count=1, epochs=10, seed=seed, workers=1,
        sg=1,  # skip-gram
    )

# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------
def embed_mp(event, model):
    vecs = [model.wv[make_token(f, event[f])] for f in FEATURE_ORDER]
    return np.mean(vecs, axis=0)

def embed_cat(event, model):
    return model.wv[device_to_concat_string(event)]

def embed_w2v(event, model):
    token = device_to_concat_string(event)
    if token in model.wv:
        return model.wv[token]
    return np.zeros(model.vector_size)

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_centroid(events, embed_fn, model):
    vecs = [embed_fn(e, model) for e in events]
    return np.mean(vecs, axis=0)

# ---------------------------------------------------------------------------
# Trivial baseline
# ---------------------------------------------------------------------------
def trivial_score(event, known_devices):
    k = device_key(event)
    for d in known_devices:
        if device_key(d) == k:
            return 0.0
    return 1.0

def evaluate_trivial(accounts):
    neg_scores, neg_labels = [], []
    for acc in accounts:
        for at in ["neg", "known"]:
            ev, lbl = acc["test"][at]
            neg_scores.append(trivial_score(ev, acc["known_devices"]))
            neg_labels.append(lbl)

    aucs = {}
    for at in ["novel", "fleet", "spoof"]:
        a_scores, a_labels = [], []
        for acc in accounts:
            ev, lbl = acc["test"][at]
            a_scores.append(trivial_score(ev, acc["known_devices"]))
            a_labels.append(lbl)
        combined = a_scores + neg_scores
        clabels = a_labels + neg_labels
        try:
            aucs[at] = roc_auc_score(clabels, combined)
        except Exception:
            aucs[at] = float("nan")
    return aucs

# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def score_all(accounts, embed_fn, model):
    """Return per-attack-type scores and labels."""
    result = {k: {"scores": [], "labels": []} for k in ["novel", "fleet", "spoof", "neg", "known"]}
    centroids = {}
    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_fn, model)
        centroids[acc["id"]] = centroid
        for at, (ev, lbl) in acc["test"].items():
            d = cosine_dist(embed_fn(ev, model), centroid)
            result[at]["scores"].append(d)
            result[at]["labels"].append(lbl)
    return result, centroids

def compute_auc(result, attack_type):
    neg_scores = result["neg"]["scores"] + result["known"]["scores"]
    neg_labels = result["neg"]["labels"] + result["known"]["labels"]
    combined = result[attack_type]["scores"] + neg_scores
    clabels = result[attack_type]["labels"] + neg_labels
    try:
        return roc_auc_score(clabels, combined)
    except Exception:
        return float("nan")

def bootstrap_auc(result, attack_type, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    neg_s = np.array(result["neg"]["scores"] + result["known"]["scores"])
    neg_l = np.array(result["neg"]["labels"] + result["known"]["labels"])
    att_s = np.array(result[attack_type]["scores"])
    att_l = np.array(result[attack_type]["labels"])
    combined_s = np.concatenate([att_s, neg_s])
    combined_l = np.concatenate([att_l, neg_l])
    n = len(combined_s)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(combined_l[idx])) < 2:
            continue
        aucs.append(roc_auc_score(combined_l[idx], combined_s[idx]))
    aucs = np.array(aucs)
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def bootstrap_silhouette(vecs, labels, n_boot=N_BOOTSTRAP, seed=SEED, sample_size=2000):
    rng = np.random.default_rng(seed)
    vecs = np.array(vecs)
    labels = np.array(labels)
    n = len(vecs)
    sils = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=min(sample_size, n), replace=True)
        s_vecs = vecs[idx]
        s_labels = labels[idx]
        if len(np.unique(s_labels)) < 2:
            continue
        try:
            sils.append(silhouette_score(s_vecs, s_labels, metric="cosine"))
        except Exception:
            continue
    sils = np.array(sils)
    return np.mean(sils), np.percentile(sils, 2.5), np.percentile(sils, 97.5)

# ---------------------------------------------------------------------------
# T6: Corrected silhouette with true device labels
# ---------------------------------------------------------------------------
def collect_embeddings_true_labels(accounts, embed_fn, model, max_events=5000):
    vecs, labels = [], []
    label_map = {}
    label_counter = [0]
    total = 0
    for acc in accounts:
        for ev in acc["train_events"]:
            if total >= max_events:
                break
            k = device_key(ev)
            if k not in label_map:
                label_map[k] = label_counter[0]
                label_counter[0] += 1
            vecs.append(embed_fn(ev, model))
            labels.append(label_map[k])
            total += 1
        if total >= max_events:
            break
    return vecs, labels

# ---------------------------------------------------------------------------
# T3: Matching-feature ablation
# ---------------------------------------------------------------------------
def generate_k_match_events(accounts, k, rng):
    """
    Generate one test event per account with exactly k features matching the primary device.
    k features chosen randomly; remaining 6-k features are randomized.
    Returns list of (event, label=1) — these are attacks.
    """
    events = []
    for acc in accounts:
        primary = acc["primary"]
        feat_order = list(FEATURE_ORDER)
        # Choose which k features to match
        match_features = rng.choice(feat_order, size=k, replace=False).tolist()
        event = {}
        for f in feat_order:
            if f in match_features:
                event[f] = primary[f]
            else:
                # Different value
                alts = [v for v in FEATURES[f] if v != primary[f]]
                if not alts:
                    event[f] = primary[f]  # only one value, must match
                else:
                    event[f] = rng.choice(alts)
        events.append((event, 1))
    return events

def k_match_auc(accounts, k, embed_fn, model, result):
    """AUC for k-match attack events vs. standard negatives."""
    rng = np.random.default_rng(SEED + k)
    attack_events = generate_k_match_events(accounts, k, rng)
    neg_scores = result["neg"]["scores"] + result["known"]["scores"]
    neg_labels = result["neg"]["labels"] + result["known"]["labels"]
    att_scores, att_labels = [], []
    for acc, (ev, lbl) in zip(accounts, attack_events):
        centroid = np.mean(
            [embed_fn(e, model) for e in acc["train_events"]], axis=0
        )
        d = cosine_dist(embed_fn(ev, model), centroid)
        att_scores.append(d)
        att_labels.append(lbl)
    combined = att_scores + neg_scores
    clabels = att_labels + neg_labels
    try:
        return roc_auc_score(clabels, combined)
    except Exception:
        return float("nan")

# ---------------------------------------------------------------------------
# T4: Tz-counterfactual attribution
# ---------------------------------------------------------------------------
def tz_counterfactual_distance(accounts, embed_fn, model):
    """
    For spoof events (tz differs from primary), compute:
    - actual cosine distance to centroid
    - counterfactual distance when tz token is replaced with primary tz
    - tz-attributable distance = actual - counterfactual
    """
    actual_dists, cf_dists, tz_attr = [], [], []
    for acc in accounts:
        centroid = np.mean(
            [embed_fn(e, model) for e in acc["train_events"]], axis=0
        )
        spoof_ev, _ = acc["test"]["spoof"]
        actual_d = cosine_dist(embed_fn(spoof_ev, model), centroid)

        # Counterfactual: spoof with primary tz
        cf_ev = dict(spoof_ev)
        cf_ev["tz"] = acc["primary"]["tz"]
        cf_d = cosine_dist(embed_fn(cf_ev, model), centroid)

        actual_dists.append(actual_d)
        cf_dists.append(cf_d)
        tz_attr.append(actual_d - cf_d)

    return np.array(actual_dists), np.array(cf_dists), np.array(tz_attr)

# ---------------------------------------------------------------------------
# T8: Co-occurrence bias in mean-pool token space
# ---------------------------------------------------------------------------
def token_similarity_matrix(model, mode="mp"):
    """Return pairwise cosine similarity matrix for all feature tokens."""
    tokens = []
    for f in FEATURE_ORDER:
        for v in FEATURES[f]:
            tokens.append(make_token(f, v))

    vecs = np.array([model.wv[t] for t in tokens])
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / (norms + 1e-10)
    sim_matrix = vecs_norm @ vecs_norm.T
    return tokens, sim_matrix

def cross_vs_within_similarity(tokens, sim_matrix):
    """
    Compare mean cosine similarity within same feature type vs. across feature types.
    """
    # Assign feature type to each token
    feat_types = []
    for f in FEATURE_ORDER:
        for v in FEATURES[f]:
            feat_types.append(f)
    feat_types = np.array(feat_types)

    n = len(tokens)
    within_sims, cross_sims = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if feat_types[i] == feat_types[j]:
                within_sims.append(sim_matrix[i, j])
            else:
                cross_sims.append(sim_matrix[i, j])

    return np.array(within_sims), np.array(cross_sims)

# ---------------------------------------------------------------------------
# T5: Fleet AUC stratified
# ---------------------------------------------------------------------------
def fleet_auc_stratified(accounts, result):
    contaminated_ids = [a["id"] for a in accounts if a["is_fleet"]]
    uncontaminated_ids = [a["id"] for a in accounts if not a["is_fleet"]]

    neg_scores = result["neg"]["scores"] + result["known"]["scores"]
    neg_labels = result["neg"]["labels"] + result["known"]["labels"]

    def fleet_auc_for_ids(ids):
        fleet_s = [result["fleet"]["scores"][i] for i in ids]
        fleet_l = [result["fleet"]["labels"][i] for i in ids]
        combined = fleet_s + neg_scores
        clabels = fleet_l + neg_labels
        try:
            return roc_auc_score(clabels, combined)
        except Exception:
            return float("nan")

    return (
        fleet_auc_for_ids(contaminated_ids),
        fleet_auc_for_ids(uncontaminated_ids),
        len(contaminated_ids),
        len(uncontaminated_ids),
    )

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_auc_bar_with_ci(ax, conditions, aucs_mean, aucs_lo, aucs_hi, title, colors=None):
    x = np.arange(len(conditions))
    if colors is None:
        colors = ["#2196F3"] * len(conditions)
    bars = ax.bar(x, aucs_mean, color=colors, alpha=0.8)
    for i, (m, lo, hi) in enumerate(zip(aucs_mean, aucs_lo, aucs_hi)):
        ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], fmt="none",
                    color="black", capsize=4, linewidth=1.5)
        ax.text(x[i], hi + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("ROC-AUC")

def save_fig(fig, name):
    path = f"{FIGURE_DIR}/{name}"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("H2 Experiment — All Empirical Tests")
    print("=" * 70)

    print("\n[Data] Generating synthetic dataset...")
    accounts, fleet_device = generate_dataset()
    print(f"  {N_ACCOUNTS} accounts, {N_TRAIN} training events each")
    print(f"  Fleet accounts: {sum(a['is_fleet'] for a in accounts)}")

    mp_corpus = build_mean_pool_corpus(accounts)
    cat_corpus = build_concat_corpus(accounts)

    # -----------------------------------------------------------------------
    # Train models
    # -----------------------------------------------------------------------
    print("\n[Models] Training FastText models...")
    mp_w6 = train_fasttext_mp(mp_corpus, window=6)
    mp_w3 = train_fasttext_mp(mp_corpus, window=3)
    mp_w2 = train_fasttext_mp(mp_corpus, window=2)
    mp_w1 = train_fasttext_mp(mp_corpus, window=1)
    cat_ft = train_fasttext_cat(cat_corpus)
    print("[Models] Training Word2Vec on concat corpus (T2 — n-gram ablation)...")
    cat_w2v = train_word2vec_cat(cat_corpus)
    print("  Done.")

    # -----------------------------------------------------------------------
    # Score all models
    # -----------------------------------------------------------------------
    print("\n[Scoring] Computing embeddings and centroid distances...")
    r_mp_w6, _ = score_all(accounts, embed_mp, mp_w6)
    r_mp_w3, _ = score_all(accounts, embed_mp, mp_w3)
    r_mp_w2, _ = score_all(accounts, embed_mp, mp_w2)
    r_mp_w1, _ = score_all(accounts, embed_mp, mp_w1)
    r_cat, _   = score_all(accounts, embed_cat, cat_ft)
    r_w2v, _   = score_all(accounts, embed_w2v, cat_w2v)
    trivial_aucs = evaluate_trivial(accounts)
    print("  Done.")

    # -----------------------------------------------------------------------
    # Bootstrap CIs — primary results
    # -----------------------------------------------------------------------
    print("\n[Bootstrap] Computing CIs (N=1000)...")

    def boot_all(result, name):
        out = {}
        for at in ["novel", "fleet", "spoof"]:
            m, lo, hi = bootstrap_auc(result, at)
            out[at] = (m, lo, hi)
        return out

    b_mp_w6 = boot_all(r_mp_w6, "mp_w6")
    b_mp_w3 = boot_all(r_mp_w3, "mp_w3")
    b_mp_w2 = boot_all(r_mp_w2, "mp_w2")
    b_mp_w1 = boot_all(r_mp_w1, "mp_w1")
    b_cat   = boot_all(r_cat,   "cat")
    b_w2v   = boot_all(r_w2v,   "w2v")
    print("  Done.")

    # -----------------------------------------------------------------------
    # T6: Corrected silhouette
    # -----------------------------------------------------------------------
    print("\n[T6] Corrected silhouette (true device labels)...")
    vecs_mp, labs_mp = collect_embeddings_true_labels(accounts, embed_mp, mp_w6)
    vecs_cat, labs_cat = collect_embeddings_true_labels(accounts, embed_cat, cat_ft)
    sil_mp_mean, sil_mp_lo, sil_mp_hi = bootstrap_silhouette(vecs_mp, labs_mp)
    sil_cat_mean, sil_cat_lo, sil_cat_hi = bootstrap_silhouette(vecs_cat, labs_cat)
    print(f"  Mean-pool silhouette: {sil_mp_mean:.4f} [{sil_mp_lo:.4f}, {sil_mp_hi:.4f}]")
    print(f"  Concat silhouette:    {sil_cat_mean:.4f} [{sil_cat_lo:.4f}, {sil_cat_hi:.4f}]")

    # -----------------------------------------------------------------------
    # T3: Matching-feature ablation
    # -----------------------------------------------------------------------
    print("\n[T3] Matching-feature ablation (k=0..5)...")
    k_aucs_mp = []
    k_aucs_cat = []
    for k in range(6):
        auc_mp_k = k_match_auc(accounts, k, embed_mp, mp_w6, r_mp_w6)
        auc_cat_k = k_match_auc(accounts, k, embed_cat, cat_ft, r_cat)
        k_aucs_mp.append(auc_mp_k)
        k_aucs_cat.append(auc_cat_k)
        print(f"  k={k}: mean-pool={auc_mp_k:.4f}, concat={auc_cat_k:.4f}, "
              f"gap={auc_mp_k - auc_cat_k:+.4f}")

    # -----------------------------------------------------------------------
    # T4: Tz-counterfactual
    # -----------------------------------------------------------------------
    print("\n[T4] Tz-counterfactual attribution...")
    act_mp, cf_mp, attr_mp = tz_counterfactual_distance(accounts, embed_mp, mp_w6)
    act_cat, cf_cat, attr_cat = tz_counterfactual_distance(accounts, embed_cat, cat_ft)
    print(f"  Mean-pool — actual dist: {act_mp.mean():.4f}, CF dist: {cf_mp.mean():.4f}, "
          f"tz-attr: {attr_mp.mean():.4f}")
    print(f"  Concat    — actual dist: {act_cat.mean():.4f}, CF dist: {cf_cat.mean():.4f}, "
          f"tz-attr: {attr_cat.mean():.4f}")

    # -----------------------------------------------------------------------
    # T5: Fleet AUC stratified
    # -----------------------------------------------------------------------
    print("\n[T5] Fleet AUC stratified...")
    mp_fleet_cont, mp_fleet_uncont, n_cont, n_uncont = fleet_auc_stratified(accounts, r_mp_w6)
    cat_fleet_cont, cat_fleet_uncont, _, _ = fleet_auc_stratified(accounts, r_cat)
    print(f"  Contaminated accounts ({n_cont}):   mp={mp_fleet_cont:.4f}, cat={cat_fleet_cont:.4f}")
    print(f"  Uncontaminated accounts ({n_uncont}): mp={mp_fleet_uncont:.4f}, cat={cat_fleet_uncont:.4f}")

    # -----------------------------------------------------------------------
    # T8: Token similarity matrix
    # -----------------------------------------------------------------------
    print("\n[T8] Co-occurrence bias in mean-pool token space...")
    tokens_mp, sim_mp = token_similarity_matrix(mp_w6)
    within_mp, cross_mp = cross_vs_within_similarity(tokens_mp, sim_mp)
    print(f"  Within-feature mean cosine similarity: {within_mp.mean():.4f}")
    print(f"  Cross-feature mean cosine similarity:  {cross_mp.mean():.4f}")

    # -----------------------------------------------------------------------
    # Print full results table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Primary AUC Results with Bootstrap 95% CI ---")
    header = f"{'Condition':<22} {'Novel':>16} {'Fleet':>16} {'Spoof':>16}"
    print(header)
    print("-" * 72)

    def fmt_auc(m, lo, hi):
        return f"{m:.3f} [{lo:.3f},{hi:.3f}]"

    conditions = [
        ("Mean-pool w=6", b_mp_w6),
        ("Mean-pool w=3", b_mp_w3),
        ("Mean-pool w=2", b_mp_w2),
        ("Mean-pool w=1", b_mp_w1),
        ("Concat FT",     b_cat),
        ("Concat W2V",    b_w2v),
    ]

    for name, b in conditions:
        row = f"{name:<22}"
        for at in ["novel", "fleet", "spoof"]:
            m, lo, hi = b[at]
            row += f"  {fmt_auc(m, lo, hi):>14}"
        print(row)

    print(f"\n{'Trivial baseline':<22}", end="")
    for at in ["novel", "fleet", "spoof"]:
        print(f"  {trivial_aucs[at]:>14.3f}", end="")
    print()

    print(f"\n--- Silhouette Score (true device labels) ---")
    print(f"  Mean-pool w=6: {sil_mp_mean:.4f} [{sil_mp_lo:.4f}, {sil_mp_hi:.4f}]")
    print(f"  Concat FT:     {sil_cat_mean:.4f} [{sil_cat_lo:.4f}, {sil_cat_hi:.4f}]")
    sil_gap = sil_mp_mean - sil_cat_mean
    print(f"  Gap (mp - cat): {sil_gap:+.4f}")

    print(f"\n--- T4: Tz-Counterfactual Attribution ---")
    print(f"  Mean-pool tz-attr: {attr_mp.mean():.4f} ± {attr_mp.std():.4f}")
    print(f"  Concat    tz-attr: {attr_cat.mean():.4f} ± {attr_cat.std():.4f}")
    attr_gap = attr_mp.mean() - attr_cat.mean()
    print(f"  Gap (mp - cat): {attr_gap:+.4f}")

    print(f"\n--- T5: Fleet AUC Stratified ---")
    print(f"  Contaminated   (n={n_cont}):   mp={mp_fleet_cont:.4f}, cat={cat_fleet_cont:.4f}")
    print(f"  Uncontaminated (n={n_uncont}): mp={mp_fleet_uncont:.4f}, cat={cat_fleet_uncont:.4f}")

    print(f"\n--- T8: Token Embedding Co-occurrence Bias ---")
    print(f"  Mean-pool — within-feature similarity: {within_mp.mean():.4f}")
    print(f"  Mean-pool — cross-feature similarity:  {cross_mp.mean():.4f}")
    print(f"  Ratio (within/cross): {within_mp.mean()/max(cross_mp.mean(), 1e-6):.3f}")

    # -----------------------------------------------------------------------
    # PRE-SPECIFIED VERDICTS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRE-SPECIFIED VERDICTS (per DEBATE.md)")
    print("=" * 70)

    # T1: Window sweep
    spoof_w1 = b_mp_w1["spoof"][0]
    spoof_cat = b_cat["spoof"][0]
    spoof_w6 = b_mp_w6["spoof"][0]
    print(f"\n[T1] Window sweep — spoof AUC:")
    print(f"  mp_w1={spoof_w1:.4f}, mp_w6={spoof_w6:.4f}, cat={spoof_cat:.4f}")
    if spoof_w1 > spoof_cat:
        print("  VERDICT: Defense right — w=1 outperforms concat, context richness not the driver")
    elif spoof_w1 < spoof_cat and spoof_w6 < spoof_cat:
        print("  VERDICT: Ambiguous — all mean-pool windows underperform concat on spoof")
    else:
        print("  VERDICT: Mixed — window sweep shows inconsistent pattern")

    # T2: N-gram ablation
    spoof_w2v = b_w2v["spoof"][0]
    spoof_cat = b_cat["spoof"][0]
    print(f"\n[T2] N-gram ablation — spoof AUC:")
    print(f"  w2v_cat={spoof_w2v:.4f}, ft_cat={spoof_cat:.4f}")
    diff_t2 = spoof_w2v - spoof_cat
    if abs(diff_t2) < 0.01:
        print("  VERDICT: Critique right — w2v ≈ ft-concat, n-grams not driving the effect")
    elif spoof_w2v < spoof_cat:
        print("  VERDICT: Mixed — w2v < ft-concat on spoof, n-grams may help (not contaminate)")
    else:
        print("  VERDICT: Defense direction — w2v > ft-concat, n-grams contaminate concat spoof")

    # T3: Matching-feature ablation
    print(f"\n[T3] Matching-feature ablation — AUC gap (mp - cat):")
    for k in range(6):
        print(f"  k={k}: {k_aucs_mp[k] - k_aucs_cat[k]:+.4f}")
    gaps = [k_aucs_mp[k] - k_aucs_cat[k] for k in range(6)]
    if gaps[-1] > gaps[0]:
        print("  VERDICT: Defense right — gap increases with k (mechanism confirmed)")
    elif max(gaps) == gaps[0]:
        print("  VERDICT: Critique right — gap does not increase with k (mechanism not confirmed)")
    else:
        print("  VERDICT: Ambiguous — non-monotone pattern")

    # T4: Tz-counterfactual
    print(f"\n[T4] Tz-counterfactual attribution:")
    print(f"  mp tz-attr={attr_mp.mean():.4f}, cat tz-attr={attr_cat.mean():.4f}")
    if attr_mp.mean() > attr_cat.mean():
        print("  VERDICT: Defense right — mean-pool more sensitive to tz differences")
    else:
        print("  VERDICT: Critique right — mean-pool NOT more sensitive to tz differences")

    # T5: Fleet stratified
    print(f"\n[T5] Fleet AUC stratified:")
    if mp_fleet_uncont >= cat_fleet_uncont:
        print(f"  Uncontaminated: mp={mp_fleet_uncont:.4f} >= cat={cat_fleet_uncont:.4f}")
        print("  VERDICT: Defense right on uncontaminated accounts")
    else:
        print(f"  Uncontaminated: mp={mp_fleet_uncont:.4f} < cat={cat_fleet_uncont:.4f}")
        print("  VERDICT: Critique right — concat outperforms mp on uncontaminated fleet")

    # T6: Silhouette
    print(f"\n[T6] Silhouette (corrected labels):")
    print(f"  mp={sil_mp_mean:.4f}, cat={sil_cat_mean:.4f}, gap={sil_gap:+.4f}")
    if sil_mp_lo > sil_cat_hi:
        print("  VERDICT: Defense right — mp silhouette CI clearly above cat (H2 confirmed)")
    elif sil_cat_lo > sil_mp_hi:
        print("  VERDICT: Critique right — cat silhouette CI clearly above mp (H2 refuted)")
    else:
        print("  VERDICT: Ambiguous — CIs overlap")

    # T7: Trivial baseline
    print(f"\n[T7] Trivial baseline comparison:")
    for at in ["novel", "fleet", "spoof"]:
        mp_auc = b_mp_w6[at][0]
        cat_auc = b_cat[at][0]
        triv = trivial_aucs[at]
        mp_beats = "YES" if mp_auc > triv else "NO"
        cat_beats = "YES" if cat_auc > triv else "NO"
        print(f"  {at:6s}: trivial={triv:.3f}, mp={mp_auc:.3f} (beats?{mp_beats}), "
              f"cat={cat_auc:.3f} (beats?{cat_beats})")

    # T8: Co-occurrence bias
    print(f"\n[T8] Co-occurrence bias:")
    ratio = within_mp.mean() / max(cross_mp.mean(), 1e-6)
    if ratio > 2.0:
        print(f"  VERDICT: Defense right — within >> cross (ratio={ratio:.2f}), "
              f"feature tokens well-separated")
    elif ratio < 1.2:
        print(f"  VERDICT: Critique right — within ≈ cross (ratio={ratio:.2f}), "
              f"co-occurrence bias present")
    else:
        print(f"  VERDICT: Ambiguous — moderate ratio={ratio:.2f}")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\n[Figures] Generating...")

    # Figure 1: T1 Window sweep
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    window_labels = ["w=1", "w=2", "w=3", "w=6", "concat", "trivial"]
    window_boots = [b_mp_w1, b_mp_w2, b_mp_w3, b_mp_w6, b_cat, None]
    window_colors = ["#1565C0", "#1976D2", "#1E88E5", "#42A5F5", "#FF7043", "#9E9E9E"]

    for ai, at in enumerate(["novel", "fleet", "spoof"]):
        ax = axes[ai]
        means, los, his = [], [], []
        for wb in window_boots:
            if wb is None:
                m = trivial_aucs[at]
                means.append(m)
                los.append(m)
                his.append(m)
            else:
                m, lo, hi = wb[at]
                means.append(m)
                los.append(lo)
                his.append(hi)
        plot_auc_bar_with_ci(ax, window_labels, means, los, his,
                             f"T1: Window sweep — {at}", colors=window_colors)
    fig.suptitle("T1: Mean-pool window sweep vs. concat FastText\n(bars=mean, error=95% CI)",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "finding_01_window_sweep.png")

    # Figure 2: T2 N-gram ablation
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    conditions_t2 = ["Mean-pool w=6", "Concat FT", "Concat W2V", "Trivial"]
    colors_t2 = ["#2196F3", "#FF5722", "#FF9800", "#9E9E9E"]
    for ai, at in enumerate(["novel", "fleet", "spoof"]):
        ax = axes[ai]
        means, los, his = [], [], []
        for b in [b_mp_w6, b_cat, b_w2v]:
            m, lo, hi = b[at]
            means.append(m)
            los.append(lo)
            his.append(hi)
        t = trivial_aucs[at]
        means.append(t); los.append(t); his.append(t)
        plot_auc_bar_with_ci(ax, conditions_t2, means, los, his,
                             f"T2: N-gram ablation — {at}", colors=colors_t2)
    fig.suptitle("T2: FastText (with n-grams) vs. Word2Vec (no n-grams) on concat\n"
                 "(bars=mean, error=95% CI)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "finding_02_ngram_ablation.png")

    # Figure 3: T3 Matching-feature ablation
    fig, ax = plt.subplots(figsize=(9, 5))
    k_vals = list(range(6))
    ax.plot(k_vals, k_aucs_mp, "o-", color="#2196F3", label="Mean-pool w=6", linewidth=2)
    ax.plot(k_vals, k_aucs_cat, "s-", color="#FF5722", label="Concat FT", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.fill_between(k_vals, k_aucs_mp, k_aucs_cat,
                    where=[mp > cat for mp, cat in zip(k_aucs_mp, k_aucs_cat)],
                    alpha=0.15, color="#2196F3", label="mp > cat")
    ax.fill_between(k_vals, k_aucs_mp, k_aucs_cat,
                    where=[mp <= cat for mp, cat in zip(k_aucs_mp, k_aucs_cat)],
                    alpha=0.15, color="#FF5722", label="cat > mp")
    ax.set_xlabel("Number of features matching primary device (k)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("T3: Matching-feature ablation\n(Does the mean-pool advantage grow with k?)")
    ax.legend()
    ax.set_xticks(k_vals)
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    fig.tight_layout()
    save_fig(fig, "finding_03_matching_feature_ablation.png")

    # Figure 4: T4 Tz-counterfactual
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, vals, label in zip(
        axes, [attr_mp, attr_cat], ["Mean-pool w=6", "Concat FT"]
    ):
        ax.hist(vals, bins=40, color="#2196F3" if "pool" in label else "#FF5722",
                alpha=0.75, edgecolor="white")
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={vals.mean():.4f}")
        ax.axvline(0, color="gray", linestyle="-", linewidth=0.8)
        ax.set_title(f"T4: {label}\nTz-attributable cosine distance")
        ax.set_xlabel("Actual dist − Counterfactual dist")
        ax.set_ylabel("Count")
        ax.legend()
    fig.suptitle("T4: Tz-counterfactual attribution\n"
                 "(Positive = tz token increases cosine distance from centroid)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "finding_04_tz_counterfactual.png")

    # Figure 5: T5 Fleet stratified
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = ["Contaminated\n(fleet in training)", "Uncontaminated\n(fleet never seen)"]
    mp_vals = [mp_fleet_cont, mp_fleet_uncont]
    cat_vals = [cat_fleet_cont, cat_fleet_uncont]
    x = np.arange(2)
    w = 0.3
    ax.bar(x - w/2, mp_vals, w, label="Mean-pool w=6", color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, cat_vals, w, label="Concat FT", color="#FF5722", alpha=0.85)
    for i, (mv, cv) in enumerate(zip(mp_vals, cat_vals)):
        ax.text(i - w/2, mv + 0.01, f"{mv:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, cv + 0.01, f"{cv:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("ROC-AUC (fleet attack)")
    ax.set_title(f"T5: Fleet AUC stratified by training contamination\n"
                 f"(contaminated n={n_cont}, uncontaminated n={n_uncont})")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "finding_05_fleet_stratified.png")

    # Figure 6: T6 Silhouette comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    labels_s = ["Mean-pool w=6", "Concat FT"]
    means_s = [sil_mp_mean, sil_cat_mean]
    los_s = [sil_mp_lo, sil_cat_lo]
    his_s = [sil_mp_hi, sil_cat_hi]
    ax.bar([0, 1], means_s, color=["#2196F3", "#FF5722"], alpha=0.8)
    for i, (m, lo, hi) in enumerate(zip(means_s, los_s, his_s)):
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none",
                    color="black", capsize=5, linewidth=2)
        ax.text(i, hi + 0.002, f"{m:.4f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels_s)
    ax.set_ylabel("Silhouette score (cosine metric)")
    ax.set_title("T6: Corrected silhouette score\n(true device labels, bootstrap 95% CI)")
    fig.tight_layout()
    save_fig(fig, "finding_06_silhouette.png")

    # Figure 7: T8 Token similarity matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_mat = axes[0]
    im = ax_mat.imshow(sim_mp, cmap="RdYlBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax_mat)
    ax_mat.set_title("T8: Token cosine similarity matrix\n(Mean-pool model)")
    ax_mat.set_xlabel("Token index")
    ax_mat.set_ylabel("Token index")
    # Add feature boundary lines
    boundaries = np.cumsum([len(FEATURES[f]) for f in FEATURE_ORDER])
    for b in boundaries[:-1]:
        ax_mat.axhline(b - 0.5, color="white", linewidth=1.5)
        ax_mat.axvline(b - 0.5, color="white", linewidth=1.5)

    ax_hist = axes[1]
    ax_hist.hist(within_mp, bins=30, alpha=0.75, color="#2196F3",
                 label=f"Within-feature\nmean={within_mp.mean():.3f}")
    ax_hist.hist(cross_mp, bins=30, alpha=0.6, color="#FF5722",
                 label=f"Cross-feature\nmean={cross_mp.mean():.3f}")
    ax_hist.set_xlabel("Cosine similarity")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("T8: Within vs. cross-feature token similarity distribution")
    ax_hist.legend()
    fig.tight_layout()
    save_fig(fig, "finding_08_token_similarity.png")

    # Figure 8: Summary — all conditions on spoof AUC (the key test)
    fig, ax = plt.subplots(figsize=(12, 6))
    summary_conditions = [
        "Trivial", "Concat FT", "Concat W2V",
        "MP w=1", "MP w=2", "MP w=3", "MP w=6",
    ]
    summary_boots = [None, b_cat, b_w2v, b_mp_w1, b_mp_w2, b_mp_w3, b_mp_w6]
    summary_colors = ["#9E9E9E", "#FF5722", "#FF9800",
                      "#BBDEFB", "#90CAF9", "#42A5F5", "#1565C0"]
    means_sum, los_sum, his_sum = [], [], []
    for wb in summary_boots:
        if wb is None:
            t = trivial_aucs["spoof"]
            means_sum.append(t); los_sum.append(t); his_sum.append(t)
        else:
            m, lo, hi = wb["spoof"]
            means_sum.append(m); los_sum.append(lo); his_sum.append(hi)
    plot_auc_bar_with_ci(ax, summary_conditions, means_sum, los_sum, his_sum,
                         "Summary: Spoof AUC — all conditions with 95% CI\n"
                         "(Spoof attack: 5/6 features match primary, tz differs)",
                         colors=summary_colors)
    fig.tight_layout()
    save_fig(fig, "summary_all_conditions_spoof.png")

    # Figure 9: Summary — all attack types for key conditions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    key_conditions = ["Trivial", "Concat FT", "Mean-pool w=6"]
    key_boots = [None, b_cat, b_mp_w6]
    key_colors = ["#9E9E9E", "#FF5722", "#2196F3"]
    for ai, at in enumerate(["novel", "fleet", "spoof"]):
        ax = axes[ai]
        means_k, los_k, his_k = [], [], []
        for wb in key_boots:
            if wb is None:
                t = trivial_aucs[at]
                means_k.append(t); los_k.append(t); his_k.append(t)
            else:
                m, lo, hi = wb[at]
                means_k.append(m); los_k.append(lo); his_k.append(hi)
        plot_auc_bar_with_ci(ax, key_conditions, means_k, los_k, his_k,
                             f"{at.capitalize()} attack", colors=key_colors)
    fig.suptitle("Summary: Key conditions — all attack types with 95% CI", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "summary_all_attack_types.png")

    print("\n=== Experiment complete ===")

if __name__ == "__main__":
    main()
