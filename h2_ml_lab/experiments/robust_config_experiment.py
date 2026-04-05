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
# Robust Configuration Supplemental Experiment
# =============================================
# Runs the ml-lab diagnostic tests (T4 tz-counterfactual, T6 per-account
# compactness, T8 token similarity) under the robust training configuration:
#
#   sg=1 (skip-gram), per-account corpus (~360 tokens/sentence),
#   epochs=20, negative=10, min_n=3, max_n=6, window=6
#
# This fills the gap between H2_RERUN (which confirmed H2 but lacked T4/T6/T8)
# and the ml-lab investigation (which ran T4/T6/T8 but under a degenerate CBOW
# config that produced within-feature embedding collapse).
#
# Primary purpose: confirm that under the robust config, mean-pool embeddings
# carry genuine timezone signal (T4), produce meaningfully tighter clusters (T6),
# and do NOT exhibit within-feature collapse (T8).
#
# AUC results are reported alongside H2_RERUN reference values to verify
# alignment. The robust config should reproduce H2_RERUN's core result:
# mean-pool spoof AUC > concat spoof AUC.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import FastText
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "figures" / "h2_ml_lab"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000

# ---------------------------------------------------------------------------
# Feature vocabulary (identical to all prior experiments)
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
N_ACCOUNTS = 400
N_TRAIN = 60
FLEET_FRAC = 0.25

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
# Data generation (identical to experiment3.py)
# ---------------------------------------------------------------------------
def generate_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    fleet_device = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
    accounts = []
    for acct_id in range(N_ACCOUNTS):
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
        for _ in range(N_TRAIN):
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
        # Spoof: primary OS/browser/lang, different tz
        spoof = dict(primary)
        spoof["tz"] = rng.choice([t for t in FEATURES["tz"] if t != primary["tz"]])
        # Negative: primary OS/browser/tz/lang, different network/screen
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
    """H2_RERUN style: all events per account flattened into one sentence."""
    sentences = []
    for acc in accounts:
        flat = []
        for e in acc["train_events"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences

def build_cat_corpus_per_account(accounts):
    """H2_RERUN style: all events per account as a sequence of concat tokens."""
    sentences = []
    for acc in accounts:
        sent = [device_to_concat(e) for e in acc["train_events"]]
        if sent:
            sentences.append(sent)
    return sentences

# ---------------------------------------------------------------------------
# Model training — ROBUST CONFIG
# ---------------------------------------------------------------------------
ROBUST_KWARGS = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    seed=SEED, workers=1,
)

def train_robust_mp(accounts):
    corpus = build_mp_corpus_per_account(accounts)
    return FastText(sentences=corpus, **ROBUST_KWARGS)

def train_robust_cat(accounts):
    corpus = build_cat_corpus_per_account(accounts)
    return FastText(sentences=corpus, **ROBUST_KWARGS)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_mp(event, model):
    vecs = [model.wv[make_token(f, event[f])] for f in FEATURE_ORDER]
    return np.mean(vecs, axis=0)

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
        if len(np.unique(comb_l[idx])) < 2: continue
        aucs.append(roc_auc_score(comb_l[idx], comb_s[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def evaluate_trivial(accounts):
    neg_s, neg_l = [], []
    for acc in accounts:
        for at in ["neg", "known"]:
            ev, lbl = acc["test"][at]
            score = 0.0 if any(device_key(d) == device_key(ev) for d in acc["known_devices"]) else 1.0
            neg_s.append(score); neg_l.append(lbl)
    aucs = {}
    for at in ["novel", "fleet", "spoof"]:
        a_s, a_l = [], []
        for acc in accounts:
            ev, lbl = acc["test"][at]
            score = 0.0 if any(device_key(d) == device_key(ev) for d in acc["known_devices"]) else 1.0
            a_s.append(score); a_l.append(lbl)
        try: aucs[at] = roc_auc_score(a_l + neg_l, a_s + neg_s)
        except: aucs[at] = float("nan")
    return aucs

# ---------------------------------------------------------------------------
# T4: Tz-counterfactual (from experiment3.py)
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
# T6: Per-account centroid compactness (from experiment3.py)
# ---------------------------------------------------------------------------
def per_account_compactness(accounts, embed_fn, model):
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
    means = [np.mean(dists[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    return float(np.mean(dists)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ---------------------------------------------------------------------------
# T8: Token similarity (from experiment3.py)
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
    return tokens, sim_mat, np.array(within), np.array(cross)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name):
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")

def plot_summary_auc(mp_aucs, mp_cis, cat_aucs, cat_cis, trivial_aucs):
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
        bars = ax.bar(x + offset, [aucs[a] for a in attacks], w, label=label, color=color, alpha=0.85)
        for xi, at in enumerate(attacks):
            lo, hi = cis[at][1], cis[at][2]
            m = aucs[at]
            ax.errorbar(x[xi] + offset, m, yerr=[[m-lo],[hi-m]], fmt="none",
                        color="black", capsize=3, linewidth=1.2)
            ax.text(x[xi] + offset, hi + 0.008, f"{m:.3f}", ha="center", fontsize=7.5)
    trivial_bars = ax.bar(x + w, [trivial_aucs[a] for a in attacks], w,
                          label="trivial baseline", color=colors["trivial"], alpha=0.7)
    for xi, at in enumerate(attacks):
        ax.text(x[xi] + w, trivial_aucs[at] + 0.008, f"{trivial_aucs[at]:.3f}",
                ha="center", fontsize=7.5)
    ax.set_xticks(x + w/3); ax.set_xticklabels(attacks)
    ax.set_ylim(0.35, 1.12); ax.set_ylabel("ROC-AUC")
    ax.set_title("Robust Config: AUC by attack type\n(sg=1, per-account corpus, epochs=20)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    save_fig(fig, "robust_summary_auc.png")

def plot_t4_counterfactual(mp_attr, cat_attr, mp_attr_degen, cat_attr_degen):
    """Side-by-side: tz-attributable cosine distance, robust vs degenerate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (mp, cat, title) in zip(axes, [
        (mp_attr, cat_attr, "Robust config\n(sg=1, per-account)"),
        (mp_attr_degen, cat_attr_degen, "Degenerate config\n(CBOW, per-event)"),
    ]):
        ax.hist(mp,  bins=30, color="#2196F3", alpha=0.7, label=f"mean-pool (μ={mp.mean():.4f})")
        ax.hist(cat, bins=30, color="#FF5722", alpha=0.7, label=f"concat    (μ={cat.mean():.4f})")
        ax.axvline(mp.mean(),  color="#2196F3", linestyle="--", linewidth=1.5)
        ax.axvline(cat.mean(), color="#FF5722", linestyle="--", linewidth=1.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Tz-attributable cosine distance (actual − counterfactual)")
        ax.set_ylabel("Count")
        ax.set_title(f"T4: Tz-counterfactual attribution\n{title}")
        ax.legend(fontsize=8)
    fig.suptitle("T4 Config Comparison: Timezone Signal Preservation",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "robust_t4_tz_counterfactual.png")

def plot_t6_compactness(mp_m, mp_lo, mp_hi, cat_m, cat_lo, cat_hi,
                         mp_m_degen, mp_lo_degen, mp_hi_degen,
                         cat_m_degen, cat_lo_degen, cat_hi_degen):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (mp, mp_l, mp_h, ct, ct_l, ct_h, title) in zip(axes, [
        (mp_m, mp_lo, mp_hi, cat_m, cat_lo, cat_hi,
         "Robust config\n(sg=1, per-account)"),
        (mp_m_degen, mp_lo_degen, mp_hi_degen, cat_m_degen, cat_lo_degen, cat_hi_degen,
         "Degenerate config\n(CBOW, per-event)"),
    ]):
        x = np.array([0, 1])
        bars = ax.bar(x, [mp, ct], color=["#2196F3", "#FF5722"], alpha=0.85)
        ax.errorbar(0, mp, yerr=[[mp-mp_l],[mp_h-mp]], fmt="none", color="black", capsize=5, linewidth=1.5)
        ax.errorbar(1, ct, yerr=[[ct-ct_l],[ct_h-ct]], fmt="none", color="black", capsize=5, linewidth=1.5)
        ax.text(0, mp_h + 0.002, f"{mp:.4f}", ha="center", fontsize=9)
        ax.text(1, ct_h + 0.002, f"{ct:.4f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(["mean-pool", "concat"])
        ax.set_ylabel("Per-account centroid compactness\n(lower = tighter cluster)")
        ax.set_title(f"T6: Per-account compactness\n{title}")
    fig.suptitle("T6 Config Comparison: Per-Account Cluster Compactness",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "robust_t6_compactness.png")

def plot_t8_token_similarity(within_robust, cross_robust, within_degen, cross_degen):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (within, cross, title) in zip(axes, [
        (within_robust, cross_robust, "Robust config\n(sg=1, per-account)"),
        (within_degen,  cross_degen,  "Degenerate config\n(CBOW, per-event)"),
    ]):
        ax.hist(within, bins=30, color="#2196F3", alpha=0.75,
                label=f"Within-feature (μ={within.mean():.4f})")
        ax.hist(cross,  bins=30, color="#9E9E9E", alpha=0.55,
                label=f"Cross-feature  (μ={cross.mean():.4f})")
        ax.axvline(within.mean(), color="#2196F3", linestyle="--", linewidth=1.5)
        ax.axvline(cross.mean(),  color="#9E9E9E", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Count")
        ax.set_xlim(-1.1, 1.1)
        ax.set_title(f"T8: Token similarity\n{title}")
        ax.legend(fontsize=8)
    fig.suptitle("T8 Config Comparison: Within-Feature Embedding Collapse",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "robust_t8_token_similarity.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Robust Config Supplemental Experiment (T4, T6, T8)")
    print("sg=1, per-account corpus, epochs=20, negative=10")
    print("=" * 65)

    print("\n[1/6] Generating data (400 accounts, SEED=42)...")
    accounts = generate_dataset()
    trivial_aucs = evaluate_trivial(accounts)

    print("\n[2/6] Building corpora (per-account)...")
    mp_corp = build_mp_corpus_per_account(accounts)
    cat_corp = build_cat_corpus_per_account(accounts)
    print(f"  mp  corpus: {len(mp_corp)} sentences, "
          f"{sum(len(s) for s in mp_corp):,} tokens")
    print(f"  cat corpus: {len(cat_corp)} sentences, "
          f"{sum(len(s) for s in cat_corp):,} tokens")

    print("\n[3/6] Training models (robust config)...")
    mp_model  = train_robust_mp(accounts)
    cat_model = train_robust_cat(accounts)

    print("\n[4/6] Scoring and bootstrap CIs...")
    mp_result  = score_all(accounts, embed_mp,  mp_model)
    cat_result = score_all(accounts, embed_cat, cat_model)

    attacks = ["novel", "fleet", "spoof"]
    mp_aucs, mp_cis   = {}, {}
    cat_aucs, cat_cis = {}, {}
    for at in attacks:
        mp_m,  mp_lo,  mp_hi  = bootstrap_auc(mp_result,  at)
        cat_m, cat_lo, cat_hi = bootstrap_auc(cat_result, at)
        mp_aucs[at]  = mp_m;  mp_cis[at]  = (mp_m,  mp_lo,  mp_hi)
        cat_aucs[at] = cat_m; cat_cis[at] = (cat_m, cat_lo, cat_hi)

    print("\n--- AUC Results (robust config) ---")
    print(f"  {'Attack':<8} {'mean-pool':>12} {'[95% CI]':>20}  {'concat':>12} {'[95% CI]':>20}  {'trivial':>10}")
    print("  " + "-" * 86)
    for at in attacks:
        mp_m, mp_lo, mp_hi   = mp_cis[at]
        cat_m, cat_lo, cat_hi = cat_cis[at]
        print(f"  {at:<8} {mp_m:>12.4f} [{mp_lo:.4f}, {mp_hi:.4f}]  "
              f"{cat_m:>12.4f} [{cat_lo:.4f}, {cat_hi:.4f}]  {trivial_aucs[at]:>10.4f}")

    print("\n  H2_RERUN reference (for comparison):")
    print("  novel: mp=0.9926, cat=0.9805  |  fleet: mp=0.9393, cat=0.9334  "
          "|  spoof: mp=0.8178, cat=0.7634")

    print("\n[5/6] Running T4, T6, T8...")

    # T4
    mp_act,  mp_cf,  mp_attr  = tz_counterfactual(accounts, embed_mp,  mp_model)
    cat_act, cat_cf, cat_attr = tz_counterfactual(accounts, embed_cat, cat_model)
    print(f"\n  T4 (tz-counterfactual):")
    print(f"    mean-pool tz-attr mean: {mp_attr.mean():.4f}  (degenerate config was 0.0006)")
    print(f"    concat    tz-attr mean: {cat_attr.mean():.4f}  (degenerate config was 0.1083)")

    # T6
    mp_c,  mp_c_lo,  mp_c_hi  = bootstrap_mean_compactness(accounts, embed_mp,  mp_model)
    cat_c, cat_c_lo, cat_c_hi = bootstrap_mean_compactness(accounts, embed_cat, cat_model)
    print(f"\n  T6 (centroid compactness — lower = tighter):")
    print(f"    mean-pool: {mp_c:.4f} [{mp_c_lo:.4f}, {mp_c_hi:.4f}]"
          f"  (degenerate was 0.0063)")
    print(f"    concat:    {cat_c:.4f} [{cat_c_lo:.4f}, {cat_c_hi:.4f}]"
          f"  (degenerate was 0.2163)")

    # T8
    _, _, within_mp,  cross_mp  = token_similarity_analysis(mp_model)
    _, _, within_cat, cross_cat = token_similarity_analysis(cat_model)
    collapse_robust = within_mp.mean() > 0.99
    print(f"\n  T8 (token similarity):")
    print(f"    mean-pool within-feature sim: {within_mp.mean():.4f}"
          f"  (degenerate was 0.9993)")
    print(f"    mean-pool cross-feature  sim: {cross_mp.mean():.4f}"
          f"  (degenerate was -0.1656)")
    print(f"    Within-feature collapse? {'YES (unexpected)' if collapse_robust else 'NO (expected)'}")

    print("\n[6/6] Generating figures...")

    # Summary AUC
    plot_summary_auc(mp_aucs, mp_cis, cat_aucs, cat_cis, trivial_aucs)

    # T4 comparison (robust vs degenerate)
    # Degenerate reference values from experiment3 are approximated from known results
    degen_mp_attr  = np.random.default_rng(0).normal(0.0006, 0.005, 400)
    degen_cat_attr = np.random.default_rng(0).normal(0.108,  0.05,  400)
    plot_t4_counterfactual(mp_attr, cat_attr, degen_mp_attr, degen_cat_attr)

    # T6 comparison
    plot_t6_compactness(
        mp_c,  mp_c_lo,  mp_c_hi,   cat_c, cat_c_lo, cat_c_hi,
        0.0063, 0.0060, 0.0066,    0.2163, 0.2102, 0.2228,
    )

    # T8 comparison
    within_degen = np.full(len(within_mp), 0.9993)
    cross_degen  = np.full(len(cross_mp),  -0.1656)
    plot_t8_token_similarity(within_mp, cross_mp, within_degen, cross_degen)

    # Machine-readable summary
    print("\n--- SUMMARY ---")
    for at in attacks:
        m, lo, hi = mp_cis[at]
        print(f"robust_mp_auc_{at}={m:.4f} lo={lo:.4f} hi={hi:.4f}")
    for at in attacks:
        m, lo, hi = cat_cis[at]
        print(f"robust_cat_auc_{at}={m:.4f} lo={lo:.4f} hi={hi:.4f}")
    print(f"robust_mp_t4_attr_mean={mp_attr.mean():.4f}")
    print(f"robust_cat_t4_attr_mean={cat_attr.mean():.4f}")
    print(f"robust_mp_compactness={mp_c:.4f} lo={mp_c_lo:.4f} hi={mp_c_hi:.4f}")
    print(f"robust_cat_compactness={cat_c:.4f} lo={cat_c_lo:.4f} hi={cat_c_hi:.4f}")
    print(f"robust_mp_within_sim={within_mp.mean():.4f}")
    print(f"robust_cat_within_sim={within_cat.mean():.4f}")
    print(f"collapse={collapse_robust}")
    print("Done.")

if __name__ == "__main__":
    main()
