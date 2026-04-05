# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.24",
#   "matplotlib>=3.7",
# ]
# ///
#
# Configuration Verification — T8 Token Similarity Under ml-lab vs. H2_RERUN Configs
#
# Purpose: Empirically confirm that the within-feature embedding collapse (T8,
# within-feature cosine sim = 0.999) observed in the ml-lab mean-pool model is
# caused by the combination of (a) CBOW training objective and (b) per-event
# corpus construction used in the ml-lab PoC — and that the H2_RERUN configuration
# (skip-gram + per-account corpus) does NOT produce this collapse.
#
# Two configs are compared:
#   ml-lab:    sg=0 (CBOW), per-event sentences (6 tokens/sentence), epochs=10
#   H2_RERUN:  sg=1 (skip-gram), per-account sentences (~360 tokens/sentence),
#              epochs=20, negative=10
#
# All other parameters are held constant: vector_size=64, window=6, min_n=3,
# max_n=6, min_count=1, seed=42, 400 accounts, 60 events/account.

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import FastText

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "figures" / "h2_ml_lab"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature vocabulary (identical to ml-lab PoC and H2_RERUN)
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
N_TRAIN_PER_ACCOUNT = 60

# ---------------------------------------------------------------------------
# Data generation (identical to ml-lab PoC)
# ---------------------------------------------------------------------------
def zipf_weights(n, s=1.5):
    w = np.array([1.0 / (k ** s) for k in range(1, n + 1)])
    return w / w.sum()

def generate_accounts(seed=SEED):
    rng = np.random.default_rng(seed)
    accounts = []
    for _ in range(N_ACCOUNTS):
        n_devices = int(rng.integers(2, 5))
        devices = []
        seen = set()
        while len(devices) < n_devices:
            d = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
            key = tuple(d[f] for f in FEATURE_ORDER)
            if key not in seen:
                seen.add(key)
                devices.append(d)
        weights = zipf_weights(n_devices)
        events = []
        for _ in range(N_TRAIN_PER_ACCOUNT):
            idx = rng.choice(n_devices, p=weights)
            events.append(dict(devices[idx]))
        accounts.append({"devices": devices, "events": events})
    return accounts

def make_token(f, v):
    return f"{f}_{v}"

def device_to_tokens(d):
    return [make_token(f, d[f]) for f in FEATURE_ORDER]

# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------
def build_per_event_corpus(accounts):
    """ml-lab style: one 6-token sentence per login event."""
    sentences = []
    for acc in accounts:
        for event in acc["events"]:
            sentences.append(device_to_tokens(event))
    return sentences

def build_per_account_corpus(accounts):
    """H2_RERUN style: all events for one account flattened into one sentence."""
    sentences = []
    for acc in accounts:
        flat = []
        for event in acc["events"]:
            flat.extend(device_to_tokens(event))
        if flat:
            sentences.append(flat)
    return sentences

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_mllab_config(corpus):
    """ml-lab PoC defaults: CBOW (sg=0), epochs=10, negative=5 (default)."""
    return FastText(
        sentences=corpus,
        vector_size=64,
        window=6,
        sg=0,           # CBOW — ml-lab default
        min_count=1,
        epochs=10,
        min_n=3,
        max_n=6,
        seed=SEED,
        workers=1,
    )

def train_h2rerun_config(corpus):
    """H2_RERUN config: skip-gram (sg=1), epochs=20, negative=10."""
    return FastText(
        sentences=corpus,
        vector_size=64,
        window=6,
        sg=1,           # Skip-gram — H2_RERUN
        negative=10,
        min_count=1,
        epochs=20,
        min_n=3,
        max_n=6,
        seed=SEED,
        workers=1,
    )

# ---------------------------------------------------------------------------
# T8: Token similarity analysis
# ---------------------------------------------------------------------------
def token_similarity_analysis(model):
    tokens, feat_types = [], []
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
        for j in range(i + 1, n):
            if feat_types[i] == feat_types[j]:
                within.append(sim_mat[i, j])
            else:
                cross.append(sim_mat[i, j])
    return tokens, feat_types, sim_mat, np.array(within), np.array(cross)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_comparison(results):
    """Two-panel figure: within-feature sim distributions for both configs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("T8 Config Verification: Within-Feature Token Similarity\n"
                 "ml-lab config (CBOW, per-event) vs. H2_RERUN config (skip-gram, per-account)",
                 fontsize=11, fontweight="bold")

    colors = {"ml-lab\n(CBOW, per-event)": "#FF5722",
              "H2_RERUN\n(skip-gram, per-account)": "#2196F3"}

    for ax, (label, r) in zip(axes, results.items()):
        within = r["within"]
        cross = r["cross"]
        color = colors[label]
        ax.hist(within, bins=30, color=color, alpha=0.8, label=f"Within-feature (n={len(within)})")
        ax.hist(cross, bins=30, color="gray", alpha=0.5, label=f"Cross-feature (n={len(cross)})")
        ax.axvline(within.mean(), color=color, linestyle="--", linewidth=1.5,
                   label=f"Within mean = {within.mean():.4f}")
        ax.axvline(cross.mean(), color="gray", linestyle="--", linewidth=1.5,
                   label=f"Cross mean = {cross.mean():.4f}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlim(-1.1, 1.1)

    plt.tight_layout()
    out = FIGURES_DIR / "config_verification_t8.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("T8 Config Verification: ml-lab vs. H2_RERUN")
    print("=" * 65)

    print("\n[1/4] Generating data (400 accounts, 60 events each)...")
    accounts = generate_accounts()

    print("\n[2/4] Building corpora...")
    per_event_corpus   = build_per_event_corpus(accounts)
    per_account_corpus = build_per_account_corpus(accounts)
    print(f"  per-event corpus:   {len(per_event_corpus):,} sentences, "
          f"{sum(len(s) for s in per_event_corpus):,} tokens")
    print(f"  per-account corpus: {len(per_account_corpus):,} sentences, "
          f"{sum(len(s) for s in per_account_corpus):,} tokens")

    print("\n[3/4] Training models...")
    print("  Training ml-lab config  (CBOW,      per-event,   epochs=10)...")
    model_mllab   = train_mllab_config(per_event_corpus)
    print("  Training H2_RERUN config (skip-gram, per-account, epochs=20)...")
    model_h2rerun = train_h2rerun_config(per_account_corpus)

    print("\n[4/4] Running T8 token similarity analysis...")
    _, _, _, within_ml,  cross_ml  = token_similarity_analysis(model_mllab)
    _, _, _, within_h2r, cross_h2r = token_similarity_analysis(model_h2rerun)

    print("\n--- T8 Results ---")
    print(f"{'Config':<40} {'Within-feature mean':>22} {'Cross-feature mean':>20}")
    print("-" * 84)
    print(f"{'ml-lab (CBOW, per-event, epochs=10)':<40} "
          f"{within_ml.mean():>22.4f} {cross_ml.mean():>20.4f}")
    print(f"{'H2_RERUN (sg=1, per-account, epochs=20)':<40} "
          f"{within_h2r.mean():>22.4f} {cross_h2r.mean():>20.4f}")

    collapse_mllab   = within_ml.mean() > 0.99
    collapse_h2rerun = within_h2r.mean() > 0.99
    print(f"\n  ml-lab   within-feature collapse (sim > 0.99)? {'YES' if collapse_mllab   else 'NO'}")
    print(f"  H2_RERUN within-feature collapse (sim > 0.99)? {'YES' if collapse_h2rerun else 'NO'}")

    if collapse_mllab and not collapse_h2rerun:
        print("\n  CONFIRMED: ml-lab config produces within-feature collapse; "
              "H2_RERUN config does not.")
        print("  The H2_RERUN results (mean-pool > concat on spoof) are valid "
              "under sg=1 + per-account corpus.")
    elif collapse_mllab and collapse_h2rerun:
        print("\n  UNEXPECTED: Both configs produce collapse. Further investigation needed.")
    elif not collapse_mllab:
        print("\n  UNEXPECTED: ml-lab config does not produce collapse. Check implementation.")

    results = {
        "ml-lab\n(CBOW, per-event)":           {"within": within_ml,  "cross": cross_ml},
        "H2_RERUN\n(skip-gram, per-account)":  {"within": within_h2r, "cross": cross_h2r},
    }
    plot_comparison(results)

    # Machine-readable summary
    print("\n--- SUMMARY ---")
    print(f"mllab_within_mean={within_ml.mean():.4f}")
    print(f"mllab_cross_mean={cross_ml.mean():.4f}")
    print(f"h2rerun_within_mean={within_h2r.mean():.4f}")
    print(f"h2rerun_cross_mean={cross_h2r.mean():.4f}")
    print(f"collapse_mllab={collapse_mllab}")
    print(f"collapse_h2rerun={collapse_h2rerun}")


if __name__ == "__main__":
    main()
