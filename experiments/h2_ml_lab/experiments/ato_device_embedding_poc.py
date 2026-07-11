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
# ATO Device Embedding PoC — H2
# Hypothesis: Mean-pooling 6 feature-token embeddings outperforms embedding a
# concatenated device string (FastText), on silhouette score and ROC-AUC for
# ATO detection. Advantage expected to be largest on spoof attacks.
#
# Deliberately leaving out:
# - Hyperparameter tuning (window, min_count, vector size, epochs)
# - Real login data — purely synthetic
# - Any feature engineering beyond OS/browser/tz/lang/network/screen tokens
# - Threshold selection / precision-recall analysis
# - Production concerns: retraining cadence, cold-start, model serialization
# - Ensemble or learned aggregation (only mean-pool vs. concat tested)
# - Per-feature silhouette analysis
# - Statistical significance beyond one run (bootstrap is in experiment2.py)

import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import FastText
from sklearn.metrics import roc_auc_score
from sklearn.metrics import silhouette_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------
FEATURES = {
    "os":       ["ios", "android", "windows", "macos", "linux"],
    "browser":  ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz":       ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang":     ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "network":  ["wifi", "lte", "5g", "broadband"],
    "screen":   ["small", "medium", "large", "xlarge"],
}

FEATURE_ORDER = ["os", "browser", "tz", "lang", "network", "screen"]

def make_token(feature, value):
    return f"{feature}_{value}"

def device_to_tokens(device: dict) -> list[str]:
    """Convert device dict to list of 6 prefixed tokens."""
    return [make_token(f, device[f]) for f in FEATURE_ORDER]

def device_to_concat_string(device: dict) -> str:
    """Convert device dict to single underscore-joined string."""
    return "_".join(device[f] for f in FEATURE_ORDER)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
N_ACCOUNTS = 400
N_TRAIN_PER_ACCOUNT = 60
N_DEVICES_PER_ACCOUNT = (2, 4)  # uniform in range
FLEET_ATTACK_FRACTION = 0.25    # fraction of accounts with fleet attacker

def sample_device(rng=None):
    if rng is None:
        rng = np.random
    return {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}

def sample_account_devices(n_devices, rng=None):
    """Sample n_devices unique device profiles for an account."""
    if rng is None:
        rng = np.random
    devices = []
    seen = set()
    while len(devices) < n_devices:
        d = sample_device(rng)
        key = tuple(d[f] for f in FEATURE_ORDER)
        if key not in seen:
            seen.add(key)
            devices.append(d)
    return devices

def zipf_weights(n, s=1.5):
    """Zipf weights for n items with exponent s."""
    weights = np.array([1.0 / (k ** s) for k in range(1, n + 1)])
    return weights / weights.sum()

def generate_dataset(seed=SEED):
    rng = np.random.default_rng(seed)

    # Fleet attacker: one device injected into 25% of accounts' training sets
    fleet_device = sample_device(rng)

    accounts = []
    for account_id in range(N_ACCOUNTS):
        n_devices = int(rng.integers(N_DEVICES_PER_ACCOUNT[0], N_DEVICES_PER_ACCOUNT[1] + 1))
        known_devices = sample_account_devices(n_devices, rng)
        primary = known_devices[0]

        # Training events: Zipf-weighted draws from known devices
        weights = zipf_weights(n_devices)
        train_events = []
        for _ in range(N_TRAIN_PER_ACCOUNT):
            idx = rng.choice(n_devices, p=weights)
            train_events.append(dict(known_devices[idx]))  # copy

        # Fleet attack injection (25% of accounts)
        is_fleet_account = rng.random() < FLEET_ATTACK_FRACTION
        if is_fleet_account:
            # Replace one training event with fleet device
            replace_idx = int(rng.integers(0, len(train_events)))
            train_events[replace_idx] = dict(fleet_device)

        # Test events:
        # 1. Novel attack: completely foreign device
        novel_device = sample_device(rng)
        while any(
            all(novel_device[f] == d[f] for f in FEATURE_ORDER)
            for d in known_devices
        ):
            novel_device = sample_device(rng)

        # 2. Fleet attack: the shared fleet device
        fleet_test = dict(fleet_device)

        # 3. Spoof attack: matches primary OS/browser/lang, differs on tz only
        spoof_device = dict(primary)
        available_tzs = [t for t in FEATURES["tz"] if t != primary["tz"]]
        spoof_device["tz"] = rng.choice(available_tzs)

        # 4. Negative (enrollment): same OS/browser/tz/lang as primary, different network/screen
        neg_device = dict(primary)
        alt_networks = [n for n in FEATURES["network"] if n != primary["network"]]
        alt_screens = [s for s in FEATURES["screen"] if s != primary["screen"]]
        neg_device["network"] = rng.choice(alt_networks)
        neg_device["screen"] = rng.choice(alt_screens)

        # 5. Known device (true positive anchor — same as a training device)
        known_test = dict(known_devices[0])

        accounts.append({
            "id": account_id,
            "known_devices": known_devices,
            "primary": primary,
            "train_events": train_events,
            "is_fleet_account": is_fleet_account,
            "fleet_device": fleet_device,
            "test": {
                "novel": (novel_device, 1),    # attack
                "fleet": (fleet_test, 1),      # attack
                "spoof": (spoof_device, 1),    # attack
                "neg": (neg_device, 0),        # legitimate enrollment
                "known": (known_test, 0),      # known legitimate
            },
        })

    return accounts, fleet_device

# ---------------------------------------------------------------------------
# FastText training helpers
# ---------------------------------------------------------------------------
def build_mean_pool_corpus(accounts):
    """Each training event → list of 6 prefixed tokens (one sentence)."""
    sentences = []
    for acc in accounts:
        for event in acc["train_events"]:
            sentences.append(device_to_tokens(event))
    return sentences

def build_concat_corpus(accounts):
    """Each training event → single-token sentence (the concatenated string)."""
    sentences = []
    for acc in accounts:
        for event in acc["train_events"]:
            sentences.append([device_to_concat_string(event)])
    return sentences

def train_fasttext(sentences, vector_size=64, window=6, min_count=1, epochs=10, seed=SEED):
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        seed=seed,
        workers=1,
    )
    return model

# ---------------------------------------------------------------------------
# Embedding and inference
# ---------------------------------------------------------------------------
def embed_event_mean_pool(event: dict, model: FastText) -> np.ndarray:
    tokens = device_to_tokens(event)
    vecs = [model.wv[t] for t in tokens]
    return np.mean(vecs, axis=0)

def embed_event_concat(event: dict, model: FastText) -> np.ndarray:
    token = device_to_concat_string(event)
    return model.wv[token]

def compute_centroid(events: list[dict], embed_fn, model) -> np.ndarray:
    vecs = [embed_fn(e, model) for e in events]
    return np.mean(vecs, axis=0)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / (norm_a * norm_b)

# ---------------------------------------------------------------------------
# Trivial baseline: exact 6/6 feature set-membership
# ---------------------------------------------------------------------------
def trivial_score(event: dict, known_devices: list[dict]) -> float:
    """Returns 0 if exact match exists (legitimate), 1 otherwise (flagged)."""
    key = tuple(event[f] for f in FEATURE_ORDER)
    for d in known_devices:
        if tuple(d[f] for f in FEATURE_ORDER) == key:
            return 0.0
    return 1.0

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(accounts, embed_fn, model):
    """Compute per-attack-type AUC scores."""
    scores = {"novel": [], "fleet": [], "spoof": [], "neg": [], "known": []}
    labels = {"novel": [], "fleet": [], "spoof": [], "neg": [], "known": []}

    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_fn, model)
        for attack_type, (event, label) in acc["test"].items():
            dist = cosine_distance(embed_fn(event, model), centroid)
            scores[attack_type].append(dist)
            labels[attack_type].append(label)

    # AUC per attack type: combine attacks (label=1) with negatives (label=0)
    neg_scores = scores["neg"] + scores["known"]
    neg_labels = labels["neg"] + labels["known"]

    aucs = {}
    for attack_type in ["novel", "fleet", "spoof"]:
        combined_scores = scores[attack_type] + neg_scores
        combined_labels = labels[attack_type] + neg_labels
        try:
            aucs[attack_type] = roc_auc_score(combined_labels, combined_scores)
        except Exception:
            aucs[attack_type] = float("nan")

    return aucs

def evaluate_trivial_baseline(accounts):
    """Evaluate trivial set-membership baseline."""
    neg_scores = []
    neg_labels = []
    for acc in accounts:
        for attack_type in ["neg", "known"]:
            event, label = acc["test"][attack_type]
            score = trivial_score(event, acc["known_devices"])
            neg_scores.append(score)
            neg_labels.append(label)

    aucs = {}
    for attack_type in ["novel", "fleet", "spoof"]:
        a_scores = []
        a_labels = []
        for acc in accounts:
            event, label = acc["test"][attack_type]
            score = trivial_score(event, acc["known_devices"])
            a_scores.append(score)
            a_labels.append(label)
        combined_scores = a_scores + neg_scores
        combined_labels = a_labels + neg_labels
        try:
            aucs[attack_type] = roc_auc_score(combined_labels, combined_scores)
        except Exception:
            aucs[attack_type] = float("nan")

    return aucs

def compute_silhouette(accounts, embed_fn, model):
    """Per-device silhouette score across all accounts (cosine metric)."""
    vecs = []
    device_labels = []
    device_id = 0
    for acc in accounts:
        for event in acc["train_events"]:
            vecs.append(embed_fn(event, model))
            device_labels.append(device_id % 10)  # group by modulo to keep tractable
        device_id += 1

    vecs = np.array(vecs)
    device_labels = np.array(device_labels)

    # Sample for speed in PoC (full version in experiment)
    idx = np.random.choice(len(vecs), min(2000, len(vecs)), replace=False)
    try:
        return silhouette_score(vecs[idx], device_labels[idx], metric="cosine")
    except Exception:
        return float("nan")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_auc_comparison(mean_pool_aucs, concat_aucs, trivial_aucs, out_path):
    attack_types = ["novel", "fleet", "spoof"]
    x = np.arange(len(attack_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, [mean_pool_aucs[a] for a in attack_types], width,
                   label="Mean-pool", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, [concat_aucs[a] for a in attack_types], width,
                   label="Concat", color="#FF5722", alpha=0.85)
    bars3 = ax.bar(x + width, [trivial_aucs[a] for a in attack_types], width,
                   label="Trivial baseline", color="#9E9E9E", alpha=0.85)

    ax.set_xlabel("Attack type")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("PoC: AUC by encoding strategy and attack type\n(no bootstrap CI — see experiment2.py)")
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== H2 PoC: Mean-pool vs. Concat FastText for ATO ===\n")

    print("Generating synthetic dataset...")
    accounts, fleet_device = generate_dataset()
    print(f"  {N_ACCOUNTS} accounts, {N_TRAIN_PER_ACCOUNT} training events each")
    print(f"  Fleet device: {fleet_device}")

    # Build corpora
    mean_pool_corpus = build_mean_pool_corpus(accounts)
    concat_corpus = build_concat_corpus(accounts)

    # Train models
    print("\nTraining FastText (mean-pool)...")
    mp_model = train_fasttext(mean_pool_corpus)
    print("Training FastText (concat)...")
    cat_model = train_fasttext(concat_corpus)

    # Evaluate
    print("\nEvaluating...")
    mp_aucs = evaluate(accounts, embed_event_mean_pool, mp_model)
    cat_aucs = evaluate(accounts, embed_event_concat, cat_model)
    trivial_aucs = evaluate_trivial_baseline(accounts)

    mp_sil = compute_silhouette(accounts, embed_event_mean_pool, mp_model)
    cat_sil = compute_silhouette(accounts, embed_event_concat, cat_model)

    # Print results
    print("\n--- Results ---")
    print(f"{'Metric':<30} {'Mean-pool':>12} {'Concat':>12} {'Trivial':>12}")
    print("-" * 68)
    for at in ["novel", "fleet", "spoof"]:
        print(f"AUC ({at:<6}){'':>16} {mp_aucs[at]:>12.4f} {cat_aucs[at]:>12.4f} {trivial_aucs[at]:>12.4f}")
    print(f"Silhouette (cosine){'':>11} {mp_sil:>12.4f} {cat_sil:>12.4f} {'N/A':>12}")

    # Key check: spoof gap
    spoof_gap = mp_aucs["spoof"] - cat_aucs["spoof"]
    print(f"\nSpoof AUC gap (mean-pool - concat): {spoof_gap:+.4f}")
    if spoof_gap > 0:
        print("  -> Mean-pool has HIGHER spoof AUC (consistent with H2)")
    else:
        print("  -> Mean-pool has LOWER or equal spoof AUC (inconsistent with H2)")

    sil_gap = mp_sil - cat_sil
    print(f"Silhouette gap (mean-pool - concat): {sil_gap:+.4f}")
    if sil_gap > 0:
        print("  -> Mean-pool has BETTER silhouette (consistent with H2)")
    else:
        print("  -> Mean-pool has WORSE silhouette (inconsistent with H2)")

    # Plot
    out_fig = Path(__file__).parent.parent / "figures" / "poc_auc_comparison.png"
    plot_auc_comparison(mp_aucs, cat_aucs, trivial_aucs, out_fig)

    print("\n=== PoC complete. Run experiment2.py for bootstrap CIs. ===")

if __name__ == "__main__":
    main()
