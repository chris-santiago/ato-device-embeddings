# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "matplotlib==3.10.8",
#   "scipy==1.15.3",
# ]
# ///
#
# H2 Co-occurrence Analysis — C1 Mechanism Diagnostic
# ====================================================
# The C1 finding is that per-event corpus construction causes within-feature
# token embeddings to collapse toward a single point (cosine similarity → 0.9992),
# while per-account corpus construction preserves healthy separation.
#
# The mechanism: in a per-event corpus each training sentence is one login event,
# so every os_* token always appears at position 0 surrounded by the same types
# of neighbour tokens (browser at position 1, timezone at position 2, ...).
# All within-feature tokens accumulate structurally identical context-word
# distributions, leaving no gradient to separate them. In a per-account corpus
# events are concatenated, so os_ios and os_android appear adjacent to different
# prior/subsequent events and accumulate divergent distributions.
#
# This script makes that mechanism observable: it builds both corpus types from
# the same synthetic dataset, computes context-word frequency distributions for
# each token within each feature, and measures Jensen-Shannon divergence between
# within-feature token pairs. Near-zero JSD in the per-event corpus confirms
# tokens are indistinguishable; high JSD in the per-account corpus confirms they
# are separable — closing the gap between "the 2×2 factorial shows corpus matters"
# and "here is why corpus matters."
#
# Outputs:
#   aggregate/figures/h2_c1_cooccurrence.png
#
# Usage:
#   uv run h2_cooccurrence.py
#   uv run h2_cooccurrence.py --seed 42
#   uv run h2_cooccurrence.py --smoke   # fast schema check, no figure written

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import jensenshannon

# ---------------------------------------------------------------------------
# Constants — match h2_rerun.py exactly
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

N_ACCOUNTS       = 400
N_TRAIN          = 60
FLEET_FRAC       = 0.25
WINDOW           = 6   # matches FastText training window

N_ACCOUNTS_SMOKE = 50

TITLE_FS  = 11
LEGEND_FS = 8

def _apply_style(ax):
    ax.spines[["top", "right"]].set_visible(False)

# ---------------------------------------------------------------------------
# Helpers — identical to h2_rerun.py
# ---------------------------------------------------------------------------
def make_token(f, v): return f"{f}_{v}"
def device_key(d): return tuple(d[f] for f in FEATURE_ORDER)
def device_to_tokens(d): return [make_token(f, d[f]) for f in FEATURE_ORDER]

def zipf_weights(n, s=1.5):
    w = np.array([1.0 / k**s for k in range(1, n+1)])
    return w / w.sum()

# ---------------------------------------------------------------------------
# Data generation — identical to h2_rerun.py
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
        weights = zipf_weights(n_dev)
        train_events = []
        for _ in range(n_train):
            idx = rng.choice(n_dev, p=weights)
            train_events.append(dict(known[idx]))
        is_fleet = rng.random() < FLEET_FRAC
        if is_fleet:
            replace_idx = int(rng.integers(0, len(train_events)))
            train_events[replace_idx] = dict(fleet_device)
        accounts.append({"id": acct_id, "train_events": train_events})
    return accounts

# ---------------------------------------------------------------------------
# Corpus construction — identical to h2_rerun.py
# ---------------------------------------------------------------------------
def build_per_event_corpus(accounts):
    """One 6-token sentence per login event (degenerate config)."""
    return [device_to_tokens(e) for acc in accounts for e in acc["train_events"]]

def build_per_account_corpus(accounts):
    """All events concatenated into one sentence per account (robust config)."""
    sentences = []
    for acc in accounts:
        flat = []
        for e in acc["train_events"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences

# ---------------------------------------------------------------------------
# Co-occurrence analysis
# ---------------------------------------------------------------------------
ALL_TOKENS = [make_token(f, v) for f in FEATURE_ORDER for v in FEATURES[f]]
TOKEN_INDEX = {t: i for i, t in enumerate(ALL_TOKENS)}
VOCAB_SIZE  = len(ALL_TOKENS)


def build_cooccurrence_matrix(corpus, window=WINDOW):
    """
    Returns a (VOCAB_SIZE, VOCAB_SIZE) co-occurrence count matrix.
    For each token in each sentence, increments counts for all tokens
    within `window` positions to the left and right.
    """
    matrix = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)
    for sentence in corpus:
        indices = [TOKEN_INDEX[t] for t in sentence if t in TOKEN_INDEX]
        for pos, center in enumerate(indices):
            lo = max(0, pos - window)
            hi = min(len(indices), pos + window + 1)
            for ctx_pos in range(lo, hi):
                if ctx_pos != pos:
                    matrix[center, indices[ctx_pos]] += 1.0
    return matrix


def row_distribution(matrix, token_idx):
    """Normalise a co-occurrence matrix row to a probability distribution."""
    row = matrix[token_idx].copy()
    total = row.sum()
    if total == 0:
        return row
    return row / total


def mean_within_feature_jsd(matrix):
    """
    For each feature, compute the mean pairwise Jensen-Shannon divergence
    between all within-feature token distributions.
    Returns dict: feature -> mean JSD (float in [0, 1]).
    """
    results = {}
    for feat in FEATURE_ORDER:
        values  = FEATURES[feat]
        tokens  = [make_token(feat, v) for v in values]
        indices = [TOKEN_INDEX[t] for t in tokens]
        dists   = [row_distribution(matrix, i) for i in indices]
        jsds = []
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                jsds.append(float(jensenshannon(dists[i], dists[j])))
        results[feat] = float(np.mean(jsds)) if jsds else 0.0
    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {path}")


def plot_cooccurrence(per_event_jsd, per_account_jsd, out_path: Path):
    features = FEATURE_ORDER
    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2,
           [per_event_jsd[f] for f in features],
           width, label="per-event corpus (degenerate)",
           color="#d62728", alpha=0.85)
    ax.bar(x + width / 2,
           [per_account_jsd[f] for f in features],
           width, label="per-account corpus (robust)",
           color="#1f77b4", alpha=0.85)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean within-feature JSD")
    ax.set_title(
        "C1 Mechanism: Within-Feature Token Co-occurrence Divergence\n"
        "per-event corpus forces identical distributions (JSD ≈ 0); "
        "per-account corpus allows divergence",
        fontsize=TITLE_FS,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend(fontsize=LEGEND_FS)
    ax.set_ylim(0, 0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    _apply_style(ax)

    # Annotate mean across all features
    mean_pe = np.mean([per_event_jsd[f] for f in features])
    mean_pa = np.mean([per_account_jsd[f] for f in features])
    ax.text(0.98, 0.95,
            f"All-feature mean JSD\nper-event:   {mean_pe:.3f}\nper-account: {mean_pa:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="C1 co-occurrence mechanism diagnostic")
    p.add_argument("--seed",  type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 50 accounts, assert schema, no figure written")
    return p.parse_args()


def main():
    args = parse_args()
    n_accounts = N_ACCOUNTS_SMOKE if args.smoke else N_ACCOUNTS

    print(f"Generating dataset  seed={args.seed}, n_accounts={n_accounts}")
    accounts = generate_dataset(args.seed, n_accounts=n_accounts)

    print("Building per-event corpus ...")
    pe_corpus = build_per_event_corpus(accounts)
    print(f"  {len(pe_corpus)} sentences")

    print("Building per-account corpus ...")
    pa_corpus = build_per_account_corpus(accounts)
    print(f"  {len(pa_corpus)} sentences")

    print("Computing co-occurrence matrices (this takes a few seconds) ...")
    pe_matrix = build_cooccurrence_matrix(pe_corpus)
    pa_matrix = build_cooccurrence_matrix(pa_corpus)

    print("Computing within-feature JSD ...")
    pe_jsd = mean_within_feature_jsd(pe_matrix)
    pa_jsd = mean_within_feature_jsd(pa_matrix)

    print("\nWithin-feature mean JSD by feature:")
    print(f"  {'Feature':<10}  {'per-event':>10}  {'per-account':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}")
    for feat in FEATURE_ORDER:
        print(f"  {feat:<10}  {pe_jsd[feat]:>10.4f}  {pa_jsd[feat]:>12.4f}")
    mean_pe = np.mean(list(pe_jsd.values()))
    mean_pa = np.mean(list(pa_jsd.values()))
    print(f"  {'mean':<10}  {mean_pe:>10.4f}  {mean_pa:>12.4f}")

    if args.smoke:
        assert mean_pe < mean_pa, \
            f"Smoke FAIL: per-event mean JSD ({mean_pe:.4f}) should be < per-account ({mean_pa:.4f})"
        print("\nSmoke PASS")
        return

    rerun_root = Path(__file__).resolve().parent.parent.parent
    out_path   = rerun_root / "aggregate" / "figures" / "h2_c1_cooccurrence.png"
    plot_cooccurrence(pe_jsd, pa_jsd, out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
