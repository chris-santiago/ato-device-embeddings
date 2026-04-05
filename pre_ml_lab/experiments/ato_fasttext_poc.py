# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.8",
#   "umap-learn>=0.5",
# ]
# ///
"""
Account Takeover Detection via FastText Device Embeddings
=========================================================
Hypothesis: FastText trained on per-account device sequences will embed
known devices near their account's centroid. Novel (takeover) devices will
land far from that centroid, creating a detectable anomaly signal.
"""

import random
import string
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def make_device_id(length: int = 16) -> str:
    """Random alphanumeric device ID, no semantic meaning."""
    return "dev_" + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=length)
    )


@dataclass
class Event:
    account_id: str
    device_id: str
    is_takeover: bool = False


def generate_corpus(
    n_accounts: int = 200,
    devices_per_account: Tuple[int, int] = (3, 8),  # distinct devices owned
    history_len: Tuple[int, int] = (20, 60),  # login events per account
    takeover_rate: float = 0.10,  # fraction of accounts get attacked
    n_takeover_devices: int = 1,
) -> Tuple[Dict[str, List[str]], List[Event], List[Event]]:
    """
    Returns
    -------
    corpus : {account_id: [device_id, ...]}   (training sentences)
    legit_events  : held-out events using known devices
    attack_events : held-out events using brand-new devices
    """
    corpus: Dict[str, List[str]] = {}
    account_devices: Dict[str, List[str]] = {}

    for i in range(n_accounts):
        acct = f"acct_{i:04d}"
        n_dev = random.randint(*devices_per_account)
        devices = [make_device_id() for _ in range(n_dev)]
        account_devices[acct] = devices

        n_events = random.randint(*history_len)
        # device usage follows a Zipf-like pattern (primary device used more)
        weights = np.array([1 / (k + 1) for k in range(n_dev)], dtype=float)
        weights /= weights.sum()
        history = random.choices(devices, weights=weights.tolist(), k=n_events)
        corpus[acct] = history

    # legit test events (known devices, seen during training — expected to score low)
    legit_events: List[Event] = []
    for acct, devices in account_devices.items():
        legit_events.append(Event(acct, random.choice(devices), is_takeover=False))

    # inject takeover events
    attack_events: List[Event] = []
    n_attacked = max(1, int(n_accounts * takeover_rate))
    attacked_accounts = random.sample(list(account_devices.keys()), n_attacked)
    for acct in attacked_accounts:
        for _ in range(n_takeover_devices):
            novel_device = make_device_id()  # never seen anywhere in corpus
            attack_events.append(Event(acct, novel_device, is_takeover=True))

    print(
        f"[data]  accounts={n_accounts}  attacked={n_attacked}  "
        f"legit_holdout={len(legit_events)}  attack_holdout={len(attack_events)}"
    )
    return corpus, legit_events, attack_events


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FASTTEXT TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

from gensim.models import FastText  # noqa: E402


def train_fasttext(corpus: Dict[str, List[str]], **kwargs) -> FastText:
    sentences = list(corpus.values())
    params = dict(
        vector_size=64,
        window=5,
        min_count=1,  # must embed every device, including rare ones
        sg=1,  # skip-gram; better for rare tokens
        workers=4,
        epochs=15,
        seed=SEED,
    )
    params.update(kwargs)
    model = FastText(sentences=sentences, **params)
    print(f"[model] vocab={len(model.wv)}  dim={model.vector_size}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 3. INFERENCE — centroid-based anomaly score
# ═══════════════════════════════════════════════════════════════════════════════

from numpy.linalg import norm  # noqa: E402


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 − cosine_similarity, in [0, 2]."""
    denom = norm(a) * norm(b)
    if denom < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def account_centroid(model: FastText, devices: List[str]) -> np.ndarray:
    vecs = [model.wv[d] for d in devices if d in model.wv]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)


def score_event(model: FastText, corpus: Dict[str, List[str]], event: Event) -> float:
    """
    Cosine distance between the candidate device's embedding and the
    account's centroid. Higher = more anomalous.

    FastText can embed OOV tokens via subword n-grams, so novel device IDs
    get a vector even if never seen during training.
    """
    centroid = account_centroid(model, corpus[event.account_id])
    device_vec = model.wv[event.device_id]  # works for OOV via subwords
    return cosine_distance(centroid, device_vec)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def evaluate(
    model: FastText,
    corpus: Dict[str, List[str]],
    legit_events: List[Event],
    attack_events: List[Event],
) -> Tuple[np.ndarray, np.ndarray, float]:
    all_events = legit_events + attack_events
    scores = np.array([score_event(model, corpus, e) for e in all_events])
    labels = np.array([int(e.is_takeover) for e in all_events])

    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    legit_scores = scores[labels == 0]
    attack_scores = scores[labels == 1]
    print(f"\n[eval]  ROC-AUC = {auc:.4f}")
    print(
        f"        legit  dist: mean={legit_scores.mean():.4f}  std={legit_scores.std():.4f}"
    )
    print(
        f"        attack dist: mean={attack_scores.mean():.4f}  std={attack_scores.std():.4f}"
    )

    # simple threshold at Youden's J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    thresh = thresholds[best_idx]
    preds = (scores >= thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    print(
        f"        threshold={thresh:.4f}  precision={precision:.3f}  recall={recall:.3f}"
    )
    print(f"        TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return fpr, tpr, auc


def plot_roc(
    fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: str = str(FIGURES_DIR / "roc_curve.png")
):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"FastText centroid (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ATO Detection — ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[plot]  saved → {save_path}")
    plt.close(fig)


def plot_score_distributions(
    model: FastText,
    corpus: Dict[str, List[str]],
    legit_events: List[Event],
    attack_events: List[Event],
    save_path: str = str(FIGURES_DIR / "score_distributions.png"),
):
    legit_scores = [score_event(model, corpus, e) for e in legit_events]
    attack_scores = [score_event(model, corpus, e) for e in attack_events]

    fig, ax = plt.subplots(figsize=(7, 4))
    all_scores = legit_scores + attack_scores
    bins = np.linspace(0, max(all_scores) * 1.05, 40)
    ax.hist(legit_scores, bins=bins, alpha=0.6, color="steelblue", label="Known device")
    ax.hist(
        attack_scores, bins=bins, alpha=0.6, color="crimson", label="Takeover device"
    )
    ax.set_xlabel("Cosine distance to account centroid")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution: Known vs Takeover Devices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[plot]  saved → {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION — UMAP embedding of device vectors
# ═══════════════════════════════════════════════════════════════════════════════


def plot_umap(
    model: FastText,
    corpus: Dict[str, List[str]],
    attack_events: List[Event],
    n_accounts_to_show: int = 20,
    save_path: str = str(FIGURES_DIR / "umap_devices.png"),
):
    try:
        import umap  # umap-learn
    except ImportError:
        print("[umap]  umap-learn not installed — skipping UMAP plot")
        return

    # select a subset of accounts for legibility
    accounts_to_show = list(corpus.keys())[:n_accounts_to_show]
    {
        e.device_id for e in attack_events if e.account_id in accounts_to_show
    }

    device_ids, vecs, colors, markers = [], [], [], []
    palette = plt.cm.tab20.colors  # 20 distinct colors

    for idx, acct in enumerate(accounts_to_show):
        color = palette[idx % len(palette)]
        for dev in set(corpus[acct]):
            if dev not in model.wv:
                continue
            device_ids.append(dev)
            vecs.append(model.wv[dev])
            colors.append(color)
            markers.append("o")

    # add attack devices for these accounts
    for e in attack_events:
        if e.account_id not in accounts_to_show:
            continue
        acct_idx = accounts_to_show.index(e.account_id)
        device_ids.append(e.device_id)
        vecs.append(model.wv[e.device_id])
        colors.append(palette[acct_idx % len(palette)])
        markers.append("X")

    if len(vecs) < 5:
        print("[umap]  too few devices to plot")
        return

    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=10, min_dist=0.1)
    embedding = reducer.fit_transform(np.array(vecs))

    fig, ax = plt.subplots(figsize=(10, 8))
    for xy, color, marker in zip(embedding, colors, markers):
        size = 120 if marker == "X" else 40
        ax.scatter(
            *xy,
            c=[color],
            marker=marker,
            s=size,
            edgecolors="black" if marker == "X" else "none",
            linewidths=0.5,
            zorder=3 if marker == "X" else 2,
        )

    # legend entries
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Known device",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            label="Takeover device",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_title(
        f"UMAP of Device Embeddings (first {n_accounts_to_show} accounts)\n"
        "Color = account, X = takeover device"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[plot]  saved → {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ATO FastText PoC")
    print("=" * 60)

    # 1. data
    corpus, legit_events, attack_events = generate_corpus(
        n_accounts=300,
        devices_per_account=(3, 10),
        history_len=(25, 80),
        takeover_rate=0.15,
    )

    # 2. train
    print("\n[train] fitting FastText …")
    model = train_fasttext(corpus)

    # 3. evaluate
    fpr, tpr, auc = evaluate(model, corpus, legit_events, attack_events)

    # 4. plots
    print("\n[plots] generating …")
    plot_roc(fpr, tpr, auc)
    plot_score_distributions(model, corpus, legit_events, attack_events)
    plot_umap(model, corpus, attack_events)

    print("\nDone.")
