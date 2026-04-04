# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.8",
# ]
# ///
"""
H2 PoC — Mean-Pool FastText vs. Concatenated-String FastText
=============================================================

Hypothesis (H2):
    Mean-pooling six feature-token embeddings will outperform concatenated-string
    FastText because (a) n-gram bleed across feature boundaries contributes spurious
    signal uncorrelated with any semantic dimension, and (b) front-loaded positional
    weighting means a mismatch at feature N penalizes n-gram overlap for all features
    that follow.

    Observable predictions:
      1. concat_fasttext silhouette score < mean_pool_fasttext silhouette score
      2. concat_fasttext AUC (novel, fleet, spoof) <= mean_pool_fasttext AUC

Two signals are compared head-to-head on otherwise identical data and evaluation:

  feature_fasttext (mean-pool)
    - Train FastText on sequences of 6 individual feature tokens per login event.
    - Embed a device as the mean of its 6 token vectors.
    - Account centroid = mean of per-device embeddings.

  concat_fasttext
    - Train FastText on the single concatenated token per login event:
      "ios_safari_utc-5_en_us_wifi_small"
    - Embed a device directly as the vector of its concatenated token.
    - Account centroid = mean of per-device embeddings.

Primary metric: ROC-AUC on novel / fleet / spoof attack types (enrollment in the
negative class — same corrected evaluation design from Experiment 3).

Secondary metric: silhouette score over per-device embeddings across 400 accounts.

Deliberate scope exclusions (become Step 3 adversarial targets):
  - No feature ordering permutation study (H2's directional claim; not tested here).
  - No statistical bootstrap CIs (PoC only; added in Step 6 experiment).
  - No Markov corpus mode (i.i.d. only for simplicity).
  - No Word2Vec baseline, OOV signals, or id_w2v (already characterised in Exp 3).
  - No hyperparameter sweep over min_n / max_n (standard gensim defaults used).
  - Concatenation order is fixed (os-browser-tz-lang-net-screen); ordering sensitivity
    is an open question surfaced in Step 3, resolved empirically in Step 6.
"""

import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics import roc_auc_score, silhouette_score
from gensim.models import FastText

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)

# ── constants (match Experiment 3 exactly) ────────────────────────────────────
N_ACCOUNTS = 400
N_FLEET_DEVICES = 80
EVENTS_PER_ACCT = 60
ATTACK_EVENTS = 80
FLEET_INJECT = 8
FLEET_TARGET_FRAC = 0.25
VEC_DIM = 64
FEAT_WINDOW = 6  # 6 feature tokens = one full login event
N_NEGATIVE = 10
EPOCHS = 20

FEATURE_KEYS = ["os", "browser", "tz", "lang", "net", "screen"]
FEATURE_VALUES = {
    "os": ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz": ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang": ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "net": ["wifi", "lte", "5g", "broadband"],
    "screen": ["small", "medium", "large", "xlarge"],
}

Profile = Tuple[str, str, str, str, str, str]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def make_device_id(prefix: str = "dev") -> str:
    suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=12))
    return f"{prefix}_{suffix}"


def sample_primary_profile() -> Profile:
    return tuple(rng.choice(FEATURE_VALUES[k]) for k in FEATURE_KEYS)  # type: ignore[return-value]


def profile_to_tokens(profile: Profile) -> List[str]:
    """Mean-pool approach: 6 separate feature tokens per device."""
    return [f"{k}_{v}" for k, v in zip(FEATURE_KEYS, profile)]


def profile_to_concat(profile: Profile) -> str:
    """Concat approach: single joined string token per device."""
    return "_".join(profile)


def vary_profile_net_screen(base: Profile) -> Profile:
    os_, browser, tz, lang = base[0], base[1], base[2], base[3]
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    denom = norm(a) * norm(b)
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Device:
    device_id: str
    profile: Profile


@dataclass
class Account:
    account_id: str
    primary_profile: Profile
    known_devices: List[Device]
    known_device_ids: Set[str] = field(default_factory=set)
    observed_profiles: Set[Profile] = field(default_factory=set)
    id_corpus: List[str] = field(default_factory=list)
    # mean-pool corpus: sequence of 6-token lists, one per login event
    feature_corpus: List[List[str]] = field(default_factory=list)
    # concat corpus: sequence of single concatenated tokens, one per login event
    concat_corpus: List[str] = field(default_factory=list)


@dataclass
class EvalEvent:
    account: Account
    device_id: str
    profile: Profile
    label: int  # 1 = attack, 0 = legitimate


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def build_fleet(n: int) -> List[Device]:
    devices = []
    for _ in range(n):
        os_ = rng.choice(["windows", "linux"])
        browser = rng.choice(["chrome", "firefox"])
        tz = rng.choice(["utc+5", "utc+8"])
        lang = rng.choice(["zh_cn", "de_de", "fr_fr"])
        net = rng.choice(FEATURE_VALUES["net"])
        screen = rng.choice(FEATURE_VALUES["screen"])
        profile: Profile = (os_, browser, tz, lang, net, screen)
        devices.append(Device(device_id=make_device_id("fleet"), profile=profile))
    return devices


def build_accounts(fleet: List[Device]) -> List[Account]:
    accounts = []
    targeted_count = int(N_ACCOUNTS * FLEET_TARGET_FRAC)
    targeted_indices = set(rng.sample(range(N_ACCOUNTS), targeted_count))

    for i in range(N_ACCOUNTS):
        primary = sample_primary_profile()
        n_devs = rng.randint(2, 4)
        known_devices = [
            Device(device_id=make_device_id("legit"), profile=vary_profile_net_screen(primary))
            for _ in range(n_devs)
        ]
        acct = Account(
            account_id=f"acct_{i:04d}",
            primary_profile=primary,
            known_devices=known_devices,
        )
        acct.known_device_ids = {d.device_id for d in known_devices}
        accounts.append(acct)

    for i, acct in enumerate(accounts):
        devices = acct.known_devices
        is_targeted = i in targeted_indices

        if is_targeted:
            for dev in rng.choices(fleet, k=FLEET_INJECT):
                acct.id_corpus.append(dev.device_id)
                acct.feature_corpus.append(profile_to_tokens(dev.profile))
                acct.concat_corpus.append(profile_to_concat(dev.profile))
                acct.observed_profiles.add(dev.profile)

        for _ in range(EVENTS_PER_ACCT):
            dev = rng.choice(devices)
            acct.id_corpus.append(dev.device_id)
            acct.feature_corpus.append(profile_to_tokens(dev.profile))
            acct.concat_corpus.append(profile_to_concat(dev.profile))
            acct.observed_profiles.add(dev.profile)

    return accounts


def make_novel_attack_profile(victim_primary: Profile) -> Profile:
    victim_os = victim_primary[0]
    other_os = [o for o in FEATURE_VALUES["os"] if o != victim_os]
    os_ = rng.choice(other_os)
    browser = rng.choice(FEATURE_VALUES["browser"])
    victim_tz_idx = FEATURE_VALUES["tz"].index(victim_primary[2])
    far_tz_indices = [j for j, _ in enumerate(FEATURE_VALUES["tz"]) if abs(j - victim_tz_idx) >= 2]
    tz = FEATURE_VALUES["tz"][rng.choice(far_tz_indices)]
    lang = rng.choice([l for l in FEATURE_VALUES["lang"] if l not in ("en_us", "en_gb")])
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def make_spoof_attack_profile(victim_primary: Profile) -> Profile:
    os_ = victim_primary[0]
    browser = victim_primary[1]
    victim_tz = victim_primary[2]
    other_tz = [t for t in FEATURE_VALUES["tz"] if t != victim_tz]
    tz = rng.choice(other_tz)
    lang = victim_primary[3]
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def make_enroll_profile(victim_primary: Profile) -> Profile:
    os_, browser, tz, lang = victim_primary[:4]
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def generate_eval_events(
    accounts: List[Account],
    fleet: List[Device],
    n_per_type: int = ATTACK_EVENTS,
) -> Dict[str, List[EvalEvent]]:
    events: Dict[str, List[EvalEvent]] = {
        "legit": [],
        "enroll": [],
        "novel": [],
        "fleet": [],
        "spoof": [],
    }
    for _ in range(n_per_type):
        acct = rng.choice(accounts)
        dev = rng.choice(acct.known_devices)
        events["legit"].append(EvalEvent(acct, dev.device_id, dev.profile, label=0))

        enroll_id = make_device_id("enroll")
        events["enroll"].append(EvalEvent(acct, enroll_id, make_enroll_profile(acct.primary_profile), label=0))

        novel_id = make_device_id("novel")
        events["novel"].append(EvalEvent(acct, novel_id, make_novel_attack_profile(acct.primary_profile), label=1))

        fleet_dev = rng.choice(fleet)
        events["fleet"].append(EvalEvent(acct, fleet_dev.device_id, fleet_dev.profile, label=1))

        spoof_id = make_device_id("spoof")
        events["spoof"].append(EvalEvent(acct, spoof_id, make_spoof_attack_profile(acct.primary_profile), label=1))

    return events


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_mean_pool_fasttext(accounts: List[Account]) -> FastText:
    """FastText on 6-token sequences (mean-pool signal from Experiment 3)."""
    sentences = []
    for acct in accounts:
        flat: List[str] = []
        for token_list in acct.feature_corpus:
            flat.extend(token_list)
        if flat:
            sentences.append(flat)
    return FastText(
        sentences=sentences,
        vector_size=VEC_DIM,
        window=FEAT_WINDOW,
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
        min_n=3,
        max_n=6,
    )


def train_concat_fasttext(accounts: List[Account]) -> FastText:
    """FastText on single concatenated token per login event."""
    sentences = []
    for acct in accounts:
        if acct.concat_corpus:
            sentences.append(acct.concat_corpus)
    return FastText(
        sentences=sentences,
        vector_size=VEC_DIM,
        window=1,  # each event is one token; context is the adjacent events
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
        min_n=3,
        max_n=6,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING & CENTROIDS
# ═══════════════════════════════════════════════════════════════════════════════


def get_mean_pool_vec(profile: Profile, model: FastText) -> np.ndarray:
    """Mean of 6 individual feature-token vectors."""
    tokens = profile_to_tokens(profile)
    return np.mean([model.wv[t] for t in tokens], axis=0)


def get_concat_vec(profile: Profile, model: FastText) -> np.ndarray:
    """Single vector for the concatenated token string."""
    return model.wv[profile_to_concat(profile)]


def compute_centroids(
    accounts: List[Account],
    model: FastText,
    embed_fn,
) -> Dict[str, np.ndarray]:
    """Per-account centroid = mean of unique observed device embeddings."""
    centroids: Dict[str, np.ndarray] = {}
    global_mean = model.wv.vectors.mean(axis=0)
    for acct in accounts:
        if not acct.observed_profiles:
            centroids[acct.account_id] = global_mean
            continue
        vecs = [embed_fn(p, model) for p in acct.observed_profiles]
        centroids[acct.account_id] = np.mean(vecs, axis=0)
    return centroids


# ═══════════════════════════════════════════════════════════════════════════════
# SILHOUETTE SCORE
# ═══════════════════════════════════════════════════════════════════════════════


def compute_silhouette(
    accounts: List[Account],
    model: FastText,
    embed_fn,
    max_accounts: int = 200,
) -> float:
    """
    Silhouette over per-device embeddings.
    Label = account index. Higher silhouette => tighter within-account clusters.
    Subsample to max_accounts to keep runtime manageable.
    """
    sample = accounts[:max_accounts]
    vecs, labels = [], []
    for i, acct in enumerate(sample):
        for p in acct.observed_profiles:
            vecs.append(embed_fn(p, model))
            labels.append(i)
    vecs_arr = np.array(vecs)
    labels_arr = np.array(labels)
    # Need at least 2 labels with 2+ samples each
    unique, counts = np.unique(labels_arr, return_counts=True)
    valid = unique[counts >= 2]
    if len(valid) < 2:
        return float("nan")
    mask = np.isin(labels_arr, valid)
    return float(silhouette_score(vecs_arr[mask], labels_arr[mask], metric="cosine"))


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def score_signal(
    eval_events: Dict[str, List[EvalEvent]],
    model: FastText,
    centroids: Dict[str, np.ndarray],
    embed_fn,
) -> Dict[str, Dict[str, list]]:
    """Score all events with a given embedding function and centroid map."""
    results: Dict[str, Dict[str, list]] = {
        et: {"scores": [], "labels": []} for et in eval_events
    }
    for event_type, events in eval_events.items():
        for ev in events:
            vec = embed_fn(ev.profile, model)
            centroid = centroids[ev.account.account_id]
            results[event_type]["scores"].append(cos_dist(vec, centroid))
            results[event_type]["labels"].append(ev.label)
    return results


def auc_vs_negative(
    results: Dict[str, Dict[str, list]],
    attack_type: str,
) -> float:
    """
    AUC for one attack type against combined legit+enroll negative class.
    Returns NaN if data is insufficient.
    """
    neg_scores = results["legit"]["scores"] + results["enroll"]["scores"]
    neg_labels = results["legit"]["labels"] + results["enroll"]["labels"]
    pos_scores = results[attack_type]["scores"]
    pos_labels = results[attack_type]["labels"]
    y_true = neg_labels + pos_labels
    y_score = neg_scores + pos_scores
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_results(
    mp_aucs: Dict[str, float],
    cc_aucs: Dict[str, float],
    mp_sil: float,
    cc_sil: float,
) -> None:
    attack_types = ["novel", "fleet", "spoof"]
    mp_vals = [mp_aucs[a] for a in attack_types]
    cc_vals = [cc_aucs[a] for a in attack_types]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # AUC comparison bar chart
    x = np.arange(len(attack_types))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width / 2, mp_vals, width, label="mean_pool_fasttext", color="#2196F3")
    ax.bar(x + width / 2, cc_vals, width, label="concat_fasttext", color="#FF5722")
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8, label="OOV ceiling (0.75)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC by attack type\n(enrollment in negative class)")
    ax.legend(fontsize=8)

    # Silhouette comparison
    ax2 = axes[1]
    names = ["mean_pool\nfasttext", "concat\nfasttext"]
    sil_vals = [mp_sil, cc_sil]
    colors = ["#2196F3", "#FF5722"]
    bars = ax2.bar(names, sil_vals, color=colors)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Silhouette score (cosine)")
    ax2.set_title("Per-account cluster separation\n(higher = tighter within-account clusters)")
    for bar, val in zip(bars, sil_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.002,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out = FIGURES_DIR / "concat_poc_fig1.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("H2 PoC — Mean-Pool vs. Concat FastText")
    print("=" * 50)

    print("\n[1/5] Generating data...")
    fleet = build_fleet(N_FLEET_DEVICES)
    accounts = build_accounts(fleet)
    eval_events = generate_eval_events(accounts, fleet)
    print(f"  Accounts: {len(accounts)}, Fleet devices: {len(fleet)}")
    print(f"  Eval events per type: {ATTACK_EVENTS}")

    print("\n[2/5] Training models...")
    mp_model = train_mean_pool_fasttext(accounts)
    cc_model = train_concat_fasttext(accounts)
    print("  Done.")

    print("\n[3/5] Computing centroids...")
    mp_centroids = compute_centroids(accounts, mp_model, get_mean_pool_vec)
    cc_centroids = compute_centroids(accounts, cc_model, get_concat_vec)

    print("\n[4/5] Silhouette scores...")
    mp_sil = compute_silhouette(accounts, mp_model, get_mean_pool_vec)
    cc_sil = compute_silhouette(accounts, cc_model, get_concat_vec)
    print(f"  mean_pool_fasttext silhouette: {mp_sil:.4f}")
    print(f"  concat_fasttext    silhouette: {cc_sil:.4f}")
    diff = mp_sil - cc_sil
    verdict = "H2 SUPPORTED" if diff > 0 else "H2 CONTRADICTED"
    print(f"  Δ (mp - concat) = {diff:+.4f}  [{verdict} on silhouette]")

    print("\n[5/5] AUC evaluation...")
    mp_scored = score_signal(eval_events, mp_model, mp_centroids, get_mean_pool_vec)
    cc_scored = score_signal(eval_events, cc_model, cc_centroids, get_concat_vec)

    attack_types = ["novel", "fleet", "spoof"]
    mp_aucs: Dict[str, float] = {}
    cc_aucs: Dict[str, float] = {}

    print(f"\n  {'Attack':<8} {'mean_pool':>12} {'concat':>12} {'Δ':>10} {'verdict':>20}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*20}")
    for at in attack_types:
        mp_auc = auc_vs_negative(mp_scored, at)
        cc_auc = auc_vs_negative(cc_scored, at)
        mp_aucs[at] = mp_auc
        cc_aucs[at] = cc_auc
        delta = mp_auc - cc_auc
        v = "H2 supported" if delta >= 0 else "H2 contradicted"
        print(f"  {at:<8} {mp_auc:>12.4f} {cc_auc:>12.4f} {delta:>+10.4f} {v:>20}")

    print("\n[6/6] Saving figure...")
    plot_results(mp_aucs, cc_aucs, mp_sil, cc_sil)

    print("\nDone.")


if __name__ == "__main__":
    main()
