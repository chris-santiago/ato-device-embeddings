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
H2 Rerun PoC — Mean-Pool FastText vs. Concatenated-String FastText
===================================================================

Hypothesis (H2):
    Mean-pooling six feature-token embeddings into a device vector will
    outperform directly embedding a single concatenated device string with
    FastText, on both silhouette score and AUC for ATO detection.

Proposed mechanism:
    (1) Cross-boundary n-gram noise: in a concatenated string like
        "ios_safari_utc-5_en_us_wifi_small", the FastText n-grams straddle
        feature boundaries (e.g., "ari_ut", "i_utc") producing spurious signal
        uncorrelated with any semantic dimension.
    (2) Front-loaded positional weighting: a tz mismatch at position 3 corrupts
        n-gram similarity for all following features (lang, net, screen) even
        when those features agree with the victim. Mean-pooling treats all six
        dimensions equally and avoids both effects.

Approach:
    mean_pool_fasttext:
        - FastText trained on 6-token sequences per login event:
          [os_ios, browser_safari, tz_utc-5, lang_en_us, net_wifi, screen_small]
        - Device embedding = mean of 6 individual token vectors.
        - Account centroid = mean of per-device embeddings over training history.

    concat_fasttext:
        - FastText trained on single concatenated string per login event:
          "ios_safari_utc-5_en_us_wifi_small"
        - Device embedding = vector of the concatenated token.
        - Account centroid = mean of per-device embeddings over training history.

Detection: cosine distance between new login embedding and account centroid.
    High distance => anomaly (attack); low distance => known device (normal).

Primary metric: ROC-AUC (novel, fleet, spoof; enrollment in negative class).
Secondary metric: silhouette score (cosine, per-device embeddings, 200-account subsample).

Deliberate exclusions (adversarial targets for Step 3 critique):
    - No bootstrap CIs (added in experiment phase).
    - No OOV token injection test.
    - No feature-ordering permutation study.
    - No prefixed-concat format (key:val|...) variant.
    - No hyperparameter sweep over min_n / max_n (gensim defaults used).
    - No Markov corpus mode (i.i.d. only).
    - No trivial frequency-baseline comparison.
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

# ── paths ─────────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)

# ── constants ─────────────────────────────────────────────────────────────────
N_ACCOUNTS       = 400
N_FLEET_DEVICES  = 80
EVENTS_PER_ACCT  = 60
ATTACK_EVENTS    = 80
FLEET_INJECT     = 8
FLEET_TARGET_FRAC = 0.25
VEC_DIM          = 64
FEAT_WINDOW      = 6   # covers one full 6-token login event for mean-pool
CONCAT_WINDOW    = 1   # each event is one token in concat corpus
N_NEGATIVE       = 10
EPOCHS           = 20
SILHOUETTE_ACCOUNTS = 200

FEATURE_KEYS = ["os", "browser", "tz", "lang", "net", "screen"]
FEATURE_VALUES: Dict[str, List[str]] = {
    "os":      ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz":      ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang":    ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "net":     ["wifi", "lte", "5g", "broadband"],
    "screen":  ["small", "medium", "large", "xlarge"],
}

Profile = Tuple[str, str, str, str, str, str]


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
    # mean-pool corpus: list of 6-token lists (one per login event)
    feature_corpus: List[List[str]] = field(default_factory=list)
    # concat corpus: list of single concatenated tokens (one per login event)
    concat_corpus: List[str] = field(default_factory=list)


@dataclass
class EvalEvent:
    account: Account
    profile: Profile
    label: int   # 1 = attack, 0 = legitimate


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _rand_id(prefix: str) -> str:
    suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=10))
    return f"{prefix}_{suffix}"


def _sample_profile() -> Profile:
    return tuple(rng.choice(FEATURE_VALUES[k]) for k in FEATURE_KEYS)  # type: ignore[return-value]


def _vary_net_screen(base: Profile) -> Profile:
    """Keep OS/browser/tz/lang fixed; randomise net and screen."""
    os_, browser, tz, lang = base[:4]
    net    = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def _to_tokens(profile: Profile) -> List[str]:
    """Mean-pool: 6 prefixed feature tokens."""
    return [f"{k}_{v}" for k, v in zip(FEATURE_KEYS, profile)]


def _to_concat(profile: Profile) -> str:
    """Concat: underscore-joined values, no key prefixes."""
    return "_".join(profile)


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    denom = norm(a) * norm(b)
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_fleet(n: int) -> List[Device]:
    """Fleet devices cluster in attacker-typical OS/tz/lang space."""
    devices = []
    for _ in range(n):
        profile: Profile = (
            rng.choice(["windows", "linux"]),
            rng.choice(["chrome", "firefox"]),
            rng.choice(["utc+5", "utc+8"]),
            rng.choice(["zh_cn", "de_de", "fr_fr"]),
            rng.choice(FEATURE_VALUES["net"]),
            rng.choice(FEATURE_VALUES["screen"]),
        )
        devices.append(Device(_rand_id("fleet"), profile))
    return devices


def _add_login(acct: Account, profile: Profile) -> None:
    acct.feature_corpus.append(_to_tokens(profile))
    acct.concat_corpus.append(_to_concat(profile))
    acct.observed_profiles.add(profile)


def build_accounts(fleet: List[Device]) -> List[Account]:
    accounts: List[Account] = []
    targeted = set(rng.sample(range(N_ACCOUNTS), int(N_ACCOUNTS * FLEET_TARGET_FRAC)))

    for i in range(N_ACCOUNTS):
        primary  = _sample_profile()
        n_devs   = rng.randint(2, 4)
        known    = [Device(_rand_id("legit"), _vary_net_screen(primary)) for _ in range(n_devs)]
        acct     = Account(
            account_id=f"acct_{i:04d}",
            primary_profile=primary,
            known_devices=known,
            known_device_ids={d.device_id for d in known},
        )
        accounts.append(acct)

    for i, acct in enumerate(accounts):
        if i in targeted:
            for dev in rng.choices(fleet, k=FLEET_INJECT):
                _add_login(acct, dev.profile)
        for _ in range(EVENTS_PER_ACCT):
            _add_login(acct, rng.choice(acct.known_devices).profile)

    return accounts


# ── attack profile generators ─────────────────────────────────────────────────

def _novel_profile(primary: Profile) -> Profile:
    """Foreign OS, far tz, non-English language."""
    other_os = [o for o in FEATURE_VALUES["os"] if o != primary[0]]
    tz_idx   = FEATURE_VALUES["tz"].index(primary[2])
    far_tzs  = [j for j in range(len(FEATURE_VALUES["tz"])) if abs(j - tz_idx) >= 2]
    return (
        rng.choice(other_os),
        rng.choice(FEATURE_VALUES["browser"]),
        FEATURE_VALUES["tz"][rng.choice(far_tzs)],
        rng.choice([l for l in FEATURE_VALUES["lang"] if l not in ("en_us", "en_gb")]),
        rng.choice(FEATURE_VALUES["net"]),
        rng.choice(FEATURE_VALUES["screen"]),
    )


def _fleet_profile(fleet: List[Device]) -> Profile:
    return rng.choice(fleet).profile


def _spoof_profile(primary: Profile) -> Profile:
    """Victim OS/browser/lang, different tz — hardest case."""
    other_tz = [t for t in FEATURE_VALUES["tz"] if t != primary[2]]
    return (
        primary[0],
        primary[1],
        rng.choice(other_tz),
        primary[3],
        rng.choice(FEATURE_VALUES["net"]),
        rng.choice(FEATURE_VALUES["screen"]),
    )


def _enroll_profile(primary: Profile) -> Profile:
    """Legit new device: same OS/browser/tz/lang, new net/screen."""
    return (primary[0], primary[1], primary[2], primary[3],
            rng.choice(FEATURE_VALUES["net"]),
            rng.choice(FEATURE_VALUES["screen"]))


def generate_eval_events(
    accounts: List[Account],
    fleet: List[Device],
    n: int = ATTACK_EVENTS,
) -> Dict[str, List[EvalEvent]]:
    events: Dict[str, List[EvalEvent]] = {
        "legit": [], "enroll": [], "novel": [], "fleet": [], "spoof": []
    }
    for _ in range(n):
        acct = rng.choice(accounts)
        events["legit"].append(EvalEvent(acct, rng.choice(acct.known_devices).profile, 0))
        events["enroll"].append(EvalEvent(acct, _enroll_profile(acct.primary_profile), 0))
        events["novel"].append(EvalEvent(acct, _novel_profile(acct.primary_profile), 1))
        events["fleet"].append(EvalEvent(acct, _fleet_profile(fleet), 1))
        events["spoof"].append(EvalEvent(acct, _spoof_profile(acct.primary_profile), 1))
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def _ft_kwargs(window: int) -> dict:
    return dict(
        vector_size=VEC_DIM,
        window=window,
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
        min_n=3,
        max_n=6,
    )


def train_mean_pool(accounts: List[Account]) -> FastText:
    """FastText on 6-token feature sequences (one sentence = one account's history)."""
    sentences = []
    for acct in accounts:
        flat: List[str] = []
        for tok_list in acct.feature_corpus:
            flat.extend(tok_list)
        if flat:
            sentences.append(flat)
    return FastText(sentences=sentences, **_ft_kwargs(FEAT_WINDOW))


def train_concat(accounts: List[Account]) -> FastText:
    """FastText on single concatenated token per login event."""
    sentences = [acct.concat_corpus for acct in accounts if acct.concat_corpus]
    return FastText(sentences=sentences, **_ft_kwargs(CONCAT_WINDOW))


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING & CENTROIDS
# ═══════════════════════════════════════════════════════════════════════════════

def embed_mean_pool(profile: Profile, model: FastText) -> np.ndarray:
    return np.mean([model.wv[t] for t in _to_tokens(profile)], axis=0)


def embed_concat(profile: Profile, model: FastText) -> np.ndarray:
    return model.wv[_to_concat(profile)]


def compute_centroids(accounts: List[Account], model: FastText, embed_fn) -> Dict[str, np.ndarray]:
    fallback = model.wv.vectors.mean(axis=0)
    centroids: Dict[str, np.ndarray] = {}
    for acct in accounts:
        if not acct.observed_profiles:
            centroids[acct.account_id] = fallback
        else:
            vecs = [embed_fn(p, model) for p in acct.observed_profiles]
            centroids[acct.account_id] = np.mean(vecs, axis=0)
    return centroids


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_silhouette(accounts: List[Account], model: FastText, embed_fn) -> float:
    sample    = accounts[:SILHOUETTE_ACCOUNTS]
    vecs, lbl = [], []
    for i, acct in enumerate(sample):
        for p in acct.observed_profiles:
            vecs.append(embed_fn(p, model))
            lbl.append(i)
    vecs_arr = np.array(vecs)
    lbl_arr  = np.array(lbl)
    uniq, cnts = np.unique(lbl_arr, return_counts=True)
    valid = uniq[cnts >= 2]
    if len(valid) < 2:
        return float("nan")
    mask = np.isin(lbl_arr, valid)
    return float(silhouette_score(vecs_arr[mask], lbl_arr[mask], metric="cosine"))


def score_events(
    eval_events: Dict[str, List[EvalEvent]],
    model: FastText,
    centroids: Dict[str, np.ndarray],
    embed_fn,
) -> Dict[str, Dict[str, list]]:
    out = {et: {"scores": [], "labels": []} for et in eval_events}
    for etype, evs in eval_events.items():
        for ev in evs:
            vec = embed_fn(ev.profile, model)
            cen = centroids[ev.account.account_id]
            out[etype]["scores"].append(_cos_dist(vec, cen))
            out[etype]["labels"].append(ev.label)
    return out


def auc_attack_vs_negatives(
    scored: Dict[str, Dict[str, list]],
    attack: str,
) -> float:
    """AUC for one attack type against combined legit+enroll negative class."""
    y_true  = scored["legit"]["labels"] + scored["enroll"]["labels"] + scored[attack]["labels"]
    y_score = scored["legit"]["scores"] + scored["enroll"]["scores"] + scored[attack]["scores"]
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary(
    mp_aucs: Dict[str, float],
    cc_aucs: Dict[str, float],
    mp_sil: float,
    cc_sil: float,
) -> None:
    attacks = ["novel", "fleet", "spoof"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("H2 Rerun PoC — Mean-Pool vs. Concat FastText", fontsize=12, fontweight="bold")

    # AUC panel
    ax   = axes[0]
    x    = np.arange(len(attacks))
    w    = 0.35
    ax.bar(x - w / 2, [mp_aucs[a] for a in attacks], w, label="mean_pool",  color="#2196F3", alpha=0.88)
    ax.bar(x + w / 2, [cc_aucs[a] for a in attacks], w, label="concat",     color="#FF5722", alpha=0.88)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, label="random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC by attack type\n(enrollment in negative class)")
    ax.legend(fontsize=8)

    # Silhouette panel
    ax2   = axes[1]
    names = ["mean_pool\nfasttext", "concat\nfasttext"]
    svals = [mp_sil, cc_sil]
    bars  = ax2.bar(names, svals, color=["#2196F3", "#FF5722"], alpha=0.88)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Silhouette score (cosine)")
    ax2.set_title("Per-account cluster separation")
    for bar, val in zip(bars, svals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.001,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    out = FIGURES_DIR / "h2_rerun_poc_fig1.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("H2 Rerun PoC — Mean-Pool vs. Concat FastText")
    print("=" * 60)

    print("\n[1/5] Generating data...")
    fleet    = build_fleet(N_FLEET_DEVICES)
    accounts = build_accounts(fleet)
    events   = generate_eval_events(accounts, fleet)
    print(f"  Accounts: {len(accounts)} | Fleet devices: {len(fleet)}")
    print(f"  Eval events per type: {ATTACK_EVENTS}")

    print("\n[2/5] Training models...")
    mp_model = train_mean_pool(accounts)
    cc_model = train_concat(accounts)
    print("  Training complete.")

    print("\n[3/5] Computing centroids...")
    mp_centroids = compute_centroids(accounts, mp_model, embed_mean_pool)
    cc_centroids = compute_centroids(accounts, cc_model, embed_concat)

    print("\n[4/5] Silhouette scores...")
    mp_sil = compute_silhouette(accounts, mp_model, embed_mean_pool)
    cc_sil = compute_silhouette(accounts, cc_model, embed_concat)
    delta_sil = mp_sil - cc_sil
    print(f"  mean_pool silhouette : {mp_sil:.4f}")
    print(f"  concat    silhouette : {cc_sil:.4f}")
    print(f"  Delta (mp - cc)      : {delta_sil:+.4f}  "
          f"[{'H2 SUPPORTED' if delta_sil > 0 else 'H2 CONTRADICTED'}]")

    print("\n[5/5] AUC evaluation...")
    mp_scored = score_events(events, mp_model, mp_centroids, embed_mean_pool)
    cc_scored = score_events(events, cc_model, cc_centroids, embed_concat)

    attacks  = ["novel", "fleet", "spoof"]
    mp_aucs: Dict[str, float] = {}
    cc_aucs: Dict[str, float] = {}

    print(f"\n  {'Attack':<8} {'mean_pool':>12} {'concat':>12} {'Delta':>10}  Verdict")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10}  {'-'*20}")
    supported = 0
    for at in attacks:
        mp_auc = auc_attack_vs_negatives(mp_scored, at)
        cc_auc = auc_attack_vs_negatives(cc_scored, at)
        mp_aucs[at] = mp_auc
        cc_aucs[at] = cc_auc
        delta = mp_auc - cc_auc
        verdict = "H2 supported" if delta >= 0 else "H2 contradicted"
        if delta >= 0:
            supported += 1
        print(f"  {at:<8} {mp_auc:>12.4f} {cc_auc:>12.4f} {delta:>+10.4f}  {verdict}")

    print(f"\n  Overall: {supported}/3 attack types support H2 on AUC")

    print("\n  Saving figure...")
    plot_summary(mp_aucs, cc_aucs, mp_sil, cc_sil)

    # Machine-readable summary for downstream agents
    print("\n--- SUMMARY ---")
    print(f"mp_sil={mp_sil:.4f} cc_sil={cc_sil:.4f} delta_sil={delta_sil:+.4f}")
    for at in attacks:
        print(f"mp_auc_{at}={mp_aucs[at]:.4f} cc_auc_{at}={cc_aucs[at]:.4f} delta_{at}={mp_aucs[at]-cc_aucs[at]:+.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
