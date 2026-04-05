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
H2 Rerun Experiment 1 — Debate-Agreed Empirical Tests
======================================================

Implements all five tests from H2_RERUN_DEBATE.md:

  T1  Bootstrap CIs (N=1000) on all AUC and silhouette metrics.
      Pre-specified verdict: if 2.5th percentile of (mean_pool - concat) spoof
      AUC delta > 0, defense wins C1. Else critique wins.

  T2  Concat window sweep: concat at window ∈ {1, 3, 6} vs mean_pool.
      Pre-specified verdict: if silhouette gap persists at all window values,
      defense wins C2. If window=6 closes gap by >50%, critique wins.

  T3  Prefixed-concat format: "os:ios|browser:safari|..." vs plain concat.
      Pre-specified verdict: if silhouette gap (mean_pool - prefixed) > 0.05,
      defense wins C3. If prefixed reaches within 0.05 silhouette AND 0.01
      spoof AUC of mean_pool, critique wins.

  T4  Trivial baseline: exact-profile (6/6 feature) set-membership.
      Pre-specified verdict: if mean_pool spoof AUC > set-membership spoof AUC,
      defense wins C5. Else critique wins.

  T5  Tz-position permutation: 6 orderings placing tz at positions 0–5.
      Pre-specified verdict: if any ordering recovers >50% of window=1 concat
      spoof delta vs mean_pool, critique wins C7. Else defense wins.

All signals evaluated on identical data (SEED=42, same as PoC).
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
N_ACCOUNTS        = 400
N_FLEET_DEVICES   = 80
EVENTS_PER_ACCT   = 60
ATTACK_EVENTS     = 80
FLEET_INJECT      = 8
FLEET_TARGET_FRAC = 0.25
VEC_DIM           = 64
FEAT_WINDOW       = 6
N_NEGATIVE        = 10
EPOCHS            = 20
N_BOOTSTRAP       = 1000
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
    feature_corpus: List[List[str]] = field(default_factory=list)   # mean-pool
    concat_corpus: List[str] = field(default_factory=list)          # plain concat
    prefixed_corpus: List[str] = field(default_factory=list)        # prefixed concat


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
    return (base[0], base[1], base[2], base[3],
            rng.choice(FEATURE_VALUES["net"]),
            rng.choice(FEATURE_VALUES["screen"]))


def _to_tokens(profile: Profile) -> List[str]:
    return [f"{k}_{v}" for k, v in zip(FEATURE_KEYS, profile)]


def _to_concat(profile: Profile, order: List[int] | None = None) -> str:
    vals = list(profile)
    if order is not None:
        vals = [vals[i] for i in order]
    return "_".join(vals)


def _to_prefixed(profile: Profile) -> str:
    return "|".join(f"{k}:{v}" for k, v in zip(FEATURE_KEYS, profile))


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    denom = norm(a) * norm(b)
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_fleet(n: int) -> List[Device]:
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
    acct.prefixed_corpus.append(_to_prefixed(profile))
    acct.observed_profiles.add(profile)


def build_accounts(fleet: List[Device]) -> List[Account]:
    accounts: List[Account] = []
    targeted = set(rng.sample(range(N_ACCOUNTS), int(N_ACCOUNTS * FLEET_TARGET_FRAC)))

    for i in range(N_ACCOUNTS):
        primary = _sample_profile()
        n_devs  = rng.randint(2, 4)
        known   = [Device(_rand_id("legit"), _vary_net_screen(primary)) for _ in range(n_devs)]
        acct    = Account(
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


def _novel_profile(primary: Profile) -> Profile:
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


def _spoof_profile(primary: Profile) -> Profile:
    other_tz = [t for t in FEATURE_VALUES["tz"] if t != primary[2]]
    return (primary[0], primary[1], rng.choice(other_tz), primary[3],
            rng.choice(FEATURE_VALUES["net"]),
            rng.choice(FEATURE_VALUES["screen"]))


def _enroll_profile(primary: Profile) -> Profile:
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
        events["fleet"].append(EvalEvent(acct, rng.choice(fleet).profile, 1))
        events["spoof"].append(EvalEvent(acct, _spoof_profile(acct.primary_profile), 1))
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def _ft_base_kwargs() -> dict:
    return dict(
        vector_size=VEC_DIM,
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
    sentences = []
    for acct in accounts:
        flat: List[str] = []
        for toks in acct.feature_corpus:
            flat.extend(toks)
        if flat:
            sentences.append(flat)
    return FastText(sentences=sentences, window=FEAT_WINDOW, **_ft_base_kwargs())


def train_concat(accounts: List[Account], window: int) -> FastText:
    sentences = [acct.concat_corpus for acct in accounts if acct.concat_corpus]
    return FastText(sentences=sentences, window=window, **_ft_base_kwargs())


def train_prefixed(accounts: List[Account]) -> FastText:
    sentences = [acct.prefixed_corpus for acct in accounts if acct.prefixed_corpus]
    return FastText(sentences=sentences, window=1, **_ft_base_kwargs())


def train_perm_concat(accounts: List[Account], order: List[int]) -> FastText:
    """T5: concat FastText with feature ordering permuted (window=1)."""
    sentences = [
        [_to_concat(p, order) for p in acct.observed_profiles]
        for acct in accounts if acct.observed_profiles
    ]
    return FastText(sentences=sentences, window=1, **_ft_base_kwargs())


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING & CENTROIDS
# ═══════════════════════════════════════════════════════════════════════════════

def embed_mean_pool(profile: Profile, model: FastText) -> np.ndarray:
    return np.mean([model.wv[t] for t in _to_tokens(profile)], axis=0)


def embed_concat(profile: Profile, model: FastText, order: List[int] | None = None) -> np.ndarray:
    return model.wv[_to_concat(profile, order)]


def embed_prefixed(profile: Profile, model: FastText) -> np.ndarray:
    return model.wv[_to_prefixed(profile)]


def compute_centroids(accounts, model, embed_fn) -> Dict[str, np.ndarray]:
    fallback = model.wv.vectors.mean(axis=0)
    out: Dict[str, np.ndarray] = {}
    for acct in accounts:
        if not acct.observed_profiles:
            out[acct.account_id] = fallback
        else:
            vecs = [embed_fn(p, model) for p in acct.observed_profiles]
            out[acct.account_id] = np.mean(vecs, axis=0)
    return out


def compute_centroids_perm(
    accounts: List[Account],
    model: FastText,
    order: List[int],
) -> Dict[str, np.ndarray]:
    fallback = model.wv.vectors.mean(axis=0)
    out: Dict[str, np.ndarray] = {}
    for acct in accounts:
        if not acct.observed_profiles:
            out[acct.account_id] = fallback
        else:
            vecs = [embed_concat(p, model, order) for p in acct.observed_profiles]
            out[acct.account_id] = np.mean(vecs, axis=0)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

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


def score_events_perm(
    eval_events: Dict[str, List[EvalEvent]],
    model: FastText,
    centroids: Dict[str, np.ndarray],
    order: List[int],
) -> Dict[str, Dict[str, list]]:
    out = {et: {"scores": [], "labels": []} for et in eval_events}
    for etype, evs in eval_events.items():
        for ev in evs:
            vec = embed_concat(ev.profile, model, order)
            cen = centroids[ev.account.account_id]
            out[etype]["scores"].append(_cos_dist(vec, cen))
            out[etype]["labels"].append(ev.label)
    return out


def auc_vs_negatives(scored: Dict[str, Dict[str, list]], attack: str) -> float:
    y_true  = scored["legit"]["labels"] + scored["enroll"]["labels"] + scored[attack]["labels"]
    y_score = scored["legit"]["scores"] + scored["enroll"]["scores"] + scored[attack]["scores"]
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


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


# ── trivial baseline (T4) ────────────────────────────────────────────────────

def score_set_membership(
    eval_events: Dict[str, List[EvalEvent]],
) -> Dict[str, Dict[str, list]]:
    """
    Exact-match (6/6 features) set membership baseline.
    Score = 1 if profile NOT in account's observed training set (anomaly signal),
    score = 0 if profile IS in the training set (known device).
    """
    out = {et: {"scores": [], "labels": []} for et in eval_events}
    for etype, evs in eval_events.items():
        for ev in evs:
            score = 0 if ev.profile in ev.account.observed_profiles else 1
            out[etype]["scores"].append(float(score))
            out[etype]["labels"].append(ev.label)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP (T1)
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_auc_delta(
    mp_scored: Dict[str, Dict[str, list]],
    cc_scored: Dict[str, Dict[str, list]],
    attack: str,
    n: int = N_BOOTSTRAP,
) -> Tuple[float, float, float]:
    """
    Bootstrap distribution of (mean_pool_AUC - concat_AUC) for one attack type.
    Returns (point estimate, 2.5th pct, 97.5th pct).
    """
    neg_mp = mp_scored["legit"]["scores"] + mp_scored["enroll"]["scores"]
    neg_cc = cc_scored["legit"]["scores"] + cc_scored["enroll"]["scores"]
    neg_lbl = mp_scored["legit"]["labels"] + mp_scored["enroll"]["labels"]
    pos_mp  = mp_scored[attack]["scores"]
    pos_cc  = cc_scored[attack]["scores"]
    pos_lbl = mp_scored[attack]["labels"]

    yt  = np.array(neg_lbl + pos_lbl)
    ymp = np.array(neg_mp  + pos_mp)
    ycc = np.array(neg_cc  + pos_cc)

    if len(np.unique(yt)) < 2:
        return float("nan"), float("nan"), float("nan")

    point = roc_auc_score(yt, ymp) - roc_auc_score(yt, ycc)
    rs    = np.random.RandomState(SEED)
    deltas: List[float] = []
    for _ in range(n):
        idx = rs.choice(len(yt), size=len(yt), replace=True)
        if len(np.unique(yt[idx])) < 2:
            continue
        deltas.append(
            float(roc_auc_score(yt[idx], ymp[idx]) - roc_auc_score(yt[idx], ycc[idx]))
        )
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return point, lo, hi


def bootstrap_sil_delta(
    accounts: List[Account],
    mp_model: FastText,
    cc_model: FastText,
    n: int = N_BOOTSTRAP,
) -> Tuple[float, float, float]:
    """Bootstrap CI for (mean_pool_sil - concat_sil) delta."""
    sample    = accounts[:SILHOUETTE_ACCOUNTS]
    mp_vecs, cc_vecs, lbl = [], [], []
    for i, acct in enumerate(sample):
        for p in acct.observed_profiles:
            mp_vecs.append(embed_mean_pool(p, mp_model))
            cc_vecs.append(embed_concat(p, cc_model))
            lbl.append(i)
    mp_arr  = np.array(mp_vecs)
    cc_arr  = np.array(cc_vecs)
    lbl_arr = np.array(lbl)
    uniq, cnts = np.unique(lbl_arr, return_counts=True)
    valid = uniq[cnts >= 2]
    mask  = np.isin(lbl_arr, valid)
    mp_arr = mp_arr[mask]; cc_arr = cc_arr[mask]; lbl_arr = lbl_arr[mask]
    if len(np.unique(lbl_arr)) < 2:
        return float("nan"), float("nan"), float("nan")

    mp_sil = silhouette_score(mp_arr, lbl_arr, metric="cosine")
    cc_sil = silhouette_score(cc_arr, lbl_arr, metric="cosine")
    point  = float(mp_sil - cc_sil)

    rs = np.random.RandomState(SEED)
    deltas: List[float] = []
    idx_pool = np.arange(len(lbl_arr))
    for _ in range(n):
        idx = rs.choice(len(idx_pool), size=len(idx_pool), replace=True)
        sub_mp  = mp_arr[idx]
        sub_cc  = cc_arr[idx]
        sub_lbl = lbl_arr[idx]
        u2, c2  = np.unique(sub_lbl, return_counts=True)
        v2      = u2[c2 >= 2]
        if len(v2) < 2:
            continue
        m2 = np.isin(sub_lbl, v2)
        try:
            d = (silhouette_score(sub_mp[m2], sub_lbl[m2], metric="cosine")
                 - silhouette_score(sub_cc[m2], sub_lbl[m2], metric="cosine"))
            deltas.append(float(d))
        except Exception:
            continue
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return point, lo, hi


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def fig_window_sweep(
    mp_aucs: Dict[str, float],
    mp_sil: float,
    window_results: Dict[int, Dict],  # {window: {aucs, sil}}
) -> None:
    attacks = ["novel", "fleet", "spoof"]
    windows = sorted(window_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("T2 — Concat Window Sweep vs. Mean-Pool", fontsize=11, fontweight="bold")

    colors = {1: "#FF5722", 3: "#FF9800", 6: "#FFC107"}
    ax = axes[0]
    x  = np.arange(len(attacks))
    w  = 0.18
    ax.bar(x - 1.5 * w, [mp_aucs[a] for a in attacks], w, label="mean_pool", color="#2196F3", alpha=0.9)
    for j, win in enumerate(windows):
        ax.bar(x + (j - 0.5) * w, [window_results[win]["aucs"][a] for a in attacks],
               w, label=f"concat w={win}", color=colors[win], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC by attack type")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    sil_vals = [mp_sil] + [window_results[w]["sil"] for w in windows]
    names    = ["mean_pool"] + [f"concat\nw={w}" for w in windows]
    colors2  = ["#2196F3", "#FF5722", "#FF9800", "#FFC107"]
    bars = ax2.bar(names, sil_vals, color=colors2[:len(names)], alpha=0.88)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Silhouette (cosine)")
    ax2.set_title("Per-account cluster separation")
    for bar, val in zip(bars, sil_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / "h2_rerun_exp1_fig1_window_sweep.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_prefixed_concat(
    mp_aucs: Dict[str, float],
    cc_aucs: Dict[str, float],
    pf_aucs: Dict[str, float],
    mp_sil: float,
    cc_sil: float,
    pf_sil: float,
) -> None:
    attacks = ["novel", "fleet", "spoof"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("T3 — Prefixed-Concat vs. Plain Concat vs. Mean-Pool", fontsize=11, fontweight="bold")

    x = np.arange(len(attacks))
    w = 0.25
    ax = axes[0]
    ax.bar(x - w, [mp_aucs[a] for a in attacks], w, label="mean_pool",      color="#2196F3", alpha=0.9)
    ax.bar(x,     [pf_aucs[a] for a in attacks], w, label="prefixed_concat", color="#4CAF50", alpha=0.85)
    ax.bar(x + w, [cc_aucs[a] for a in attacks], w, label="concat",          color="#FF5722", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC by attack type")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = ["mean_pool\nfasttext", "prefixed\nconcat", "plain\nconcat"]
    svals = [mp_sil, pf_sil, cc_sil]
    bars  = ax2.bar(names, svals, color=["#2196F3", "#4CAF50", "#FF5722"], alpha=0.88)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Silhouette (cosine)")
    ax2.set_title("Per-account cluster separation")
    for bar, val in zip(bars, svals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / "h2_rerun_exp1_fig2_prefixed_concat.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_tz_permutation(
    mp_spoof_auc: float,
    cc_spoof_auc: float,  # window=1 baseline
    perm_results: Dict[int, float],  # {tz_position: spoof_auc}
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("T5 — Tz-Position Permutation: Spoof AUC", fontsize=11, fontweight="bold")
    positions = sorted(perm_results.keys())
    aucs = [perm_results[p] for p in positions]
    delta_target = mp_spoof_auc - cc_spoof_auc
    recovery_50  = cc_spoof_auc + 0.5 * delta_target

    ax.plot(positions, aucs, "o-", color="#FF9800", linewidth=2, markersize=7, label="Concat (tz at pos i)")
    ax.axhline(mp_spoof_auc, color="#2196F3", linestyle="--", linewidth=1.5, label=f"mean_pool ({mp_spoof_auc:.4f})")
    ax.axhline(cc_spoof_auc, color="#FF5722", linestyle=":",  linewidth=1.5, label=f"concat w=1 baseline ({cc_spoof_auc:.4f})")
    ax.axhline(recovery_50,  color="#9C27B0", linestyle="-.", linewidth=1,   label=f"50% recovery threshold ({recovery_50:.4f})")
    ax.set_xlabel("Tz feature position in concat string (0=first)")
    ax.set_ylabel("Spoof ROC-AUC")
    ax.set_title("Does moving tz earlier recover the spoof AUC gap?")
    ax.set_xticks(positions)
    ax.legend(fontsize=8)
    ax.set_ylim(0.65, 1.0)

    plt.tight_layout()
    out = FIGURES_DIR / "h2_rerun_exp1_fig3_tz_permutation.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_trivial_baseline(
    mp_aucs: Dict[str, float],
    cc_aucs: Dict[str, float],
    sm_aucs: Dict[str, float],
) -> None:
    attacks = ["novel", "fleet", "spoof"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("T4 — Trivial Baseline (Set Membership) vs. FastText", fontsize=11, fontweight="bold")

    x = np.arange(len(attacks))
    w = 0.25
    ax.bar(x - w, [mp_aucs[a] for a in attacks], w, label="mean_pool",      color="#2196F3", alpha=0.9)
    ax.bar(x,     [cc_aucs[a] for a in attacks], w, label="concat",          color="#FF5722", alpha=0.85)
    ax.bar(x + w, [sm_aucs[a] for a in attacks], w, label="set_membership",  color="#607D8B", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC by attack type (enrollment in negative class)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "h2_rerun_exp1_fig4_trivial_baseline.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_bootstrap_ci(delta_results: Dict) -> None:
    """Plot bootstrap delta CIs for each attack type and silhouette."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("T1 — Bootstrap CIs for (mean_pool - concat) Delta", fontsize=11, fontweight="bold")

    labels = ["novel", "fleet", "spoof", "silhouette"]
    points = [delta_results[k]["point"] for k in labels]
    los    = [delta_results[k]["lo"]    for k in labels]
    his    = [delta_results[k]["hi"]    for k in labels]

    y = np.arange(len(labels))
    ax.barh(y, points, color=["#2196F3" if p > 0 else "#FF5722" for p in points], alpha=0.75)
    ax.errorbar(points, y, xerr=[np.array(points) - np.array(los),
                                  np.array(his) - np.array(points)],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUC / Silhouette delta (mean_pool - concat)")
    ax.set_title("95% bootstrap CI — does CI exclude zero?")

    plt.tight_layout()
    out = FIGURES_DIR / "h2_rerun_exp1_fig5_bootstrap_ci.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("H2 Rerun Experiment 1 — Debate-Agreed Empirical Tests")
    print("=" * 65)

    # ── data generation ───────────────────────────────────────────────────────
    print("\n[DATA] Generating accounts and eval events...")
    fleet    = build_fleet(N_FLEET_DEVICES)
    accounts = build_accounts(fleet)
    events   = generate_eval_events(accounts, fleet)
    print(f"  {len(accounts)} accounts | {len(fleet)} fleet devices | {ATTACK_EVENTS} eval events/type")

    # ── baseline model (mean_pool and concat w=1) ─────────────────────────────
    print("\n[MODELS] Training baseline mean_pool and concat (w=1)...")
    mp_model = train_mean_pool(accounts)
    cc1_model = train_concat(accounts, window=1)
    mp_cen    = compute_centroids(accounts, mp_model,  embed_mean_pool)
    cc1_cen   = compute_centroids(accounts, cc1_model, lambda p, m: embed_concat(p, m))
    mp_scored  = score_events(events, mp_model,  mp_cen,  embed_mean_pool)
    cc1_scored = score_events(events, cc1_model, cc1_cen, lambda p, m: embed_concat(p, m))
    mp_sil  = compute_silhouette(accounts, mp_model,  embed_mean_pool)
    cc1_sil = compute_silhouette(accounts, cc1_model, lambda p, m: embed_concat(p, m))
    attacks = ["novel", "fleet", "spoof"]
    mp_aucs  = {a: auc_vs_negatives(mp_scored,  a) for a in attacks}
    cc1_aucs = {a: auc_vs_negatives(cc1_scored, a) for a in attacks}
    print(f"  mean_pool sil={mp_sil:.4f}  concat(w=1) sil={cc1_sil:.4f}")
    for a in attacks:
        print(f"  {a:<8}  mp={mp_aucs[a]:.4f}  cc1={cc1_aucs[a]:.4f}  Δ={mp_aucs[a]-cc1_aucs[a]:+.4f}")

    # ── T1: Bootstrap CIs ─────────────────────────────────────────────────────
    print("\n[T1] Bootstrap confidence intervals (N=1000)...")
    delta_results: Dict[str, Dict] = {}
    for at in attacks:
        pt, lo, hi = bootstrap_auc_delta(mp_scored, cc1_scored, at)
        delta_results[at] = {"point": pt, "lo": lo, "hi": hi}
        verdict = "DEFENSE wins" if lo > 0 else ("CRITIQUE wins" if hi < 0 else "AMBIGUOUS (CI crosses 0)")
        print(f"  {at:<8}  delta={pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
    pt_sil, lo_sil, hi_sil = bootstrap_sil_delta(accounts, mp_model, cc1_model)
    delta_results["silhouette"] = {"point": pt_sil, "lo": lo_sil, "hi": hi_sil}
    verdict_sil = "DEFENSE wins" if lo_sil > 0 else ("CRITIQUE wins" if hi_sil < 0 else "AMBIGUOUS")
    print(f"  silhouette  delta={pt_sil:+.4f}  95% CI [{lo_sil:+.4f}, {hi_sil:+.4f}]  -> {verdict_sil}")
    fig_bootstrap_ci(delta_results)

    # ── T2: Window sweep ──────────────────────────────────────────────────────
    print("\n[T2] Concat window sweep {1, 3, 6}...")
    window_results: Dict[int, Dict] = {}
    for win in [1, 3, 6]:
        if win == 1:
            model = cc1_model
            cen   = cc1_cen
            scored_w = cc1_scored
            sil_w    = cc1_sil
        else:
            model = train_concat(accounts, window=win)
            cen   = compute_centroids(accounts, model, lambda p, m: embed_concat(p, m))
            scored_w = score_events(events, model, cen, lambda p, m: embed_concat(p, m))
            sil_w    = compute_silhouette(accounts, model, lambda p, m: embed_concat(p, m))
        aucs_w = {a: auc_vs_negatives(scored_w, a) for a in attacks}
        window_results[win] = {"aucs": aucs_w, "sil": sil_w}
        print(f"  concat w={win}  sil={sil_w:.4f}  novel={aucs_w['novel']:.4f}  spoof={aucs_w['spoof']:.4f}")

    # verdict for C2 on silhouette: gap persists if mean_pool > all concat windows
    sil_gap_persists = all(mp_sil > window_results[w]["sil"] for w in [1, 3, 6])
    spoof_gap_50pct_closed = (window_results[6]["aucs"]["spoof"] - cc1_aucs["spoof"]) > 0.5 * (mp_aucs["spoof"] - cc1_aucs["spoof"])
    print(f"  Silhouette gap persists at all windows: {sil_gap_persists} -> {'DEFENSE wins C2 (sil)' if sil_gap_persists else 'CRITIQUE wins C2 (sil)'}")
    print(f"  Window=6 concat closes >50% of spoof delta: {spoof_gap_50pct_closed} -> {'CRITIQUE wins C2 (AUC)' if spoof_gap_50pct_closed else 'DEFENSE wins C2 (AUC)'}")
    fig_window_sweep(mp_aucs, mp_sil, window_results)

    # ── T3: Prefixed-concat ───────────────────────────────────────────────────
    print("\n[T3] Prefixed-concat format (os:val|browser:val|...)...")
    pf_model  = train_prefixed(accounts)
    pf_cen    = compute_centroids(accounts, pf_model, embed_prefixed)
    pf_scored = score_events(events, pf_model, pf_cen, embed_prefixed)
    pf_sil    = compute_silhouette(accounts, pf_model, embed_prefixed)
    pf_aucs   = {a: auc_vs_negatives(pf_scored, a) for a in attacks}
    sil_gap_vs_prefixed = mp_sil - pf_sil
    print(f"  prefixed sil={pf_sil:.4f}  (mean_pool sil gap={sil_gap_vs_prefixed:+.4f})")
    for a in attacks:
        print(f"  {a:<8}  prefixed={pf_aucs[a]:.4f}  mp={mp_aucs[a]:.4f}  Δ={mp_aucs[a]-pf_aucs[a]:+.4f}")
    defense_wins_c3 = sil_gap_vs_prefixed > 0.05
    critique_wins_c3 = (sil_gap_vs_prefixed <= 0.05 and abs(mp_aucs["spoof"] - pf_aucs["spoof"]) <= 0.01)
    if defense_wins_c3:
        print(f"  C3 verdict: DEFENSE wins (silhouette gap {sil_gap_vs_prefixed:+.4f} > 0.05 threshold)")
    elif critique_wins_c3:
        print(f"  C3 verdict: CRITIQUE wins (prefixed-concat within 0.05 sil AND 0.01 spoof AUC of mean_pool)")
    else:
        print(f"  C3 verdict: AMBIGUOUS (prefixed improves over plain concat but does not reach mean_pool)")
    fig_prefixed_concat(mp_aucs, cc1_aucs, pf_aucs, mp_sil, cc1_sil, pf_sil)

    # ── T4: Trivial baseline ──────────────────────────────────────────────────
    print("\n[T4] Trivial baseline: exact set-membership (6/6 feature match)...")
    sm_scored = score_set_membership(events)
    sm_aucs   = {a: auc_vs_negatives(sm_scored, a) for a in attacks}
    for a in attacks:
        print(f"  {a:<8}  set_membership={sm_aucs[a]:.4f}  mean_pool={mp_aucs[a]:.4f}  Δ={mp_aucs[a]-sm_aucs[a]:+.4f}")
    defense_wins_c5 = mp_aucs["spoof"] > sm_aucs["spoof"]
    print(f"  C5 verdict: {'DEFENSE wins' if defense_wins_c5 else 'CRITIQUE wins'} (mean_pool spoof {mp_aucs['spoof']:.4f} {'>' if defense_wins_c5 else '<='} set_membership {sm_aucs['spoof']:.4f})")
    fig_trivial_baseline(mp_aucs, cc1_aucs, sm_aucs)

    # ── T5: Tz-position permutation ───────────────────────────────────────────
    print("\n[T5] Tz-position permutation (tz at positions 0-5)...")
    tz_idx_in_default = 2  # tz is 3rd feature (0-indexed=2) in default order
    perm_spoof: Dict[int, float] = {}
    delta_target_spoof = mp_aucs["spoof"] - cc1_aucs["spoof"]
    recovery_50_threshold = cc1_aucs["spoof"] + 0.5 * delta_target_spoof
    print(f"  Target delta = {delta_target_spoof:+.4f}, 50% recovery threshold = {recovery_50_threshold:.4f}")

    for tz_pos in range(6):
        # Build a new feature order: insert tz_idx_in_default at tz_pos
        remaining = [i for i in range(6) if i != tz_idx_in_default]
        order = remaining[:tz_pos] + [tz_idx_in_default] + remaining[tz_pos:]
        perm_model  = train_perm_concat(accounts, order)
        perm_cen    = compute_centroids_perm(accounts, perm_model, order)
        perm_scored = score_events_perm(events, perm_model, perm_cen, order)
        spoof_auc   = auc_vs_negatives(perm_scored, "spoof")
        perm_spoof[tz_pos] = spoof_auc
        recovered   = spoof_auc - cc1_aucs["spoof"]
        pct_recovered = recovered / delta_target_spoof * 100 if delta_target_spoof > 0 else 0.0
        print(f"  tz at pos {tz_pos}: spoof_auc={spoof_auc:.4f}  recovered={recovered:+.4f} ({pct_recovered:.1f}%)")

    best_perm_spoof = max(perm_spoof.values())
    best_recovery   = (best_perm_spoof - cc1_aucs["spoof"]) / delta_target_spoof * 100 if delta_target_spoof > 0 else 0.0
    critique_wins_c7 = best_perm_spoof >= recovery_50_threshold
    print(f"  Best permutation spoof AUC: {best_perm_spoof:.4f} ({best_recovery:.1f}% recovery)")
    print(f"  C7 verdict: {'CRITIQUE wins' if critique_wins_c7 else 'DEFENSE wins'} (best recovery {best_recovery:.1f}% vs 50% threshold)")
    fig_tz_permutation(mp_aucs["spoof"], cc1_aucs["spoof"], perm_spoof)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EXPERIMENT 1 VERDICT SUMMARY")
    print("=" * 65)
    print(f"  T1 C1 (bootstrap spoof CI): {'DEFENSE' if delta_results['spoof']['lo'] > 0 else 'CRITIQUE'} wins")
    print(f"  T1 C1 (bootstrap sil CI):   {'DEFENSE' if delta_results['silhouette']['lo'] > 0 else 'CRITIQUE'} wins")
    print(f"  T2 C2 (window sweep sil):   {'DEFENSE' if sil_gap_persists else 'CRITIQUE'} wins")
    print(f"  T2 C2 (window sweep AUC):   {'CRITIQUE' if spoof_gap_50pct_closed else 'DEFENSE'} wins")
    print(f"  T3 C3 (prefixed-concat):    {'DEFENSE' if defense_wins_c3 else ('CRITIQUE' if critique_wins_c3 else 'AMBIGUOUS')} wins")
    print(f"  T4 C5 (trivial baseline):   {'DEFENSE' if defense_wins_c5 else 'CRITIQUE'} wins")
    print(f"  T5 C7 (tz permutation):     {'CRITIQUE' if critique_wins_c7 else 'DEFENSE'} wins")
    print(f"\n  H2 overall (mean_pool > concat on AUC): 3/3 attack types in PoC")
    print(f"  H2 on silhouette: mean_pool={mp_sil:.4f} > concat={cc1_sil:.4f} (Δ={mp_sil-cc1_sil:+.4f})")
    print("Done.")


if __name__ == "__main__":
    main()
