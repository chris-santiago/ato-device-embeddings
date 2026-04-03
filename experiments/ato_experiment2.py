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
ATO Experiment 2 — Debate-Driven Empirical Tests
=================================================
Addresses the six concrete additions agreed in DEBATE.md:

  1. Cross-account in-vocab baseline   (Point 1 — mechanism test)
  2. OOV binary baseline               (Point 5 — does FastText beat a lookup?)
  3. Word2Vec ablation                 (Point 2 — isolate subword contribution)
  4. Silhouette score in 64D           (Point 9 — do account clusters actually exist?)
  5. Bootstrap CIs on ROC-AUC         (Point 6 — quantify uncertainty)
  6. AUC stratified by history length  (Point 3 — where does signal degrade?)
  + Centroid norm vs FPR              (Point 4 — is ||μ|| a useful confidence signal?)
  + Markov session model vs i.i.d.    (Point 8 — is the i.i.d. AUC an upper bound?)
"""

import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import norm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import silhouette_score
from gensim.models import FastText, Word2Vec

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_ACCOUNTS = 300
TAKEOVER_RATE = 0.15
DEV_RANGE = (3, 10)
HIST_RANGE = (25, 80)
VEC_DIM = 64
WINDOW = 5
EPOCHS = 15
N_BOOTSTRAP = 1000


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════


def make_device_id(length: int = 16) -> str:
    return "dev_" + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=length)
    )


@dataclass
class Event:
    account_id: str
    device_id: str
    label: Literal["known", "cross_account", "oov_attack"]


def _zipf_weights(n: int) -> List[float]:
    w = np.array([1 / (k + 1) for k in range(n)], dtype=float)
    return (w / w.sum()).tolist()


def generate_history_iid(devices: List[str], n_events: int) -> List[str]:
    """i.i.d. draws with Zipf usage weights — PoC baseline."""
    return random.choices(devices, weights=_zipf_weights(len(devices)), k=n_events)


def generate_history_markov(
    devices: List[str], n_events: int, stay_prob: float = 0.70
) -> List[str]:
    """
    Markov device-switching model.
    With probability `stay_prob` the user stays on the current device;
    otherwise a new device is drawn from the Zipf marginal.
    This creates session-like runs while preserving the long-run Zipf distribution
    and reduces the effective number of independent skip-gram training pairs.
    """
    zipf = _zipf_weights(len(devices))
    current = random.choices(devices, weights=zipf, k=1)[0]
    history = [current]
    for _ in range(n_events - 1):
        if random.random() < stay_prob:
            history.append(current)
        else:
            current = random.choices(devices, weights=zipf, k=1)[0]
            history.append(current)
    return history


def generate_corpus(
    mode: Literal["iid", "markov"] = "iid",
    n_accounts: int = N_ACCOUNTS,
    devices_per_account: Tuple[int, int] = DEV_RANGE,
    history_len: Tuple[int, int] = HIST_RANGE,
    takeover_rate: float = TAKEOVER_RATE,
    markov_stay: float = 0.70,
) -> Tuple[Dict, Dict, List[Event], List[Event], List[Event]]:
    """
    Returns
    -------
    corpus          : {account_id: [device_id, ...]}  — training sentences
    account_devices : {account_id: [device_id, ...]}  — distinct owned devices
    known_events    : legit events using own in-vocab devices
    cross_events    : in-vocab devices from a DIFFERENT account (mechanism test)
    attack_events   : novel OOV devices (takeover simulation)
    """
    corpus: Dict[str, List[str]] = {}
    account_devices: Dict[str, List[str]] = {}
    account_history_len: Dict[str, int] = {}

    for i in range(n_accounts):
        acct = f"acct_{i:04d}"
        n_dev = random.randint(*devices_per_account)
        devices = [make_device_id() for _ in range(n_dev)]
        account_devices[acct] = devices

        n_events = random.randint(*history_len)
        account_history_len[acct] = n_events

        if mode == "markov":
            history = generate_history_markov(devices, n_events, stay_prob=markov_stay)
        else:
            history = generate_history_iid(devices, n_events)
        corpus[acct] = history

    accounts = list(account_devices.keys())

    # known events: a device from the account's own fleet (in-vocab, own context)
    known_events = [
        Event(acct, random.choice(account_devices[acct]), "known") for acct in accounts
    ]

    # cross-account events: a device belonging to a DIFFERENT account (in-vocab, foreign context)
    # This is the key mechanistic test: does account clustering exist, or is it just OOV detection?
    cross_events = []
    for acct in accounts:
        other = random.choice([a for a in accounts if a != acct])
        cross_events.append(
            Event(acct, random.choice(account_devices[other]), "cross_account")
        )

    # attack events: novel OOV devices never seen in the corpus
    n_attacked = max(1, int(n_accounts * takeover_rate))
    attacked = random.sample(accounts, n_attacked)
    attack_events = [Event(acct, make_device_id(), "oov_attack") for acct in attacked]

    return corpus, account_devices, known_events, cross_events, attack_events


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

_base_params = dict(
    vector_size=VEC_DIM,
    window=WINDOW,
    min_count=1,
    sg=1,
    workers=4,
    epochs=EPOCHS,
    seed=SEED,
)


def train_fasttext(corpus: Dict) -> FastText:
    return FastText(sentences=list(corpus.values()), **_base_params)


def train_word2vec(corpus: Dict) -> Word2Vec:
    return Word2Vec(sentences=list(corpus.values()), **_base_params)


def wv_mean_vector(wv) -> np.ndarray:
    """Global mean of all in-vocab vectors — used as Word2Vec OOV fallback."""
    keys = list(wv.key_to_index.keys())
    sample = keys[: min(1000, len(keys))]
    return np.mean([wv[k] for k in sample], axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = norm(a) * norm(b)
    if denom < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def account_centroid(wv, history: List[str]) -> Tuple[np.ndarray, float]:
    """Returns (centroid_vector, centroid_norm). Norm encodes cluster coherence."""
    vecs = [wv[d] for d in history if d in wv]
    if not vecs:
        centroid = np.zeros(wv.vector_size)
    else:
        centroid = np.mean(vecs, axis=0)
    return centroid, float(norm(centroid))


def score_fasttext(model: FastText, corpus: Dict, event: Event) -> Tuple[float, float]:
    """Returns (cosine_distance, centroid_norm). FastText handles OOV via subwords."""
    centroid, mu_norm = account_centroid(model.wv, corpus[event.account_id])
    device_vec = model.wv[event.device_id]
    return cosine_distance(centroid, device_vec), mu_norm


def score_word2vec(
    model: Word2Vec, corpus: Dict, event: Event, oov_vec: np.ndarray
) -> float:
    """Word2Vec: novel devices receive the global mean vector (principled OOV fallback)."""
    centroid, _ = account_centroid(model.wv, corpus[event.account_id])
    device_vec = model.wv[event.device_id] if event.device_id in model.wv else oov_vec
    return cosine_distance(centroid, device_vec)


def score_oov_binary(model: FastText, event: Event) -> float:
    """
    Trivial baseline: 1.0 for OOV tokens, 0.0 for in-vocab tokens.
    Must check key_to_index explicitly — gensim's FastText __contains__ returns True
    for any string with valid character n-grams, making `in model.wv` useless for
    distinguishing OOV from in-vocab tokens.
    """
    return 0.0 if event.device_id in model.wv.key_to_index else 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, n: int = N_BOOTSTRAP, ci: float = 0.95
) -> Tuple[float, float, float]:
    """Returns (auc, lower_ci, upper_ci) via percentile bootstrap."""
    rng = np.random.default_rng(SEED)
    boot_aucs = []
    idx = np.arange(len(labels))
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if labels[sample].sum() == 0 or labels[sample].sum() == len(sample):
            continue
        boot_aucs.append(roc_auc_score(labels[sample], scores[sample]))
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_aucs, 100 * alpha)
    hi = np.percentile(boot_aucs, 100 * (1 - alpha))
    return roc_auc_score(labels, scores), lo, hi


def silhouette_in_64d(model_wv, corpus: Dict, n_accounts: int = 100) -> float:
    """
    Compute sklearn silhouette score over account-labeled device embeddings in 64D.
    Positive score means within-account distance < nearest cross-account distance.
    Only distinct devices are used (de-duplicated per account).
    """
    accounts = list(corpus.keys())[:n_accounts]
    vecs, labels = [], []
    for label_idx, acct in enumerate(accounts):
        for dev in set(corpus[acct]):
            if dev in model_wv:
                vecs.append(model_wv[dev])
                labels.append(label_idx)
    vecs = np.array(vecs)
    labels = np.array(labels)
    # Need at least 2 labels and 2 samples per label on average for silhouette
    unique, counts = np.unique(labels, return_counts=True)
    valid = unique[counts >= 2]
    mask = np.isin(labels, valid)
    if mask.sum() < 10:
        return float("nan")
    return silhouette_score(vecs[mask], labels[mask], metric="cosine")


def stratified_auc(
    scores_known: np.ndarray,
    scores_attack: np.ndarray,
    history_lens_known: np.ndarray,
    n_strata: int = 3,
) -> List[Tuple]:
    """
    AUC for known vs. attack events, stratified by the account's history length.
    Returns [(stratum_label, n_known, n_attack, auc), ...].
    """
    q = np.quantile(history_lens_known, np.linspace(0, 1, n_strata + 1))
    results = []
    for i in range(n_strata):
        lo, hi = q[i], q[i + 1]
        mask_k = (history_lens_known >= lo) & (history_lens_known <= hi)
        s_k = scores_known[mask_k]
        s_a = scores_attack[: len(s_k)]  # match sizes approximately
        if len(s_k) < 5 or len(s_a) < 2:
            continue
        n = min(len(s_k), len(s_a))
        combined = np.concatenate([s_k[:n], s_a[:n]])
        lbls = np.array([0] * n + [1] * n)
        if lbls.sum() == 0 or lbls.sum() == len(lbls):
            continue
        auc = roc_auc_score(lbls, combined)
        label = f"len∈[{int(lo)},{int(hi)}]"
        results.append((label, int(mask_k.sum()), n, auc))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════


def plot_all(results: dict, save_path: str = "experiment2_results.png"):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "ATO Experiment 2 — Debate-Driven Empirical Tests", fontsize=13, y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: ROC curves for all models ───────────────────────────────────
    ax_roc = fig.add_subplot(gs[0, 0])
    for name, (fpr, tpr, auc, lo, hi) in results["roc"].items():
        ax_roc.plot(
            fpr, tpr, lw=1.8, label=f"{name}\n  AUC={auc:.3f} [{lo:.3f},{hi:.3f}]"
        )
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("A. ROC curves (known vs. OOV attack)\nw/ 95% bootstrap CI")
    ax_roc.legend(fontsize=7, loc="lower right")

    # ── Panel B: Three-class score distributions ──────────────────────────────
    ax_dist = fig.add_subplot(gs[0, 1])
    all_scores = np.concatenate(list(results["three_class_scores"].values()))
    bins = np.linspace(0, np.percentile(all_scores, 99) * 1.05, 45)
    colors = {
        "known": "steelblue",
        "cross_account": "goldenrod",
        "oov_attack": "crimson",
    }
    labels_map = {
        "known": "Known (own)",
        "cross_account": "Cross-account (in-vocab)",
        "oov_attack": "Novel OOV (attack)",
    }
    for cls, scores in results["three_class_scores"].items():
        ax_dist.hist(
            scores, bins=bins, alpha=0.55, color=colors[cls], label=labels_map[cls]
        )
    ax_dist.set_xlabel("Cosine distance to centroid")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title(
        "B. Three-class score distributions\n(FastText; key mechanism test)"
    )
    ax_dist.legend(fontsize=7)

    # ── Panel C: Model AUC bar chart ─────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 2])
    model_names = list(results["model_aucs"].keys())
    aucs = [results["model_aucs"][m][0] for m in model_names]
    cis = [
        (
            results["model_aucs"][m][0] - results["model_aucs"][m][1],
            results["model_aucs"][m][2] - results["model_aucs"][m][0],
        )
        for m in model_names
    ]
    y = range(len(model_names))
    ax_bar.barh(
        y,
        aucs,
        xerr=np.array(cis).T,
        capsize=4,
        color=["steelblue", "seagreen", "coral", "mediumpurple"],
        alpha=0.8,
        error_kw=dict(ecolor="black", lw=1.2),
    )
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(model_names, fontsize=8)
    ax_bar.set_xlabel("ROC-AUC")
    ax_bar.set_xlim(0.4, 1.0)
    ax_bar.axvline(0.5, color="black", lw=0.8, ls="--")
    ax_bar.set_title("C. Model comparison\n(known vs. OOV attack, 95% CI)")

    # ── Panel D: Stratified AUC by history length ─────────────────────────────
    ax_strat = fig.add_subplot(gs[1, 0])
    for model_name, strata in results["stratified"].items():
        labels_s = [s[0] for s in strata]
        aucs_s = [s[3] for s in strata]
        ax_strat.plot(range(len(labels_s)), aucs_s, marker="o", label=model_name)
    if results["stratified"]:
        first = list(results["stratified"].values())[0]
        ax_strat.set_xticks(range(len(first)))
        ax_strat.set_xticklabels([s[0] for s in first], fontsize=7, rotation=15)
    ax_strat.axhline(0.5, color="black", lw=0.8, ls="--")
    ax_strat.set_ylabel("ROC-AUC")
    ax_strat.set_title("D. AUC stratified by history length\n(thin → rich accounts)")
    ax_strat.legend(fontsize=7)

    # ── Panel E: Centroid norm distribution & FPR correlation ─────────────────
    ax_norm = fig.add_subplot(gs[1, 1])
    norms_known = results["centroid_norms"]["known"]
    is_fp = results["centroid_norms"]["is_fp"]
    ax_norm.scatter(
        norms_known[~is_fp],
        np.zeros(np.sum(~is_fp)) + 0.0,
        c="steelblue",
        alpha=0.3,
        s=15,
        label="True negative",
    )
    ax_norm.scatter(
        norms_known[is_fp],
        np.zeros(np.sum(is_fp)) + 0.05,
        c="crimson",
        alpha=0.6,
        s=25,
        label="False positive",
    )
    ax_norm.set_xlabel("Centroid norm ||μ||")
    ax_norm.set_yticks([])
    ax_norm.set_title(
        "E. Centroid norm vs. false positives\n(does low ||μ|| predict FP?)"
    )
    ax_norm.legend(fontsize=7)

    # ── Panel F: i.i.d. vs Markov AUC comparison ─────────────────────────────
    ax_markov = fig.add_subplot(gs[1, 2])
    markov_data = results["markov_comparison"]
    models_m = list(markov_data["iid"].keys())
    x = np.arange(len(models_m))
    width = 0.35
    iid_aucs = [markov_data["iid"][m] for m in models_m]
    markov_aucs = [markov_data["markov"][m] for m in models_m]
    ax_markov.bar(
        x - width / 2, iid_aucs, width, label="i.i.d.", color="steelblue", alpha=0.8
    )
    ax_markov.bar(
        x + width / 2, markov_aucs, width, label="Markov", color="coral", alpha=0.8
    )
    ax_markov.set_xticks(x)
    ax_markov.set_xticklabels(models_m, fontsize=8)
    ax_markov.axhline(0.5, color="black", lw=0.8, ls="--")
    ax_markov.set_ylabel("ROC-AUC")
    ax_markov.set_title(
        "F. i.i.d. vs. Markov session model\n(is i.i.d. AUC an upper bound?)"
    )
    ax_markov.legend(fontsize=8)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  saved → {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment(mode: Literal["iid", "markov"]) -> dict:
    label = mode.upper()
    print(f"\n{'─' * 60}")
    print(f"[{label}] generating corpus …")
    corpus, account_devices, known_evts, cross_evts, attack_evts = generate_corpus(
        mode=mode
    )
    print(
        f"[{label}] accounts={N_ACCOUNTS}  "
        f"known={len(known_evts)}  cross={len(cross_evts)}  attack={len(attack_evts)}"
    )

    print(f"[{label}] training FastText …")
    ft = train_fasttext(corpus)
    print(f"[{label}] training Word2Vec …")
    w2v = train_word2vec(corpus)
    w2v_oov = wv_mean_vector(w2v.wv)

    # ── score all event classes ───────────────────────────────────────────────
    scores_known_ft = np.array([score_fasttext(ft, corpus, e)[0] for e in known_evts])
    scores_attack_ft = np.array([score_fasttext(ft, corpus, e)[0] for e in attack_evts])
    scores_cross_ft = np.array([score_fasttext(ft, corpus, e)[0] for e in cross_evts])

    scores_known_w2v = np.array(
        [score_word2vec(w2v, corpus, e, w2v_oov) for e in known_evts]
    )
    scores_attack_w2v = np.array(
        [score_word2vec(w2v, corpus, e, w2v_oov) for e in attack_evts]
    )

    scores_known_oov = np.array([score_oov_binary(ft, e) for e in known_evts])
    scores_attack_oov = np.array([score_oov_binary(ft, e) for e in attack_evts])

    mu_norms = np.array([score_fasttext(ft, corpus, e)[1] for e in known_evts])

    # ── centroid norm correlation with FP at Youden's threshold ───────────────
    combined_scores_ft = np.concatenate([scores_known_ft, scores_attack_ft])
    combined_labels_ft = np.array([0] * len(known_evts) + [1] * len(attack_evts))
    fpr_ft, tpr_ft, thresh_ft = roc_curve(combined_labels_ft, combined_scores_ft)
    j_idx = np.argmax(tpr_ft - fpr_ft)
    thresh = thresh_ft[j_idx]
    is_fp = scores_known_ft >= thresh  # known device flagged as attack

    # ── bootstrap AUC for each model ─────────────────────────────────────────
    def auc_trio(s_known, s_attack):
        n = min(len(s_known), len(s_attack))
        s = np.concatenate([s_known[:n], s_attack[:n]])
        l = np.array([0] * n + [1] * n)
        return bootstrap_auc(l, s)

    auc_ft, lo_ft, hi_ft = auc_trio(scores_known_ft, scores_attack_ft)
    auc_w2v, lo_w2v, hi_w2v = auc_trio(scores_known_w2v, scores_attack_w2v)
    auc_oov, lo_oov, hi_oov = auc_trio(scores_known_oov, scores_attack_oov)

    # cross-account test: how does cross-account in-vocab score compare?
    # Compare cross-account vs OOV attack:
    n_cross_atk = min(len(scores_cross_ft), len(scores_attack_ft))
    auc_cross_vs_oov, lo_c, hi_c = bootstrap_auc(
        np.array([0] * n_cross_atk + [1] * n_cross_atk),
        np.concatenate([scores_cross_ft[:n_cross_atk], scores_attack_ft[:n_cross_atk]]),
    )

    # ── stratified AUC by history length ─────────────────────────────────────
    hist_lens = np.array([len(corpus[e.account_id]) for e in known_evts])
    strat_ft = stratified_auc(scores_known_ft, scores_attack_ft, hist_lens)
    strat_oov = stratified_auc(scores_known_oov, scores_attack_oov, hist_lens)

    # ── silhouette score ──────────────────────────────────────────────────────
    sil_ft = silhouette_in_64d(ft.wv, corpus)
    sil_w2v = silhouette_in_64d(w2v.wv, corpus)

    # ── print report ─────────────────────────────────────────────────────────
    print(f"\n[{label}] ── MODEL COMPARISON (known vs. OOV attack) ──────────")
    print(f"  FastText centroid:    AUC={auc_ft:.4f}  95%CI=[{lo_ft:.3f},{hi_ft:.3f}]")
    print(
        f"  Word2Vec centroid:    AUC={auc_w2v:.4f}  95%CI=[{lo_w2v:.3f},{hi_w2v:.3f}]"
    )
    print(
        f"  OOV binary baseline:  AUC={auc_oov:.4f}  95%CI=[{lo_oov:.3f},{hi_oov:.3f}]"
    )

    print(f"\n[{label}] ── MECHANISM TEST (Point 1) ──────────────────────────")
    print("  Score means (FastText cosine distance):")
    print(f"    Known device (own account):       {scores_known_ft.mean():.4f}")
    print(f"    Cross-account device (in-vocab):  {scores_cross_ft.mean():.4f}")
    print(f"    Novel OOV device (attack):        {scores_attack_ft.mean():.4f}")
    print(
        f"  AUC(cross-account vs. OOV attack):  {auc_cross_vs_oov:.4f}  [{lo_c:.3f},{hi_c:.3f}]"
    )
    print("  → If AUC≈0.5: cross-account ≈ OOV → clustering is real")
    print("  → If AUC≫0.5: OOV devices score much higher → OOV is dominant signal")

    print(f"\n[{label}] ── CLUSTER VALIDATION (Point 9) ──────────────────────")
    print(f"  Silhouette score (64D, FastText):  {sil_ft:.4f}")
    print(f"  Silhouette score (64D, Word2Vec):  {sil_w2v:.4f}")
    print("  → Positive = within-account < cross-account distance")

    print(f"\n[{label}] ── STRATIFIED AUC (Point 3) ──────────────────────────")
    print("  FastText:")
    for stratum, n_k, n_a, auc in strat_ft:
        print(f"    {stratum:20s}  n_known={n_k:3d}  n_attack={n_a:2d}  AUC={auc:.4f}")
    print("  OOV binary:")
    for stratum, n_k, n_a, auc in strat_oov:
        print(f"    {stratum:20s}  n_known={n_k:3d}  n_attack={n_a:2d}  AUC={auc:.4f}")

    print(f"\n[{label}] ── CENTROID NORM (Point 4) ───────────────────────────")
    fp_norms = mu_norms[is_fp]
    tn_norms = mu_norms[~is_fp]
    print(f"  ||μ|| mean (true negatives): {tn_norms.mean():.4f}")
    print(f"  ||μ|| mean (false positives): {fp_norms.mean():.4f}")
    if len(fp_norms) > 0:
        from scipy.stats import mannwhitneyu

        stat, p = mannwhitneyu(tn_norms, fp_norms, alternative="two-sided")
        print(f"  Mann-Whitney p-value: {p:.4f}  (is ||μ|| predictive of FP?)")

    return {
        "roc": {
            "FastText": (fpr_ft, tpr_ft, auc_ft, lo_ft, hi_ft),
        },
        "three_class_scores": {
            "known": scores_known_ft,
            "cross_account": scores_cross_ft,
            "oov_attack": scores_attack_ft,
        },
        "model_aucs": {
            "FastText": (auc_ft, lo_ft, hi_ft),
            "Word2Vec": (auc_w2v, lo_w2v, hi_w2v),
            "OOV binary": (auc_oov, lo_oov, hi_oov),
            "Cross-acct\nvs OOV": (auc_cross_vs_oov, lo_c, hi_c),
        },
        "stratified": {
            "FastText": strat_ft,
            "OOV binary": strat_oov,
        },
        "centroid_norms": {
            "known": mu_norms,
            "is_fp": is_fp,
        },
        "silhouette": {"FastText": sil_ft, "Word2Vec": sil_w2v},
        "markov_comparison_aucs": {
            "FastText": auc_ft,
            "OOV binary": auc_oov,
            "Word2Vec": auc_w2v,
        },
    }


if __name__ == "__main__":
    print("=" * 60)
    print("ATO Experiment 2 — Debate-Driven Empirical Tests")
    print("=" * 60)

    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        # graceful fallback if scipy not available
        import types

        def _dummy_mwu(a, b, alternative="two-sided"):
            return types.SimpleNamespace(), float("nan")


        # monkey-patch locally so run_experiment works
        import sys

        scipy_mock = types.ModuleType("scipy")
        scipy_stats = types.ModuleType("scipy.stats")
        scipy_stats.mannwhitneyu = _dummy_mwu
        scipy_mock.stats = scipy_stats
        sys.modules["scipy"] = scipy_mock
        sys.modules["scipy.stats"] = scipy_stats

    # run both corpus modes
    res_iid = run_experiment("iid")
    res_markov = run_experiment("markov")

    # assemble markov comparison for panel F
    res_iid["markov_comparison"] = {
        "iid": res_iid["markov_comparison_aucs"],
        "markov": res_markov["markov_comparison_aucs"],
    }

    print("\n[plots] generating experiment2_results.png …")
    plot_all(res_iid, save_path=str(FIGURES_DIR / "experiment2_results.png"))

    # summary
    print("\n" + "=" * 60)
    print("SUMMARY — answers to debate questions")
    print("=" * 60)

    ft_auc = res_iid["model_aucs"]["FastText"][0]
    oov_auc = res_iid["model_aucs"]["OOV binary"][0]
    w2v_auc = res_iid["model_aucs"]["Word2Vec"][0]
    cross_auc = res_iid["model_aucs"]["Cross-acct\nvs OOV"][0]
    sil = res_iid["silhouette"]["FastText"]

    print("\nPoint 1 — OOV vs. account clustering (mechanism test):")
    print(f"  AUC(cross-account in-vocab vs. OOV attack) = {cross_auc:.3f}")
    if cross_auc < 0.60:
        verdict = "cross-account in-vocab ≈ OOV attack → account clusters ARE real"
    elif cross_auc > 0.75:
        verdict = "OOV attack >> cross-account → OOV detection is dominant signal"
    else:
        verdict = "intermediate — both mechanisms contribute"
    print(f"  Verdict: {verdict}")

    print("\nPoint 2 — Subword n-gram contribution:")
    delta = ft_auc - w2v_auc
    print(f"  FastText AUC={ft_auc:.3f}  Word2Vec AUC={w2v_auc:.3f}  Δ={delta:+.3f}")
    if abs(delta) < 0.02:
        print("  Verdict: subword n-grams have negligible effect (|Δ| < 0.02)")
    elif delta > 0:
        print("  Verdict: FastText outperforms Word2Vec — subword n-grams add signal")
    else:
        print("  Verdict: Word2Vec outperforms FastText — subword n-grams add noise")

    print("\nPoint 5 — Does FastText beat the OOV binary baseline?")
    delta_oov = ft_auc - oov_auc
    print(
        f"  FastText AUC={ft_auc:.3f}  OOV binary AUC={oov_auc:.3f}  Δ={delta_oov:+.3f}"
    )
    if delta_oov > 0.03:
        print(
            "  Verdict: FastText meaningfully exceeds binary baseline — embeddings add value"
        )
    elif delta_oov > 0:
        print(
            "  Verdict: FastText marginally exceeds binary baseline — weak embedding contribution"
        )
    else:
        print(
            "  Verdict: OOV binary ≥ FastText — embeddings add no value over a lookup"
        )

    print("\nPoint 8 — i.i.d. vs Markov (is i.i.d. AUC an upper bound?):")
    for model in ["FastText", "OOV binary", "Word2Vec"]:
        iid_a = res_iid["markov_comparison_aucs"][model]
        markov_a = res_markov["markov_comparison_aucs"][model]
        print(
            f"  {model:12s}  i.i.d.={iid_a:.3f}  Markov={markov_a:.3f}  Δ={markov_a - iid_a:+.3f}"
        )

    print("\nPoint 9 — Account clusters in 64D:")
    print(f"  Silhouette score (FastText) = {sil:.4f}")
    if sil > 0.1:
        print(
            "  Verdict: positive silhouette — account clusters exist in embedding space"
        )
    elif sil > 0:
        print("  Verdict: weakly positive — clusters exist but are diffuse")
    else:
        print("  Verdict: non-positive silhouette — no clear account-level clusters")

    print("\nDone.")
