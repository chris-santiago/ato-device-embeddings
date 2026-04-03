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
Generate conclusion-focused plots from the Experiment 2 results.
Reproduces the full experiment with a fixed seed so all numbers match.
Outputs individual, presentation-quality figures for CONCLUSIONS.md.
"""

import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.linalg import norm
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score
from gensim.models import FastText, Word2Vec
import umap

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

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

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

# ── data & model helpers (identical to experiment2.py) ───────────────────────


def make_device_id(n=16):
    return "dev_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


@dataclass
class Event:
    account_id: str
    device_id: str
    label: Literal["known", "cross_account", "oov_attack"]


def _zipf(n):
    w = np.array([1 / (k + 1) for k in range(n)], dtype=float)
    return (w / w.sum()).tolist()


def gen_iid(devices, n):
    return random.choices(devices, weights=_zipf(len(devices)), k=n)


def gen_markov(devices, n, stay=0.70):
    z = _zipf(len(devices))
    cur = random.choices(devices, weights=z, k=1)[0]
    h = [cur]
    for _ in range(n - 1):
        if random.random() < stay:
            h.append(cur)
        else:
            cur = random.choices(devices, weights=z, k=1)[0]
            h.append(cur)
    return h


def generate_corpus(mode="iid"):
    corpus, acct_devs = {}, {}
    for i in range(N_ACCOUNTS):
        acct = f"acct_{i:04d}"
        n_dev = random.randint(*DEV_RANGE)
        devs = [make_device_id() for _ in range(n_dev)]
        acct_devs[acct] = devs
        n_ev = random.randint(*HIST_RANGE)
        corpus[acct] = gen_iid(devs, n_ev) if mode == "iid" else gen_markov(devs, n_ev)
    accounts = list(acct_devs)
    known = [Event(a, random.choice(acct_devs[a]), "known") for a in accounts]
    cross = [
        Event(
            a,
            random.choice(acct_devs[random.choice([x for x in accounts if x != a])]),
            "cross_account",
        )
        for a in accounts
    ]
    n_atk = max(1, int(N_ACCOUNTS * TAKEOVER_RATE))
    attack = [
        Event(a, make_device_id(), "oov_attack") for a in random.sample(accounts, n_atk)
    ]
    return corpus, acct_devs, known, cross, attack


_base = dict(
    vector_size=VEC_DIM,
    window=WINDOW,
    min_count=1,
    sg=1,
    workers=4,
    epochs=EPOCHS,
    seed=SEED,
)


def train_ft(corpus):
    return FastText(sentences=list(corpus.values()), **_base)


def train_w2v(corpus):
    return Word2Vec(sentences=list(corpus.values()), **_base)


def wv_mean(wv):
    ks = list(wv.key_to_index)[:1000]
    return np.mean([wv[k] for k in ks], axis=0)


def cos_dist(a, b):
    d = norm(a) * norm(b)
    return 1.0 if d < 1e-12 else float(1 - np.dot(a, b) / d)


def centroid(wv, history):
    vs = [wv[d] for d in history if d in wv.key_to_index]
    return np.mean(vs, axis=0) if vs else np.zeros(wv.vector_size)


def score_ft(model, corpus, e):
    c = centroid(model.wv, corpus[e.account_id])
    return cos_dist(c, model.wv[e.device_id])


def score_w2v(model, corpus, e, oov_vec):
    c = centroid(model.wv, corpus[e.account_id])
    v = model.wv[e.device_id] if e.device_id in model.wv.key_to_index else oov_vec
    return cos_dist(c, v)


def score_oov(model, e):
    return 0.0 if e.device_id in model.wv.key_to_index else 1.0


def bstrap_auc(labels, scores, n=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
    aucs = []
    idx = np.arange(len(labels))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if 0 < labels[s].sum() < len(s):
            aucs.append(roc_auc_score(labels[s], scores[s]))
    return (
        roc_auc_score(labels, scores),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
    )


def auc_trio(sk, sa):
    n = min(len(sk), len(sa))
    return bstrap_auc(np.array([0] * n + [1] * n), np.concatenate([sk[:n], sa[:n]]))


def silhouette_64d(wv, corpus, n=100):
    accts = list(corpus)[:n]
    vecs, labs = [], []
    for i, a in enumerate(accts):
        for d in set(corpus[a]):
            if d in wv.key_to_index:
                vecs.append(wv[d])
                labs.append(i)
    vecs, labs = np.array(vecs), np.array(labs)
    u, c = np.unique(labs, return_counts=True)
    mask = np.isin(labs, u[c >= 2])
    return (
        silhouette_score(vecs[mask], labs[mask], metric="cosine")
        if mask.sum() > 10
        else float("nan")
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN BOTH CORPUS MODES
# ═══════════════════════════════════════════════════════════════════════════════

results = {}
for mode in ("iid", "markov"):
    print(f"[{mode}] building corpus + training …")
    corpus, acct_devs, known, cross, attack = generate_corpus(mode)
    ft = train_ft(corpus)
    w2v = train_w2v(corpus)
    oov = wv_mean(w2v.wv)

    sk_ft = np.array([score_ft(ft, corpus, e) for e in known])
    sa_ft = np.array([score_ft(ft, corpus, e) for e in attack])
    sc_ft = np.array([score_ft(ft, corpus, e) for e in cross])
    sk_w2v = np.array([score_w2v(w2v, corpus, e, oov) for e in known])
    sa_w2v = np.array([score_w2v(w2v, corpus, e, oov) for e in attack])
    sk_oov = np.array([score_oov(ft, e) for e in known])
    sa_oov = np.array([score_oov(ft, e) for e in attack])

    fpr_ft, tpr_ft, _ = roc_curve(
        np.array([0] * len(known) + [1] * len(attack)), np.concatenate([sk_ft, sa_ft])
    )

    results[mode] = dict(
        sk_ft=sk_ft,
        sa_ft=sa_ft,
        sc_ft=sc_ft,
        sk_w2v=sk_w2v,
        sa_w2v=sa_w2v,
        sk_oov=sk_oov,
        sa_oov=sa_oov,
        fpr_ft=fpr_ft,
        tpr_ft=tpr_ft,
        auc_ft=auc_trio(sk_ft, sa_ft),
        auc_w2v=auc_trio(sk_w2v, sa_w2v),
        auc_oov=auc_trio(sk_oov, sa_oov),
        auc_cross=auc_trio(sc_ft, sa_ft),
        sil_ft=silhouette_64d(ft.wv, corpus),
        sil_w2v=silhouette_64d(w2v.wv, corpus),
        corpus=corpus,
        ft=ft,
        w2v=w2v,
        known=known,
        attack=attack,
    )

iid = results["iid"]
mkv = results["markov"]

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Three-class score distributions (mechanism test, Point 1)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, res, title in [
    (axes[0], iid, "i.i.d. corpus"),
    (axes[1], mkv, "Markov session corpus"),
]:
    all_s = np.concatenate([res["sk_ft"], res["sc_ft"], res["sa_ft"]])
    bins = np.linspace(0, np.percentile(all_s, 99) * 1.08, 50)
    ax.hist(
        res["sk_ft"],
        bins=bins,
        alpha=0.65,
        color="#4878d0",
        label="Known (own account)",
    )
    ax.hist(
        res["sa_ft"], bins=bins, alpha=0.65, color="#d65f5f", label="Novel OOV (attack)"
    )
    ax.hist(
        res["sc_ft"],
        bins=bins,
        alpha=0.65,
        color="#ee854a",
        label="Cross-account (in-vocab)",
    )

    for arr, col in [
        (res["sk_ft"], "#4878d0"),
        (res["sa_ft"], "#d65f5f"),
        (res["sc_ft"], "#ee854a"),
    ]:
        ax.axvline(arr.mean(), color=col, lw=1.5, ls="--")

    ax.set_xlabel("Cosine distance to account centroid")
    ax.set_ylabel("Count")
    ax.set_title(f"Three-class score distributions\n({title})", fontsize=10)
    auc_c = res["auc_cross"][0]
    ax.text(
        0.97,
        0.96,
        f"AUC(cross vs OOV) = {auc_c:.3f}\n"
        f"← below 0.5 means cross-account\n   scores HIGHER than OOV attack",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey", alpha=0.9),
    )

axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle(
    "Fig 1 — Mechanism Test: Account Clustering vs OOV Detection (Point 1)",
    fontsize=11,
    y=1.01,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_mechanism.png", dpi=150, bbox_inches="tight")
print("[saved] fig1_mechanism.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Model AUC comparison with bootstrap CIs, i.i.d. and Markov (Points 2, 5, 8)
# ═══════════════════════════════════════════════════════════════════════════════

model_labels = ["FastText\ncentroid", "Word2Vec\ncentroid", "OOV binary\nbaseline"]
colors_bar = ["#4878d0", "#6acc65", "#d65f5f"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, res, title in [
    (axes[0], iid, "i.i.d. corpus"),
    (axes[1], mkv, "Markov session corpus"),
]:
    aucs = [res["auc_ft"], res["auc_w2v"], res["auc_oov"]]
    vals = [a[0] for a in aucs]
    errs_lo = [a[0] - a[1] for a in aucs]
    errs_hi = [a[2] - a[0] for a in aucs]

    bars = ax.barh(
        range(3),
        vals,
        xerr=[errs_lo, errs_hi],
        capsize=5,
        color=colors_bar,
        alpha=0.82,
        error_kw=dict(ecolor="black", lw=1.3, capthick=1.3),
    )
    ax.set_yticks(range(3))
    ax.set_yticklabels(model_labels, fontsize=9)
    ax.set_xlim(0.45, 1.03)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xlabel("ROC-AUC (95% bootstrap CI)")
    ax.set_title(f"Model comparison\n({title})", fontsize=10)

    for i, (auc, lo, hi) in enumerate(aucs):
        ax.text(
            auc + max(errs_hi[i], 0.005) + 0.005,
            i,
            f"{auc:.3f} [{lo:.3f}–{hi:.3f}]",
            va="center",
            fontsize=7.5,
        )

fig.suptitle(
    "Fig 2 — Model Comparison: FastText vs Word2Vec vs OOV Binary (Points 2, 5, 8)",
    fontsize=11,
    y=1.01,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig2_model_comparison.png", dpi=150, bbox_inches="tight")
print("[saved] fig2_model_comparison.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Silhouette scores + UMAP side-by-side (Point 9)
# ═══════════════════════════════════════════════════════════════════════════════

print("[umap] fitting FastText UMAP …")
corpus_iid = iid["corpus"]
ft_iid = iid["ft"]
w2v_iid = iid["w2v"]
N_SHOW = 20
show_accounts = list(corpus_iid.keys())[:N_SHOW]
palette = plt.cm.tab20.colors


def collect_vecs(wv, corpus, accounts, use_key_to_index=True):
    vecs, cols, markers = [], [], []
    for idx, acct in enumerate(accounts):
        col = palette[idx % len(palette)]
        for dev in set(corpus[acct]):
            check = dev in wv.key_to_index if use_key_to_index else dev in wv
            if not check:
                continue
            vecs.append(wv[dev])
            cols.append(col)
            markers.append("o")
    return np.array(vecs), cols, markers


vecs_ft, cols_ft, _ = collect_vecs(ft_iid.wv, corpus_iid, show_accounts)
vecs_w2v, cols_w2v, _ = collect_vecs(
    w2v_iid.wv, corpus_iid, show_accounts, use_key_to_index=False
)

reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=10, min_dist=0.15)
emb_ft = reducer.fit_transform(vecs_ft)
emb_w2v = reducer.fit_transform(vecs_w2v)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, emb, cols, sil, model_name in [
    (axes[0], emb_ft, cols_ft, iid["sil_ft"], "FastText"),
    (axes[1], emb_w2v, cols_w2v, iid["sil_w2v"], "Word2Vec"),
]:
    for xy, col in zip(emb, cols):
        ax.scatter(*xy, c=[col], s=22, alpha=0.75, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    sil_color = "#d65f5f" if sil < 0 else "#6acc65"
    ax.set_title(
        f"{model_name}  (first {N_SHOW} accounts, i.i.d.)\nSilhouette = ", fontsize=10
    )
    # append colored silhouette value
    ax.set_title(
        f"{model_name} — UMAP of device embeddings\n"
        f"64D Silhouette = {sil:+.4f}  "
        f"({'clusters exist' if sil > 0 else 'no cluster structure'})",
        fontsize=9.5,
    )
    # colour-code the silhouette number
    ax.text(
        0.02,
        0.98,
        f"Silhouette = {sil:+.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        fontweight="bold",
        color=sil_color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=sil_color, lw=1.5),
    )

# shared legend: each colour = one account
handles = [
    mpatches.Patch(color=palette[i], label=f"Account {i}") for i in range(N_SHOW)
]
fig.legend(
    handles=handles,
    ncol=5,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    fontsize=7,
    title="Account (color)",
)
fig.suptitle(
    "Fig 3 — Cluster Structure in Embedding Space (Point 9): "
    "FastText destroys clusters; Word2Vec preserves them",
    fontsize=11,
    y=1.02,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig3_cluster_structure.png", dpi=150, bbox_inches="tight")
print("[saved] fig3_cluster_structure.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — i.i.d. vs Markov degradation + ROC curves (Point 8)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: grouped bar chart of AUC degradation
ax = axes[0]
models = ["FastText", "Word2Vec", "OOV binary"]
iid_vals = [iid["auc_ft"][0], iid["auc_w2v"][0], iid["auc_oov"][0]]
markov_vals = [mkv["auc_ft"][0], mkv["auc_w2v"][0], mkv["auc_oov"][0]]
x = np.arange(3)
w = 0.35
b1 = ax.bar(x - w / 2, iid_vals, w, label="i.i.d.", color="#4878d0", alpha=0.85)
b2 = ax.bar(x + w / 2, markov_vals, w, label="Markov", color="#ee854a", alpha=0.85)

for bars in (b1, b2):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

# draw delta arrows
for i, (iv, mv) in enumerate(zip(iid_vals, markov_vals)):
    delta = mv - iv
    ax.annotate(
        "",
        xy=(i + w / 2, mv),
        xytext=(i + w / 2, iv),
        arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
    )
    ax.text(
        i + w / 2 + 0.18,
        (iv + mv) / 2,
        f"Δ={delta:+.3f}",
        ha="left",
        va="center",
        fontsize=8,
        color="#555",
    )

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylim(0.45, 1.06)
ax.axhline(0.5, color="black", ls="--", lw=0.8)
ax.set_ylabel("ROC-AUC")
ax.legend(fontsize=9)
ax.set_title("A. AUC degradation under session autocorrelation", fontsize=10)

# Panel B: ROC curves for all three models, both corpus modes
ax = axes[1]
line_styles = {"iid": "-", "markov": "--"}
model_colors = {"ft": "#4878d0", "w2v": "#6acc65", "oov": "#d65f5f"}

for mode_label, res in [("i.i.d.", iid), ("Markov", mkv)]:
    ls = "-" if "i.i.d" in mode_label else "--"
    # FastText ROC
    fpr_ft, tpr_ft, _ = roc_curve(
        np.array([0] * len(res["sk_ft"]) + [1] * len(res["sa_ft"])),
        np.concatenate([res["sk_ft"], res["sa_ft"]]),
    )
    ax.plot(
        fpr_ft,
        tpr_ft,
        ls=ls,
        color=model_colors["ft"],
        lw=1.5,
        label=f"FastText {mode_label} ({res['auc_ft'][0]:.3f})",
    )
    # Word2Vec ROC
    fpr_w2v, tpr_w2v, _ = roc_curve(
        np.array([0] * len(res["sk_w2v"]) + [1] * len(res["sa_w2v"])),
        np.concatenate([res["sk_w2v"], res["sa_w2v"]]),
    )
    ax.plot(
        fpr_w2v,
        tpr_w2v,
        ls=ls,
        color=model_colors["w2v"],
        lw=1.5,
        label=f"Word2Vec {mode_label} ({res['auc_w2v'][0]:.3f})",
    )
    # OOV binary ROC
    fpr_oov, tpr_oov, _ = roc_curve(
        np.array([0] * len(res["sk_oov"]) + [1] * len(res["sa_oov"])),
        np.concatenate([res["sk_oov"], res["sa_oov"]]),
    )
    ax.plot(
        fpr_oov,
        tpr_oov,
        ls=ls,
        color=model_colors["oov"],
        lw=1.5,
        label=f"OOV binary {mode_label} ({res['auc_oov'][0]:.3f})",
    )

ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.legend(fontsize=6.5, loc="lower right")
ax.set_title("B. ROC curves — all models × corpus modes", fontsize=10)

fig.suptitle(
    "Fig 4 — i.i.d. vs Markov: Session Autocorrelation Degrades FastText Most (Point 8)",
    fontsize=11,
    y=1.01,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig4_markov_degradation.png", dpi=150, bbox_inches="tight")
print("[saved] fig4_markov_degradation.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Stratified AUC by history length (Point 3)
# ═══════════════════════════════════════════════════════════════════════════════


def stratified_auc(sk, sa, hist_lens, n_strata=4):
    q = np.quantile(hist_lens, np.linspace(0, 1, n_strata + 1))
    out = []
    for i in range(n_strata):
        lo, hi = q[i], q[i + 1]
        mask = (hist_lens >= lo) & (hist_lens <= hi)
        s_k = sk[mask]
        s_a = sa[: len(s_k)]
        if len(s_k) < 5 or len(s_a) < 2:
            continue
        n = min(len(s_k), len(s_a))
        s = np.concatenate([s_k[:n], s_a[:n]])
        l = np.array([0] * n + [1] * n)
        if l.sum() in (0, len(l)):
            continue
        out.append((f"[{int(lo)}–{int(hi)}]", roc_auc_score(l, s)))
    return out


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, res, title in [(axes[0], iid, "i.i.d."), (axes[1], mkv, "Markov")]:
    hist_lens = np.array([len(res["corpus"][e.account_id]) for e in iid["known"]])
    strat_ft = stratified_auc(res["sk_ft"], res["sa_ft"], hist_lens)
    strat_w2v = stratified_auc(res["sk_w2v"], res["sa_w2v"], hist_lens)
    strat_oov = stratified_auc(res["sk_oov"], res["sa_oov"], hist_lens)

    labels = [s[0] for s in strat_ft]
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(
        x - w,
        [s[1] for s in strat_ft],
        w,
        label="FastText",
        color="#4878d0",
        alpha=0.85,
    )
    ax.bar(
        x, [s[1] for s in strat_w2v], w, label="Word2Vec", color="#6acc65", alpha=0.85
    )
    ax.bar(
        x + w,
        [s[1] for s in strat_oov],
        w,
        label="OOV binary",
        color="#d65f5f",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.5, color="black", ls="--", lw=0.8)
    ax.set_ylim(0.45, 1.06)
    ax.set_xlabel("Account history length (login events)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Stratified AUC by history length\n({title} corpus)", fontsize=10)
    if ax == axes[0]:
        ax.legend(fontsize=8)

fig.suptitle(
    "Fig 5 — Does Signal Survive Thin Histories? Stratified AUC (Point 3)",
    fontsize=11,
    y=1.01,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig5_stratified_auc.png", dpi=150, bbox_inches="tight")
print("[saved] fig5_stratified_auc.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Centroid norm vs. false positives (Point 4)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, res, title in [(axes[0], iid, "i.i.d."), (axes[1], mkv, "Markov")]:
    mu_norms = np.array(
        [
            norm(centroid(res["ft"].wv, res["corpus"][e.account_id]))
            for e in res["known"]
        ]
    )
    sk = res["sk_ft"]
    sa = res["sa_ft"]
    _, _, thresh_arr = roc_curve(
        np.array([0] * len(sk) + [1] * len(sa)), np.concatenate([sk, sa])
    )
    j_thresh = thresh_arr[
        np.argmax(
            np.array(
                [
                    t
                    for t in roc_curve(
                        np.array([0] * len(sk) + [1] * len(sa)),
                        np.concatenate([sk, sa]),
                    )[1]
                ]
            )
            - np.array(
                [
                    f
                    for f in roc_curve(
                        np.array([0] * len(sk) + [1] * len(sa)),
                        np.concatenate([sk, sa]),
                    )[0]
                ]
            )
        )
    ]

    is_fp = sk >= j_thresh
    tn_norms = mu_norms[~is_fp]
    fp_norms = mu_norms[is_fp]

    bins = np.linspace(mu_norms.min(), mu_norms.max(), 30)
    ax.hist(
        tn_norms,
        bins=bins,
        alpha=0.6,
        color="#4878d0",
        label=f"True negative (n={len(tn_norms)})",
    )
    ax.hist(
        fp_norms,
        bins=bins,
        alpha=0.6,
        color="#d65f5f",
        label=f"False positive (n={len(fp_norms)})",
    )
    ax.axvline(tn_norms.mean(), color="#4878d0", ls="--", lw=1.5)
    ax.axvline(
        fp_norms.mean() if len(fp_norms) else 0, color="#d65f5f", ls="--", lw=1.5
    )
    ax.set_xlabel("Centroid norm ||μ||")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Centroid norm distribution\n({title} corpus, Youden threshold)", fontsize=10
    )
    ax.legend(fontsize=8)

    # report means
    if len(fp_norms):
        ax.text(
            0.97,
            0.97,
            f"TN mean ||μ|| = {tn_norms.mean():.3f}\n"
            f"FP mean ||μ|| = {fp_norms.mean():.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"),
        )

fig.suptitle(
    "Fig 6 — Centroid Norm vs False Positives: ||μ|| is not predictive (Point 4)",
    fontsize=11,
    y=1.01,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig6_centroid_norm.png", dpi=150, bbox_inches="tight")
print("[saved] fig6_centroid_norm.png")
plt.close(fig)

print("\nAll figures saved.")
