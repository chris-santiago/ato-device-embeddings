# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
#   "matplotlib>=3.7",
#   "polars>=0.20",
# ]
# ///
#
# Per-Feature Normalized (PFN) Scoring Experiment v2 — H3
# ========================================================
# Revision over pfn_experiment.py addressing three critic findings:
#
#   F1 (discrete score support): The proportional split yields median n_calib=2 per user,
#     so rank-normalized scores can only take {0.0, 0.5, 1.0}. The originally proposed fix
#     (MIN_TRAIN_EVENTS=61) was diagnosed to eliminate ALL ATO test users (the 9 ATO users
#     in the test window all have 5–9 training events; no ATO user has >= 30 training events).
#     Fix applied: seed-controlled tie-breaking perturbation (+uniform(0, 1e-9)) before
#     thresholding. This breaks threshold ties without changing ranking, making top-1%
#     cutoffs exact at 1% of the test set while preserving the 9 ATO test users.
#
#   F2 (centroid data asymmetry): mp-raw used ALL training events for its centroid while
#     mp-rank-norm and PFN used only the centroid_evts subset. Added mp-raw-split baseline
#     that uses the identical centroid_evts split as the other rank-norm variants.
#
#   F6 (non-stratified bootstrap): Replaced unstratified bootstrap with stratified
#     bootstrap for AUC/AP metrics (resample positives and negatives separately at their
#     original counts before combining). Top-1% bootstrap also stratified.
#
# Findings NOT fixed (hard dataset constraints, documented below):
#   F3 (stability weight instability): Consequence of n_calib=2 (same as F1). Calib sizes
#     cannot be increased without losing ATO test users. pfn-stab results should be
#     interpreted cautiously.
#   F4 (9 ATO test events): Hard dataset constraint. Cannot be fixed within this RBA
#     dataset and temporal split. ROC-AUC is more reliable than top-1% threshold metrics.
#   F5 (cross-feature co-occurrence): Intentional by design — embeddings encode device
#     fingerprint correlations, which is the discriminative signal. Acceptable as-is.
#
# Usage:
#   uv run h3_pfn/experiments/pfn_experiment2.py
#   uv run h3_pfn/experiments/pfn_experiment2.py --smoke
#   uv run h3_pfn/experiments/pfn_experiment2.py --full

import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import FastText
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import polars as pl
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "rba" / "rba.parquet"
FIGURES_DIR  = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP       = 1000
MIN_TRAIN_EVENTS  = 5     # minimum events to build centroid + calibration
                          # NOTE: raising this would eliminate ATO test users;
                          # RBA ATO users have 5-9 training events (median=6).
N_BENIGN_USERS    = 50_000
SMOKE_N_USERS     = 5_000

# Feature schema — must match rba_rerun.py exactly
FEATURE_ORDER = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]
DEVICE_KEY_FEATURES = ["os", "browser", "device_type", "country", "region", "asn_bucket"]
N_FEATURES = len(FEATURE_ORDER)

# FastText config — identical to H2 robust config
ROBUST_KWARGS = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    seed=SEED, workers=1,
)

# ---------------------------------------------------------------------------
# Calibration split: proportional, last max(2, n//3) events as calibration.
# Called ONCE per user; all scorers share the same split to avoid confounds.
# ---------------------------------------------------------------------------
def split_calib(events: list) -> tuple[list, list]:
    """Return (centroid_events, calib_events). Min 2 calib events."""
    n = len(events)
    n_calib = max(2, n // 3)
    n_centroid = n - n_calib
    if n_centroid < 1:
        n_centroid = 1
        n_calib = n - 1
    return events[:n_centroid], events[n_centroid:]

# ---------------------------------------------------------------------------
# Tie-breaking perturbation (F1 fix): adds a tiny seed-controlled random
# offset before percentile thresholding. Breaks discrete ties without
# changing global ranking meaningfully.
# ---------------------------------------------------------------------------
def add_tie_break(scores: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Return scores + uniform(0, 1e-9) perturbation — breaks ties for exact top-1%."""
    rng = np.random.default_rng(seed)
    return scores + rng.uniform(0, 1e-9, size=len(scores))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_token(f, v):
    return f"{f}_{v}"

def device_key(d):
    return tuple(str(d.get(f) or "missing") for f in DEVICE_KEY_FEATURES)

def device_to_tokens(d):
    return [make_token(f, d.get(f) or "missing") for f in FEATURE_ORDER]

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))

def embed_mp(event: dict, model) -> np.ndarray:
    """Mean-pool: average of all feature token vectors."""
    vecs = [model.wv[make_token(f, event.get(f) or "missing")] for f in FEATURE_ORDER]
    return np.mean(vecs, axis=0)

def embed_feature(event: dict, feature: str, model) -> np.ndarray:
    """Single-feature vector for one feature."""
    return model.wv[make_token(feature, event.get(feature) or "missing")]

def compute_centroid(events, embed_fn, model) -> np.ndarray:
    return np.mean([embed_fn(e, model) for e in events], axis=0)

def compute_per_feature_centroids(events, model) -> dict:
    """Per-feature centroids: mean embedding for each feature over training events."""
    centroids = {}
    for f in FEATURE_ORDER:
        vecs = [embed_feature(e, f, model) for e in events]
        centroids[f] = np.mean(vecs, axis=0)
    return centroids

def rank_normalize(raw_score: float, calib_dist: np.ndarray) -> float:
    """Empirical CDF rank: fraction of calibration events with distance < raw_score."""
    return float(np.mean(calib_dist < raw_score))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_users(parquet_path: Path, smoke: bool, full: bool,
               seed: int = SEED, split_pct: float = 0.50) -> dict:
    print(f"  Loading parquet: {parquet_path}")
    raw = pl.read_parquet(parquet_path)
    if raw["login_timestamp"].dtype not in (pl.Int64, pl.Float64, pl.Int32):
        raise RuntimeError(
            f"login_timestamp dtype is {raw['login_timestamp'].dtype} — expected numeric. "
            "Delete data/rba/rba.parquet and re-run data_prep.py to rebuild."
        )
    df = (
        raw
        .with_columns(pl.col("login_timestamp").cast(pl.Int64, strict=False))
        .drop_nulls(subset=["user_id", "login_timestamp"])
        .sort("login_timestamp")
    )

    rng = np.random.default_rng(seed)
    if smoke:
        all_uids = df["user_id"].unique().to_list()
        keep_smoke = set(
            rng.choice(all_uids, min(SMOKE_N_USERS, len(all_uids)), replace=False).tolist()
        )
        df = df.filter(pl.col("user_id").is_in(keep_smoke))
        print(f"  Smoke pre-filter: {len(keep_smoke):,} users")

    cutoff = df["login_timestamp"].quantile(split_pct, interpolation="nearest")
    train_df = df.filter(pl.col("login_timestamp") <= cutoff)
    test_df  = df.filter(pl.col("login_timestamp") >  cutoff)
    print(f"  Split at timestamp {cutoff}: "
          f"{len(train_df):,} train / {len(test_df):,} test rows")

    ato_user_ids = set(test_df.filter(pl.col("is_ato"))["user_id"].to_list())
    print(f"  ATO users with test-window events: {len(ato_user_ids):,}")

    if not smoke and not full:
        all_user_ids = set(train_df["user_id"].unique().to_list())
        benign_pool = sorted(all_user_ids - ato_user_ids)
        sampled_benign = set(
            rng.choice(benign_pool, min(N_BENIGN_USERS, len(benign_pool)), replace=False)
        )
        keep = ato_user_ids | sampled_benign
        train_df = train_df.filter(pl.col("user_id").is_in(keep))
        test_df  = test_df.filter(pl.col("user_id").is_in(keep))
        print(f"  Subsampled: {len(ato_user_ids):,} ATO + {len(sampled_benign):,} benign users")

    feat_cols = [c for c in FEATURE_ORDER if c in df.columns]

    train_rows = (
        train_df
        .filter(~pl.col("is_ato"))
        .with_columns([pl.col(c).cast(pl.Utf8).fill_null("missing") for c in feat_cols])
        .select(["user_id"] + feat_cols)
        .to_dicts()
    )
    train_by_user: dict[str, list] = defaultdict(list)
    for row in train_rows:
        uid = row.pop("user_id")
        train_by_user[uid].append(row)

    users: dict = {}
    dropped = 0
    for uid, events in train_by_user.items():
        if len(events) < MIN_TRAIN_EVENTS:
            dropped += 1
            continue
        centroid_evts, calib_evts = split_calib(events)
        users[uid] = {
            "train": events,
            "centroid_evts": centroid_evts,  # shared across all scorers (F2 fix)
            "calib_evts": calib_evts,        # shared across all scorers (F2 fix)
            "test_pos": [],
            "test_neg": [],
            "known_devices": {device_key(e) for e in events},
        }

    test_rows = (
        test_df
        .with_columns(pl.col("is_ato").cast(pl.Boolean).fill_null(False))
        .with_columns([pl.col(c).cast(pl.Utf8).fill_null("missing") for c in feat_cols])
        .select(["user_id", "is_ato"] + feat_cols)
        .to_dicts()
    )
    for row in test_rows:
        uid = row.pop("user_id")
        is_ato = row.pop("is_ato")
        if uid not in users:
            continue
        if is_ato:
            users[uid]["test_pos"].append(row)
        else:
            users[uid]["test_neg"].append(row)

    users = {uid: u for uid, u in users.items() if u["test_pos"] or u["test_neg"]}

    total_pos = sum(len(u["test_pos"]) for u in users.values())
    total_neg = sum(len(u["test_neg"]) for u in users.values())
    ato_users = sum(1 for u in users.values() if u["test_pos"])
    print(f"  Eligible users: {len(users):,}  (dropped {dropped:,} below {MIN_TRAIN_EVENTS}-event floor)")
    print(f"  ATO users in eval: {ato_users:,}")
    print(f"  Test events: {total_pos:,} ATO  +  {total_neg:,} benign")
    if total_pos == 0:
        print("  WARNING: no ATO test events — AUC will be undefined.")
    return users

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def build_corpus(users: dict) -> list:
    """Per-account concatenated corpus (skip-gram invariant from H2)."""
    sentences = []
    for u in users.values():
        flat = []
        for e in u["train"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences

# ---------------------------------------------------------------------------
# Scoring — all scorers use pre-computed centroid_evts / calib_evts (F2 fix)
# ---------------------------------------------------------------------------
def score_mp_raw(users: dict, model) -> tuple[list, list]:
    """Mean-pool cosine distance using ALL training events for centroid.
    This is the ORIGINAL mp-raw — centroid from full training set."""
    scores, labels = [], []
    for u in users.values():
        centroid = compute_centroid(u["train"], embed_mp, model)
        for ev in u["test_pos"]:
            scores.append(cosine_dist(embed_mp(ev, model), centroid))
            labels.append(1)
        for ev in u["test_neg"]:
            scores.append(cosine_dist(embed_mp(ev, model), centroid))
            labels.append(0)
    return scores, labels

def score_mp_raw_split(users: dict, model) -> tuple[list, list]:
    """Mean-pool cosine distance using SAME centroid_evts as rank-norm variants.
    F2 fix: controls for data quantity confound vs mp-rank-norm."""
    scores, labels = [], []
    for u in users.values():
        centroid = compute_centroid(u["centroid_evts"], embed_mp, model)
        for ev in u["test_pos"]:
            scores.append(cosine_dist(embed_mp(ev, model), centroid))
            labels.append(1)
        for ev in u["test_neg"]:
            scores.append(cosine_dist(embed_mp(ev, model), centroid))
            labels.append(0)
    return scores, labels

def score_mp_rank_norm(users: dict, model) -> tuple[list, list]:
    """Mean-pool cosine distance, rank-normalized per-user."""
    scores, labels = [], []
    for u in users.values():
        centroid = compute_centroid(u["centroid_evts"], embed_mp, model)
        calib_dists = np.array([cosine_dist(embed_mp(e, model), centroid) for e in u["calib_evts"]])
        for ev in u["test_pos"]:
            raw = cosine_dist(embed_mp(ev, model), centroid)
            scores.append(rank_normalize(raw, calib_dists))
            labels.append(1)
        for ev in u["test_neg"]:
            raw = cosine_dist(embed_mp(ev, model), centroid)
            scores.append(rank_normalize(raw, calib_dists))
            labels.append(0)
    return scores, labels

def score_pfn(users: dict, model, aggregation: str = "mean") -> tuple[list, list]:
    """Per-feature normalized scoring.

    For each user:
    1. Build per-feature centroids from shared centroid_evts.
    2. Build per-feature calibration distributions from shared calib_evts.
    3. For each test event: compute per-feature rank-normalized distances.
    4. Aggregate: uniform mean / stability-weighted / max.

    aggregation: "mean" | "stab" | "max"
    """
    scores, labels = [], []
    for u in users.values():
        pf_centroids = compute_per_feature_centroids(u["centroid_evts"], model)
        pf_calib = {}
        for f in FEATURE_ORDER:
            pf_calib[f] = np.array([
                cosine_dist(embed_feature(e, f, model), pf_centroids[f])
                for e in u["calib_evts"]
            ])
        if aggregation == "stab":
            stab_weights = {}
            for f in FEATURE_ORDER:
                std_f = float(pf_calib[f].std())
                stab_weights[f] = 1.0 / (std_f + 1e-6)
            total_w = sum(stab_weights.values())
            stab_weights = {f: w / total_w for f, w in stab_weights.items()}

        def score_event(ev):
            per_feat = []
            for f in FEATURE_ORDER:
                raw_f = cosine_dist(embed_feature(ev, f, model), pf_centroids[f])
                per_feat.append(rank_normalize(raw_f, pf_calib[f]))
            if aggregation == "mean":
                return float(np.mean(per_feat))
            elif aggregation == "stab":
                return float(sum(stab_weights[f] * per_feat[i] for i, f in enumerate(FEATURE_ORDER)))
            elif aggregation == "max":
                return float(max(per_feat))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

        for ev in u["test_pos"]:
            scores.append(score_event(ev))
            labels.append(1)
        for ev in u["test_neg"]:
            scores.append(score_event(ev))
            labels.append(0)
    return scores, labels

def score_trivial(users: dict) -> tuple[list, list]:
    """Exact set-membership: 1.0 if device not seen in training, 0.0 if seen."""
    scores, labels = [], []
    for u in users.values():
        for ev in u["test_pos"]:
            score = 0.0 if device_key(ev) in u["known_devices"] else 1.0
            scores.append(score)
            labels.append(1)
        for ev in u["test_neg"]:
            score = 0.0 if device_key(ev) in u["known_devices"] else 1.0
            scores.append(score)
            labels.append(0)
    return scores, labels

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def threshold_metrics_at_top_pct(scores: list, labels: list, top_pct: float = 0.01,
                                  tie_break_seed: int = SEED) -> dict:
    """Threshold at top-pct. F1 fix: applies tie-breaking perturbation before thresholding
    to ensure exactly top_pct fraction is flagged even when scores are discrete."""
    sc = np.array(scores)
    lb = np.array(labels)
    sc_perturbed = add_tie_break(sc, seed=tie_break_seed)
    cutoff = np.percentile(sc_perturbed, 100 * (1 - top_pct))
    preds = (sc_perturbed >= cutoff).astype(int)
    tp = int(((preds == 1) & (lb == 1)).sum())
    fp = int(((preds == 1) & (lb == 0)).sum())
    fn = int(((preds == 0) & (lb == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    flagged_exact = int(preds.sum())
    expected_flagged = max(1, int(len(sc) * top_pct))
    return {
        "threshold": float(cutoff), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "flagged": flagged_exact, "expected_flagged": expected_flagged, "total": len(sc),
    }

def bootstrap_metrics_stratified(scores: list, labels: list,
                                  n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple:
    """Stratified bootstrap for AUC/AP — resamples pos/neg separately at original counts.
    F6 fix: avoids optimistic CIs when positives are rare."""
    rng = np.random.default_rng(seed)
    sc  = np.array(scores)
    lb  = np.array(labels)
    pos_idx = np.where(lb == 1)[0]
    neg_idx = np.where(lb == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return (float("nan"),) * 6
    roc_aucs, pr_aucs = [], []
    for _ in range(n_boot):
        boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([boot_pos, boot_neg])
        lb_b, sc_b = lb[idx], sc[idx]
        if len(np.unique(lb_b)) < 2:
            continue
        roc_aucs.append(roc_auc_score(lb_b, sc_b))
        pr_aucs.append(average_precision_score(lb_b, sc_b))
    if not roc_aucs:
        return (float("nan"),) * 6
    ra, pa = np.array(roc_aucs), np.array(pr_aucs)
    return (
        float(np.mean(ra)), float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5)),
        float(np.mean(pa)), float(np.percentile(pa, 2.5)), float(np.percentile(pa, 97.5)),
    )

def bootstrap_top1pct_stratified(scores: list, labels: list,
                                  n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Stratified bootstrap CIs for top-1% precision, recall, F1.
    F6 fix: resamples pos/neg separately at original counts."""
    rng = np.random.default_rng(seed)
    sc  = np.array(scores)
    lb  = np.array(labels)
    pos_idx = np.where(lb == 1)[0]
    neg_idx = np.where(lb == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0:
        nan3 = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return {"precision": nan3, "recall": nan3, "f1": nan3}
    precisions, recalls, f1s = [], [], []
    for b in range(n_boot):
        boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([boot_pos, boot_neg])
        sc_b, lb_b = sc[idx], lb[idx]
        if lb_b.sum() == 0:
            continue
        m = threshold_metrics_at_top_pct(sc_b.tolist(), lb_b.tolist(), tie_break_seed=SEED + b)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])
    if not precisions:
        nan3 = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return {"precision": nan3, "recall": nan3, "f1": nan3}
    def ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "lo":   float(np.percentile(arr, 2.5)),
            "hi":   float(np.percentile(arr, 97.5)),
        }
    return {
        "precision": ci(precisions),
        "recall":    ci(recalls),
        "f1":        ci(f1s),
    }

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name: str, figures_dir: Path = FIGURES_DIR) -> None:
    out = figures_dir / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")

def plot_summary_auc(results: dict) -> None:
    """Bar chart of ROC-AUC and PR-AUC for all scoring variants."""
    variants = list(results.keys())
    colors = {
        "mp-raw":        "#9E9E9E",
        "mp-raw-split":  "#BDBDBD",
        "mp-rank-norm":  "#2196F3",
        "pfn-mean":      "#4CAF50",
        "pfn-stab":      "#FF9800",
        "pfn-max":       "#E91E63",
        "trivial":       "#607D8B",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, ["roc_auc", "pr_auc"]):
        vals = [results[v][metric]["mean"] for v in variants]
        los  = [results[v][metric]["lo"]   for v in variants]
        his  = [results[v][metric]["hi"]   for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, vals, color=[colors.get(v, "#777") for v in variants], alpha=0.85)
        for i, (v, lo, hi) in enumerate(zip(vals, los, his)):
            if not np.isnan(lo):
                ax.errorbar(i, v, yerr=[[v - lo], [hi - v]],
                            fmt="none", color="black", capsize=4, linewidth=1.3)
            ax.text(i, (hi if not np.isnan(hi) else v) + 0.01, f"{v:.3f}",
                    ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.18)
        label = "ROC-AUC" if metric == "roc_auc" else "PR-AUC"
        ax.set_ylabel(label)
        ax.set_title(f"H3 PFN v2: {label}\n(RBA dataset, sg=1, per-account corpus, stratified bootstrap)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    save_fig(fig, "pfn_v2_summary_auc.png")

def plot_top1pct(results: dict) -> None:
    """Bar chart of top-1% precision and recall for all scoring variants."""
    variants = list(results.keys())
    colors = {
        "mp-raw":        "#9E9E9E",
        "mp-raw-split":  "#BDBDBD",
        "mp-rank-norm":  "#2196F3",
        "pfn-mean":      "#4CAF50",
        "pfn-stab":      "#FF9800",
        "pfn-max":       "#E91E63",
        "trivial":       "#607D8B",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, ["precision", "recall"]):
        vals = [results[v]["top1pct"][metric]["mean"] for v in variants]
        los  = [results[v]["top1pct"][metric]["lo"]   for v in variants]
        his  = [results[v]["top1pct"][metric]["hi"]   for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, vals, color=[colors.get(v, "#777") for v in variants], alpha=0.85)
        for i, (v, lo, hi) in enumerate(zip(vals, los, his)):
            if not np.isnan(lo):
                ax.errorbar(i, v, yerr=[[v - lo], [hi - v]],
                            fmt="none", color="black", capsize=4, linewidth=1.3)
            ax.text(i, (hi if not np.isnan(hi) else v) + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, min(1.18, max([h for h in his if not np.isnan(h)] or [0.2]) * 1.3 + 0.05))
        ax.set_ylabel(f"Top-1% {metric.capitalize()}")
        ax.set_title(f"H3 PFN v2: Top-1% {metric.capitalize()}\n"
                     f"(RBA dataset, tie-breaking perturbation, stratified bootstrap)")
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    save_fig(fig, "pfn_v2_top1pct.png")

def plot_precision_recall_curve(results_raw: dict) -> None:
    """PR curves for all scoring variants."""
    colors = {
        "mp-raw":        "#9E9E9E",
        "mp-raw-split":  "#BDBDBD",
        "mp-rank-norm":  "#2196F3",
        "pfn-mean":      "#4CAF50",
        "pfn-stab":      "#FF9800",
        "pfn-max":       "#E91E63",
        "trivial":       "#607D8B",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(results_raw.keys())
    baseline_rate = np.mean(results_raw[labels[0]][1])
    ax.axhline(baseline_rate, color="gray", linestyle=":",
               linewidth=0.8, label=f"random ({baseline_rate:.4f})")
    for name, (scores, lbls) in results_raw.items():
        p, r, _ = precision_recall_curve(lbls, scores)
        ap = average_precision_score(lbls, scores)
        ax.plot(r, p, color=colors.get(name, "#777"), linewidth=1.5,
                label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("H3 PFN v2: Precision-Recall Curves\n(RBA dataset, ATO label)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_fig(fig, "pfn_v2_pr_curves.png")

def plot_feature_importance(users: dict, model) -> dict:
    """Per-feature mean AUC — shows which features carry the most signal."""
    feature_aucs = {}
    for f in FEATURE_ORDER:
        scores, labels = [], []
        for u in users.values():
            fc = np.mean([embed_feature(e, f, model) for e in u["centroid_evts"]], axis=0)
            calib_d = np.array([cosine_dist(embed_feature(e, f, model), fc) for e in u["calib_evts"]])
            for ev in u["test_pos"]:
                raw = cosine_dist(embed_feature(ev, f, model), fc)
                scores.append(rank_normalize(raw, calib_d))
                labels.append(1)
            for ev in u["test_neg"]:
                raw = cosine_dist(embed_feature(ev, f, model), fc)
                scores.append(rank_normalize(raw, calib_d))
                labels.append(0)
        if len(set(labels)) == 2:
            try:
                feature_aucs[f] = roc_auc_score(labels, scores)
            except Exception:
                feature_aucs[f] = float("nan")
        else:
            feature_aucs[f] = float("nan")

    fig, ax = plt.subplots(figsize=(8, 4))
    features = FEATURE_ORDER
    aucs = [feature_aucs[f] for f in features]
    x = np.arange(len(features))
    ax.bar(x, aucs, color="#4CAF50", alpha=0.85)
    for i, a in enumerate(aucs):
        ax.text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=15, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("ROC-AUC (single-feature rank-norm)")
    ax.set_title("H3 PFN v2: Per-Feature AUC\n(shows which features carry ATO signal)")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "pfn_v2_feature_importance.png")
    return feature_aucs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="H3 PFN v2 experiment on RBA dataset")
    parser.add_argument("--smoke", action="store_true",
                        help=f"fast check on {SMOKE_N_USERS} users")
    parser.add_argument("--full",  action="store_true",
                        help="use all eligible users (no subsampling)")
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH)
    parser.add_argument("--split-pct", type=float, default=0.50)
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"ERROR: parquet not found: {args.parquet}")
        print("Run: uv run h2_rba/experiments/data_prep.py")
        return

    tag = "smoke" if args.smoke else ("full" if args.full else "default")
    print("=" * 72)
    print("H3 Per-Feature Normalized (PFN) Scoring Experiment v2")
    print(f"mode={tag} | sg=1, per-account corpus, proportional calib split")
    print("Fixes: F1=tie-break perturbation, F2=mp-raw-split, F6=stratified bootstrap")
    print("=" * 72)

    t0 = time.time()
    print("\n[1/6] Loading data ...")
    users = load_users(args.parquet, smoke=args.smoke, full=args.full,
                       split_pct=args.split_pct)

    print("\n[2/6] Calibration split statistics ...")
    calib_sizes = [len(u["calib_evts"]) for u in users.values()]
    centroid_sizes = [len(u["centroid_evts"]) for u in users.values()]
    print(f"  Calibration events  — min: {min(calib_sizes)}, "
          f"median: {int(np.median(calib_sizes))}, max: {max(calib_sizes)}")
    print(f"  Centroid events     — min: {min(centroid_sizes)}, "
          f"median: {int(np.median(centroid_sizes))}, max: {max(centroid_sizes)}")
    n_ato_test = sum(len(u["test_pos"]) for u in users.values())
    print(f"  ATO test events: {n_ato_test}")
    print(f"  NOTE: ATO users have 5-9 training events (median calib=2).")
    print(f"  Raising MIN_TRAIN_EVENTS would eliminate ATO test users entirely.")
    print(f"  F1 fix: tie-breaking perturbation instead of larger calibration sets.")

    print("\n[3/6] Training FastText (single model, per-account corpus) ...")
    corpus = build_corpus(users)
    print(f"  Corpus: {len(corpus):,} sentences, "
          f"{sum(len(s) for s in corpus):,} tokens")
    model = FastText(sentences=corpus, **ROBUST_KWARGS)
    print(f"  Vocabulary: {len(model.wv):,} tokens")

    print("\n[4/6] Scoring all variants ...")
    print("  mp-raw (all training events) ...")
    raw_s,  raw_l  = score_mp_raw(users, model)
    print("  mp-raw-split (centroid_evts only — F2 fair comparison) ...")
    rns_s,  rns_l  = score_mp_raw_split(users, model)
    print("  mp-rank-norm ...")
    rn_s,   rn_l   = score_mp_rank_norm(users, model)
    print("  pfn-mean ...")
    pfnm_s, pfnm_l = score_pfn(users, model, aggregation="mean")
    print("  pfn-stab ...")
    pfns_s, pfns_l = score_pfn(users, model, aggregation="stab")
    print("  pfn-max ...")
    pfnx_s, pfnx_l = score_pfn(users, model, aggregation="max")
    print("  trivial ...")
    triv_s, triv_l = score_trivial(users)
    assert raw_l == triv_l, "Label vectors mismatched across scorers"

    variants = {
        "mp-raw":        (raw_s,  raw_l),
        "mp-raw-split":  (rns_s,  rns_l),
        "mp-rank-norm":  (rn_s,   rn_l),
        "pfn-mean":      (pfnm_s, pfnm_l),
        "pfn-stab":      (pfns_s, pfns_l),
        "pfn-max":       (pfnx_s, pfnx_l),
        "trivial":       (triv_s, triv_l),
    }

    print("\n[5/6] Computing metrics (stratified bootstrap) ...")
    results = {}
    for name, (sc, lb) in variants.items():
        roc_m, roc_lo, roc_hi, pr_m, pr_lo, pr_hi = bootstrap_metrics_stratified(sc, lb)
        top1 = threshold_metrics_at_top_pct(sc, lb)
        top1_boot = bootstrap_top1pct_stratified(sc, lb)
        results[name] = {
            "roc_auc": {"mean": roc_m, "lo": roc_lo, "hi": roc_hi},
            "pr_auc":  {"mean": pr_m,  "lo": pr_lo,  "hi": pr_hi},
            "top1pct": {
                "precision": top1_boot["precision"],
                "recall":    top1_boot["recall"],
                "f1":        top1_boot["f1"],
                "point": top1,
            },
        }

    # Print summary tables
    print(f"\n  {'Model':<16} {'ROC-AUC':>10} {'[95% CI]':>22}  {'PR-AUC':>8} {'[95% CI]':>22}")
    print("  " + "-" * 88)
    for name in results:
        r = results[name]
        roc, rlo, rhi = r["roc_auc"]["mean"], r["roc_auc"]["lo"], r["roc_auc"]["hi"]
        pr, plo, phi  = r["pr_auc"]["mean"],  r["pr_auc"]["lo"],  r["pr_auc"]["hi"]
        ci_r = f"[{rlo:.4f}, {rhi:.4f}]" if not np.isnan(rlo) else "      —      "
        ci_p = f"[{plo:.4f}, {phi:.4f}]" if not np.isnan(plo) else "      —      "
        print(f"  {name:<16} {roc:>10.4f} {ci_r:>22}  {pr:>8.4f} {ci_p:>22}")

    print(f"\n  Top-1% Threshold Metrics (point estimate + stratified bootstrap CI):")
    print(f"  {'Model':<16} {'TP':>4} {'FP':>6} {'Flagged':>8}  "
          f"{'Prec':>7} {'[CI]':>16}  {'Rec':>7} {'[CI]':>16}")
    print("  " + "-" * 98)
    for name in results:
        r = results[name]["top1pct"]
        pt = r["point"]
        p_ci = f"[{r['precision']['lo']:.3f},{r['precision']['hi']:.3f}]"
        re_ci = f"[{r['recall']['lo']:.3f},{r['recall']['hi']:.3f}]"
        print(f"  {name:<16} {pt['tp']:>4} {pt['fp']:>6} {pt['flagged']:>8}  "
              f"{pt['precision']:>7.4f} {p_ci:>16}  {pt['recall']:>7.4f} {re_ci:>16}")

    # Headline verdict
    mp_rn_prec = results["mp-rank-norm"]["top1pct"]["precision"]["mean"]
    mp_rn_rec  = results["mp-rank-norm"]["top1pct"]["recall"]["mean"]
    best_pfn = None
    best_improvement = -np.inf
    for pfn_name in ["pfn-mean", "pfn-stab", "pfn-max"]:
        pfn_prec = results[pfn_name]["top1pct"]["precision"]["mean"]
        pfn_rec  = results[pfn_name]["top1pct"]["recall"]["mean"]
        improvement = (pfn_prec - mp_rn_prec) + (pfn_rec - mp_rn_rec)
        if improvement > best_improvement:
            best_improvement = improvement
            best_pfn = pfn_name
    pfn_beats_baseline = (
        results[best_pfn]["top1pct"]["precision"]["mean"] > mp_rn_prec and
        results[best_pfn]["top1pct"]["recall"]["mean"]    > mp_rn_rec
    )
    # Additional AUC check
    pfn_beats_auc = results[best_pfn]["roc_auc"]["mean"] > results["mp-rank-norm"]["roc_auc"]["mean"]
    triv_roc = results["trivial"]["roc_auc"]["mean"]
    best_pfn_roc = results[best_pfn]["roc_auc"]["mean"]
    mp_rn_roc = results["mp-rank-norm"]["roc_auc"]["mean"]
    print(f"\n  HEADLINE:")
    print(f"  mp-rank-norm: ROC-AUC={mp_rn_roc:.4f}, "
          f"top-1% precision={mp_rn_prec:.4f}, recall={mp_rn_rec:.4f}")
    pfn_prec = results[best_pfn]["top1pct"]["precision"]["mean"]
    pfn_rec  = results[best_pfn]["top1pct"]["recall"]["mean"]
    print(f"  best PFN ({best_pfn}): ROC-AUC={best_pfn_roc:.4f}, "
          f"top-1% precision={pfn_prec:.4f}, recall={pfn_rec:.4f}")
    verdict = "H3 HYPOTHESIS CONFIRMED" if pfn_beats_baseline else "H3 HYPOTHESIS NOT CONFIRMED"
    print(f"  {verdict}")
    print(f"  Trivial ROC-AUC: {triv_roc:.4f}  |  mp-rank-norm: {mp_rn_roc:.4f}  |  "
          f"Best-PFN: {best_pfn_roc:.4f}")
    print(f"  PFN beats trivial AUC: {best_pfn_roc > triv_roc}")
    print(f"  PFN beats mp-rank-norm AUC: {pfn_beats_auc}")
    print(f"  NOTE: Test set has only {n_ato_test} ATO events — results have high variance.")

    print("\n[6/6] Saving figures and metrics ...")
    plot_summary_auc(results)
    plot_top1pct(results)
    plot_precision_recall_curve(variants)
    print("  Computing per-feature importance ...")
    feature_aucs = plot_feature_importance(users, model)
    print(f"  Per-feature AUCs: { {f: f'{v:.4f}' for f, v in feature_aucs.items()} }")

    elapsed = time.time() - t0
    metrics_out = {
        "version": "v2",
        "mode": tag, "split_pct": args.split_pct,
        "n_users": len(users),
        "n_ato_test": n_ato_test,
        "n_benign_test": sum(len(u["test_neg"]) for u in users.values()),
        "calib_sizes_median": int(np.median(calib_sizes)),
        "fixes_applied": ["F1_tie_break_perturbation", "F2_mp_raw_split", "F6_stratified_bootstrap"],
        "unfixed_limitations": {
            "F3": "pfn-stab stability weights computed on n_calib=2 — numerically unstable",
            "F4": f"only {n_ato_test} ATO test events — fundamental dataset constraint",
            "F5": "per-feature distances correlated via co-occurrence — intentional by design",
        },
        "results": {
            k: {
                "roc_auc": v["roc_auc"],
                "pr_auc":  v["pr_auc"],
                "top1pct_point": v["top1pct"]["point"],
                "top1pct_boot":  {
                    "precision": v["top1pct"]["precision"],
                    "recall":    v["top1pct"]["recall"],
                    "f1":        v["top1pct"]["f1"],
                },
            }
            for k, v in results.items()
        },
        "feature_aucs": feature_aucs,
        "best_pfn": best_pfn,
        "hypothesis_confirmed": pfn_beats_baseline,
        "elapsed_s": round(elapsed, 1),
    }
    metrics_path = FIGURES_DIR / "pfn_v2_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics: {metrics_path}")
    print(f"\n  Elapsed: {elapsed:.0f}s")
    print("Done.")

if __name__ == "__main__":
    main()
