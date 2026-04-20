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
# Per-Feature Normalized (PFN) Scoring Experiment v3 — H3 Follow-on
# =================================================================
# Targeted ablation of pfn_experiment2.py: excludes rtt_bucket from the
# PFN feature set to test whether dropping the below-chance feature
# (rtt_bucket AUC=0.461) recovers the PFN mechanism.
#
# Motivation from v2 diagnosis:
#   Root cause of PFN failure: rtt_bucket (per-feature AUC=0.461) actively
#   degrades the uniform mean aggregate. The remaining 6 features all score
#   above chance (0.61–0.83). Excluding rtt_bucket removes the anti-signal
#   while keeping the dilution-prevention mechanism intact.
#
# Changes vs v2:
#   - Added PFN_FEATURES_6 = FEATURE_ORDER without rtt_bucket
#   - score_pfn() and compute_per_feature_centroids() accept optional features= arg
#   - New variant: pfn-mean-6f (6-feature uniform mean, rtt_bucket excluded)
#   - Verdict comparison uses mp-raw-split (fair baseline) not mp-rank-norm
#   - Figure/metrics prefix: pfn_v3_*
#
# All v2 fixes retained:
#   F1: tie-breaking perturbation for top-1% threshold
#   F2: shared centroid/calib split; mp-raw-split baseline
#   F6: stratified bootstrap
#
# Usage:
#   uv run h3_pfn/experiments/pfn_experiment3.py
#   uv run h3_pfn/experiments/pfn_experiment3.py --smoke
#   uv run h3_pfn/experiments/pfn_experiment3.py --full

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
MIN_TRAIN_EVENTS  = 5
N_BENIGN_USERS    = 50_000
SMOKE_N_USERS     = 5_000

# Full feature schema (model trains on all 7)
FEATURE_ORDER       = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]
DEVICE_KEY_FEATURES = ["os", "browser", "device_type", "country", "region", "asn_bucket"]

# 6-feature set: rtt_bucket excluded (AUC=0.461, below chance in v2)
PFN_FEATURES_6 = [f for f in FEATURE_ORDER if f != "rtt_bucket"]

ROBUST_KWARGS = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    seed=SEED, workers=1,
)

# ---------------------------------------------------------------------------
# Calibration split
# ---------------------------------------------------------------------------
def split_calib(events: list) -> tuple[list, list]:
    n = len(events)
    n_calib = max(2, n // 3)
    n_centroid = max(1, n - n_calib)
    n_calib = n - n_centroid
    return events[:n_centroid], events[n_centroid:]

def add_tie_break(scores: np.ndarray, seed: int = SEED) -> np.ndarray:
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
    vecs = [model.wv[make_token(f, event.get(f) or "missing")] for f in FEATURE_ORDER]
    return np.mean(vecs, axis=0)

def embed_feature(event: dict, feature: str, model) -> np.ndarray:
    return model.wv[make_token(feature, event.get(feature) or "missing")]

def compute_centroid(events, embed_fn, model) -> np.ndarray:
    return np.mean([embed_fn(e, model) for e in events], axis=0)

def compute_per_feature_centroids(events, model, features=None) -> dict:
    if features is None:
        features = FEATURE_ORDER
    return {
        f: np.mean([embed_feature(e, f, model) for e in events], axis=0)
        for f in features
    }

def rank_normalize(raw_score: float, calib_dist: np.ndarray) -> float:
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
    print(f"  Split at {cutoff}: {len(train_df):,} train / {len(test_df):,} test rows")

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
        print(f"  Subsampled: {len(ato_user_ids):,} ATO + {len(sampled_benign):,} benign")

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
            "train":        events,
            "centroid_evts": centroid_evts,
            "calib_evts":    calib_evts,
            "test_pos":     [],
            "test_neg":     [],
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
    ato_users_count = sum(1 for u in users.values() if u["test_pos"])
    print(f"  Eligible users: {len(users):,}  (dropped {dropped:,})")
    print(f"  ATO users in eval: {ato_users_count:,}")
    print(f"  Test events: {total_pos:,} ATO  +  {total_neg:,} benign")
    return users

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def build_corpus(users: dict) -> list:
    sentences = []
    for u in users.values():
        flat = []
        for e in u["train"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_mp_raw(users: dict, model) -> tuple[list, list]:
    scores, labels = [], []
    for u in users.values():
        centroid = compute_centroid(u["train"], embed_mp, model)
        for ev in u["test_pos"]:
            scores.append(cosine_dist(embed_mp(ev, model), centroid)); labels.append(1)
        for ev in u["test_neg"]:
            scores.append(cosine_dist(embed_mp(ev, model), centroid)); labels.append(0)
    return scores, labels

def score_mp_raw_split(users: dict, model) -> tuple[list, list]:
    """Mean-pool using same centroid_evts as rank-norm — fair comparison baseline."""
    scores, labels = [], []
    for u in users.values():
        centroid = compute_centroid(u["centroid_evts"], embed_mp, model)
        for ev in u["test_pos"]:
            scores.append(cosine_dist(embed_mp(ev, model), centroid)); labels.append(1)
        for ev in u["test_neg"]:
            scores.append(cosine_dist(embed_mp(ev, model), centroid)); labels.append(0)
    return scores, labels

def score_mp_rank_norm(users: dict, model) -> tuple[list, list]:
    scores, labels = [], []
    for u in users.values():
        centroid = compute_centroid(u["centroid_evts"], embed_mp, model)
        calib_d = np.array([cosine_dist(embed_mp(e, model), centroid) for e in u["calib_evts"]])
        for ev in u["test_pos"]:
            scores.append(rank_normalize(cosine_dist(embed_mp(ev, model), centroid), calib_d))
            labels.append(1)
        for ev in u["test_neg"]:
            scores.append(rank_normalize(cosine_dist(embed_mp(ev, model), centroid), calib_d))
            labels.append(0)
    return scores, labels

def score_pfn(users: dict, model, aggregation: str = "mean",
              features: list | None = None) -> tuple[list, list]:
    """Per-feature normalized scoring over a given feature subset.

    features: list of feature names to decompose (default: all 7).
              Pass PFN_FEATURES_6 to exclude rtt_bucket.
    aggregation: "mean" | "stab" | "max"
    """
    if features is None:
        features = FEATURE_ORDER
    scores, labels = [], []
    for u in users.values():
        pf_centroids = compute_per_feature_centroids(u["centroid_evts"], model, features)
        pf_calib = {
            f: np.array([cosine_dist(embed_feature(e, f, model), pf_centroids[f])
                         for e in u["calib_evts"]])
            for f in features
        }
        if aggregation == "stab":
            stds = np.array([pf_calib[f].std() for f in features])
            weights = 1.0 / (stds + 1e-6)
            weights = weights / weights.sum()

        def score_event(ev):
            per_feat = np.array([
                rank_normalize(cosine_dist(embed_feature(ev, f, model), pf_centroids[f]),
                               pf_calib[f])
                for f in features
            ])
            if aggregation == "max":
                return float(per_feat.max())
            elif aggregation == "stab":
                return float(np.dot(per_feat, weights))
            return float(per_feat.mean())

        for ev in u["test_pos"]:
            scores.append(score_event(ev)); labels.append(1)
        for ev in u["test_neg"]:
            scores.append(score_event(ev)); labels.append(0)
    return scores, labels

def score_trivial(users: dict) -> tuple[list, list]:
    scores, labels = [], []
    for u in users.values():
        for ev in u["test_pos"]:
            scores.append(0.0 if device_key(ev) in u["known_devices"] else 1.0); labels.append(1)
        for ev in u["test_neg"]:
            scores.append(0.0 if device_key(ev) in u["known_devices"] else 1.0); labels.append(0)
    return scores, labels

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def threshold_metrics_at_top_pct(scores: list, labels: list, top_pct: float = 0.01,
                                  tie_break_seed: int = SEED) -> dict:
    sc = np.array(scores)
    lb = np.array(labels)
    sc_p = add_tie_break(sc, seed=tie_break_seed)
    cutoff = np.percentile(sc_p, 100 * (1 - top_pct))
    preds = (sc_p >= cutoff).astype(int)
    tp = int(((preds == 1) & (lb == 1)).sum())
    fp = int(((preds == 1) & (lb == 0)).sum())
    fn = int(((preds == 0) & (lb == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "threshold": float(cutoff), "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "flagged": int(preds.sum()), "expected_flagged": max(1, int(len(sc) * top_pct)),
        "total": len(sc),
    }

def bootstrap_metrics_stratified(scores: list, labels: list,
                                  n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple:
    rng = np.random.default_rng(seed)
    sc, lb = np.array(scores), np.array(labels)
    pos_idx, neg_idx = np.where(lb == 1)[0], np.where(lb == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return (float("nan"),) * 6
    roc_aucs, pr_aucs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
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
    rng = np.random.default_rng(seed)
    sc, lb = np.array(scores), np.array(labels)
    pos_idx, neg_idx = np.where(lb == 1)[0], np.where(lb == 0)[0]
    if len(pos_idx) == 0:
        nan3 = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return {"precision": nan3, "recall": nan3, "f1": nan3}
    precisions, recalls, f1s = [], [], []
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        m = threshold_metrics_at_top_pct(sc[idx].tolist(), lb[idx].tolist(),
                                         tie_break_seed=SEED + b)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

    def ci(arr):
        return {"mean": float(np.mean(arr)),
                "lo":   float(np.percentile(arr, 2.5)),
                "hi":   float(np.percentile(arr, 97.5))}
    return {"precision": ci(precisions), "recall": ci(recalls), "f1": ci(f1s)}

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name: str) -> None:
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")

COLORS = {
    "mp-raw":         "#9E9E9E",
    "mp-raw-split":   "#BDBDBD",
    "mp-rank-norm":   "#2196F3",
    "pfn-mean":       "#4CAF50",
    "pfn-mean-6f":    "#1B5E20",  # darker green — the ablation variant
    "pfn-stab":       "#FF9800",
    "pfn-max":        "#E91E63",
    "trivial":        "#607D8B",
}

def plot_summary_auc(results: dict) -> None:
    variants = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, ["roc_auc", "pr_auc"]):
        vals = [results[v][metric]["mean"] for v in variants]
        los  = [results[v][metric]["lo"]   for v in variants]
        his  = [results[v][metric]["hi"]   for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, vals, color=[COLORS.get(v, "#777") for v in variants], alpha=0.85)
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
        ax.set_title(f"H3 PFN v3 (rtt_bucket ablation): {label}\n"
                     f"(RBA dataset, darker green = pfn-mean-6f)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    save_fig(fig, "pfn_v3_summary_auc.png")

def plot_top1pct(results: dict) -> None:
    variants = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, ["precision", "recall"]):
        vals = [results[v]["top1pct"][metric]["mean"] for v in variants]
        los  = [results[v]["top1pct"][metric]["lo"]   for v in variants]
        his  = [results[v]["top1pct"][metric]["hi"]   for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, vals, color=[COLORS.get(v, "#777") for v in variants], alpha=0.85)
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
        ax.set_title(f"H3 PFN v3: Top-1% {metric.capitalize()}\n"
                     f"(pfn-mean-6f = pfn-mean excluding rtt_bucket)")
    plt.tight_layout()
    save_fig(fig, "pfn_v3_top1pct.png")

def plot_precision_recall_curve(results_raw: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(results_raw.keys())
    baseline_rate = float(np.mean(results_raw[labels[0]][1]))
    ax.axhline(baseline_rate, color="gray", linestyle=":",
               linewidth=0.8, label=f"random ({baseline_rate:.4f})")
    for name, (scores, lbls) in results_raw.items():
        p, r, _ = precision_recall_curve(lbls, scores)
        ap = average_precision_score(lbls, scores)
        ax.plot(r, p, color=COLORS.get(name, "#777"), linewidth=1.5,
                label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("H3 PFN v3: Precision-Recall Curves\n(RBA dataset)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_fig(fig, "pfn_v3_pr_curves.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="H3 PFN v3: rtt_bucket ablation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full",  action="store_true")
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH)
    parser.add_argument("--split-pct", type=float, default=0.50)
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"ERROR: parquet not found: {args.parquet}")
        print("Run: uv run h2_rba/experiments/data_prep.py")
        return

    tag = "smoke" if args.smoke else ("full" if args.full else "default")
    print("=" * 72)
    print("H3 PFN v3 — rtt_bucket Ablation")
    print(f"mode={tag}  |  PFN_FEATURES_6 = {PFN_FEATURES_6}")
    print("=" * 72)

    t0 = time.time()

    print("\n[1/5] Loading data ...")
    users = load_users(args.parquet, smoke=args.smoke, full=args.full,
                       split_pct=args.split_pct)
    n_ato_test = sum(len(u["test_pos"]) for u in users.values())
    calib_sizes = [len(u["calib_evts"]) for u in users.values()]
    print(f"  Calibration  — min: {min(calib_sizes)}, "
          f"median: {int(np.median(calib_sizes))}, max: {max(calib_sizes)}")

    print("\n[2/5] Training FastText (all 7 features, per-account corpus) ...")
    corpus = build_corpus(users)
    model = FastText(sentences=corpus, **ROBUST_KWARGS)
    print(f"  Vocabulary: {len(model.wv):,} tokens")

    print("\n[3/5] Scoring all variants ...")
    print("  mp-raw ...")
    raw_s,  raw_l  = score_mp_raw(users, model)
    print("  mp-raw-split (fair baseline) ...")
    rns_s,  rns_l  = score_mp_raw_split(users, model)
    print("  mp-rank-norm ...")
    rn_s,   rn_l   = score_mp_rank_norm(users, model)
    print("  pfn-mean (7 features, v2 reference) ...")
    pfnm_s, pfnm_l = score_pfn(users, model, aggregation="mean", features=FEATURE_ORDER)
    print("  pfn-mean-6f (6 features, rtt_bucket excluded) ...")
    pfn6_s, pfn6_l = score_pfn(users, model, aggregation="mean", features=PFN_FEATURES_6)
    print("  pfn-stab-6f ...")
    pfns_s, pfns_l = score_pfn(users, model, aggregation="stab", features=PFN_FEATURES_6)
    print("  pfn-max-6f ...")
    pfnx_s, pfnx_l = score_pfn(users, model, aggregation="max",  features=PFN_FEATURES_6)
    print("  trivial ...")
    triv_s, triv_l = score_trivial(users)
    assert raw_l == triv_l, "Label vectors mismatched"

    variants = {
        "mp-raw":        (raw_s,  raw_l),
        "mp-raw-split":  (rns_s,  rns_l),
        "mp-rank-norm":  (rn_s,   rn_l),
        "pfn-mean":      (pfnm_s, pfnm_l),
        "pfn-mean-6f":   (pfn6_s, pfn6_l),
        "pfn-stab-6f":   (pfns_s, pfns_l),
        "pfn-max-6f":    (pfnx_s, pfnx_l),
        "trivial":       (triv_s, triv_l),
    }

    print("\n[4/5] Computing metrics (stratified bootstrap) ...")
    results = {}
    for name, (sc, lb) in variants.items():
        roc_m, roc_lo, roc_hi, pr_m, pr_lo, pr_hi = bootstrap_metrics_stratified(sc, lb)
        top1       = threshold_metrics_at_top_pct(sc, lb)
        top1_boot  = bootstrap_top1pct_stratified(sc, lb)
        results[name] = {
            "roc_auc": {"mean": roc_m, "lo": roc_lo, "hi": roc_hi},
            "pr_auc":  {"mean": pr_m,  "lo": pr_lo,  "hi": pr_hi},
            "top1pct": {
                "precision": top1_boot["precision"],
                "recall":    top1_boot["recall"],
                "f1":        top1_boot["f1"],
                "point":     top1,
            },
        }

    # Print ROC-AUC table
    print(f"\n  {'Model':<16} {'ROC-AUC':>10} {'[95% CI]':>22}  {'PR-AUC':>8} {'[95% CI]':>22}")
    print("  " + "-" * 88)
    for name in results:
        r = results[name]
        roc, rlo, rhi = r["roc_auc"]["mean"], r["roc_auc"]["lo"], r["roc_auc"]["hi"]
        pr, plo, phi  = r["pr_auc"]["mean"],  r["pr_auc"]["lo"],  r["pr_auc"]["hi"]
        ci_r = f"[{rlo:.4f}, {rhi:.4f}]" if not np.isnan(rlo) else "      —      "
        ci_p = f"[{plo:.4f}, {phi:.4f}]" if not np.isnan(plo) else "      —      "
        print(f"  {name:<16} {roc:>10.4f} {ci_r:>22}  {pr:>8.4f} {ci_p:>22}")

    # Print top-1% table
    print(f"\n  Top-1% (point + stratified bootstrap CI):")
    print(f"  {'Model':<16} {'TP':>4} {'FP':>6} {'Flagged':>8}  "
          f"{'Prec':>7} {'[CI]':>16}  {'Rec':>7} {'[CI]':>16}")
    print("  " + "-" * 98)
    for name in results:
        r = results[name]["top1pct"]
        pt = r["point"]
        p_ci  = f"[{r['precision']['lo']:.3f},{r['precision']['hi']:.3f}]"
        re_ci = f"[{r['recall']['lo']:.3f},{r['recall']['hi']:.3f}]"
        print(f"  {name:<16} {pt['tp']:>4} {pt['fp']:>6} {pt['flagged']:>8}  "
              f"{pt['precision']:>7.4f} {p_ci:>16}  {pt['recall']:>7.4f} {re_ci:>16}")

    # Verdict: compare pfn-mean-6f vs mp-raw-split (fair baseline)
    baseline = "mp-raw-split"
    target   = "pfn-mean-6f"
    b_prec = results[baseline]["top1pct"]["precision"]["mean"]
    b_rec  = results[baseline]["top1pct"]["recall"]["mean"]
    b_roc  = results[baseline]["roc_auc"]["mean"]
    t_prec = results[target]["top1pct"]["precision"]["mean"]
    t_rec  = results[target]["top1pct"]["recall"]["mean"]
    t_roc  = results[target]["roc_auc"]["mean"]
    beats_threshold = t_prec > b_prec and t_rec > b_rec
    beats_auc       = t_roc > b_roc
    print(f"\n  VERDICT (pfn-mean-6f vs {baseline}):")
    print(f"  {baseline:<16}: ROC={b_roc:.4f}  prec={b_prec:.4f}  rec={b_rec:.4f}")
    print(f"  {target:<16}: ROC={t_roc:.4f}  prec={t_prec:.4f}  rec={t_rec:.4f}")
    print(f"  pfn-mean-6f beats baseline on top-1%: {beats_threshold}")
    print(f"  pfn-mean-6f beats baseline on ROC-AUC: {beats_auc}")
    verdict = "FOLLOW-ON CONFIRMED" if beats_threshold else "FOLLOW-ON NOT CONFIRMED"
    print(f"  {verdict}")
    print(f"  NOTE: n_ato_test={n_ato_test} — 1 TP swing = {1/n_ato_test:.3f} recall change")

    print("\n[5/5] Saving figures and metrics ...")
    plot_summary_auc(results)
    plot_top1pct(results)
    plot_precision_recall_curve(variants)

    elapsed = time.time() - t0
    metrics_out = {
        "version": "v3",
        "ablation": "rtt_bucket_excluded_from_pfn",
        "pfn_features_7": FEATURE_ORDER,
        "pfn_features_6": PFN_FEATURES_6,
        "mode": tag, "split_pct": args.split_pct,
        "n_users": len(users),
        "n_ato_test": n_ato_test,
        "n_benign_test": sum(len(u["test_neg"]) for u in users.values()),
        "calib_sizes_median": int(np.median(calib_sizes)),
        "verdict": verdict,
        "pfn_mean_6f_beats_baseline_top1pct": beats_threshold,
        "pfn_mean_6f_beats_baseline_auc": beats_auc,
        "results": {
            k: {
                "roc_auc": v["roc_auc"],
                "pr_auc":  v["pr_auc"],
                "top1pct_point": v["top1pct"]["point"],
                "top1pct_boot": {
                    "precision": v["top1pct"]["precision"],
                    "recall":    v["top1pct"]["recall"],
                    "f1":        v["top1pct"]["f1"],
                },
            }
            for k, v in results.items()
        },
        "elapsed_s": round(elapsed, 1),
    }
    metrics_path = FIGURES_DIR / "pfn_v3_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics: {metrics_path}")
    print(f"\n  Elapsed: {elapsed:.0f}s")
    print("Done.")

if __name__ == "__main__":
    main()
