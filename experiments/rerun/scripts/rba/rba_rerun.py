# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim==4.4.0",
#   "numpy==2.2.6",
#   "scikit-learn==1.7.2",
#   "matplotlib==3.10.8",
#   "polars==1.40.0",
# ]
# ///
#
# RBA Rerun — H2 Replication on Real Data (Seed-Parameterized)
# =============================================================
# Replicates the H2_RERUN pipeline (robust_config_experiment.py) on the
# DAS Group RBA dataset instead of synthetic data. Tests whether FastText
# skip-gram mean-pool centroid scoring beats trivial set-membership on
# real-world ATO detection.
#
# Addresses issues:
#   983dfc46 — no --seed CLI arg (SEED=42 hardcoded)
#   b0218ada — output schema missing seed/timestamp/split_percentile
#   c4d6de92 — results.json not written to rerun seeds directory
#   27ef710e — dep version pins were >= not ==
#   a2b73375 — T8 collapse threshold canonicalized to 0.9
#   9511d90f — h2_replicated uses CI-lower-bound criterion
#
# Requires: data/rba/rba.parquet (run data_prep.py first)
#
# Usage:
#   uv run experiments/rerun/scripts/rba/rba_rerun.py --seed 42
#   uv run experiments/rerun/scripts/rba/rba_rerun.py --seed 42 --smoke
#   uv run experiments/rerun/scripts/rba/rba_rerun.py --seed 42 --full

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from gensim.models import FastText
from sklearn.metrics import roc_auc_score, average_precision_score
import polars as pl
from collections import defaultdict

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent.parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "rba" / "rba.parquet"
SEEDS_ROOT   = Path(__file__).resolve().parent.parent.parent / "seeds"

N_BOOTSTRAP       = 1000
MIN_TRAIN_EVENTS  = 5   # lowered from 10: RBA ATO users have few events; 10 leaves only 2 in eval
N_BENIGN_USERS    = 50_000   # benign-only users to include in default mode
SMOKE_N_USERS     = 5_000    # random users for smoke test; verifies pipeline (AUC may be NaN)

# Feature schema — schema-driven, no closed vocabulary
FEATURE_ORDER = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]
DEVICE_KEY_FEATURES = ["os", "browser", "device_type", "country", "region", "asn_bucket"]

# Base config — seed injected per-run in main()
ROBUST_KWARGS_BASE = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    workers=1,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_token(f, v):
    return f"{f}_{v}"

def device_key(d):
    return tuple(str(d.get(f) or "missing") for f in DEVICE_KEY_FEATURES)

def device_to_tokens(d):
    return [make_token(f, d.get(f) or "missing") for f in FEATURE_ORDER]

def device_to_concat(d):
    return "_".join(str(d.get(f) or "missing") for f in FEATURE_ORDER)

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_centroid(events, embed_fn, model):
    return np.mean([embed_fn(e, model) for e in events], axis=0)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_users(parquet_path: Path, smoke: bool, full: bool, seed: int, split_pct: float = 0.50) -> dict:
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

    # Smoke: random early user sample before split — fast pipeline check (AUC may be NaN)
    rng = np.random.default_rng(seed)
    if smoke:
        all_uids = df["user_id"].unique().to_list()
        keep_smoke = set(
            rng.choice(all_uids, min(SMOKE_N_USERS, len(all_uids)), replace=False).tolist()
        )
        df = df.filter(pl.col("user_id").is_in(keep_smoke))
        print(f"  Smoke pre-filter: {len(keep_smoke):,} users")

    # Temporal split: train = earliest 50%, test = last 50%.
    # All 141 ATO events occur before the 70th percentile — the original 80/20
    # split put every ATO event in the training window.  50/50 preserves ~34
    # ATO events in the test split while still giving each user a meaningful
    # training history.
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
        print(f"  Subsampled: {len(ato_user_ids):,} ATO users + {len(sampled_benign):,} benign users")

    feat_cols = [c for c in FEATURE_ORDER if c in df.columns]
    if len(feat_cols) < len(FEATURE_ORDER):
        missing_cols = set(FEATURE_ORDER) - set(feat_cols)
        print(f"  WARNING: missing columns {missing_cols}; those features will use 'missing' tokens")

    # Build training records: batch to_dicts() then group in Python — fast single pass
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
        users[uid] = {
            "train": events,
            "test_pos": [],
            "test_neg": [],
            "known_devices": {device_key(e) for e in events},
        }

    # Test records: fill is_ato (Boolean) separately from string feature columns
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
    print(f"  Eligible users: {len(users):,}  (dropped {dropped:,} below min-train floor)")
    print(f"  ATO users in eval: {ato_users:,}")
    print(f"  Test events: {total_pos:,} ATO (pos)  +  {total_neg:,} benign (neg)")
    if total_pos == 0:
        print("  WARNING: no ATO test events — AUC will be undefined.")
    return users

# ---------------------------------------------------------------------------
# Corpus construction (per-account, identical invariant to H2_RERUN)
# ---------------------------------------------------------------------------
def build_mp_corpus(users: dict) -> list:
    sentences = []
    for u in users.values():
        flat = []
        for e in u["train"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences

def build_cat_corpus(users: dict) -> list:
    sentences = []
    for u in users.values():
        sent = [device_to_concat(e) for e in u["train"]]
        if sent:
            sentences.append(sent)
    return sentences

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_mp(event: dict, model) -> np.ndarray:
    vecs = [model.wv[make_token(f, event.get(f) or "missing")] for f in FEATURE_ORDER]
    return np.mean(vecs, axis=0)

def embed_cat(event: dict, model) -> np.ndarray:
    return model.wv[device_to_concat(event)]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_all(users: dict, embed_fn, model) -> tuple[list, list]:
    scores, labels = [], []
    for u in users.values():
        centroid = compute_centroid(u["train"], embed_fn, model)
        for ev in u["test_pos"]:
            scores.append(cosine_dist(embed_fn(ev, model), centroid))
            labels.append(1)
        for ev in u["test_neg"]:
            scores.append(cosine_dist(embed_fn(ev, model), centroid))
            labels.append(0)
    return scores, labels

def score_trivial(users: dict) -> tuple[list, list]:
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
def threshold_metrics_at_top_pct(scores: list, labels: list, top_pct: float = 0.01) -> dict:
    sc = np.array(scores)
    lb = np.array(labels)
    cutoff = np.percentile(sc, 100 * (1 - top_pct))
    preds = (sc >= cutoff).astype(int)
    tp = int(((preds == 1) & (lb == 1)).sum())
    fp = int(((preds == 1) & (lb == 0)).sum())
    fn = int(((preds == 0) & (lb == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(cutoff), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "flagged": int(preds.sum()), "total": len(sc),
    }

def bootstrap_metrics(
    scores: list, labels: list, n_boot: int, seed: int
) -> tuple:
    """Returns: roc_mean, roc_lo, roc_hi, pr_mean, pr_lo, pr_hi"""
    rng = np.random.default_rng(seed)
    sc  = np.array(scores)
    lb  = np.array(labels)
    n   = len(sc)
    roc_aucs, pr_aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(lb[idx])) < 2:
            continue
        roc_aucs.append(roc_auc_score(lb[idx], sc[idx]))
        pr_aucs.append(average_precision_score(lb[idx], sc[idx]))
    if not roc_aucs:
        return (float("nan"),) * 6
    ra, pa = np.array(roc_aucs), np.array(pr_aucs)
    return (
        float(np.mean(ra)), float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5)),
        float(np.mean(pa)), float(np.percentile(pa, 2.5)), float(np.percentile(pa, 97.5)),
    )

# ---------------------------------------------------------------------------
# T6: Per-user centroid compactness
# ---------------------------------------------------------------------------
def per_user_compactness(users: dict, embed_fn, model, seed: int, max_users: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uids = list(users.keys())
    if len(uids) > max_users:
        uids = rng.choice(uids, max_users, replace=False).tolist()
    dists = []
    for uid in uids:
        u = users[uid]
        centroid = compute_centroid(u["train"], embed_fn, model)
        d = [cosine_dist(embed_fn(e, model), centroid) for e in u["train"]]
        dists.append(float(np.mean(d)))
    return np.array(dists)

def bootstrap_mean_compactness(
    users: dict, embed_fn, model, n_boot: int, seed: int
) -> tuple[float, float, float]:
    dists = per_user_compactness(users, embed_fn, model, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(dists)
    means = [float(np.mean(dists[rng.integers(0, n, size=n)])) for _ in range(n_boot)]
    return float(np.mean(dists)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ---------------------------------------------------------------------------
# T8: Within- vs. cross-feature token similarity
# ---------------------------------------------------------------------------
def token_similarity_analysis(model, seed: int, max_pairs: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    token_feats = {}
    for token in model.wv.key_to_index:
        for f in FEATURE_ORDER:
            if token.startswith(f + "_"):
                token_feats[token] = f
                break
    tokens = list(token_feats.keys())
    if len(tokens) < 2:
        return np.array([]), np.array([])
    vecs  = np.array([model.wv[t] for t in tokens])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / (norms + 1e-10)
    sim_mat = vecs_n @ vecs_n.T
    feat_arr = np.array([token_feats[t] for t in tokens])
    n = len(tokens)
    within, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if feat_arr[i] == feat_arr[j]:
                within.append(float(sim_mat[i, j]))
            else:
                cross.append(float(sim_mat[i, j]))
    rng = np.random.default_rng(seed)
    if len(within) > max_pairs:
        within = rng.choice(within, max_pairs, replace=False).tolist()
    if len(cross) > max_pairs:
        cross = rng.choice(cross, max_pairs, replace=False).tolist()
    return np.array(within), np.array(cross)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name: str, figures_dir: Path) -> None:
    out = figures_dir / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")

def plot_summary_auc(
    mp_roc, mp_roc_lo, mp_roc_hi,
    cat_roc, cat_roc_lo, cat_roc_hi,
    triv_roc,
    mp_pr, mp_pr_lo, mp_pr_hi,
    cat_pr, cat_pr_lo, cat_pr_hi,
    triv_pr,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels  = ["mean-pool", "concat", "trivial"]
    colors  = ["#2196F3", "#FF5722", "#9E9E9E"]
    roc_vals = [mp_roc, cat_roc, triv_roc]
    roc_cis  = [(mp_roc_lo, mp_roc_hi), (cat_roc_lo, cat_roc_hi), (None, None)]
    pr_vals  = [mp_pr, cat_pr, triv_pr]
    pr_cis   = [(mp_pr_lo, mp_pr_hi), (cat_pr_lo, cat_pr_hi), (None, None)]
    for ax, vals, cis, metric in zip(axes, [roc_vals, pr_vals], [roc_cis, pr_cis], ["ROC-AUC", "PR-AUC"]):
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors, alpha=0.85)
        for i, (v, (lo, hi)) in enumerate(zip(vals, cis)):
            if lo is not None:
                ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none",
                            color="black", capsize=4, linewidth=1.3)
            ax.text(i, (hi or v) + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric)
        ax.set_title(f"RBA Rerun: {metric}\n(sg=1, per-account corpus, binary ATO)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    save_fig(fig, "rba_summary_auc.png", figures_dir)

def plot_pr_curve(mp_scores, cat_scores, triv_scores, labels, figures_dir: Path) -> None:
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = np.mean(labels)
    ax.axhline(baseline, color="gray", linestyle=":", linewidth=0.8,
               label=f"random baseline (AP={baseline:.4f})")
    for sc, name, color in zip(
        [mp_scores, cat_scores, triv_scores],
        ["mean-pool", "concat", "trivial"],
        ["#2196F3", "#FF5722", "#9E9E9E"],
    ):
        p, r, _ = precision_recall_curve(labels, sc)
        ap = average_precision_score(labels, sc)
        ax.plot(r, p, color=color, linewidth=1.5, label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("RBA Rerun: Precision-Recall Curve\n(Is Account Takeover)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_fig(fig, "rba_pr_curve.png", figures_dir)

def plot_t6_compactness(
    mp_m, mp_lo, mp_hi, cat_m, cat_lo, cat_hi, figures_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.array([0, 1])
    ax.bar(x, [mp_m, cat_m], color=["#2196F3", "#FF5722"], alpha=0.85)
    ax.errorbar(0, mp_m, yerr=[[mp_m - mp_lo], [mp_hi - mp_m]],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.errorbar(1, cat_m, yerr=[[cat_m - cat_lo], [cat_hi - cat_m]],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.text(0, mp_hi + 0.002, f"{mp_m:.4f}", ha="center", fontsize=10)
    ax.text(1, cat_hi + 0.002, f"{cat_m:.4f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["mean-pool", "concat"])
    ax.set_ylabel("Per-user centroid compactness\n(lower = tighter cluster)")
    ax.set_title("T6: Per-User Compactness\n(RBA dataset, sg=1 per-account corpus)")
    plt.tight_layout()
    save_fig(fig, "rba_t6_compactness.png", figures_dir)

def plot_t8_token_similarity(within_mp, cross_mp, within_cat, cross_cat, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (within, cross, title) in zip(axes, [
        (within_mp, cross_mp, "mean-pool model"),
        (within_cat, cross_cat, "concat model"),
    ]):
        if len(within) and len(cross):
            ax.hist(within, bins=40, color="#2196F3", alpha=0.75,
                    label=f"Within-feature (μ={within.mean():.4f})")
            ax.hist(cross,  bins=40, color="#9E9E9E", alpha=0.55,
                    label=f"Cross-feature  (μ={cross.mean():.4f})")
            ax.axvline(within.mean(), color="#2196F3", linestyle="--", linewidth=1.5)
            ax.axvline(cross.mean(),  color="#9E9E9E", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        ax.set_xlim(-1.1, 1.1)
        ax.set_title(f"T8: Token Similarity\n{title}")
        ax.legend(fontsize=8)
    fig.suptitle("T8: Within- vs. Cross-Feature Embedding Similarity (RBA)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "rba_t8_token_similarity.png", figures_dir)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_split(
    users: dict,
    mp_model,
    cat_model,
    split_pct: float,
    seed: int,
    n_bootstrap: int,
) -> dict:
    """Score a test split and return Section 4 RBA metrics block."""
    mp_scores,   mp_labels   = score_all(users, embed_mp,  mp_model)
    cat_scores,  cat_labels  = score_all(users, embed_cat, cat_model)
    triv_scores, triv_labels = score_trivial(users)
    assert mp_labels == triv_labels, "label vectors mismatched"

    mp_roc,  mp_roc_lo,  mp_roc_hi,  mp_pr,  _,  _ = bootstrap_metrics(
        mp_scores,   mp_labels,   n_boot=n_bootstrap, seed=seed)
    cat_roc, cat_roc_lo, cat_roc_hi, cat_pr, _,  _ = bootstrap_metrics(
        cat_scores,  cat_labels,  n_boot=n_bootstrap, seed=seed)
    triv_roc, _, _, triv_pr, _, _ = bootstrap_metrics(
        triv_scores, triv_labels, n_boot=n_bootstrap, seed=seed)

    n_ato = sum(len(u["test_pos"]) for u in users.values())
    # CI-lower-bound criterion per decision 9511d90f
    h2_replicated = bool(mp_roc_lo > triv_roc)

    return {
        "split_percentile": int(split_pct * 100),
        "n_ato_test_events": n_ato,
        "auc": {
            "mean_pool": {
                "roc_auc":  mp_roc,
                "pr_auc":   mp_pr,
                "ci_lower": mp_roc_lo,
                "ci_upper": mp_roc_hi,
            },
            "concat": {
                "roc_auc":  cat_roc,
                "pr_auc":   cat_pr,
                "ci_lower": cat_roc_lo,
                "ci_upper": cat_roc_hi,
            },
            "trivial": {
                "roc_auc": triv_roc,
                "pr_auc":  triv_pr,
            },
        },
        "h2_replicated": h2_replicated,
        "_mp_roc_lo": mp_roc_lo,   # retained for abort check only; not in schema
        "_triv_roc":  triv_roc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RBA rerun of H2 robust config (seed-parameterized)")
    parser.add_argument("--seed", type=int, required=True,
                        help="random seed for FastText and bootstrap sampling")
    parser.add_argument("--smoke", action="store_true",
                        help=f"fast pipeline check on {SMOKE_N_USERS} random users (AUC may be NaN)")
    parser.add_argument("--full",  action="store_true",
                        help="use all eligible users (no subsampling)")
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH,
                        help="path to rba.parquet (default: data/rba/rba.parquet)")
    args = parser.parse_args()

    seed = args.seed
    n_bootstrap = 200 if args.smoke else N_BOOTSTRAP

    if not args.parquet.exists():
        print(f"ERROR: parquet not found: {args.parquet}")
        print("Run: uv run experiments/rerun/scripts/rba/data_prep.py")
        raise SystemExit(1)

    seed_dir = SEEDS_ROOT / f"seed_{seed}" / "rba"
    seed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = seed_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    robust_kwargs = {**ROBUST_KWARGS_BASE, "seed": seed}

    tag = "smoke" if args.smoke else ("full" if args.full else "default")
    print("=" * 65)
    print(f"RBA Rerun — H2 Replication  [seed={seed}  mode={tag}]")
    print("sg=1, per-account corpus, binary ATO label")
    print("=" * 65)

    # ── 1. Load data on primary 50/50 split (training set is fixed across sensitivity runs)
    print("\n[1/7] Loading data (primary 50/50 split) ...")
    users_50 = load_users(args.parquet, smoke=args.smoke, full=args.full,
                          seed=seed, split_pct=0.50)

    # ── 2. Build corpora from primary training split
    print("\n[2/7] Building corpora (per-account) ...")
    mp_corp  = build_mp_corpus(users_50)
    cat_corp = build_cat_corpus(users_50)
    print(f"  mp  corpus: {len(mp_corp):,} sentences, "
          f"{sum(len(s) for s in mp_corp):,} tokens")
    print(f"  cat corpus: {len(cat_corp):,} sentences, "
          f"{sum(len(s) for s in cat_corp):,} tokens")

    # ── 3. Train FastText once on primary training split
    print(f"\n[3/7] Training FastText models (robust config, seed={seed}) ...")
    print(f"  kwargs: {robust_kwargs}")
    if not mp_corp:
        raise RuntimeError(
            f"Empty mp corpus — check load_users() output above. "
            f"users={len(users_50)}, "
            f"ato_users={sum(1 for u in users_50.values() if u['test_pos'])}"
        )
    mp_model  = FastText(sentences=mp_corp,  **robust_kwargs)
    cat_model = FastText(sentences=cat_corp, **robust_kwargs)
    print(f"  Vocabulary: {len(mp_model.wv):,} tokens (mp), "
          f"{len(cat_model.wv):,} tokens (cat)")

    # ── 4. T6 compactness and T8 token similarity (model-level, not split-dependent)
    print("\n[4/7] T6 compactness, T8 token similarity ...")
    mp_c,  mp_c_lo,  mp_c_hi  = bootstrap_mean_compactness(
        users_50, embed_mp,  mp_model,  n_boot=n_bootstrap, seed=seed)
    cat_c, cat_c_lo, cat_c_hi = bootstrap_mean_compactness(
        users_50, embed_cat, cat_model, n_boot=n_bootstrap, seed=seed)
    print(f"  T6 mean-pool compactness: {mp_c:.4f} [{mp_c_lo:.4f}, {mp_c_hi:.4f}]")
    print(f"  T6 concat    compactness: {cat_c:.4f} [{cat_c_lo:.4f}, {cat_c_hi:.4f}]")

    within_mp,  cross_mp  = token_similarity_analysis(mp_model,  seed=seed)
    within_cat, cross_cat = token_similarity_analysis(cat_model, seed=seed)

    w_mp_mean  = float(within_mp.mean())  if len(within_mp)  else float("nan")
    cr_mp_mean = float(cross_mp.mean())   if len(cross_mp)   else float("nan")
    # Canonical collapse threshold = 0.9 (decision a2b73375)
    collapse = w_mp_mean > 0.9
    if not (w_mp_mean != w_mp_mean):   # not NaN
        print(f"  T8 mean-pool within-feature sim: {w_mp_mean:.4f} "
              f"({'COLLAPSE — abort!' if collapse else 'ok'})")
        print(f"  T8 mean-pool cross-feature  sim: {cr_mp_mean:.4f}")
        if collapse:
            print("  ABORT: T8 collapse detected under robust config — "
                  "debug corpus construction before continuing.")
            raise SystemExit(1)

    t8_block = {
        "within_feature_mean": w_mp_mean,
        "cross_feature_mean":  cr_mp_mean,
        "within_cross_ratio":  (w_mp_mean / cr_mp_mean) if cr_mp_mean > 1e-10 else float("nan"),
        "collapse_detected":   collapse,
    }
    t6_block = {
        "mean_pool": mp_c,
        "concat":    cat_c,
    }

    # ── 5. Primary 50/50 metrics
    print("\n[5/7] Primary split (50/50) metrics ...")
    primary = _run_split(users_50, mp_model, cat_model, 0.50, seed, n_bootstrap)
    print(f"  mean-pool ROC-AUC: {primary['auc']['mean_pool']['roc_auc']:.4f}  "
          f"CI: [{primary['auc']['mean_pool']['ci_lower']:.4f}, "
          f"{primary['auc']['mean_pool']['ci_upper']:.4f}]")
    print(f"  trivial   ROC-AUC: {primary['auc']['trivial']['roc_auc']:.4f}")
    print(f"  h2_replicated (CI-lower > trivial): {primary['h2_replicated']}")

    # ── 6. Sensitivity splits (same models, different test windows)
    print("\n[6/7] Sensitivity splits (40/60 and 60/40, same trained model) ...")
    users_40 = load_users(args.parquet, smoke=args.smoke, full=args.full,
                          seed=seed, split_pct=0.40)
    split40 = _run_split(users_40, mp_model, cat_model, 0.40, seed, n_bootstrap)

    users_60 = load_users(args.parquet, smoke=args.smoke, full=args.full,
                          seed=seed, split_pct=0.60)
    split60 = _run_split(users_60, mp_model, cat_model, 0.60, seed, n_bootstrap)

    # ── 7. Save figures and results
    print("\n[7/7] Saving figures and results ...")
    plot_summary_auc(
        primary["auc"]["mean_pool"]["roc_auc"],
        primary["auc"]["mean_pool"]["ci_lower"],
        primary["auc"]["mean_pool"]["ci_upper"],
        primary["auc"]["concat"]["roc_auc"],
        primary["auc"]["concat"]["ci_lower"],
        primary["auc"]["concat"]["ci_upper"],
        primary["auc"]["trivial"]["roc_auc"],
        primary["auc"]["mean_pool"]["pr_auc"],
        None, None,
        primary["auc"]["concat"]["pr_auc"],
        None, None,
        primary["auc"]["trivial"]["pr_auc"],
        figures_dir=figures_dir,
    )

    mp_scores,  _           = score_all(users_50, embed_mp,  mp_model)
    cat_scores, _           = score_all(users_50, embed_cat, cat_model)
    triv_scores, triv_labels = score_trivial(users_50)
    plot_pr_curve(mp_scores, cat_scores, triv_scores, triv_labels,
                  figures_dir=figures_dir)
    plot_t6_compactness(mp_c, mp_c_lo, mp_c_hi, cat_c, cat_c_lo, cat_c_hi,
                        figures_dir=figures_dir)
    if len(within_mp) and len(within_cat):
        plot_t8_token_similarity(within_mp, cross_mp, within_cat, cross_cat,
                                 figures_dir=figures_dir)

    timestamp = datetime.now(timezone.utc).isoformat()

    def _make_results(split_block: dict) -> dict:
        out = {
            "seed": seed,
            "timestamp": timestamp,
            "split_percentile": split_block["split_percentile"],
            "n_ato_test_events": split_block["n_ato_test_events"],
            "auc": split_block["auc"],
            "t6_compactness": t6_block,
            "t8_token_similarity": t8_block,
            "h2_replicated": split_block["h2_replicated"],
        }
        return out

    results_primary = _make_results(primary)
    results_40      = _make_results(split40)
    results_60      = _make_results(split60)

    def _write_json(obj: dict, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"  Wrote: {path}")

    _write_json(results_primary, seed_dir / "results.json")
    _write_json(results_40,      seed_dir / "results_split40.json")
    _write_json(results_60,      seed_dir / "results_split60.json")

    verdict = "REPLICATED" if results_primary["h2_replicated"] else "NOT REPLICATED"
    print(f"\n  H2 finding {verdict} on real RBA data (seed={seed}).")
    if not results_primary["h2_replicated"]:
        raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()
