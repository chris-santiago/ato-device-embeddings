# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
#   "matplotlib>=3.7",
#   "polars>=0.20",
#   "torch>=2.0",
# ]
# ///
#
# GRU Temporal Sequence Experiment — H4
# ======================================
# Tests whether a GRU trained on benign event sequences improves ATO detection
# over static mean-pool cosine distance by modeling temporal patterns.
#
# Scoring variants:
#   mp-raw-split  — mean-pool cosine dist (centroid from centroid_evts); H3 baseline
#   recency-mean  — exponentially-weighted mean-pool (λ=0.5); cheap temporal ablation
#   gru-predict   — GRU trained to predict next embedding; cosine dist vs prediction
#   trivial       — exact device-key set-membership
#
# GRU design:
#   1-layer GRU, hidden_dim=64, input_dim=64, nn.Linear(64, 64) projection
#   Trained on benign sequences only (no labeled ATOs available at train time)
#   Loss: 1 - mean(cosine_similarity(pred, target))  — teacher forcing
#   Input = seq[:-1], target = seq[1:]
#   90/10 train/val split on users; early stopping patience=5, max_epochs=50
#
# Scoring: feed centroid_evts through GRU → project final hidden state →
#          cosine_dist(projected_state, embed_mp(test_event))
#
# Primary metric: ROC-AUC (n=9 ATO test events; top-1% metrics also reported)
#
# Usage:
#   uv run h4_gru/experiments/gru_experiment.py
#   uv run h4_gru/experiments/gru_experiment.py --smoke
#   uv run h4_gru/experiments/gru_experiment.py --full

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "rba" / "rba.parquet"
FIGURES_DIR  = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP      = 1000
MIN_TRAIN_EVENTS = 5       # raising eliminates all ATO test users (they have 5-9 train events)
N_BENIGN_USERS   = 50_000
SMOKE_N_USERS    = 5_000

FEATURE_ORDER       = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]
DEVICE_KEY_FEATURES = ["os", "browser", "device_type", "country", "region", "asn_bucket"]

ROBUST_KWARGS = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    seed=SEED, workers=1,
)

GRU_INPUT_DIM  = 64
GRU_HIDDEN_DIM = 64
GRU_LAMBDA     = 0.5   # recency decay rate
GRU_MAX_EPOCHS = 50
GRU_PATIENCE   = 5
GRU_LR         = 1e-3
GRU_BATCH      = 64

COLORS = {
    "mp-raw-split": "#9E9E9E",
    "recency-mean": "#2196F3",
    "gru-predict":  "#4CAF50",
    "trivial":      "#607D8B",
}


# ---------------------------------------------------------------------------
# GRU model
# ---------------------------------------------------------------------------
class GRUPredictor(nn.Module):
    def __init__(self, input_dim: int = GRU_INPUT_DIM, hidden_dim: int = GRU_HIDDEN_DIM):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D) next-step predictions via teacher forcing."""
        out, _ = self.gru(x)
        return self.proj(out)

    def encode(self, x: torch.Tensor) -> np.ndarray:
        """Feed sequence x → project final hidden state → numpy (D,).
        Used at scoring time to predict the next event embedding."""
        _, h = self.gru(x)                          # h: (1, 1, H)
        pred = self.proj(h.squeeze(0).squeeze(0))   # (D,)
        return pred.cpu().numpy()


# ---------------------------------------------------------------------------
# Calibration split + helpers (identical to H3)
# ---------------------------------------------------------------------------
def split_calib(events: list) -> tuple[list, list]:
    n = len(events)
    n_calib = max(2, n // 3)
    n_centroid = max(1, n - n_calib)
    return events[:n_centroid], events[n_centroid:]

def add_tie_break(scores: np.ndarray, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return scores + rng.uniform(0, 1e-9, size=len(scores))

def make_token(f: str, v) -> str:
    return f"{f}_{v}"

def device_key(d: dict) -> tuple:
    return tuple(str(d.get(f) or "missing") for f in DEVICE_KEY_FEATURES)

def device_to_tokens(d: dict) -> list[str]:
    return [make_token(f, d.get(f) or "missing") for f in FEATURE_ORDER]

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))

def embed_mp(event: dict, model) -> np.ndarray:
    """Mean-pool all feature token vectors into one 64-dim embedding."""
    vecs = [model.wv[make_token(f, event.get(f) or "missing")] for f in FEATURE_ORDER]
    return np.mean(vecs, axis=0)


# ---------------------------------------------------------------------------
# Data loading (same structure as H3; temporal order guaranteed by global sort)
# ---------------------------------------------------------------------------
def load_users(parquet_path: Path, smoke: bool, full: bool,
               seed: int = SEED, split_pct: float = 0.50) -> dict:
    print(f"  Loading parquet: {parquet_path}")
    raw = pl.read_parquet(parquet_path)
    if raw["login_timestamp"].dtype not in (pl.Int64, pl.Float64, pl.Int32):
        raise RuntimeError(
            f"login_timestamp dtype: {raw['login_timestamp'].dtype} — expected numeric"
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
        keep = set(rng.choice(all_uids, min(SMOKE_N_USERS, len(all_uids)), replace=False).tolist())
        df = df.filter(pl.col("user_id").is_in(keep))
        print(f"  Smoke: {len(keep):,} users")

    cutoff = df["login_timestamp"].quantile(split_pct, interpolation="nearest")
    train_df = df.filter(pl.col("login_timestamp") <= cutoff)
    test_df  = df.filter(pl.col("login_timestamp") >  cutoff)
    print(f"  Split at {cutoff}: {len(train_df):,} train / {len(test_df):,} test rows")

    ato_user_ids = set(test_df.filter(pl.col("is_ato"))["user_id"].to_list())
    print(f"  ATO users with test events: {len(ato_user_ids):,}")

    if not smoke and not full:
        benign_pool = sorted(set(train_df["user_id"].unique().to_list()) - ato_user_ids)
        sampled_benign = set(
            rng.choice(benign_pool, min(N_BENIGN_USERS, len(benign_pool)), replace=False)
        )
        keep = ato_user_ids | sampled_benign
        train_df = train_df.filter(pl.col("user_id").is_in(keep))
        test_df  = test_df.filter(pl.col("user_id").is_in(keep))
        print(f"  Subsampled: {len(ato_user_ids):,} ATO + {len(sampled_benign):,} benign users")

    feat_cols = [c for c in FEATURE_ORDER if c in df.columns]

    # Include login_timestamp to assert temporal order per user
    train_rows = (
        train_df
        .filter(~pl.col("is_ato"))
        .with_columns([pl.col(c).cast(pl.Utf8).fill_null("missing") for c in feat_cols])
        .select(["user_id", "login_timestamp"] + feat_cols)
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
        ts_list = [e["login_timestamp"] for e in events]
        assert ts_list == sorted(ts_list), f"Events for user {uid} not in temporal order"
        events_clean = [{k: v for k, v in e.items() if k != "login_timestamp"} for e in events]
        centroid_evts, calib_evts = split_calib(events_clean)
        users[uid] = {
            "train":         events_clean,
            "centroid_evts": centroid_evts,
            "calib_evts":    calib_evts,
            "test_pos":      [],
            "test_neg":      [],
            "known_devices": {device_key(e) for e in events_clean},
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
        (users[uid]["test_pos"] if is_ato else users[uid]["test_neg"]).append(row)

    users = {uid: u for uid, u in users.items() if u["test_pos"] or u["test_neg"]}
    total_pos = sum(len(u["test_pos"]) for u in users.values())
    total_neg = sum(len(u["test_neg"]) for u in users.values())
    ato_users = sum(1 for u in users.values() if u["test_pos"])
    print(f"  Eligible: {len(users):,} users (dropped {dropped:,} below {MIN_TRAIN_EVENTS}-event floor)")
    print(f"  ATO users in eval: {ato_users:,}")
    print(f"  Test events: {total_pos:,} ATO + {total_neg:,} benign")
    return users


# ---------------------------------------------------------------------------
# FastText corpus (same as H3 — per-account concatenated, sg=1)
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
# GRU dataset and training
# ---------------------------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs: list[np.ndarray]):
        self.seqs = [s for s in seqs if len(s) >= 2]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.seqs[idx]
        return (
            torch.tensor(s[:-1], dtype=torch.float32),
            torch.tensor(s[1:],  dtype=torch.float32),
        )


def collate_pad(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length (input, target) pairs; return (inp, tgt, mask)."""
    inputs, targets = zip(*batch)
    max_len = max(x.shape[0] for x in inputs)
    D, B    = inputs[0].shape[1], len(inputs)
    inp_pad = torch.zeros(B, max_len, D)
    tgt_pad = torch.zeros(B, max_len, D)
    mask    = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = inp.shape[0]
        inp_pad[i, :L] = inp
        tgt_pad[i, :L] = tgt
        mask[i, :L]    = True
    return inp_pad, tgt_pad, mask


def build_train_seqs(users: dict, model_ft) -> list[np.ndarray]:
    """Embed each user's full training sequence as (L, 64) float array."""
    return [
        np.array([embed_mp(e, model_ft) for e in u["train"]])
        for u in users.values()
        if len(u["train"]) >= 2
    ]


def train_gru(users: dict, model_ft, device: str = "cpu") -> tuple[GRUPredictor, dict]:
    """Train GRU on benign training sequences; return (model, loss_history)."""
    print("  Embedding training sequences ...")
    seqs = build_train_seqs(users, model_ft)
    seq_lens = [len(s) for s in seqs]
    print(f"  {len(seqs):,} sequences | min={min(seq_lens)} "
          f"median={int(np.median(seq_lens))} max={max(seq_lens)}")

    rng   = np.random.default_rng(SEED)
    perm  = rng.permutation(len(seqs))
    n_trn = int(0.9 * len(seqs))
    trn_seqs = [seqs[i] for i in perm[:n_trn]]
    val_seqs = [seqs[i] for i in perm[n_trn:]]

    trn_dl = DataLoader(
        SeqDataset(trn_seqs), batch_size=GRU_BATCH, shuffle=True, collate_fn=collate_pad,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_dl = DataLoader(
        SeqDataset(val_seqs), batch_size=GRU_BATCH, shuffle=False, collate_fn=collate_pad,
    )

    gru = GRUPredictor().to(device)
    opt = torch.optim.Adam(gru.parameters(), lr=GRU_LR)

    best_val   = float("inf")
    pat_count  = 0
    best_state = None
    trn_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(GRU_MAX_EPOCHS):
        gru.train()
        t_loss = 0.0
        for inp, tgt, mask in trn_dl:
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            pred = gru(inp)
            loss = (1.0 - F.cosine_similarity(pred, tgt, dim=-1))[mask].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_loss += loss.item()

        gru.train(False)   # equivalent to gru.eval()
        v_loss = 0.0
        with torch.no_grad():
            for inp, tgt, mask in val_dl:
                inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                pred   = gru(inp)
                v_loss += (1.0 - F.cosine_similarity(pred, tgt, dim=-1))[mask].mean().item()

        avg_t = t_loss / max(1, len(trn_dl))
        avg_v = v_loss / max(1, len(val_dl))
        trn_history.append(avg_t)
        val_history.append(avg_v)

        if avg_v < best_val - 1e-4:
            best_val  = avg_v
            pat_count = 0
            best_state = {k: v.cpu().clone() for k, v in gru.state_dict().items()}
        else:
            pat_count += 1

        if (epoch + 1) % 5 == 0 or pat_count >= GRU_PATIENCE:
            print(f"  epoch {epoch+1:3d} | train={avg_t:.4f} | val={avg_v:.4f} "
                  f"| patience={pat_count}/{GRU_PATIENCE}")

        if pat_count >= GRU_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        gru.load_state_dict(best_state)
        print(f"  Loaded best weights (val_loss={best_val:.4f})")

    gru.to(device)
    return gru, {"train": trn_history, "val": val_history}


# ---------------------------------------------------------------------------
# Scoring variants
# ---------------------------------------------------------------------------
def score_mp_raw_split(users: dict, model_ft) -> tuple[list, list]:
    """Mean-pool cosine dist using centroid_evts — H3 fair baseline."""
    scores, labels = [], []
    for u in users.values():
        centroid = np.mean([embed_mp(e, model_ft) for e in u["centroid_evts"]], axis=0)
        for ev in u["test_pos"]:
            scores.append(cosine_dist(embed_mp(ev, model_ft), centroid))
            labels.append(1)
        for ev in u["test_neg"]:
            scores.append(cosine_dist(embed_mp(ev, model_ft), centroid))
            labels.append(0)
    return scores, labels


def score_recency_mean(users: dict, model_ft, lam: float = GRU_LAMBDA) -> tuple[list, list]:
    """Exponentially-weighted mean-pool: w_t = exp(-λ*(n-1-t)), most recent weight = 1."""
    scores, labels = [], []
    for u in users.values():
        evts = u["centroid_evts"]
        n    = len(evts)
        w    = np.array([np.exp(-lam * (n - 1 - t)) for t in range(n)])
        w   /= w.sum()
        embs = np.array([embed_mp(e, model_ft) for e in evts])
        centroid = (w[:, None] * embs).sum(axis=0)
        for ev in u["test_pos"]:
            scores.append(cosine_dist(embed_mp(ev, model_ft), centroid))
            labels.append(1)
        for ev in u["test_neg"]:
            scores.append(cosine_dist(embed_mp(ev, model_ft), centroid))
            labels.append(0)
    return scores, labels


def score_gru_predict(users: dict, model_ft, gru: GRUPredictor,
                      device: str = "cpu") -> tuple[list, list]:
    """GRU scorer: encode centroid_evts history → predict next → cosine_dist."""
    gru.train(False)   # inference mode
    scores, labels = [], []
    with torch.no_grad():
        for u in users.values():
            evts = u["centroid_evts"]
            if not evts:
                continue
            seq = torch.tensor(
                np.array([embed_mp(e, model_ft) for e in evts]),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)  # (1, L, D)
            pred_np = gru.encode(seq)  # (D,)
            for ev in u["test_pos"]:
                scores.append(cosine_dist(embed_mp(ev, model_ft), pred_np))
                labels.append(1)
            for ev in u["test_neg"]:
                scores.append(cosine_dist(embed_mp(ev, model_ft), pred_np))
                labels.append(0)
    return scores, labels


def score_trivial(users: dict) -> tuple[list, list]:
    scores, labels = [], []
    for u in users.values():
        for ev in u["test_pos"]:
            scores.append(0.0 if device_key(ev) in u["known_devices"] else 1.0)
            labels.append(1)
        for ev in u["test_neg"]:
            scores.append(0.0 if device_key(ev) in u["known_devices"] else 1.0)
            labels.append(0)
    return scores, labels


# ---------------------------------------------------------------------------
# Metrics (same as H3 — stratified bootstrap, tie-breaking perturbation)
# ---------------------------------------------------------------------------
def threshold_metrics_at_top_pct(scores: list, labels: list, top_pct: float = 0.01,
                                  tie_break_seed: int = SEED) -> dict:
    sc = add_tie_break(np.array(scores), seed=tie_break_seed)
    lb = np.array(labels)
    cutoff = np.percentile(sc, 100 * (1 - top_pct))
    preds  = (sc >= cutoff).astype(int)
    tp = int(((preds == 1) & (lb == 1)).sum())
    fp = int(((preds == 1) & (lb == 0)).sum())
    fn = int(((preds == 0) & (lb == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "threshold": float(cutoff), "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "flagged": int(preds.sum()),
        "expected_flagged": max(1, int(len(sc) * top_pct)),
        "total": len(sc),
    }


def bootstrap_metrics_stratified(scores: list, labels: list,
                                  n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple:
    rng = np.random.default_rng(seed)
    sc, lb = np.array(scores), np.array(labels)
    pos_idx = np.where(lb == 1)[0]
    neg_idx = np.where(lb == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return (float("nan"),) * 6
    roc_aucs, pr_aucs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
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
    pos_idx = np.where(lb == 1)[0]
    neg_idx = np.where(lb == 0)[0]
    if len(pos_idx) == 0:
        nan3 = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return {"precision": nan3, "recall": nan3, "f1": nan3}
    precisions, recalls, f1s = [], [], []
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
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
    def ci(arr: list) -> dict:
        return {
            "mean": float(np.mean(arr)),
            "lo":   float(np.percentile(arr, 2.5)),
            "hi":   float(np.percentile(arr, 97.5)),
        }
    return {"precision": ci(precisions), "recall": ci(recalls), "f1": ci(f1s)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name: str) -> None:
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")


def plot_summary_auc(results: dict) -> None:
    variants = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
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
            ax.text(i, (hi if not np.isnan(hi) else v) + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.18)
        label = "ROC-AUC" if metric == "roc_auc" else "PR-AUC"
        ax.set_ylabel(label)
        ax.set_title(f"H4 GRU: {label}\n(RBA dataset, stratified bootstrap N={N_BOOTSTRAP})")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    save_fig(fig, "gru_summary_auc.png")


def plot_top1pct(results: dict) -> None:
    variants = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
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
                    f"{v:.4f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(f"Top-1% {metric.capitalize()}")
        ax.set_title(f"H4 GRU: Top-1% {metric.capitalize()}")
        ax.axhline(0.0, color="gray", linewidth=0.5)
    plt.tight_layout()
    save_fig(fig, "gru_top1pct.png")


def plot_pr_curves(variants_raw: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    lb_ref   = list(variants_raw.values())[0][1]
    baseline = np.mean(lb_ref)
    ax.axhline(baseline, color="gray", linestyle=":", linewidth=0.8,
               label=f"random ({baseline:.4f})")
    for name, (sc, lb) in variants_raw.items():
        p, r, _ = precision_recall_curve(lb, sc)
        ap = average_precision_score(lb, sc)
        ax.plot(r, p, color=COLORS.get(name, "#777"), linewidth=1.5,
                label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("H4 GRU: Precision-Recall Curves\n(RBA dataset, ATO label)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_fig(fig, "gru_pr_curves.png")


def plot_training_loss(loss_history: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history["train"], label="train", color="#4CAF50", linewidth=1.5)
    ax.plot(loss_history["val"],   label="val",   color="#E91E63", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("1 − mean cosine similarity")
    ax.set_title("H4 GRU: Training Loss")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "gru_training_loss.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="H4 GRU temporal sequence experiment")
    parser.add_argument("--smoke",   action="store_true", help=f"fast check on {SMOKE_N_USERS} users")
    parser.add_argument("--full",    action="store_true", help="use all eligible users")
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH)
    parser.add_argument("--split-pct", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (cpu / cuda / mps)")
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"ERROR: parquet not found: {args.parquet}")
        print("Run: uv run h2_rba/experiments/data_prep.py")
        return

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("  MPS not available, falling back to cpu")
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available, falling back to cpu")
        device = "cpu"

    tag = "smoke" if args.smoke else ("full" if args.full else "default")
    print("=" * 72)
    print("H4 GRU Temporal Sequence Experiment")
    print(f"mode={tag} | sg=1 FastText | GRU hidden_dim={GRU_HIDDEN_DIM} | device={device}")
    print("Variants: mp-raw-split (H3 baseline), recency-mean (λ=0.5), gru-predict, trivial")
    print("=" * 72)

    t0 = time.time()

    print("\n[1/6] Loading data ...")
    users = load_users(args.parquet, smoke=args.smoke, full=args.full,
                       split_pct=args.split_pct)

    calib_sizes    = [len(u["calib_evts"])    for u in users.values()]
    centroid_sizes = [len(u["centroid_evts"]) for u in users.values()]
    n_ato_test     = sum(len(u["test_pos"]) for u in users.values())
    n_ben_test     = sum(len(u["test_neg"]) for u in users.values())
    print(f"\n[2/6] Data split stats ...")
    print(f"  Centroid evts — min: {min(centroid_sizes)}  "
          f"median: {int(np.median(centroid_sizes))}  max: {max(centroid_sizes)}")
    print(f"  Calib evts    — min: {min(calib_sizes)}  "
          f"median: {int(np.median(calib_sizes))}  max: {max(calib_sizes)}")
    print(f"  ATO test events: {n_ato_test} | benign test events: {n_ben_test:,}")

    print("\n[3/6] Training FastText (single model, per-account corpus) ...")
    corpus = build_corpus(users)
    print(f"  Corpus: {len(corpus):,} sentences  {sum(len(s) for s in corpus):,} tokens")
    model_ft = FastText(sentences=corpus, **ROBUST_KWARGS)
    print(f"  Vocabulary: {len(model_ft.wv):,} tokens")

    print("\n[4/6] Training GRU ...")
    gru, loss_history = train_gru(users, model_ft, device=device)

    print("\n[5/6] Scoring variants ...")
    print("  mp-raw-split ...")
    mp_s,  mp_l  = score_mp_raw_split(users, model_ft)
    print("  recency-mean ...")
    rec_s, rec_l = score_recency_mean(users, model_ft)
    print("  gru-predict ...")
    gru_s, gru_l = score_gru_predict(users, model_ft, gru, device=device)
    print("  trivial ...")
    tri_s, tri_l = score_trivial(users)
    assert mp_l == tri_l, "Label vectors mismatched across scorers"

    variants = {
        "mp-raw-split": (mp_s,  mp_l),
        "recency-mean": (rec_s, rec_l),
        "gru-predict":  (gru_s, gru_l),
        "trivial":      (tri_s, tri_l),
    }

    print("\n[6/6] Computing metrics and saving figures ...")
    results = {}
    for name, (sc, lb) in variants.items():
        roc_m, roc_lo, roc_hi, pr_m, pr_lo, pr_hi = bootstrap_metrics_stratified(sc, lb)
        top1  = threshold_metrics_at_top_pct(sc, lb)
        top1b = bootstrap_top1pct_stratified(sc, lb)
        results[name] = {
            "roc_auc": {"mean": roc_m, "lo": roc_lo, "hi": roc_hi},
            "pr_auc":  {"mean": pr_m,  "lo": pr_lo,  "hi": pr_hi},
            "top1pct": {
                "precision": top1b["precision"],
                "recall":    top1b["recall"],
                "f1":        top1b["f1"],
                "point":     top1,
            },
        }

    print(f"\n  {'Model':<18} {'ROC-AUC':>10} {'[95% CI]':>22}  {'PR-AUC':>8} {'[95% CI]':>22}")
    print("  " + "-" * 90)
    for name, r in results.items():
        roc, rlo, rhi = r["roc_auc"]["mean"], r["roc_auc"]["lo"], r["roc_auc"]["hi"]
        pr,  plo, phi = r["pr_auc"]["mean"],  r["pr_auc"]["lo"],  r["pr_auc"]["hi"]
        ci_r = f"[{rlo:.4f}, {rhi:.4f}]" if not np.isnan(rlo) else "      —      "
        ci_p = f"[{plo:.4f}, {phi:.4f}]" if not np.isnan(plo) else "      —      "
        print(f"  {name:<18} {roc:>10.4f} {ci_r:>22}  {pr:>8.4f} {ci_p:>22}")

    print(f"\n  Top-1% Threshold Metrics:")
    print(f"  {'Model':<18} {'TP':>4} {'FP':>6}  {'Prec':>7} {'[CI]':>16}  {'Rec':>7} {'[CI]':>16}")
    print("  " + "-" * 88)
    for name, r in results.items():
        pt   = r["top1pct"]["point"]
        p_ci = f"[{r['top1pct']['precision']['lo']:.3f},{r['top1pct']['precision']['hi']:.3f}]"
        r_ci = f"[{r['top1pct']['recall']['lo']:.3f},{r['top1pct']['recall']['hi']:.3f}]"
        print(f"  {name:<18} {pt['tp']:>4} {pt['fp']:>6}  "
              f"{pt['precision']:>7.4f} {p_ci:>16}  {pt['recall']:>7.4f} {r_ci:>16}")

    mp_roc  = results["mp-raw-split"]["roc_auc"]["mean"]
    gru_roc = results["gru-predict"]["roc_auc"]["mean"]
    hypothesis_confirmed = gru_roc > mp_roc
    print(f"\n  HEADLINE:")
    print(f"  mp-raw-split ROC-AUC : {mp_roc:.4f}")
    print(f"  gru-predict  ROC-AUC : {gru_roc:.4f}")
    print(f"  GRU beats baseline   : {hypothesis_confirmed}")
    print(f"  NOTE: n={n_ato_test} ATO test events — high-variance result.")

    plot_summary_auc(results)
    plot_top1pct(results)
    plot_pr_curves(variants)
    plot_training_loss(loss_history)

    elapsed = time.time() - t0
    metrics_out = {
        "version": "v1",
        "mode": tag,
        "split_pct": args.split_pct,
        "n_users": len(users),
        "n_ato_test": n_ato_test,
        "n_benign_test": n_ben_test,
        "gru_config": {
            "hidden_dim": GRU_HIDDEN_DIM,
            "input_dim":  GRU_INPUT_DIM,
            "max_epochs": GRU_MAX_EPOCHS,
            "patience":   GRU_PATIENCE,
            "lr":         GRU_LR,
            "batch_size": GRU_BATCH,
        },
        "recency_lambda": GRU_LAMBDA,
        "fasttext_kwargs": ROBUST_KWARGS,
        "results": {
            k: {
                "roc_auc":       v["roc_auc"],
                "pr_auc":        v["pr_auc"],
                "top1pct_point": v["top1pct"]["point"],
                "top1pct_boot":  {
                    "precision": v["top1pct"]["precision"],
                    "recall":    v["top1pct"]["recall"],
                    "f1":        v["top1pct"]["f1"],
                },
            }
            for k, v in results.items()
        },
        "hypothesis_confirmed": hypothesis_confirmed,
        "elapsed_s": round(elapsed, 1),
    }
    metrics_path = FIGURES_DIR / "gru_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics: {metrics_path}")
    print(f"\n  Elapsed: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
