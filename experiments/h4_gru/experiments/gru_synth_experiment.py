# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
#   "matplotlib>=3.7",
#   "torch>=2.0",
# ]
# ///
#
# GRU Temporal Sequence Experiment — H4 (Synthetic Dataset)
# ===========================================================
# Same GRU hypothesis as gru_experiment.py but run on the synthetic dataset
# from H2 variable_spoof_experiment.py. Synthetic data provides:
#   - 400 positives per spoof level (vs 9 in RBA) → tight bootstrap CIs
#   - Controlled attack difficulty: k=1 (tz only), k=2 (tz+network), k=3 (tz+network+screen)
#   - Reproducible generation without parquet dependency
#
# Scoring variants (same four as gru_experiment.py):
#   mp-raw-split  — mean-pool cosine dist using centroid_evts; H3/H4-RBA baseline
#   recency-mean  — exponentially-weighted mean-pool (λ=0.5)
#   gru-predict   — GRU trained to predict next embedding; cosine dist vs prediction
#   trivial       — exact device-key set-membership
#
# Synthetic data schema (6 features, matches H2 variable_spoof_experiment.py):
#   os, browser, tz, lang, network, screen
#   N_ACCOUNTS=400, N_TRAIN=60, CENTROID_N=40, CALIB_N=20, FLEET_FRAC=0.25
#
# Spoof attacks:
#   k=1 — tz only changed (hardest; single-field mismatch, near-perfect spoof)
#   k=2 — tz + network changed
#   k=3 — tz + network + screen changed (easiest; three-field mismatch)
# Negatives: new legitimate device (different network/screen) + known primary device
#
# Primary metric: spoof AUC per k (400 pos vs 800 neg — reliable with tight CIs)
#
# Usage:
#   uv run h4_gru/experiments/gru_synth_experiment.py
#   uv run h4_gru/experiments/gru_synth_experiment.py --accounts 800

import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import FastText
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
N_ACCOUNTS  = 400
N_TRAIN     = 60
CENTROID_N  = 40
CALIB_N     = 20
FLEET_FRAC  = 0.25

FEATURES = {
    "os":      ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz":      ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang":    ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "network": ["wifi", "lte", "5g", "broadband"],
    "screen":  ["small", "medium", "large", "xlarge"],
}
FEATURE_ORDER = ["os", "browser", "tz", "lang", "network", "screen"]

ROBUST_KWARGS = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    seed=SEED, workers=1,
)

GRU_INPUT_DIM  = 64
GRU_HIDDEN_DIM = 64
GRU_LAMBDA     = 0.5
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
# GRU model (identical to gru_experiment.py)
# ---------------------------------------------------------------------------
class GRUPredictor(nn.Module):
    def __init__(self, input_dim: int = GRU_INPUT_DIM, hidden_dim: int = GRU_HIDDEN_DIM):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.proj(out)

    def encode(self, x: torch.Tensor) -> np.ndarray:
        _, h = self.gru(x)
        return self.proj(h.squeeze(0).squeeze(0)).cpu().numpy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_token(f: str, v: str) -> str:
    return f"{f}_{v}"

def device_key(d: dict) -> tuple:
    return tuple(d[f] for f in FEATURE_ORDER)

def device_to_tokens(d: dict) -> list[str]:
    return [make_token(f, d[f]) for f in FEATURE_ORDER]

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))

def embed_mp(event: dict, model) -> np.ndarray:
    return np.mean([model.wv[make_token(f, event[f])] for f in FEATURE_ORDER], axis=0)

def zipf_weights(n: int, s: float = 1.5) -> np.ndarray:
    w = np.array([1.0 / k**s for k in range(1, n + 1)])
    return w / w.sum()


# ---------------------------------------------------------------------------
# Synthetic data generation (same as H2 variable_spoof_experiment.py)
# ---------------------------------------------------------------------------
def make_spoof(primary: dict, rng, k: int) -> dict:
    s = dict(primary)
    s["tz"] = rng.choice([t for t in FEATURES["tz"] if t != primary["tz"]])
    if k >= 2:
        s["network"] = rng.choice(FEATURES["network"])
    if k >= 3:
        s["screen"] = rng.choice(FEATURES["screen"])
    return s


def generate_dataset(n_accounts: int = N_ACCOUNTS, seed: int = SEED) -> list:
    rng = np.random.default_rng(seed)
    fleet_device = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
    accounts = []
    for acct_id in range(n_accounts):
        n_dev = int(rng.integers(2, 5))
        known, seen = [], set()
        while len(known) < n_dev:
            d = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
            k = device_key(d)
            if k not in seen:
                seen.add(k)
                known.append(d)
        primary = known[0]
        weights = zipf_weights(n_dev)
        train_events = []
        for _ in range(N_TRAIN):
            idx = rng.choice(n_dev, p=weights)
            train_events.append(dict(known[idx]))
        if rng.random() < FLEET_FRAC:
            replace_idx = int(rng.integers(0, len(train_events)))
            train_events[replace_idx] = dict(fleet_device)
        novel = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
        attempts = 0
        while (novel["os"] == primary["os"] or novel["tz"] == primary["tz"]
               or novel["lang"] == primary["lang"]) and attempts < 1000:
            novel = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
            attempts += 1
        neg = dict(primary)
        neg["network"] = rng.choice([n for n in FEATURES["network"] if n != primary["network"]])
        neg["screen"]  = rng.choice([s for s in FEATURES["screen"]  if s != primary["screen"]])
        accounts.append({
            "id":            acct_id,
            "known_devices": known,
            "primary":       primary,
            "train_events":  train_events,   # all N_TRAIN events in temporal order
            "centroid_evts": train_events[:CENTROID_N],
            "calib_evts":    train_events[CENTROID_N:],
            "test": {
                "spoof_k1": (make_spoof(primary, rng, 1), 1),
                "spoof_k2": (make_spoof(primary, rng, 2), 1),
                "spoof_k3": (make_spoof(primary, rng, 3), 1),
                "neg":      (neg, 0),
                "known":    (dict(primary), 0),
            },
        })
    return accounts


# ---------------------------------------------------------------------------
# FastText corpus
# ---------------------------------------------------------------------------
def build_corpus(accounts: list) -> list:
    sentences = []
    for acc in accounts:
        flat = []
        for e in acc["train_events"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences


# ---------------------------------------------------------------------------
# GRU dataset and training (same architecture as gru_experiment.py)
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


def train_gru(accounts: list, model_ft, device: str = "cpu") -> tuple[GRUPredictor, dict]:
    """Train GRU on all training sequences; return (model, loss_history)."""
    seqs = [
        np.array([embed_mp(e, model_ft) for e in acc["train_events"]])
        for acc in accounts
    ]
    print(f"  {len(seqs):,} training sequences | len={N_TRAIN} each")

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

        gru.train(False)
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
# Scoring variants — return flat (scores, labels) over all attack types
# ---------------------------------------------------------------------------
NEG_KEYS    = ("neg", "known")
ATTACK_KEYS = ("spoof_k1", "spoof_k2", "spoof_k3")


def _collect_scores(accounts: list, score_fn) -> dict:
    """Apply score_fn(account, event) -> float to all test events.
    Returns dict[event_type] -> (scores, labels)."""
    all_keys = ATTACK_KEYS + NEG_KEYS
    result = {k: ([], []) for k in all_keys}
    for acc in accounts:
        for key in all_keys:
            ev, lbl = acc["test"][key]
            result[key][0].append(score_fn(acc, ev))
            result[key][1].append(lbl)
    return result


def score_mp_raw_split(accounts: list, model_ft) -> dict:
    def _score(acc, ev):
        centroid = np.mean([embed_mp(e, model_ft) for e in acc["centroid_evts"]], axis=0)
        return cosine_dist(embed_mp(ev, model_ft), centroid)
    return _collect_scores(accounts, _score)


def score_recency_mean(accounts: list, model_ft, lam: float = GRU_LAMBDA) -> dict:
    def _score(acc, ev):
        evts = acc["centroid_evts"]
        n    = len(evts)
        w    = np.array([np.exp(-lam * (n - 1 - t)) for t in range(n)])
        w   /= w.sum()
        embs = np.array([embed_mp(e, model_ft) for e in evts])
        centroid = (w[:, None] * embs).sum(axis=0)
        return cosine_dist(embed_mp(ev, model_ft), centroid)
    return _collect_scores(accounts, _score)


def score_gru_predict(accounts: list, model_ft, gru: GRUPredictor,
                      device: str = "cpu") -> dict:
    gru.train(False)
    pred_cache: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for acc in accounts:
            seq = torch.tensor(
                np.array([embed_mp(e, model_ft) for e in acc["centroid_evts"]]),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            pred_cache[acc["id"]] = gru.encode(seq)

    def _score(acc, ev):
        return cosine_dist(embed_mp(ev, model_ft), pred_cache[acc["id"]])

    return _collect_scores(accounts, _score)


def score_trivial(accounts: list) -> dict:
    def _score(acc, ev):
        known_keys = {device_key(d) for d in acc["known_devices"]}
        return 0.0 if device_key(ev) in known_keys else 1.0
    return _collect_scores(accounts, _score)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def spoof_auc(result: dict, attack_key: str,
              n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    """Stratified bootstrap AUC: attack_key positives vs neg+known negatives."""
    rng = np.random.default_rng(seed)
    neg_s = np.concatenate([result[k][0] for k in NEG_KEYS])
    neg_l = np.concatenate([result[k][1] for k in NEG_KEYS])
    att_s = np.array(result[attack_key][0])
    att_l = np.array(result[attack_key][1])
    comb_s = np.concatenate([att_s, neg_s])
    comb_l = np.concatenate([att_l, neg_l])
    pos_idx = np.where(comb_l == 1)[0]
    neg_idx = np.where(comb_l == 0)[0]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
        lb_b, sc_b = comb_l[idx], comb_s[idx]
        if len(np.unique(lb_b)) < 2:
            continue
        aucs.append(roc_auc_score(lb_b, sc_b))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    a = np.array(aucs)
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def compute_all_aucs(result: dict) -> dict:
    """Return {attack_key: {mean, lo, hi}} for all three spoof levels."""
    return {
        k: dict(zip(["mean", "lo", "hi"], spoof_auc(result, k)))
        for k in ATTACK_KEYS
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name: str) -> None:
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")


def plot_spoof_aucs(all_results: dict) -> None:
    """Grouped bar chart: AUC by spoof level for each scoring variant."""
    scorers = list(all_results.keys())
    attack_labels = {"spoof_k1": "k=1 (tz)", "spoof_k2": "k=2 (tz+net)", "spoof_k3": "k=3 (tz+net+screen)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, atk in zip(axes, ATTACK_KEYS):
        vals = [all_results[s][atk]["mean"] for s in scorers]
        los  = [all_results[s][atk]["lo"]   for s in scorers]
        his  = [all_results[s][atk]["hi"]   for s in scorers]
        x = np.arange(len(scorers))
        ax.bar(x, vals, color=[COLORS.get(s, "#777") for s in scorers], alpha=0.85)
        for i, (v, lo, hi) in enumerate(zip(vals, los, his)):
            if not np.isnan(lo):
                ax.errorbar(i, v, yerr=[[v - lo], [hi - v]],
                            fmt="none", color="black", capsize=4, linewidth=1.3)
            ax.text(i, (hi if not np.isnan(hi) else v) + 0.005,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(scorers, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0.4, 1.15)
        ax.set_ylabel("Spoof AUC")
        ax.set_title(f"H4 GRU Synth: {attack_labels[atk]}\n(N={N_ACCOUNTS} accounts, stratified bootstrap)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    save_fig(fig, "gru_synth_spoof_auc.png")


def plot_training_loss(loss_history: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history["train"], label="train", color="#4CAF50", linewidth=1.5)
    ax.plot(loss_history["val"],   label="val",   color="#E91E63", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("1 − mean cosine similarity")
    ax.set_title("H4 GRU Synth: Training Loss")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "gru_synth_training_loss.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="H4 GRU synthetic dataset experiment")
    parser.add_argument("--accounts", type=int, default=N_ACCOUNTS,
                        help=f"number of synthetic accounts (default {N_ACCOUNTS})")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (cpu / cuda / mps)")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("  MPS not available, falling back to cpu")
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available, falling back to cpu")
        device = "cpu"

    print("=" * 72)
    print("H4 GRU Temporal Sequence Experiment — Synthetic Dataset")
    print(f"N_ACCOUNTS={args.accounts} | N_TRAIN={N_TRAIN} | "
          f"CENTROID_N={CENTROID_N} | CALIB_N={CALIB_N} | device={device}")
    print("Variants: mp-raw-split, recency-mean (λ=0.5), gru-predict, trivial")
    print("=" * 72)

    t0 = time.time()

    print("\n[1/5] Generating synthetic dataset ...")
    accounts = generate_dataset(n_accounts=args.accounts, seed=SEED)
    print(f"  {len(accounts):,} accounts | {N_TRAIN} train events each | "
          f"features: {FEATURE_ORDER}")

    print("\n[2/5] Training FastText (per-account concatenated corpus) ...")
    corpus = build_corpus(accounts)
    print(f"  Corpus: {len(corpus):,} sentences  {sum(len(s) for s in corpus):,} tokens")
    model_ft = FastText(sentences=corpus, **ROBUST_KWARGS)
    print(f"  Vocabulary: {len(model_ft.wv):,} tokens")

    print("\n[3/5] Training GRU ...")
    gru, loss_history = train_gru(accounts, model_ft, device=device)

    print("\n[4/5] Scoring variants ...")
    print("  mp-raw-split ...")
    mp_result  = score_mp_raw_split(accounts, model_ft)
    print("  recency-mean ...")
    rec_result = score_recency_mean(accounts, model_ft)
    print("  gru-predict ...")
    gru_result = score_gru_predict(accounts, model_ft, gru, device=device)
    print("  trivial ...")
    tri_result = score_trivial(accounts)

    all_results_raw = {
        "mp-raw-split": mp_result,
        "recency-mean": rec_result,
        "gru-predict":  gru_result,
        "trivial":      tri_result,
    }

    print("\n[5/5] Computing metrics and saving figures ...")
    all_results: dict[str, dict] = {}
    for scorer, result in all_results_raw.items():
        all_results[scorer] = compute_all_aucs(result)

    # Print summary table
    print(f"\n  {'Scorer':<18}  "
          f"{'k=1 AUC':>10} {'[95% CI]':>18}  "
          f"{'k=2 AUC':>10} {'[95% CI]':>18}  "
          f"{'k=3 AUC':>10} {'[95% CI]':>18}")
    print("  " + "-" * 102)
    for scorer, aucs in all_results.items():
        row = f"  {scorer:<18}"
        for atk in ATTACK_KEYS:
            m, lo, hi = aucs[atk]["mean"], aucs[atk]["lo"], aucs[atk]["hi"]
            ci = f"[{lo:.3f},{hi:.3f}]" if not np.isnan(lo) else "      —     "
            row += f"  {m:>10.4f} {ci:>18}"
        print(row)

    mp_k1  = all_results["mp-raw-split"]["spoof_k1"]["mean"]
    gru_k1 = all_results["gru-predict"]["spoof_k1"]["mean"]
    hypothesis_confirmed = gru_k1 > mp_k1
    print(f"\n  HEADLINE (hardest case — k=1):")
    print(f"  mp-raw-split : {mp_k1:.4f}")
    print(f"  gru-predict  : {gru_k1:.4f}")
    print(f"  GRU beats baseline: {hypothesis_confirmed} (Δ={gru_k1-mp_k1:+.4f})")

    plot_spoof_aucs(all_results)
    plot_training_loss(loss_history)

    elapsed = time.time() - t0
    metrics_out = {
        "version": "v1-synth",
        "n_accounts": args.accounts,
        "n_train": N_TRAIN,
        "centroid_n": CENTROID_N,
        "calib_n": CALIB_N,
        "gru_config": {
            "hidden_dim": GRU_HIDDEN_DIM,
            "input_dim":  GRU_INPUT_DIM,
            "max_epochs": GRU_MAX_EPOCHS,
            "patience":   GRU_PATIENCE,
            "lr":         GRU_LR,
            "batch_size": GRU_BATCH,
        },
        "recency_lambda": GRU_LAMBDA,
        "results": {
            scorer: {
                atk: aucs[atk]
                for atk in ATTACK_KEYS
            }
            for scorer, aucs in all_results.items()
        },
        "hypothesis_confirmed_k1": hypothesis_confirmed,
        "elapsed_s": round(elapsed, 1),
    }
    metrics_path = FIGURES_DIR / "gru_synth_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics: {metrics_path}")
    print(f"\n  Elapsed: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
