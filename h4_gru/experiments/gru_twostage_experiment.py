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
# GRU Two-Stage Experiment — H4
# ==============================
# Tests the practical two-stage architecture:
#
#   Stage 1 (set-membership): if the device was seen in training → score = 0, allow.
#   Stage 2 (embedding model): only unknown devices reach here; score = embedding distance.
#
# The combined score for any test event is:
#   known device   → 0.0  (Stage 1 catch; never reaches embedding model)
#   unknown device → embedding_model_score  (Stage 2 refines)
#
# This reflects real deployment: known-device checks are cheap and run first;
# the ML model only pays compute on genuinely novel login attempts.
#
# The practical consequence for evaluation:
#   - "known" negatives are trivially scored at 0 → never falsely flagged
#   - Only hard case remains: spoof (unknown malicious) vs neg (unknown legitimate enrollment)
#   - Trivial baseline has no signal here — both spoof and neg are unknown to it
#
# Uses the temporal synthetic generator from gru_temporal_synth_experiment.py
# (Markov session clustering, device drift, travel episodes) for realistic sequences.
#
# Compares four Stage-2 scorers in both single-stage and two-stage modes:
#   mp-raw-split, recency-mean, gru-predict, trivial (as reference)
#
# Primary metric: spoof AUC per k on two-stage combined scores.
#
# Usage:
#   uv run h4_gru/experiments/gru_twostage_experiment.py
#   uv run h4_gru/experiments/gru_twostage_experiment.py --accounts 800

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

N_BOOTSTRAP    = 1000
N_ACCOUNTS     = 400
N_TRAIN        = 60
CENTROID_N     = 40
CALIB_N        = 20
FLEET_FRAC     = 0.25

P_STAY         = 0.70
DRIFT_INTERVAL = 20
P_DRIFT        = 0.25
P_TRAVEL       = 0.20
TRAVEL_LEN_MIN = 4
TRAVEL_LEN_MAX = 10

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

NEG_KEYS    = ("neg", "known")
ATTACK_KEYS = ("spoof_k1", "spoof_k2", "spoof_k3")
ALL_KEYS    = ATTACK_KEYS + NEG_KEYS

COLORS_S1 = {   # single-stage
    "mp-raw-split": "#BDBDBD",
    "recency-mean": "#90CAF9",
    "gru-predict":  "#A5D6A7",
    "trivial":      "#B0BEC5",
}
COLORS_S2 = {   # two-stage
    "mp-raw-split": "#616161",
    "recency-mean": "#1565C0",
    "gru-predict":  "#2E7D32",
    "trivial":      "#455A64",
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

def make_spoof(primary: dict, rng, k: int) -> dict:
    s = dict(primary)
    s["tz"] = rng.choice([t for t in FEATURES["tz"] if t != primary["tz"]])
    if k >= 2:
        s["network"] = rng.choice(FEATURES["network"])
    if k >= 3:
        s["screen"] = rng.choice(FEATURES["screen"])
    return s


# ---------------------------------------------------------------------------
# Temporal data generator (same as gru_temporal_synth_experiment.py)
# ---------------------------------------------------------------------------
def generate_temporal_events(known_devices: list[dict], rng, n_events: int = N_TRAIN) -> list[dict]:
    n_dev  = len(known_devices)
    base_w = zipf_weights(n_dev)
    ranked = list(range(n_dev))

    travel_start = travel_end = -1
    travel_tz = travel_net = None
    if n_dev >= 1 and rng.random() < P_TRAVEL:
        lo = max(1, n_events // 10)
        hi = max(lo + 1, int(n_events * 0.6))
        travel_start = int(rng.integers(lo, hi))
        travel_len   = int(rng.integers(TRAVEL_LEN_MIN, TRAVEL_LEN_MAX + 1))
        travel_end   = min(travel_start + travel_len, n_events)
        home_tz  = known_devices[ranked[0]]["tz"]
        home_net = known_devices[ranked[0]]["network"]
        travel_tz  = rng.choice([t for t in FEATURES["tz"]     if t != home_tz])
        travel_net = rng.choice([t for t in FEATURES["network"] if t != home_net])

    events: list[dict] = []
    last_dev: int | None = None

    for i in range(n_events):
        if i > 0 and i % DRIFT_INTERVAL == 0 and n_dev > 1:
            if rng.random() < P_DRIFT:
                swap = int(rng.integers(1, n_dev))
                ranked[0], ranked[swap] = ranked[swap], ranked[0]

        if last_dev is not None and rng.random() < P_STAY:
            chosen = last_dev
        else:
            pos    = rng.choice(n_dev, p=base_w)
            chosen = ranked[pos]

        event = dict(known_devices[chosen])
        if travel_start <= i < travel_end:
            event["tz"]      = travel_tz
            event["network"] = travel_net

        events.append(event)
        last_dev = chosen

    return events


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

        train_events = generate_temporal_events(known, rng)
        if rng.random() < FLEET_FRAC:
            replace_idx = int(rng.integers(0, len(train_events)))
            train_events[replace_idx] = dict(fleet_device)

        neg = dict(primary)
        neg["network"] = rng.choice([n for n in FEATURES["network"] if n != primary["network"]])
        neg["screen"]  = rng.choice([s for s in FEATURES["screen"]  if s != primary["screen"]])

        accounts.append({
            "id":            acct_id,
            "known_devices": known,
            "primary":       primary,
            "train_events":  train_events,
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
# FastText corpus and GRU training
# ---------------------------------------------------------------------------
def build_corpus(accounts: list) -> list:
    return [
        flat for acc in accounts
        if (flat := [t for e in acc["train_events"] for t in device_to_tokens(e)])
    ]


class SeqDataset(Dataset):
    def __init__(self, seqs: list[np.ndarray]):
        self.seqs = [s for s in seqs if len(s) >= 2]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.seqs[idx]
        return (torch.tensor(s[:-1], dtype=torch.float32),
                torch.tensor(s[1:],  dtype=torch.float32))


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
    seqs = [np.array([embed_mp(e, model_ft) for e in acc["train_events"]]) for acc in accounts]
    rng   = np.random.default_rng(SEED)
    perm  = rng.permutation(len(seqs))
    n_trn = int(0.9 * len(seqs))
    trn_dl = DataLoader(SeqDataset([seqs[i] for i in perm[:n_trn]]),
                        batch_size=GRU_BATCH, shuffle=True, collate_fn=collate_pad,
                        generator=torch.Generator().manual_seed(SEED))
    val_dl = DataLoader(SeqDataset([seqs[i] for i in perm[n_trn:]]),
                        batch_size=GRU_BATCH, shuffle=False, collate_fn=collate_pad)

    gru = GRUPredictor().to(device)
    opt = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    best_val, pat_count, best_state = float("inf"), 0, None
    trn_h, val_h = [], []

    for epoch in range(GRU_MAX_EPOCHS):
        gru.train()
        t_loss = sum(
            (loss := (1.0 - F.cosine_similarity(
                gru(inp.to(device)), tgt.to(device), dim=-1
            ))[mask.to(device)].mean(), opt.zero_grad(), loss.backward(), opt.step(), loss.item()
            )[-1]
            for inp, tgt, mask in trn_dl
        )
        gru.train(False)
        v_loss = 0.0
        with torch.no_grad():
            for inp, tgt, mask in val_dl:
                inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                v_loss += (1.0 - F.cosine_similarity(gru(inp), tgt, dim=-1))[mask].mean().item()

        avg_t = t_loss / max(1, len(trn_dl))
        avg_v = v_loss / max(1, len(val_dl))
        trn_h.append(avg_t); val_h.append(avg_v)

        if avg_v < best_val - 1e-4:
            best_val, pat_count = avg_v, 0
            best_state = {k: v.cpu().clone() for k, v in gru.state_dict().items()}
        else:
            pat_count += 1

        if (epoch + 1) % 5 == 0 or pat_count >= GRU_PATIENCE:
            print(f"  epoch {epoch+1:3d} | train={avg_t:.4f} | val={avg_v:.4f} "
                  f"| patience={pat_count}/{GRU_PATIENCE}")
        if pat_count >= GRU_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}"); break

    if best_state:
        gru.load_state_dict(best_state)
        print(f"  Best weights loaded (val_loss={best_val:.4f})")
    return gru.to(device), {"train": trn_h, "val": val_h}


# ---------------------------------------------------------------------------
# Single-stage scorers (raw embedding distance)
# ---------------------------------------------------------------------------
def _collect(accounts: list, score_fn) -> dict:
    result = {k: ([], []) for k in ALL_KEYS}
    for acc in accounts:
        for key in ALL_KEYS:
            ev, lbl = acc["test"][key]
            result[key][0].append(score_fn(acc, ev))
            result[key][1].append(lbl)
    return result


def score_mp_raw_split(accounts: list, model_ft) -> dict:
    def _s(acc, ev):
        c = np.mean([embed_mp(e, model_ft) for e in acc["centroid_evts"]], axis=0)
        return cosine_dist(embed_mp(ev, model_ft), c)
    return _collect(accounts, _s)


def score_recency_mean(accounts: list, model_ft) -> dict:
    def _s(acc, ev):
        evts = acc["centroid_evts"]
        n = len(evts)
        w = np.array([np.exp(-GRU_LAMBDA * (n - 1 - t)) for t in range(n)])
        w /= w.sum()
        c = (w[:, None] * np.array([embed_mp(e, model_ft) for e in evts])).sum(axis=0)
        return cosine_dist(embed_mp(ev, model_ft), c)
    return _collect(accounts, _s)


def score_gru_predict(accounts: list, model_ft, gru: GRUPredictor, device: str) -> dict:
    gru.train(False)
    cache: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for acc in accounts:
            seq = torch.tensor(
                np.array([embed_mp(e, model_ft) for e in acc["centroid_evts"]]),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            cache[acc["id"]] = gru.encode(seq)

    def _s(acc, ev):
        return cosine_dist(embed_mp(ev, model_ft), cache[acc["id"]])
    return _collect(accounts, _s)


def score_trivial(accounts: list) -> dict:
    def _s(acc, ev):
        return 0.0 if device_key(ev) in {device_key(d) for d in acc["known_devices"]} else 1.0
    return _collect(accounts, _s)


# ---------------------------------------------------------------------------
# Two-stage gate: override score to 0 for known devices (Stage 1 catch)
# ---------------------------------------------------------------------------
def apply_stage1_gate(result: dict, accounts: list) -> dict:
    """Post-process single-stage scores: known devices → score = 0.

    Simulates the sequential architecture:
      known device  → Stage 1 catches it, score = 0 (not anomalous)
      unknown device → Stage 2 score passes through unchanged
    """
    gated = {k: (list(scores), list(labels)) for k, (scores, labels) in result.items()}
    for i, acc in enumerate(accounts):
        known_keys = {device_key(d) for d in acc["known_devices"]}
        for key in ALL_KEYS:
            ev, _ = acc["test"][key]
            if device_key(ev) in known_keys:
                gated[key][0][i] = 0.0
    return gated


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def spoof_auc(result: dict, attack_key: str,
              n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
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
    return {k: dict(zip(["mean", "lo", "hi"], spoof_auc(result, k))) for k in ATTACK_KEYS}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name: str) -> None:
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")


def plot_comparison(single: dict, twostage: dict) -> None:
    """Side-by-side: single-stage vs two-stage AUC for each spoof level."""
    scorers = list(single.keys())
    attack_labels = {
        "spoof_k1": "k=1 (tz only — hardest)",
        "spoof_k2": "k=2 (tz + network)",
        "spoof_k3": "k=3 (tz + network + screen)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    x = np.arange(len(scorers))
    w = 0.38

    for ax, atk in zip(axes, ATTACK_KEYS):
        s_vals = [single[s][atk]["mean"]    for s in scorers]
        s_his  = [single[s][atk]["hi"]      for s in scorers]
        s_los  = [single[s][atk]["lo"]      for s in scorers]
        t_vals = [twostage[s][atk]["mean"]  for s in scorers]
        t_his  = [twostage[s][atk]["hi"]    for s in scorers]
        t_los  = [twostage[s][atk]["lo"]    for s in scorers]

        ax.bar(x - w/2, s_vals, width=w, label="single-stage",
               color=[COLORS_S1.get(s, "#aaa") for s in scorers], alpha=0.9)
        ax.bar(x + w/2, t_vals, width=w, label="two-stage (trivial gate)",
               color=[COLORS_S2.get(s, "#555") for s in scorers], alpha=0.9)

        for i, (sv, slo, shi, tv, tlo, thi) in enumerate(
                zip(s_vals, s_los, s_his, t_vals, t_los, t_his)):
            if not np.isnan(slo):
                ax.errorbar(i - w/2, sv, yerr=[[sv-slo], [shi-sv]],
                            fmt="none", color="black", capsize=3, linewidth=1)
            if not np.isnan(tlo):
                ax.errorbar(i + w/2, tv, yerr=[[tv-tlo], [thi-tv]],
                            fmt="none", color="black", capsize=3, linewidth=1)
            ax.text(i - w/2, (shi if not np.isnan(shi) else sv) + 0.01,
                    f"{sv:.3f}", ha="center", fontsize=7.5, color="#444")
            ax.text(i + w/2, (thi if not np.isnan(thi) else tv) + 0.01,
                    f"{tv:.3f}", ha="center", fontsize=7.5, color="#000")

        ax.set_xticks(x)
        ax.set_xticklabels(scorers, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0.4, 1.18)
        ax.set_ylabel("Spoof AUC")
        ax.set_title(f"H4 Two-Stage: {attack_labels[atk]}\n"
                     f"(N={N_ACCOUNTS}, temporal synth, stratified bootstrap)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        if atk == "spoof_k1":
            ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, "gru_twostage_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="H4 two-stage architecture experiment")
    parser.add_argument("--accounts", type=int, default=N_ACCOUNTS)
    parser.add_argument("--device",   type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 72)
    print("H4 GRU Two-Stage Experiment (Stage 1: set-membership → Stage 2: embedding)")
    print(f"N_ACCOUNTS={args.accounts} | device={device}")
    print("Temporal synthetic data: p_stay=0.70, p_drift=0.25, p_travel=0.20")
    print("=" * 72)

    t0 = time.time()

    print("\n[1/5] Generating temporal synthetic dataset ...")
    accounts = generate_dataset(n_accounts=args.accounts)
    known_pass = sum(
        1 for acc in accounts
        if device_key(acc["test"]["known"][0]) in {device_key(d) for d in acc["known_devices"]}
    )
    unknown_neg = sum(
        1 for acc in accounts
        if device_key(acc["test"]["neg"][0]) not in {device_key(d) for d in acc["known_devices"]}
    )
    unknown_spoof = {
        k: sum(
            1 for acc in accounts
            if device_key(acc["test"][k][0]) not in {device_key(d) for d in acc["known_devices"]}
        )
        for k in ATTACK_KEYS
    }
    print(f"  Stage 1 routing (per account):")
    print(f"    known device → caught by Stage 1: {known_pass}/{args.accounts} "
          f"({100*known_pass/args.accounts:.0f}%)")
    print(f"    neg (new enrollment) → Stage 2: {unknown_neg}/{args.accounts} "
          f"({100*unknown_neg/args.accounts:.0f}%)")
    for k, cnt in unknown_spoof.items():
        print(f"    {k} → Stage 2: {cnt}/{args.accounts} ({100*cnt/args.accounts:.0f}%)")

    print("\n[2/5] Training FastText ...")
    corpus = build_corpus(accounts)
    print(f"  Corpus: {len(corpus):,} sentences  {sum(len(s) for s in corpus):,} tokens")
    model_ft = FastText(sentences=corpus, **ROBUST_KWARGS)
    print(f"  Vocabulary: {len(model_ft.wv):,} tokens")

    print("\n[3/5] Training GRU ...")
    gru, _ = train_gru(accounts, model_ft, device=device)

    print("\n[4/5] Scoring all variants (single-stage + two-stage gate) ...")
    raw_scores = {
        "mp-raw-split": score_mp_raw_split(accounts, model_ft),
        "recency-mean": score_recency_mean(accounts, model_ft),
        "gru-predict":  score_gru_predict(accounts, model_ft, gru, device),
        "trivial":      score_trivial(accounts),
    }
    gated_scores = {
        name: apply_stage1_gate(result, accounts)
        for name, result in raw_scores.items()
    }

    print("\n[5/5] Computing metrics and saving figures ...")
    single_aucs  = {name: compute_all_aucs(r) for name, r in raw_scores.items()}
    twostage_aucs = {name: compute_all_aucs(r) for name, r in gated_scores.items()}

    for label, aucs in [("Single-stage", single_aucs), ("Two-stage  ", twostage_aucs)]:
        print(f"\n  {label}")
        print(f"  {'Scorer':<18}  "
              f"{'k=1 AUC':>10} {'[95% CI]':>18}  "
              f"{'k=2 AUC':>10} {'[95% CI]':>18}  "
              f"{'k=3 AUC':>10} {'[95% CI]':>18}")
        print("  " + "-" * 102)
        for scorer, a in aucs.items():
            row = f"  {scorer:<18}"
            for atk in ATTACK_KEYS:
                m, lo, hi = a[atk]["mean"], a[atk]["lo"], a[atk]["hi"]
                row += f"  {m:>10.4f} [{lo:.3f},{hi:.3f}]"
            print(row)

    print("\n  Delta (two-stage − single-stage):")
    print(f"  {'Scorer':<18}  {'Δ k=1':>10}  {'Δ k=2':>10}  {'Δ k=3':>10}")
    print("  " + "-" * 54)
    for scorer in single_aucs:
        row = f"  {scorer:<18}"
        for atk in ATTACK_KEYS:
            d = twostage_aucs[scorer][atk]["mean"] - single_aucs[scorer][atk]["mean"]
            row += f"  {d:>+10.4f}"
        print(row)

    mp_ts  = twostage_aucs["mp-raw-split"]["spoof_k1"]["mean"]
    gru_ts = twostage_aucs["gru-predict"]["spoof_k1"]["mean"]
    print(f"\n  HEADLINE — two-stage k=1 (hardest):")
    print(f"  trivial+mp-raw-split : {mp_ts:.4f}")
    print(f"  trivial+gru-predict  : {gru_ts:.4f}  (Δ={gru_ts-mp_ts:+.4f})")
    print(f"  GRU wins two-stage k=1: {gru_ts > mp_ts}")

    plot_comparison(single_aucs, twostage_aucs)

    elapsed = time.time() - t0
    metrics_out = {
        "version": "v1-twostage",
        "n_accounts": args.accounts,
        "stage1_routing": {
            "known_caught": known_pass,
            "neg_to_stage2": unknown_neg,
            **{k: v for k, v in unknown_spoof.items()},
        },
        "single_stage": {
            scorer: {atk: aucs[atk] for atk in ATTACK_KEYS}
            for scorer, aucs in single_aucs.items()
        },
        "two_stage": {
            scorer: {atk: aucs[atk] for atk in ATTACK_KEYS}
            for scorer, aucs in twostage_aucs.items()
        },
        "hypothesis_confirmed_k1": gru_ts > mp_ts,
        "elapsed_s": round(elapsed, 1),
    }
    out_path = FIGURES_DIR / "gru_twostage_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics: {out_path}")
    print(f"\n  Elapsed: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
