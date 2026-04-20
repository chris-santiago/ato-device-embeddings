# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
#   "matplotlib>=3.7",
# ]
# ///
#
# H5 Stress Test: H2 Mean-Pool FastText across k=1/2/3, single-stage vs two-stage
# Reports ROC-AUC and PR-AUC with bootstrap CIs.

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import FastText
from sklearn.metrics import roc_auc_score, average_precision_score

SEED        = 42
N_BOOTSTRAP = 1000
N_ACCOUNTS  = 400
N_TRAIN     = 60
CENTROID_N  = 40
CALIB_N     = 20
FLEET_FRAC  = 0.25

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

ALL_KEYS   = ["spoof_k1", "spoof_k2", "spoof_k3", "neg", "known"]
SPOOF_KEYS = ["spoof_k1", "spoof_k2", "spoof_k3"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_token(f, v): return f"{f}_{v}"
def device_key(d): return tuple(d[f] for f in FEATURE_ORDER)
def device_to_tokens(d): return [make_token(f, d[f]) for f in FEATURE_ORDER]
def zipf_weights(n, s=1.5):
    w = np.array([1.0 / k**s for k in range(1, n + 1)])
    return w / w.sum()

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))

def embed_mp(event, model):
    return np.mean([model.wv[make_token(f, event[f])] for f in FEATURE_ORDER], axis=0)

# ---------------------------------------------------------------------------
# Data generation  (identical to variable_spoof_experiment.py)
# ---------------------------------------------------------------------------
def make_spoof(primary, rng, k):
    s = dict(primary)
    s["tz"] = rng.choice([t for t in FEATURES["tz"] if t != primary["tz"]])
    if k >= 2:
        s["network"] = rng.choice(FEATURES["network"])
    if k >= 3:
        s["screen"] = rng.choice(FEATURES["screen"])
    return s

def generate_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    fleet_device = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
    accounts = []
    for acct_id in range(N_ACCOUNTS):
        n_dev = int(rng.integers(2, 5))
        known, seen = [], set()
        while len(known) < n_dev:
            d = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
            dk = device_key(d)
            if dk not in seen:
                seen.add(dk); known.append(d)
        primary = known[0]
        weights = zipf_weights(n_dev)
        train_events = [dict(known[rng.choice(n_dev, p=weights)]) for _ in range(N_TRAIN)]
        if rng.random() < FLEET_FRAC:
            train_events[int(rng.integers(0, len(train_events)))] = dict(fleet_device)
        neg = dict(primary)
        neg["network"] = rng.choice([n for n in FEATURES["network"] if n != primary["network"]])
        neg["screen"]  = rng.choice([s for s in FEATURES["screen"]  if s != primary["screen"]])
        accounts.append({
            "id": acct_id, "known_devices": known, "primary": primary,
            "train_events": train_events,
            "test": {
                "spoof_k1": (make_spoof(primary, rng, 1), 1),
                "spoof_k2": (make_spoof(primary, rng, 2), 1),
                "spoof_k3": (make_spoof(primary, rng, 3), 1),
                "neg":      (neg, 0),
                "known":    (dict(primary), 0),
            },
        })
    return accounts

def build_corpus(accounts):
    return [[tok for e in acc["train_events"] for tok in device_to_tokens(e)] for acc in accounts]

# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
def score_mp_raw(accounts, model):
    """Centroid over all N_TRAIN events — original H2 scorer."""
    result = {k: ([], []) for k in ALL_KEYS}
    for acc in accounts:
        centroid = np.mean([embed_mp(e, model) for e in acc["train_events"]], axis=0)
        for at, (ev, lbl) in acc["test"].items():
            result[at][0].append(cosine_dist(embed_mp(ev, model), centroid))
            result[at][1].append(lbl)
    return result

def score_mp_split(accounts, model):
    """Centroid over first CENTROID_N events — fair split (H3/H4 convention)."""
    result = {k: ([], []) for k in ALL_KEYS}
    for acc in accounts:
        centroid = np.mean([embed_mp(e, model) for e in acc["train_events"][:CENTROID_N]], axis=0)
        for at, (ev, lbl) in acc["test"].items():
            result[at][0].append(cosine_dist(embed_mp(ev, model), centroid))
            result[at][1].append(lbl)
    return result

def score_trivial(accounts):
    result = {k: ([], []) for k in ALL_KEYS}
    for acc in accounts:
        known_keys = {device_key(d) for d in acc["known_devices"]}
        for at, (ev, lbl) in acc["test"].items():
            result[at][0].append(0.0 if device_key(ev) in known_keys else 1.0)
            result[at][1].append(lbl)
    return result

def apply_stage1_gate(result, accounts):
    """Post-process: known devices → score=0 (Stage 1 catch, never reaches Stage 2)."""
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
def bootstrap_metrics(result, attack_key, neg_keys=("neg", "known"),
                      n_boot=N_BOOTSTRAP, seed=SEED):
    rng   = np.random.default_rng(seed)
    pos_s = np.array(result[attack_key][0])
    pos_l = np.array(result[attack_key][1])
    neg_s = np.concatenate([np.array(result[k][0]) for k in neg_keys])
    neg_l = np.concatenate([np.array(result[k][1]) for k in neg_keys])
    all_s = np.concatenate([pos_s, neg_s])
    all_l = np.concatenate([pos_l, neg_l])
    n = len(all_s)
    roc_vals, pr_vals = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys, ss = all_l[idx], all_s[idx]
        if len(np.unique(ys)) < 2: continue
        roc_vals.append(roc_auc_score(ys, ss))
        pr_vals.append(average_precision_score(ys, ss))
    roc_arr = np.array(roc_vals)
    pr_arr  = np.array(pr_vals)
    return {
        "roc_auc": {"mean": float(np.mean(roc_arr)),
                    "lo":   float(np.percentile(roc_arr, 2.5)),
                    "hi":   float(np.percentile(roc_arr, 97.5))},
        "pr_auc":  {"mean": float(np.mean(pr_arr)),
                    "lo":   float(np.percentile(pr_arr, 2.5)),
                    "hi":   float(np.percentile(pr_arr, 97.5))},
    }

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_results(all_metrics):
    scorers    = ["mp-raw", "mp-split", "trivial"]
    spoof_keys = SPOOF_KEYS
    xlabels    = ["k=1 (tz)", "k=2 (tz+net)", "k=3 (tz+net+scr)"]
    colors     = {
        "mp-raw":   {"single": "#1565C0", "two":    "#42A5F5"},
        "mp-split": {"single": "#2E7D32", "two":    "#66BB6A"},
        "trivial":  {"single": "#757575", "two":    "#BDBDBD"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metric_keys = ["roc_auc", "pr_auc"]
    metric_labels = ["ROC-AUC", "PR-AUC"]

    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        x = np.arange(len(spoof_keys))
        n_scorers = len(scorers)
        total_bars = n_scorers * 2
        w = 0.8 / total_bars
        for si, scorer in enumerate(scorers):
            for stage_i, stage in enumerate(["single", "two"]):
                bar_idx = si * 2 + stage_i
                offset  = (bar_idx - total_bars / 2 + 0.5) * w
                vals = [all_metrics[stage][scorer][sk][mk]["mean"] for sk in spoof_keys]
                los  = [all_metrics[stage][scorer][sk][mk]["lo"]   for sk in spoof_keys]
                his  = [all_metrics[stage][scorer][sk][mk]["hi"]   for sk in spoof_keys]
                label = f"{scorer} ({'2-stage' if stage == 'two' else '1-stage'})"
                bars = ax.bar(x + offset, vals, w, label=label,
                              color=colors[scorer][stage], alpha=0.85)
                for xi, (lo, hi) in enumerate(zip(los, his)):
                    cx = x[xi] + offset
                    ax.plot([cx, cx], [lo, hi], color="black", linewidth=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x); ax.set_xticklabels(xlabels)
        ax.set_ylabel(ml); ax.set_xlabel("Spoof difficulty")
        ax.set_title(f"{ml} — single-stage vs two-stage")
        ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8, label="trivial=0.75")
        ax.axhline(0.5,  color="gray", linestyle=":",  linewidth=0.6)
        if mk == "roc_auc":
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle("H5 Stress Test: H2 Mean-Pool FastText (k=1/2/3, single vs two-stage)", fontsize=11)
    plt.tight_layout()
    out = FIGURES_DIR / "h2_stress_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("H5 Stress Test — H2 Mean-Pool FastText: k=1/2/3 × single/two-stage")
    print(f"N_ACCOUNTS={N_ACCOUNTS} | ROC-AUC + PR-AUC | bootstrap N={N_BOOTSTRAP}")
    print("=" * 70)

    print("\n[1/4] Generating dataset ...")
    accounts = generate_dataset()
    print(f"  {len(accounts)} accounts, {N_TRAIN} train events each")

    print("\n[2/4] Training FastText (H2 robust config) ...")
    corpus = build_corpus(accounts)
    model  = FastText(sentences=corpus, **ROBUST_KWARGS)
    print(f"  vocab: {len(model.wv):,} tokens")

    print("\n[3/4] Scoring ...")
    raw_single    = score_mp_raw(accounts, model)
    split_single  = score_mp_split(accounts, model)
    triv_single   = score_trivial(accounts)
    raw_two       = apply_stage1_gate(raw_single,   accounts)
    split_two     = apply_stage1_gate(split_single, accounts)
    triv_two      = apply_stage1_gate(triv_single,  accounts)

    print("\n[4/4] Computing metrics ...")
    scorers_map = {
        "single": {"mp-raw": raw_single,  "mp-split": split_single, "trivial": triv_single},
        "two":    {"mp-raw": raw_two,      "mp-split": split_two,    "trivial": triv_two},
    }
    all_metrics = {"single": {}, "two": {}}
    for stage, scorers in scorers_map.items():
        for name, result in scorers.items():
            all_metrics[stage][name] = {}
            for sk in SPOOF_KEYS:
                all_metrics[stage][name][sk] = bootstrap_metrics(result, sk)

    # Print table
    for stage in ["single", "two"]:
        print(f"\n  {'='*20} {stage.upper()}-STAGE {'='*20}")
        hdr = f"  {'Scorer':<12}  {'k':>3}  {'ROC-AUC':>8}  [95% CI]            {'PR-AUC':>8}  [95% CI]"
        print(hdr); print("  " + "-" * 75)
        for name in ["mp-raw", "mp-split", "trivial"]:
            for sk in SPOOF_KEYS:
                m = all_metrics[stage][name][sk]
                roc = m["roc_auc"]; pr = m["pr_auc"]
                k_label = sk.replace("spoof_", "")
                print(f"  {name:<12}  {k_label:>3}  "
                      f"{roc['mean']:.4f}  [{roc['lo']:.4f},{roc['hi']:.4f}]  "
                      f"{pr['mean']:.4f}  [{pr['lo']:.4f},{pr['hi']:.4f}]")

    # Delta table
    print("\n  Delta (two-stage − single-stage) — ROC-AUC:")
    print(f"  {'Scorer':<12}" + "".join(f"  {sk.replace('spoof_',''):>8}" for sk in SPOOF_KEYS))
    print("  " + "-" * 45)
    for name in ["mp-raw", "mp-split", "trivial"]:
        deltas = [all_metrics["two"][name][sk]["roc_auc"]["mean"]
                - all_metrics["single"][name][sk]["roc_auc"]["mean"]
                for sk in SPOOF_KEYS]
        print(f"  {name:<12}" + "".join(f"  {d:+.4f}" for d in deltas))

    plot_results(all_metrics)

    out_json = FIGURES_DIR / "h2_stress_metrics.json"
    with open(out_json, "w") as f:
        json.dump({"version": "h5-stress", "n_accounts": N_ACCOUNTS,
                   "n_bootstrap": N_BOOTSTRAP, "results": all_metrics}, f, indent=2)
    print(f"  Metrics: {out_json}")
    print("\nDone.")

if __name__ == "__main__":
    main()
