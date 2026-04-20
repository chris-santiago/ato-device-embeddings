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
# Per-User Rank-Normalized Scoring Experiment
# ============================================
# Replicates the robust_config_experiment.py H2 results (mean-pool raw, concat
# raw, trivial) and adds per-user rank-normalized mean-pool as a new comparison.
#
# Normalization approach: empirical-CDF (rank-based).
# For each test event: score = fraction of calibration distances below test distance.
# This avoids z-score sigma-collapse (common when most events come from one device)
# and the Gaussian assumption.
#
# Train split to avoid circularity:
#   First CENTROID_N=40 events  → build centroid
#   Last  CALIB_N=20 events     → calibrate baseline distance distribution
# Raw mean-pool uses all 60 events for centroid (matching robust_config_experiment.py).
# The normalized variant has fewer centroid events; a flat result is not conclusive
# evidence against normalization — synthetic users are i.i.d. with N_TRAIN=60 each,
# giving less heterogeneity than real data (the condition where normalization shines).

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import FastText
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP  = 1000
CENTROID_N   = 40   # events used to build centroid (normalized variant)
CALIB_N      = 20   # events used to calibrate baseline distribution

# ---------------------------------------------------------------------------
# Feature vocabulary (identical to all prior experiments)
# ---------------------------------------------------------------------------
FEATURES = {
    "os":      ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz":      ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang":    ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "network": ["wifi", "lte", "5g", "broadband"],
    "screen":  ["small", "medium", "large", "xlarge"],
}
FEATURE_ORDER = ["os", "browser", "tz", "lang", "network", "screen"]
N_ACCOUNTS = 400
N_TRAIN    = 60
FLEET_FRAC = 0.25

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_token(f, v): return f"{f}_{v}"
def device_key(d): return tuple(d[f] for f in FEATURE_ORDER)
def device_to_tokens(d): return [make_token(f, d[f]) for f in FEATURE_ORDER]
def device_to_concat(d): return "_".join(d[f] for f in FEATURE_ORDER)

def zipf_weights(n, s=1.5):
    w = np.array([1.0 / k**s for k in range(1, n+1)])
    return w / w.sum()

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_centroid(events, embed_fn, model):
    return np.mean([embed_fn(e, model) for e in events], axis=0)

# ---------------------------------------------------------------------------
# Data generation (identical to robust_config_experiment.py)
# ---------------------------------------------------------------------------
def generate_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    fleet_device = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
    accounts = []
    for acct_id in range(N_ACCOUNTS):
        n_dev = int(rng.integers(2, 5))
        known, seen = [], set()
        while len(known) < n_dev:
            d = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
            k = device_key(d)
            if k not in seen:
                seen.add(k); known.append(d)
        primary = known[0]
        weights = zipf_weights(n_dev)
        train_events = []
        for _ in range(N_TRAIN):
            idx = rng.choice(n_dev, p=weights)
            train_events.append(dict(known[idx]))
        is_fleet = rng.random() < FLEET_FRAC
        if is_fleet:
            replace_idx = int(rng.integers(0, len(train_events)))
            train_events[replace_idx] = dict(fleet_device)
        novel = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
        attempts = 0
        while (novel["os"] == primary["os"] or novel["tz"] == primary["tz"]
               or novel["lang"] == primary["lang"]) and attempts < 1000:
            novel = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
            attempts += 1
        # Spoof: primary OS/browser/lang, different tz + randomised network/screen
        # Matches h2_rerun_experiment1.py _spoof_profile() definition.
        spoof = dict(primary)
        spoof["tz"]      = rng.choice([t for t in FEATURES["tz"] if t != primary["tz"]])
        spoof["network"] = rng.choice(FEATURES["network"])
        spoof["screen"]  = rng.choice(FEATURES["screen"])
        neg = dict(primary)
        neg["network"] = rng.choice([n for n in FEATURES["network"] if n != primary["network"]])
        neg["screen"]  = rng.choice([s for s in FEATURES["screen"]  if s != primary["screen"]])
        accounts.append({
            "id": acct_id, "known_devices": known, "primary": primary,
            "train_events": train_events, "is_fleet": is_fleet,
            "fleet_device": fleet_device,
            "test": {
                "novel": (novel, 1), "fleet": (dict(fleet_device), 1),
                "spoof": (spoof, 1), "neg": (neg, 0), "known": (dict(primary), 0),
            },
        })
    return accounts

# ---------------------------------------------------------------------------
# Corpus construction (per-account)
# ---------------------------------------------------------------------------
def build_mp_corpus(accounts):
    sentences = []
    for acc in accounts:
        flat = []
        for e in acc["train_events"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences

def build_cat_corpus(accounts):
    sentences = []
    for acc in accounts:
        sent = [device_to_concat(e) for e in acc["train_events"]]
        if sent:
            sentences.append(sent)
    return sentences

# ---------------------------------------------------------------------------
# Model training (robust config)
# ---------------------------------------------------------------------------
ROBUST_KWARGS = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    seed=SEED, workers=1,
)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_mp(event, model):
    vecs = [model.wv[make_token(f, event[f])] for f in FEATURE_ORDER]
    return np.mean(vecs, axis=0)

def embed_cat(event, model):
    return model.wv[device_to_concat(event)]

# ---------------------------------------------------------------------------
# Scoring — raw (all 60 events → centroid, identical to robust_config_experiment)
# ---------------------------------------------------------------------------
def score_all_raw(accounts, embed_fn, model):
    result = {k: {"scores": [], "labels": []} for k in ["novel", "fleet", "spoof", "neg", "known"]}
    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_fn, model)
        for at, (ev, lbl) in acc["test"].items():
            result[at]["scores"].append(cosine_dist(embed_fn(ev, model), centroid))
            result[at]["labels"].append(lbl)
    return result

# ---------------------------------------------------------------------------
# Scoring — rank-normalized (40 → centroid, 20 → calibration baseline)
# ---------------------------------------------------------------------------
def score_all_rank_normalized(accounts, embed_fn, model):
    result = {k: {"scores": [], "labels": []} for k in ["novel", "fleet", "spoof", "neg", "known"]}
    calib_sigmas = []
    for acc in accounts:
        centroid_events = acc["train_events"][:CENTROID_N]
        calib_events    = acc["train_events"][CENTROID_N:]
        centroid = compute_centroid(centroid_events, embed_fn, model)
        baseline = np.array([cosine_dist(embed_fn(e, model), centroid) for e in calib_events])
        calib_sigmas.append(float(baseline.std()))
        for at, (ev, lbl) in acc["test"].items():
            raw_dist = cosine_dist(embed_fn(ev, model), centroid)
            rank_score = float(np.mean(baseline < raw_dist))
            result[at]["scores"].append(rank_score)
            result[at]["labels"].append(lbl)
    sigmas = np.array(calib_sigmas)
    print(f"  Calibration sigma quartiles: "
          f"P25={np.percentile(sigmas, 25):.4f} "
          f"P50={np.percentile(sigmas, 50):.4f} "
          f"P75={np.percentile(sigmas, 75):.4f}")
    return result

# ---------------------------------------------------------------------------
# Trivial baseline (set-membership)
# ---------------------------------------------------------------------------
def evaluate_trivial(accounts):
    neg_s, neg_l = [], []
    for acc in accounts:
        for at in ["neg", "known"]:
            ev, lbl = acc["test"][at]
            score = 0.0 if any(device_key(d) == device_key(ev) for d in acc["known_devices"]) else 1.0
            neg_s.append(score); neg_l.append(lbl)
    aucs = {}
    for at in ["novel", "fleet", "spoof"]:
        a_s, a_l = [], []
        for acc in accounts:
            ev, lbl = acc["test"][at]
            score = 0.0 if any(device_key(d) == device_key(ev) for d in acc["known_devices"]) else 1.0
            a_s.append(score); a_l.append(lbl)
        try: aucs[at] = roc_auc_score(a_l + neg_l, a_s + neg_s)
        except: aucs[at] = float("nan")
    return aucs

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def bootstrap_auc(result, attack_type, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    neg_s = np.array(result["neg"]["scores"] + result["known"]["scores"])
    neg_l = np.array(result["neg"]["labels"] + result["known"]["labels"])
    att_s = np.array(result[attack_type]["scores"])
    att_l = np.array(result[attack_type]["labels"])
    comb_s = np.concatenate([att_s, neg_s])
    comb_l = np.concatenate([att_l, neg_l])
    n = len(comb_s)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(comb_l[idx])) < 2: continue
        aucs.append(roc_auc_score(comb_l[idx], comb_s[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, name):
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")

def plot_auc_comparison(results_by_model, trivial_aucs):
    attacks = ["novel", "fleet", "spoof"]
    x = np.arange(len(attacks))
    w = 0.18
    colors = {
        "mp-raw":        "#2196F3",
        "mp-rank-norm":  "#4CAF50",
        "cat-raw":       "#FF5722",
        "trivial":       "#9E9E9E",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for (name, result), offset in zip(results_by_model, offsets):
        aucs = [bootstrap_auc(result, at)[0] for at in attacks]
        cis  = [bootstrap_auc(result, at)[1:] for at in attacks]
        bars = ax.bar(x + offset * w, aucs, w, label=name,
                      color=colors.get(name, "#607D8B"), alpha=0.85)
        for xi, (lo, hi) in enumerate(cis):
            cx = x[xi] + offset * w
            ax.plot([cx, cx], [lo, hi], color="black", linewidth=1.2)
    # trivial
    triv_vals = [trivial_aucs[at] for at in attacks]
    ax.bar(x + 1.5 * w, triv_vals, w, label="trivial",
           color=colors["trivial"], alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x + 0.0); ax.set_xticklabels(attacks)
    ax.set_ylabel("ROC-AUC"); ax.set_xlabel("Attack type")
    ax.set_title("Normalized vs Raw Scoring — AUC by Attack Type\n"
                 "(mp-rank-norm: 40-event centroid + 20-event calibration)")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    save_fig(fig, "normalized_score_auc.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Per-User Rank-Normalized Scoring Experiment")
    print("Replication base: robust_config_experiment.py (H2)")
    print("=" * 60)

    print("\n[1/5] Generating synthetic dataset ...")
    accounts = generate_dataset()
    print(f"  {len(accounts)} accounts, {N_TRAIN} train events each")

    print("\n[2/5] Building corpora and training models ...")
    mp_corpus  = build_mp_corpus(accounts)
    cat_corpus = build_cat_corpus(accounts)
    mp_model   = FastText(sentences=mp_corpus,  **ROBUST_KWARGS)
    cat_model  = FastText(sentences=cat_corpus, **ROBUST_KWARGS)
    print(f"  mp  vocab: {len(mp_model.wv):,}   cat vocab: {len(cat_model.wv):,}")

    print("\n[3/5] Scoring ...")
    print("  mp-raw ...")
    mp_raw     = score_all_raw(accounts, embed_mp, mp_model)
    print("  mp-rank-norm ...")
    mp_norm    = score_all_rank_normalized(accounts, embed_mp, mp_model)
    print("  cat-raw ...")
    cat_raw    = score_all_raw(accounts, embed_cat, cat_model)
    trivial_aucs = evaluate_trivial(accounts)

    print("\n[4/5] Bootstrap AUC ...")
    attacks = ["novel", "fleet", "spoof"]
    models = [
        ("mp-raw",       mp_raw),
        ("mp-rank-norm", mp_norm),
        ("cat-raw",      cat_raw),
    ]

    header = f"  {'Model':<16}" + "".join(f"  {at:^22}" for at in attacks)
    print(header)
    print("  " + "-" * (16 + 24 * len(attacks)))
    for name, result in models:
        row = f"  {name:<16}"
        for at in attacks:
            mu, lo, hi = bootstrap_auc(result, at)
            row += f"  {mu:.4f} [{lo:.4f},{hi:.4f}]"
        print(row)
    triv_row = f"  {'trivial':<16}" + "".join(
        f"  {trivial_aucs[at]:.4f}        —       " for at in attacks
    )
    print(triv_row)

    print("\n[5/5] Saving figures ...")
    plot_auc_comparison(models, trivial_aucs)
    print("\nDone.")

if __name__ == "__main__":
    main()
