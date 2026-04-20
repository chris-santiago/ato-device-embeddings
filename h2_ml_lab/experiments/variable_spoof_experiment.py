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
# Variable-Spoof Experiment
# =========================
# Tests whether mean-pool + rank-normalization detects spoofed logins across
# a spectrum of attacker sophistication, measured by the number of device
# fields the attacker fails to match (1–3 correlated field mistakes).
#
# Spoof variants (per account):
#   k=1  tz only changed          — VPN with spoofed timezone; everything else matches
#   k=2  tz + network changed     — datacenter VPN; network type exposes hosting provider
#   k=3  tz + network + screen    — emulated device; screen resolution also mismatches
#
# Negative class (same in all comparisons):
#   "neg"   — primary OS/browser/tz/lang, different network/screen (new device enrollment)
#   "known" — exact primary device
#
# Comparisons: mp-raw, mp-rank-norm, trivial (set-membership)
# Rank-norm: first CENTROID_N=40 events → centroid; last CALIB_N=20 → baseline CDF

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

N_BOOTSTRAP = 1000
CENTROID_N  = 40
CALIB_N     = 20

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
# Data generation
# ---------------------------------------------------------------------------
def make_spoof(primary, rng, k):
    """Return a spoof device that differs from primary in exactly k fields.

    k=1: tz only (VPN timezone mismatch)
    k=2: tz + network (datacenter VPN)
    k=3: tz + network + screen (emulated/different device)
    """
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
        neg = dict(primary)
        neg["network"] = rng.choice([n for n in FEATURES["network"] if n != primary["network"]])
        neg["screen"]  = rng.choice([s for s in FEATURES["screen"]  if s != primary["screen"]])
        accounts.append({
            "id": acct_id, "known_devices": known, "primary": primary,
            "train_events": train_events, "is_fleet": is_fleet,
            "fleet_device": fleet_device,
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
# Corpus + model
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

ROBUST_KWARGS = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    seed=SEED, workers=1,
)

def embed_mp(event, model):
    return np.mean([model.wv[make_token(f, event[f])] for f in FEATURE_ORDER], axis=0)

# ---------------------------------------------------------------------------
# Scoring — raw
# ---------------------------------------------------------------------------
def score_all_raw(accounts, model):
    keys = ["spoof_k1", "spoof_k2", "spoof_k3", "neg", "known"]
    result = {k: {"scores": [], "labels": []} for k in keys}
    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_mp, model)
        for at, (ev, lbl) in acc["test"].items():
            result[at]["scores"].append(cosine_dist(embed_mp(ev, model), centroid))
            result[at]["labels"].append(lbl)
    return result

# ---------------------------------------------------------------------------
# Scoring — rank-normalised
# ---------------------------------------------------------------------------
def score_all_rank_norm(accounts, model):
    keys = ["spoof_k1", "spoof_k2", "spoof_k3", "neg", "known"]
    result = {k: {"scores": [], "labels": []} for k in keys}
    calib_sigmas = []
    for acc in accounts:
        centroid_events = acc["train_events"][:CENTROID_N]
        calib_events    = acc["train_events"][CENTROID_N:]
        centroid = compute_centroid(centroid_events, embed_mp, model)
        baseline = np.array([cosine_dist(embed_mp(e, model), centroid) for e in calib_events])
        calib_sigmas.append(float(baseline.std()))
        for at, (ev, lbl) in acc["test"].items():
            raw_dist   = cosine_dist(embed_mp(ev, model), centroid)
            rank_score = float(np.mean(baseline < raw_dist))
            result[at]["scores"].append(rank_score)
            result[at]["labels"].append(lbl)
    sigmas = np.array(calib_sigmas)
    print(f"  Calibration sigma: P25={np.percentile(sigmas,25):.4f} "
          f"P50={np.percentile(sigmas,50):.4f} P75={np.percentile(sigmas,75):.4f}")
    return result

# ---------------------------------------------------------------------------
# Trivial baseline
# ---------------------------------------------------------------------------
def evaluate_trivial(accounts):
    neg_s, neg_l = [], []
    for acc in accounts:
        for at in ["neg", "known"]:
            ev, lbl = acc["test"][at]
            score = 0.0 if any(device_key(d) == device_key(ev) for d in acc["known_devices"]) else 1.0
            neg_s.append(score); neg_l.append(lbl)
    aucs = {}
    for at in ["spoof_k1", "spoof_k2", "spoof_k3"]:
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
def bootstrap_auc(result, attack_type, neg_keys=("neg", "known"),
                  n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    neg_s = np.concatenate([result[k]["scores"] for k in neg_keys])
    neg_l = np.concatenate([result[k]["labels"] for k in neg_keys])
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
# Figure
# ---------------------------------------------------------------------------
def plot_results(raw_result, norm_result, trivial_aucs):
    spoof_keys  = ["spoof_k1", "spoof_k2", "spoof_k3"]
    labels      = ["k=1 (tz only)", "k=2 (tz+net)", "k=3 (tz+net+screen)"]
    colors      = {"mp-raw": "#2196F3", "mp-rank-norm": "#4CAF50", "trivial": "#9E9E9E"}
    x = np.arange(len(spoof_keys))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, result, offset) in [
        ("mp-raw",       raw_result,  -1),
        ("mp-rank-norm", norm_result,  0),
    ]:
        aucs = [bootstrap_auc(result, at)[0] for at in spoof_keys]
        cis  = [bootstrap_auc(result, at)[1:] for at in spoof_keys]
        ax.bar(x + offset * w, aucs, w, label=name,
               color=colors[name], alpha=0.85)
        for xi, (lo, hi) in enumerate(cis):
            cx = x[xi] + offset * w
            ax.plot([cx, cx], [lo, hi], color="black", linewidth=1.2)

    triv = [trivial_aucs[at] for at in spoof_keys]
    ax.bar(x + 1 * w, triv, w, label="trivial", color=colors["trivial"], alpha=0.85)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("ROC-AUC"); ax.set_xlabel("Spoof difficulty (fields mismatched)")
    ax.set_title("Spoof Detection vs. Attacker Sophistication\n"
                 "mean-pool raw vs rank-norm vs trivial")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    out = FIGURES_DIR / "variable_spoof_auc.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Variable-Spoof Experiment (k=1,2,3 field mistakes)")
    print("mean-pool raw vs rank-norm  |  SEED=42, N=400 accounts")
    print("=" * 60)

    print("\n[1/5] Generating dataset ...")
    accounts = generate_dataset()
    print(f"  {len(accounts)} accounts, {N_TRAIN} train events each")

    print("\n[2/5] Building corpus and training model ...")
    mp_corpus = build_mp_corpus(accounts)
    model     = FastText(sentences=mp_corpus, **ROBUST_KWARGS)
    print(f"  vocab: {len(model.wv):,} tokens")

    print("\n[3/5] Scoring ...")
    print("  mp-raw ...")
    raw_result  = score_all_raw(accounts, model)
    print("  mp-rank-norm ...")
    norm_result = score_all_rank_norm(accounts, model)
    trivial_aucs = evaluate_trivial(accounts)

    print("\n[4/5] Bootstrap AUC ...")
    spoof_keys = ["spoof_k1", "spoof_k2", "spoof_k3"]
    labels     = {
        "spoof_k1": "k=1  tz only",
        "spoof_k2": "k=2  tz+net",
        "spoof_k3": "k=3  tz+net+scr",
    }
    header = f"  {'Spoof type':<18}" + "".join(f"  {'mp-raw':^22}  {'mp-rank-norm':^22}  {'trivial':>8}" for _ in [""])
    print(header)
    print("  " + "-" * 82)
    for sk in spoof_keys:
        raw_mu,  raw_lo,  raw_hi  = bootstrap_auc(raw_result,  sk)
        norm_mu, norm_lo, norm_hi = bootstrap_auc(norm_result, sk)
        triv = trivial_aucs[sk]
        print(f"  {labels[sk]:<18}  "
              f"{raw_mu:.4f} [{raw_lo:.4f},{raw_hi:.4f}]  "
              f"{norm_mu:.4f} [{norm_lo:.4f},{norm_hi:.4f}]  "
              f"{triv:.4f}")

    print("\n[5/5] Saving figures ...")
    plot_results(raw_result, norm_result, trivial_aucs)
    print("\nDone.")

if __name__ == "__main__":
    main()
