# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim==4.4.0",
#   "numpy==2.2.6",
#   "scikit-learn==1.7.2",
# ]
# ///
#
# H2 Degenerate-Config Downstream Scoring — C1 Consequence Diagnostic
# ====================================================================
# The 2x2 factorial (h2_rerun.py, T8) established that per-event corpus
# construction collapses within-feature embeddings on 5/5 seeds. What the
# rerun never measured is the DOWNSTREAM consequence of that collapse:
# detection AUC per attack type under each factorial cell. The only prior
# evidence was single-seed (h2_ml_lab REPORT.md: spoof 0.384, novel 0.880,
# fleet 0.922 under the degenerate config) — issue 006ae95f.
#
# This script closes that gap. For each seed it regenerates the identical
# H2 dataset (same generate_dataset as h2_rerun.py), trains the four
# factorial cells plus the historical degenerate PoC config (CBOW +
# per-event, epochs=10), scores every model downstream with mean-pool
# centroid-cosine, and reports ROC/PR AUC per attack type.
#
# Consistency cross-checks (both asserted):
#   - T8 collapse flags per cell must match h2_rerun.py results.json.
#   - The sg+per-account cell's spoof AUC must match results.json's
#     t2.mean_pool_spoof_auc within tolerance (bootstrap-mean vs point).
#
# Outputs:
#   aggregate/h2_degenerate_downstream.json
#
# Usage:
#   uv run h2_degenerate_downstream.py
#   uv run h2_degenerate_downstream.py --smoke   # 50 accounts, seed 42 only

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gensim.models import FastText
from sklearn.metrics import average_precision_score, roc_auc_score

SEEDS = [42, 123, 456, 789, 2024]

N_ACCOUNTS = 400
N_TRAIN = 60
FLEET_FRAC = 0.25
N_ACCOUNTS_SMOKE = 50

T8_COLLAPSE_THRESHOLD = 0.9  # decision a2b73375
AUC_MATCH_TOLERANCE = 0.02   # bootstrap-mean (results.json) vs point AUC (here)

FEATURES = {
    "os":      ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz":      ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang":    ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "network": ["wifi", "lte", "5g", "broadband"],
    "screen":  ["small", "medium", "large", "xlarge"],
}
FEATURE_ORDER = ["os", "browser", "tz", "lang", "network", "screen"]

# ---------------------------------------------------------------------------
# Helpers — identical to h2_rerun.py
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
# Data generation — identical to h2_rerun.py
# ---------------------------------------------------------------------------
def generate_dataset(seed, n_accounts=N_ACCOUNTS, n_train=N_TRAIN):
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
                seen.add(k); known.append(d)
        primary = known[0]
        weights = zipf_weights(n_dev)
        train_events = []
        for _ in range(n_train):
            idx = rng.choice(n_dev, p=weights)
            train_events.append(dict(known[idx]))
        is_fleet = rng.random() < FLEET_FRAC
        if is_fleet:
            replace_idx = int(rng.integers(0, len(train_events)))
            train_events[replace_idx] = dict(fleet_device)
        # Novel: foreign OS, tz, language
        novel = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
        attempts = 0
        while (novel["os"] == primary["os"] or novel["tz"] == primary["tz"]
               or novel["lang"] == primary["lang"]) and attempts < 1000:
            novel = {f: rng.choice(FEATURES[f]) for f in FEATURE_ORDER}
            attempts += 1
        # Spoof: 3-field — tz changed + randomised network/screen (issue ce3a6063)
        spoof = dict(primary)
        spoof["tz"]      = rng.choice([t for t in FEATURES["tz"] if t != primary["tz"]])
        spoof["network"] = rng.choice(FEATURES["network"])
        spoof["screen"]  = rng.choice(FEATURES["screen"])
        # Negative: same OS/browser/tz/lang, different network/screen
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
# Corpora — identical to h2_rerun.py
# ---------------------------------------------------------------------------
def build_mp_corpus_per_account(accounts):
    sentences = []
    for acc in accounts:
        flat = []
        for e in acc["train_events"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences

def build_per_event_corpus(accounts):
    return [device_to_tokens(e) for acc in accounts for e in acc["train_events"]]

# ---------------------------------------------------------------------------
# Model cells — kwargs identical to h2_rerun.py
# ---------------------------------------------------------------------------
def _factorial_kwargs(seed, sg):
    return dict(
        vector_size=64, window=6, sg=sg, negative=10,
        min_count=1, epochs=20, min_n=3, max_n=6,
        seed=seed, workers=1,
    )

def _degenerate_kwargs(seed):
    return dict(
        vector_size=64, window=6, sg=0,
        min_count=1, epochs=10, min_n=3, max_n=6,
        seed=seed, workers=1,
    )

CELLS = ["sg_per_account", "cbow_per_account", "sg_per_event", "cbow_per_event",
         "degenerate_poc"]

def train_cell(cell, accounts, seed):
    if cell == "degenerate_poc":
        return FastText(sentences=build_per_event_corpus(accounts), **_degenerate_kwargs(seed))
    sg = 1 if cell.startswith("sg") else 0
    corpus = (build_mp_corpus_per_account(accounts) if cell.endswith("per_account")
              else build_per_event_corpus(accounts))
    return FastText(sentences=corpus, **_factorial_kwargs(seed, sg))

# ---------------------------------------------------------------------------
# Downstream scoring — identical mean-pool path to h2_rerun.py
# ---------------------------------------------------------------------------
def embed_mp(event, model):
    return np.mean([model.wv[make_token(f, event[f])] for f in FEATURE_ORDER], axis=0)

def score_all(accounts, model):
    result = {k: {"scores": [], "labels": []} for k in ["novel", "fleet", "spoof", "neg", "known"]}
    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_mp, model)
        for at, (ev, lbl) in acc["test"].items():
            result[at]["scores"].append(cosine_dist(embed_mp(ev, model), centroid))
            result[at]["labels"].append(lbl)
    return result

def compute_aucs(result, attack_type):
    neg_s = result["neg"]["scores"] + result["known"]["scores"]
    neg_l = result["neg"]["labels"] + result["known"]["labels"]
    comb_s = result[attack_type]["scores"] + neg_s
    comb_l = result[attack_type]["labels"] + neg_l
    return (float(roc_auc_score(comb_l, comb_s)),
            float(average_precision_score(comb_l, comb_s)))

# ---------------------------------------------------------------------------
# T8 — identical to h2_rerun.py (cross-check against results.json)
# ---------------------------------------------------------------------------
def within_feature_mean(model):
    tokens, feat_types = [], []
    for f in FEATURE_ORDER:
        for v in FEATURES[f]:
            tokens.append(make_token(f, v)); feat_types.append(f)
    vecs = np.array([model.wv[t] for t in tokens])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / (norms + 1e-10)
    sim_mat = vecs_n @ vecs_n.T
    feat_types = np.array(feat_types)
    within = [sim_mat[i, j]
              for i in range(len(tokens)) for j in range(i+1, len(tokens))
              if feat_types[i] == feat_types[j]]
    return float(np.mean(within))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Degenerate-config downstream AUC across the 2x2 factorial")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 50 accounts, seed 42 only, assert schema, no JSON written")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [42] if args.smoke else SEEDS
    n_accounts = N_ACCOUNTS_SMOKE if args.smoke else N_ACCOUNTS

    rerun_root = Path(__file__).resolve().parent.parent.parent
    attacks = ["novel", "fleet", "spoof"]
    per_seed: dict = {cell: {at: {"roc": [], "pr": []} for at in attacks} for cell in CELLS}
    collapse_flags: dict = {cell: [] for cell in CELLS}

    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        accounts = generate_dataset(seed, n_accounts=n_accounts)
        for cell in CELLS:
            model = train_cell(cell, accounts, seed)
            wfm = within_feature_mean(model)
            collapsed = wfm > T8_COLLAPSE_THRESHOLD
            collapse_flags[cell].append(collapsed)
            result = score_all(accounts, model)
            line = f"  {cell:18s} within={wfm:.4f} collapse={collapsed!s:5s} "
            for at in attacks:
                roc, pr = compute_aucs(result, at)
                per_seed[cell][at]["roc"].append(roc)
                per_seed[cell][at]["pr"].append(pr)
                line += f" {at}={roc:.4f}"
            print(line)

        # Cross-check the sg+per-account cell against the rerun's results.json
        # (full runs only — smoke uses 50 accounts, so AUCs legitimately differ)
        results_path = rerun_root / "seeds" / f"seed_{seed}" / "h2" / "results.json"
        if not args.smoke and results_path.exists():
            ref = json.loads(results_path.read_text())
            ref_spoof = ref["t2_window_sweep"]["mean_pool_spoof_auc"]
            here_spoof = per_seed["sg_per_account"]["spoof"]["roc"][-1]
            delta = abs(here_spoof - ref_spoof)
            assert delta < AUC_MATCH_TOLERANCE, (
                f"seed {seed}: sg_per_account spoof AUC {here_spoof:.4f} deviates "
                f"from results.json {ref_spoof:.4f} by {delta:.4f}")
            ref_collapse = {
                "sg_per_account":  ref["t8_token_similarity"]["factorial_2x2"]["sg_per_account"]["collapse_detected"],
                "cbow_per_account": ref["t8_token_similarity"]["factorial_2x2"]["cbow_per_account"]["collapse_detected"],
                "sg_per_event":    ref["t8_token_similarity"]["factorial_2x2"]["sg_per_event"]["collapse_detected"],
                "cbow_per_event":  ref["t8_token_similarity"]["factorial_2x2"]["cbow_per_event"]["collapse_detected"],
                "degenerate_poc":  ref["t8_token_similarity"]["degenerate_config"]["collapse_detected"],
            }
            for cell in CELLS:
                assert collapse_flags[cell][-1] == ref_collapse[cell], (
                    f"seed {seed}: {cell} collapse flag mismatch vs results.json")
            print(f"  cross-check vs results.json: OK "
                  f"(spoof delta {delta:.4f}, collapse flags match)")
        elif not args.smoke:
            print(f"  WARNING: {results_path} missing — cross-check skipped")

    print("\n=== 5-seed aggregate (mean ± std, ddof=0) ===")
    print(f"  {'cell':18s} {'collapse':>9s}  " +
          "  ".join(f"{at+'_roc':>12s}" for at in attacks) +
          "  " + "  ".join(f"{at+'_pr':>12s}" for at in attacks))
    summary: dict = {}
    for cell in CELLS:
        n_col = sum(collapse_flags[cell])
        row = f"  {cell:18s} {n_col}/{len(seeds):>6}  "
        summary[cell] = {"collapse_seeds": n_col}
        for kind in ["roc", "pr"]:
            for at in attacks:
                a = np.array(per_seed[cell][at][kind])
                summary[cell][f"{at}_{kind}_auc"] = {
                    "mean": float(a.mean()), "std": float(a.std(ddof=0)),
                    "min": float(a.min()), "max": float(a.max()),
                }
                row += f" {a.mean():.4f}±{a.std(ddof=0):.3f}"
        print(row)

    if args.smoke:
        assert summary["cbow_per_event"]["collapse_seeds"] == len(seeds), "smoke: cbow_per_event should collapse"
        assert summary["sg_per_account"]["collapse_seeds"] == 0, "smoke: sg_per_account should not collapse"
        assert (summary["sg_per_account"]["spoof_roc_auc"]["mean"]
                > summary["cbow_per_event"]["spoof_roc_auc"]["mean"]), \
            "smoke: robust spoof AUC should exceed collapsed spoof AUC"
        print("\nSMOKE PASS")
        return

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "description": ("Downstream mean-pool centroid-cosine AUC per attack type for each "
                        "2x2 factorial cell plus the historical degenerate PoC config "
                        "(CBOW + per-event, epochs=10). Closes the gap flagged in issue "
                        "006ae95f: the collapse->downstream-damage claim was previously "
                        "single-seed (h2_ml_lab)."),
        "per_seed": {cell: {at: {k: per_seed[cell][at][k] for k in ["roc", "pr"]}
                            for at in attacks} for cell in CELLS},
        "aggregate": summary,
    }
    out_path = rerun_root / "aggregate" / "h2_degenerate_downstream.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
