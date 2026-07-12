# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "matplotlib==3.10.8",
# ]
# ///
#
# Figures for the OOV-regime sweep. Reads aggregate/ablations_summary.json (produced by
# aggregate_ablations.py after oov_regime.py has run across seeds) and writes:
#   - oov_degradation_<arm>.png : spoof_k1 ROC-AUC vs OOV level, mp_raw vs lik_best vs
#     trivial, with cross-seed ±std bands.
#   - oov_delta_attribution.png : FastText−incumbent dROC per arm, and the subword-
#     attribution curve gap(morphological) − gap(arbitrary).
#
# Usage:
#   uv run experiments/rerun/ablations/oov_figures.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "aggregate" / "ablations_summary.json"
FIG_DIR = ROOT / "figures"

SERIES = [
    ("mp_raw", "FastText (mp_raw)", "#1f77b4"),
    ("lik_best", "Incumbent (lik_best)", "#d62728"),
    ("trivial", "Trivial (set-membership)", "#7f7f7f"),
]
METRICS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


def _xy(oov, arm, key):
    xs = [float(lvl) for lvl in oov["levels"]]
    means = [oov["metrics"][arm][lvl]["spoof_k1"][key]["mean"] for lvl in oov["levels"]]
    stds = [oov["metrics"][arm][lvl]["spoof_k1"][key]["std"] for lvl in oov["levels"]]
    return np.array(xs), np.array(means), np.array(stds)


def _prevalence(oov):
    nr = oov.get("neg_ratio")
    return f"  (1:{nr} imbalance)" if nr else ""


def degradation_figure(oov, arm, metric):
    fig, ax = plt.subplots(figsize=(7, 5))
    ymin = 1.0
    for name, label, color in SERIES:
        xs, means, stds = _xy(oov, arm, f"{name}_{metric}")
        ax.plot(xs, means, "-o", color=color, label=label, linewidth=2)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)
        ymin = min(ymin, float(np.min(means - stds)))
    ax.set_xlabel("OOV injection level  p  (fraction of events, both classes)")
    ax.set_ylabel(f"spoof_k1 {METRICS[metric]}  (mean ± std across seeds)")
    ax.set_title(f"OOV degradation — {arm} arm — {METRICS[metric]}{_prevalence(oov)}")
    ax.set_ylim(max(0.0, ymin - 0.05), 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = FIG_DIR / f"oov_degradation_{arm}_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def delta_attribution_figure(oov):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    arm_colors = {"morphological": "#2ca02c", "arbitrary": "#9467bd",
                  "region_morph": "#17becf", "region_arb": "#e377c2"}
    for arm in oov["arms"]:
        xs = np.array([float(lvl) for lvl in oov["levels"]])
        pts = np.array([oov["deltas"][arm][lvl]["pr"]["point"]["mean"] for lvl in oov["levels"]])
        sds = np.array([oov["deltas"][arm][lvl]["pr"]["point"]["std"] for lvl in oov["levels"]])
        c = arm_colors.get(arm)
        ax1.plot(xs, pts, "-o", color=c, label=arm, linewidth=2)
        ax1.fill_between(xs, pts - sds, pts + sds, color=c, alpha=0.15)
    ax1.axhline(0.0, ls="--", color="black", lw=1)
    ax1.set_xlabel("OOV level  p")
    ax1.set_ylabel("dPR  (FastText − incumbent), spoof_k1")
    ax1.set_title(f"FastText advantage vs OOV level{_prevalence(oov)}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    pair_colors = {"cross_feature": "#ff7f0e", "same_feature_region": "#1f77b4"}
    for pname, per_lvl in oov["subword_attribution"].items():
        levels = list(per_lvl)
        xs = np.array([float(lvl) for lvl in levels])
        means = np.array([per_lvl[lvl]["pr"]["mean"] for lvl in levels])
        stds = np.array([per_lvl[lvl]["pr"]["std"] for lvl in levels])
        c = pair_colors.get(pname)
        ax2.plot(xs, means, "-o", color=c, label=pname, linewidth=2)
        ax2.fill_between(xs, means - stds, means + stds, color=c, alpha=0.15)
    ax2.axhline(0.0, ls="--", color="black", lw=1)
    ax2.set_xlabel("OOV level  p")
    ax2.set_ylabel("gap(morph) − gap(arbitrary),  dPR")
    ax2.set_title("Subword attribution\n(same_feature isolates morphology; ~0 => dilution, not n-grams)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    out = FIG_DIR / "oov_delta_attribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    if not SUMMARY.exists():
        raise SystemExit(f"ERROR: {SUMMARY} not found — run aggregate_ablations.py first.")
    with open(SUMMARY) as f:
        summary = json.load(f)
    oov = summary.get("oov_regime")
    if not oov:
        raise SystemExit("ERROR: no oov_regime section in summary — run oov_regime.py across seeds.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for arm in oov["arms"]:
        for metric in METRICS:
            degradation_figure(oov, arm, metric)
    delta_attribution_figure(oov)


if __name__ == "__main__":
    main()
