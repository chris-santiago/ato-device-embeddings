# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "matplotlib==3.10.8",
# ]
# ///
#
# H6 Score-Margin Analysis — C2 Mechanism Diagnostic
# ====================================================
# The C2 finding is that per-user CDF rank-normalization drops spoof PR-AUC
# from 0.888 to 0.224 at 1:100 attack:benign imbalance, while ROC-AUC declines
# by only 0.021 — concealing the collapse.
#
# The mechanism: the CDF rank transform compresses the effective margin between
# attack and benign score distributions. Crucially this is a cross-user effect:
# because rank-normalization maps each user's scores into [0,1] by their own
# CDF, roughly 5% of every user's benign events map to rank > 0.95 — even if
# those events are not anomalous in absolute terms. At 1:100 imbalance, these
# benign events at high percentile swamp the attack events at similar percentile,
# collapsing precision at any useful operating point. ROC-AUC marginalizes over
# the majority class and is insensitive to this effect; PR-AUC is not.
#
# This script makes the mechanism observable through two scale-invariant metrics:
#
#   contamination_rate — fraction of benign events scoring above p10(attack),
#       i.e. above the lower tail of the useful operating range. Using the
#       10th percentile (not median) is critical: after rank-normalization,
#       attacks saturate rank=1.0, making the median threshold the ceiling.
#       At p10(attack)=0.90, ~5% of benign events exceed this threshold
#       (4.98% ± 0.13% across the 5 seeds) — vs 1.28% ± 0.69% for raw scores —
#       a 3.9× increase. (Raw contamination is 0.15% at seed 42 specifically;
#       cross-seed values range 0.15%–2.3%.) Low = good separation.
#
#   robust_margin — 10th percentile of attack scores minus 90th percentile of
#       benign scores. Positive = clean air at the decision boundary; smaller
#       values mean more overlap at the precision-critical threshold.
#
# Both metrics should worsen after rank-normalization, providing direct evidence
# that the PR-AUC collapse is caused by score-margin compression and is not an
# artifact of metric choice.
#
# Data: seeds/seed_{S}/h6/scores.npz — already computed by h6_rerun.py.
#   No experiment re-run is required.
#
# Outputs:
#   aggregate/figures/h6_c2_score_margin.png
#
# Usage:
#   uv run h6_score_margin.py
#   uv run h6_score_margin.py --smoke   # fast schema check, no figure written

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SEEDS       = [42, 123, 456, 789, 2024]
ATTACK_TYPE = "spoof_k1"     # primary C2 metric
SCORERS     = ["mp_raw", "mp_rank_norm"]
COLORS      = {"mp_raw": "#1f77b4", "mp_rank_norm": "#d62728"}

TITLE_FS  = 11
LEGEND_FS = 8

def _apply_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
LABELS      = {"mp_raw": "mp_raw (no normalization)", "mp_rank_norm": "mp_rank_norm (CDF rank)"}


# ---------------------------------------------------------------------------
# Margin computation
# ---------------------------------------------------------------------------
def compute_margins(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    Scale-invariant separation metrics for positive vs. negative score distributions.

    contamination_rate: fraction of benign events scoring above p10(attack). Using the
        10th percentile of attacks (the lower tail of the useful operating range) as
        the threshold reveals compression that median-based thresholds miss — after
        rank normalization, attacks saturate rank=1.0 so the median threshold is the
        ceiling, hiding the 5% benign contamination that floods rank > 0.90.

    robust_margin: p10(attack) - p90(benign). Positive = clean separation at the
        decision boundary; zero or negative = distributions overlap at the critical point.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    p10_attack        = float(np.percentile(pos, 10))
    contamination     = float(np.mean(neg > p10_attack))
    robust_margin     = float(p10_attack - np.percentile(neg, 90))
    return {
        "contamination_rate": contamination,
        "robust_margin":      robust_margin,
        "p10_attack":         p10_attack,
        "mean_pos":           float(pos.mean()),
        "mean_neg":           float(neg.mean()),
        "n_pos":              int((labels == 1).sum()),
        "n_neg":              int((labels == 0).sum()),
    }


def load_seed_margins(npz_path: Path) -> dict:
    results = {}
    with np.load(npz_path) as npz:
        for scorer in SCORERS:
            s_key = f"{ATTACK_TYPE}__{scorer}__s"
            l_key = f"{ATTACK_TYPE}__{scorer}__l"
            if s_key not in npz.files:
                raise KeyError(f"Missing key {s_key} in {npz_path}")
            results[scorer] = compute_margins(npz[s_key], npz[l_key])
    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {path}")


def plot_score_margin(per_seed: list[dict], out_path: Path):
    """
    Two-panel figure showing contamination rate and robust margin per seed.
    Both panels should show rank_norm worse than raw, visually confirming
    the score-margin compression mechanism.
    """
    n = len(per_seed)
    x = np.arange(n)
    width = 0.35
    seed_labels = [str(s) for s in SEEDS[:n]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: contamination rate (higher = worse)
    ax = axes[0]
    for i, scorer in enumerate(SCORERS):
        vals = [r[scorer]["contamination_rate"] for r in per_seed]
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=LABELS[scorer], color=COLORS[scorer], alpha=0.85)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Contamination rate")
    ax.set_title("Fraction of benign events above p10(attack)\n"
                 "(higher = more benign events flood the useful operating range)",
                 fontsize=TITLE_FS)
    ax.set_xticks(x); ax.set_xticklabels(seed_labels)
    ax.legend(fontsize=LEGEND_FS)
    _apply_style(ax)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--",
               label="random baseline (0.50)")
    ax.set_ylim(0, 0.6)

    # Right panel: robust margin (lower = worse)
    ax = axes[1]
    for i, scorer in enumerate(SCORERS):
        vals = [r[scorer]["robust_margin"] for r in per_seed]
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=LABELS[scorer], color=COLORS[scorer], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Robust margin  (p10 attack − p90 benign)")
    ax.set_title("Overlap-free separation at the decision boundary\n"
                 "(lower = more overlap; negative = distributions cross)",
                 fontsize=TITLE_FS)
    ax.set_xticks(x); ax.set_xticklabels(seed_labels)
    ax.legend(fontsize=LEGEND_FS)
    _apply_style(ax)

    # Annotate mean contamination ratio
    raw_cr = np.mean([r["mp_raw"]["contamination_rate"]      for r in per_seed])
    rn_cr  = np.mean([r["mp_rank_norm"]["contamination_rate"] for r in per_seed])
    ratio  = rn_cr / raw_cr if raw_cr > 1e-9 else float("inf")
    axes[0].text(0.98, 0.97,
                 f"Mean contamination\nraw:       {raw_cr:.3f}\nrank_norm: {rn_cr:.3f}\nratio:     {ratio:.1f}×",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle(
        "C2 Mechanism: Score-Margin Compression Under CDF Rank-Normalization\n"
        "spoof k=1, 1:100 imbalance — 5 seeds",
        fontsize=11
    )
    fig.tight_layout()
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="C2 score-margin mechanism diagnostic")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: check schema on seed 42 only, no figure written")
    return p.parse_args()


def main():
    args  = parse_args()
    seeds = [42] if args.smoke else SEEDS

    rerun_root = Path(__file__).resolve().parent.parent.parent
    per_seed   = []

    print(f"{'Seed':<6}  {'Scorer':<22}  {'n_pos':>6}  {'n_neg':>7}  "
          f"{'contam@p10':>14}  {'robust_margin':>14}")
    print("-" * 80)

    for seed in seeds:
        npz_path = rerun_root / "seeds" / f"seed_{seed}" / "h6" / "scores.npz"
        if not npz_path.exists():
            print(f"  WARNING: missing {npz_path}", file=sys.stderr)
            continue
        margins = load_seed_margins(npz_path)
        per_seed.append(margins)

        for scorer in SCORERS:
            m = margins[scorer]
            print(f"{seed:<6}  {scorer:<22}  {m['n_pos']:>6}  {m['n_neg']:>7}  "
                  f"{m['contamination_rate']:>14.4f}  {m['robust_margin']:>14.5f}")

    if not per_seed:
        print("ERROR: no seeds loaded", file=sys.stderr)
        sys.exit(1)

    print("\nCross-seed mean ± std:")
    for scorer in SCORERS:
        cr = np.array([r[scorer]["contamination_rate"] for r in per_seed])
        rm = np.array([r[scorer]["robust_margin"]      for r in per_seed])
        print(f"  {scorer:<22}  contamination={cr.mean():.4f}±{cr.std():.4f}  "
              f"robust_margin={rm.mean():.5f}±{rm.std():.5f}")

    raw_cr = np.mean([r["mp_raw"]["contamination_rate"]      for r in per_seed])
    rn_cr  = np.mean([r["mp_rank_norm"]["contamination_rate"] for r in per_seed])
    if raw_cr > 1e-9:
        print(f"\nMean contamination ratio (rank_norm / raw): {rn_cr / raw_cr:.1f}×")

    if args.smoke:
        raw = per_seed[0]["mp_raw"]
        rn  = per_seed[0]["mp_rank_norm"]
        assert rn["contamination_rate"] > raw["contamination_rate"], \
            (f"Smoke FAIL: rank_norm contamination ({rn['contamination_rate']:.4f}) "
             f"should exceed raw ({raw['contamination_rate']:.4f})")
        assert rn["robust_margin"] < raw["robust_margin"], \
            (f"Smoke FAIL: rank_norm robust margin ({rn['robust_margin']:.5f}) "
             f"should be lower than raw ({raw['robust_margin']:.5f})")
        print("\nSmoke PASS")
        return

    out_path = rerun_root / "aggregate" / "figures" / "h6_c2_score_margin.png"
    plot_score_margin(per_seed, out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
