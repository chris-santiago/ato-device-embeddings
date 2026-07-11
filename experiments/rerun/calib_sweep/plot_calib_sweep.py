# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "matplotlib==3.10.8",
# ]
# ///
#
# Plot the calibration-window ablation.
# Panel A: spoof-k1 PR-AUC vs calibration window size, mp_raw (embedding-quality
#          control) vs mp_rank_norm, mean +/- std across seeds.
# Panel B: rank-norm contamination rate vs calib size, with the 1/calib pure-
#          quantization prediction overlaid -- if observed contamination tracks
#          1/calib the floor is a small-sample artifact; if it plateaus above,
#          the floor is structural.
# Style mirrors experiments/rerun/scripts/h6/h6_figures.py (fig_c2_pr_curves).

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT    = Path(__file__).resolve().parent
AGG_DIR = ROOT / "aggregate"
FIG_DIR = AGG_DIR / "figures"

COLORS = {"mp_raw": "#2166ac", "mp_rank_norm": "#d6604d"}
LABELS = {"mp_raw": "mp_raw (no normalization — embedding control)",
          "mp_rank_norm": "mp_rank_norm (per-user CDF rank)"}
QUANT_COLOR = "#4d4d4d"
TITLE_FS, LEGEND_FS = 11, 8


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _series(summary, scorer, key):
    calibs = summary["calib_sizes"]
    means, stds = [], []
    for c in calibs:
        blk = summary["by_calib"][str(c)]["scorers"][scorer]
        s = blk[key]
        means.append(s["mean"])
        stds.append(s["std"])
    return np.array(calibs), np.array(means), np.array(stds)


def main() -> None:
    summary = json.loads((AGG_DIR / "calib_sweep_summary.json").read_text())
    calibs = summary["calib_sizes"]
    n_seeds = summary["by_calib"][str(calibs[0])]["gap_pr_auc"]["n_seeds"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    # -- Panel A: PR-AUC vs calib size --
    for scorer in ("mp_raw", "mp_rank_norm"):
        x, m, s = _series(summary, scorer, "pr_auc")
        axA.errorbar(x, m, yerr=s, marker="o", lw=1.8, capsize=4,
                     color=COLORS[scorer], label=LABELS[scorer],
                     ecolor="#555555", elinewidth=1.2)
    axA.set_xscale("log")
    axA.set_xticks(calibs)
    axA.set_xticklabels([str(c) for c in calibs])
    axA.set_xlabel("Calibration window size (events per account)")
    axA.set_ylabel("Spoof-k1 PR-AUC")
    axA.set_ylim(0, 1.02)
    axA.set_title("PR-AUC vs. calibration window (1:100 imbalance)", fontsize=TITLE_FS)
    axA.legend(fontsize=LEGEND_FS, loc="center right")
    _style(axA)

    # -- Panel B: contamination vs calib size, with 1/calib prediction --
    x, m, s = _series(summary, "mp_rank_norm", "contamination_rate")
    axB.errorbar(x, m, yerr=s, marker="o", lw=1.8, capsize=4,
                 color=COLORS["mp_rank_norm"], label="observed contamination (rank_norm)",
                 ecolor="#555555", elinewidth=1.2)
    axB.plot(x, 1.0 / x, ls="--", lw=1.5, color=QUANT_COLOR, marker="s", ms=4,
             label="1/calib (pure-quantization prediction)")
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xticks(calibs)
    axB.set_xticklabels([str(c) for c in calibs])
    axB.set_xlabel("Calibration window size (events per account)")
    axB.set_ylabel("Benign fraction above p10(attack)")
    axB.set_title("Rank floor vs. quantization prediction", fontsize=TITLE_FS)
    axB.legend(fontsize=LEGEND_FS, loc="upper right")
    _style(axB)

    fig.text(0.99, 0.01, f"n={n_seeds} seeds", ha="right", va="bottom",
             fontsize=7, color="#888888")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "calib_sweep_pr_auc.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
