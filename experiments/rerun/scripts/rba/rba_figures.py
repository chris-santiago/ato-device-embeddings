# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "matplotlib==3.10.8",
# ]
# ///
#
# RBA Publication Figures — Aggregate Across Seeds
# =================================================
# Reads results.json (and results_split40/60.json) from each completed seed
# under experiments/rerun/seeds/seed_*/rba/ and produces 4 summary figures
# in experiments/rerun/aggregate/figures/.
#
# Works with any number of completed seeds (1–5). Error bars appear once ≥2
# seeds are present; with 1 seed, point values are shown without error bars.
#
# Figures produced:
#   rba_summary_auc.png       — ROC-AUC and PR-AUC bar chart, mean ± std
#   rba_t6_compactness.png    — T6 compactness bar chart, mean ± std
#   rba_t8_token_similarity.png — T8 within vs cross-feature sim, mean ± std
#   rba_split_sensitivity.png — ROC-AUC across 50/50, 40/60, 60/40 splits
#
# Usage:
#   uv run experiments/rerun/scripts/rba/rba_figures.py
#   uv run experiments/rerun/scripts/rba/rba_figures.py --seeds-root /path/to/seeds

import argparse
import json
import shutil
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RERUN_ROOT  = Path(__file__).resolve().parent.parent.parent
SEEDS_ROOT  = RERUN_ROOT / "seeds"
AGG_FIGURES = RERUN_ROOT / "aggregate" / "figures"

COLORS = {
    "mean_pool": "#2196F3",
    "concat":    "#FF5722",
    "trivial":   "#9E9E9E",
    "within":    "#2196F3",
    "cross":     "#9E9E9E",
}

SEEDS_ORDER = [42, 123, 456, 789, 2024]

EXPECTED_FIGURES = [
    "rba_summary_auc.png",
    "rba_t6_compactness.png",
    "rba_t8_token_similarity.png",
    "rba_split_sensitivity.png",
]

# ---------------------------------------------------------------------------
# Smoke helpers
# ---------------------------------------------------------------------------

def make_smoke_record(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    def j(base, w=0.005): return float(base + rng.uniform(-w, w))
    return {
        "seed": seed, "timestamp": "2026-04-21T00:00:00Z",
        "split_percentile": 50, "n_ato_test_events": 500,
        "auc": {
            "mean_pool": {"roc_auc": j(0.850), "pr_auc": j(0.032), "ci_lower": j(0.025), "ci_upper": j(0.040)},
            "concat":    {"roc_auc": j(0.820), "pr_auc": j(0.025), "ci_lower": j(0.018), "ci_upper": j(0.032)},
            "trivial":   {"roc_auc": j(0.780), "pr_auc": j(0.003)},
        },
        "t6_compactness": {"mean_pool": j(0.036), "concat": j(0.047)},
        "t8_token_similarity": {
            "within_feature_mean": j(0.563), "cross_feature_mean": j(0.490),
            "within_cross_ratio": j(1.149), "collapse_detected": False,
        },
        "h2_replicated": True,
    }

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seed_results(seeds_root: Path) -> list[dict]:
    """Load results.json from every completed seed, in canonical seed order."""
    records = []
    for seed in SEEDS_ORDER:
        path = seeds_root / f"seed_{seed}" / "rba" / "results.json"
        if path.exists():
            with open(path) as f:
                records.append(json.load(f))
    if not records:
        sys.exit(f"ERROR: no completed RBA seed results found under {seeds_root}")
    print(f"  Loaded {len(records)} seed(s): "
          f"{[r['seed'] for r in records]}")
    return records


def load_split_results(seeds_root: Path, filename: str) -> list[dict]:
    """Load a split-variant results file (results_split40.json etc.) for completed seeds."""
    records = []
    for seed in SEEDS_ORDER:
        path = seeds_root / f"seed_{seed}" / "rba" / filename
        if path.exists():
            with open(path) as f:
                records.append(json.load(f))
    return records


def stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, std). std=0 when only 1 value."""
    a = np.array(values, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=0))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _errbar(ax, x, mean, std, n_seeds):
    """Draw an error bar only when ≥2 seeds are present."""
    if n_seeds >= 2 and std > 0:
        ax.errorbar(x, mean, yerr=std, fmt="none",
                    color="black", capsize=4, linewidth=1.2, zorder=5)


def _label(ax, x, mean, std, n_seeds, offset=0.01):
    label = f"{mean:.3f}" if n_seeds == 1 else f"{mean:.3f}\n±{std:.3f}"
    ax.text(x, mean + std + offset if n_seeds >= 2 else mean + offset,
            label, ha="center", va="bottom", fontsize=7.5)


# ---------------------------------------------------------------------------
# Figure 1: Summary AUC (ROC-AUC and PR-AUC)
# ---------------------------------------------------------------------------

def fig_summary_auc(records: list[dict], out_dir: Path) -> None:
    n = len(records)
    scorers = ["mean_pool", "concat", "trivial"]
    labels  = ["mean-pool", "concat", "trivial"]
    colors  = [COLORS["mean_pool"], COLORS["concat"], COLORS["trivial"]]
    x = np.arange(len(scorers))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, metric_key, metric_label in [
        (axes[0], "roc_auc", "ROC-AUC"),
        (axes[1], "pr_auc",  "PR-AUC"),
    ]:
        means, stds = [], []
        for s in scorers:
            vals = [r["auc"][s][metric_key] for r in records]
            m, sd = stats(vals)
            means.append(m)
            stds.append(sd)

        ax.bar(x, means, color=colors, alpha=0.85, zorder=3)
        for i, (m, sd) in enumerate(zip(means, stds)):
            _errbar(ax, x[i], m, sd, n)
            _label(ax, x[i], m, sd, n)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel(metric_label)
        ax.set_title(f"RBA: {metric_label} (n={n} seeds)\n"
                     f"sg=1, per-account corpus, 50/50 split")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.suptitle("RBA Replication Check — AUC Summary", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "rba_summary_auc.png", out_dir)


# ---------------------------------------------------------------------------
# Figure 2: T6 Compactness
# ---------------------------------------------------------------------------

def fig_t6_compactness(records: list[dict], out_dir: Path) -> None:
    n = len(records)
    scorers = ["mean_pool", "concat"]
    labels  = ["mean-pool", "concat"]
    colors  = [COLORS["mean_pool"], COLORS["concat"]]
    x = np.arange(len(scorers))

    means, stds = [], []
    for s in scorers:
        vals = [r["t6_compactness"][s] for r in records]
        m, sd = stats(vals)
        means.append(m)
        stds.append(sd)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x, means, color=colors, alpha=0.85, zorder=3)
    for i, (m, sd) in enumerate(zip(means, stds)):
        _errbar(ax, x[i], m, sd, n)
        _label(ax, x[i], m, sd, n, offset=0.001)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-user centroid compactness\n(mean cosine dist, lower = tighter)")
    ax.set_title(f"T6: Per-User Compactness — RBA (n={n} seeds)\n"
                 "sg=1, per-account corpus")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    save_fig(fig, "rba_t6_compactness.png", out_dir)


# ---------------------------------------------------------------------------
# Figure 3: T8 Token Similarity
# ---------------------------------------------------------------------------

def fig_t8_token_similarity(records: list[dict], out_dir: Path) -> None:
    n = len(records)

    within_vals = [r["t8_token_similarity"]["within_feature_mean"] for r in records]
    cross_vals  = [r["t8_token_similarity"]["cross_feature_mean"]  for r in records]
    collapse_count = sum(r["t8_token_similarity"]["collapse_detected"] for r in records)

    w_mean, w_std = stats(within_vals)
    c_mean, c_std = stats(cross_vals)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.array([0, 1])
    ax.bar(x, [w_mean, c_mean],
           color=[COLORS["within"], COLORS["cross"]], alpha=0.85, zorder=3)
    _errbar(ax, 0, w_mean, w_std, n)
    _errbar(ax, 1, c_mean, c_std, n)
    _label(ax, 0, w_mean, w_std, n, offset=0.005)
    _label(ax, 1, c_mean, c_std, n, offset=0.005)

    # Collapse threshold line
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1.0,
               alpha=0.7, label="collapse threshold (0.9)")

    ax.set_xticks(x)
    ax.set_xticklabels(["Within-feature", "Cross-feature"])
    ax.set_ylabel("Mean cosine similarity")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"T8: Token Similarity — RBA mean-pool model (n={n} seeds)\n"
                 f"Collapse detected: {collapse_count}/{n} seeds")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    save_fig(fig, "rba_t8_token_similarity.png", out_dir)


# ---------------------------------------------------------------------------
# Figure 4: Split Sensitivity
# ---------------------------------------------------------------------------

def fig_split_sensitivity(
    rec_40: list[dict],
    rec_50: list[dict],
    rec_60: list[dict],
    out_dir: Path,
) -> None:
    """ROC-AUC for mean-pool and trivial across three train/test splits."""
    splits = []
    for label, records in [("40/60", rec_40), ("50/50", rec_50), ("60/40", rec_60)]:
        if not records:
            continue
        mp_m,   mp_sd   = stats([r["auc"]["mean_pool"]["roc_auc"] for r in records])
        triv_m, triv_sd = stats([r["auc"]["trivial"]["roc_auc"]   for r in records])
        splits.append((label, mp_m, mp_sd, triv_m, triv_sd, len(records)))

    if not splits:
        print("  No split sensitivity data found — skipping rba_split_sensitivity.png")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(splits))
    width = 0.32

    for i, (label, mp_m, mp_sd, triv_m, triv_sd, n) in enumerate(splits):
        ax.bar(x[i] - width/2, mp_m,   width, color=COLORS["mean_pool"],
               alpha=0.85, label="mean-pool" if i == 0 else "_")
        ax.bar(x[i] + width/2, triv_m, width, color=COLORS["trivial"],
               alpha=0.85, label="trivial" if i == 0 else "_")
        _errbar(ax, x[i] - width/2, mp_m,   mp_sd,   n)
        _errbar(ax, x[i] + width/2, triv_m, triv_sd, n)
        _label(ax, x[i] - width/2, mp_m,   mp_sd,   n, offset=0.004)
        _label(ax, x[i] + width/2, triv_m, triv_sd, n, offset=0.004)

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in splits])
    ax.set_xlabel("Train / Test split")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.18)
    ax.set_title("RBA: Split Sensitivity — mean-pool vs trivial\n"
                 "Same trained model (50/50), varied test window")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    save_fig(fig, "rba_split_sensitivity.png", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RBA aggregate figures from completed seed results")
    parser.add_argument("--seeds-root", type=Path, default=SEEDS_ROOT,
                        help=f"path to seeds/ directory (default: {SEEDS_ROOT})")
    parser.add_argument("--out-dir", type=Path, default=AGG_FIGURES,
                        help=f"output directory (default: {AGG_FIGURES})")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: synthetic data in temp dir, assert all figures written")
    args = parser.parse_args()

    print("=" * 60)
    print("RBA Aggregate Figures")
    print("=" * 60)

    if args.smoke:
        smoke_seeds = [42, 123, 456]
        rec_50   = [make_smoke_record(s) for s in smoke_seeds]
        rec_40   = [make_smoke_record(s) for s in smoke_seeds]
        rec_60   = [make_smoke_record(s) for s in smoke_seeds]
        tmp_root = Path(tempfile.mkdtemp())
        out_dir  = tmp_root / "rba_figures_smoke"
    else:
        print("\n[1/5] Loading primary results (50/50 split) ...")
        rec_50 = load_seed_results(args.seeds_root)
        print("\n[2/5] Loading sensitivity split results ...")
        rec_40 = load_split_results(args.seeds_root, "results_split40.json")
        rec_60 = load_split_results(args.seeds_root, "results_split60.json")
        print(f"  40/60 split: {len(rec_40)} seed(s)  |  "
              f"60/40 split: {len(rec_60)} seed(s)")
        tmp_root = None
        out_dir  = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/5] Summary AUC figure ...")
    fig_summary_auc(rec_50, out_dir)

    print("\n[4/5] T6 compactness + T8 token similarity figures ...")
    fig_t6_compactness(rec_50, out_dir)
    fig_t8_token_similarity(rec_50, out_dir)

    print("\n[5/5] Split sensitivity figure ...")
    fig_split_sensitivity(rec_40, rec_50, rec_60, out_dir)

    if args.smoke:
        missing = [f for f in EXPECTED_FIGURES if not (out_dir / f).exists()]
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)
        if missing:
            for m in missing:
                print(f"ASSERTION FAILED: {m} not created", file=sys.stderr)
            sys.exit(1)
        print("\nSMOKE PASS — all 4 figures created.")
    else:
        print(f"\nDone. {len(rec_50)} seed(s) aggregated.")
        if len(rec_50) < 5:
            print(f"  NOTE: only {len(rec_50)}/5 seeds complete — "
                  "re-run after all seeds finish for final figures.")


if __name__ == "__main__":
    main()
