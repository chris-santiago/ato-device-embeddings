# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "scikit-learn==1.7.2",
#   "matplotlib==3.10.8",
# ]
# ///
#
# H6 Figures — Publication figures for Contributions 2, 3, and 4 (H6 component)
# ==============================================================================
# Reads:
#   experiments/rerun/seeds/seed_*/h6/results.json  (aggregate metrics, all seeds)
#   experiments/rerun/seeds/seed_{rep}/h6/scores.npz (representative seed raw arrays)
# Writes (to experiments/rerun/aggregate/figures/):
#   h6_c2_pr_curves.png        — C2: PR curves, mp_raw vs mp_rank_norm (spoof k=1)
#   h6_c2_roc_curves.png       — C2: ROC curves, same comparison
#   h6_c2_score_dist.png       — C2: Score distributions showing CDF compression
#   h6_c3_population_decomp.png — C3: Fleet population decomposition (pre/post lag)
#   h6_c3_topk_precision.png   — C3: Top-k precision on fleet residual at 1–10%
#   h6_c3_pr_curves.png        — C3: PR curves fleet residual, mp_raw vs two_stage
#   h6_c4_spoof_gradient.png   — C4: PR-AUC at spoof k=1/2/3 (monotonic gradient)
#
# Prerequisite: run h6_rerun.py for at least seed 42 before running this script.
#
# Usage:
#   uv run experiments/rerun/scripts/h6/h6_figures.py
#   uv run experiments/rerun/scripts/h6/h6_figures.py --rep-seed 42
#   uv run experiments/rerun/scripts/h6/h6_figures.py --smoke

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

RERUN_ROOT  = Path(__file__).resolve().parent.parent.parent
SEEDS_ROOT  = RERUN_ROOT / "seeds"
FIGURES_DIR = RERUN_ROOT / "aggregate" / "figures"
ALL_SEEDS   = [42, 123, 456, 789, 2024]

COLORS = {
    "mp_raw":       "#2166ac",
    "mp_rank_norm": "#d6604d",
    "two_stage":    "#4dac26",
    "trivial":      "#878787",
}
DISPLAY_NAMES = {
    "mp_raw":       "mp_raw",
    "mp_rank_norm": "mp_rank_norm",
    "two_stage":    "two_stage",
    "trivial":      "trivial (exact match)",
}

TITLE_FS  = 11
LEGEND_FS = 8

EXPECTED_FIGURES = [
    "h6_c2_pr_curves.png",
    "h6_c2_roc_curves.png",
    "h6_c2_score_dist.png",
    "h6_c3_population_decomp.png",
    "h6_c3_topk_precision.png",
    "h6_c3_pr_curves.png",
    "h6_c4_spoof_gradient.png",
]


# ---------------------------------------------------------------------------
# Smoke helpers
# ---------------------------------------------------------------------------

class FakeNpz:
    """Dict-backed drop-in for np.load(scores.npz) used in smoke mode."""
    def __init__(self, data: dict): self._data = data
    @property
    def files(self): return list(self._data.keys())
    def __getitem__(self, k): return self._data[k]
    def __contains__(self, k): return k in self._data


def make_smoke_seed_result(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    def j(base, w=0.008): return float(base + rng.uniform(-w, w))
    def sb(roc, pr): return {"roc_auc": j(roc), "pr_auc": j(pr), "ci_lower": j(pr - 0.05), "ci_upper": j(pr + 0.05)}
    def fb(roc, pr, tp, prec): return {"roc_auc": j(roc), "pr_auc": j(pr), "top1pct_tp": tp, "top1pct_prec": j(prec)}
    return {
        "seed": seed, "timestamp": "2026-04-21T00:00:00Z",
        "config": {"n_accounts": 400, "vocab_size": 229, "imbalance_ratio": 100,
                   "fleet_frac": 0.25, "blocklist_lag_days": 10, "attack_window_days": 30,
                   "neg_ratio": 100, "n_fleet_accounts": 100, "n_pre_lag": 60, "n_post_lag": 40},
        "spoof_k1": {"mp_raw": sb(0.972, 0.892), "two_stage": sb(0.970, 0.889),
                     "mp_rank_norm": sb(0.995, 0.215), "two_stage_rank_norm": sb(0.993, 0.210),
                     "trivial": sb(0.750, 0.050)},
        "spoof_k2": {"mp_raw": sb(0.990, 0.950), "trivial": sb(0.750, 0.050)},
        "spoof_k3": {"mp_raw": sb(0.999, 0.990), "trivial": sb(0.750, 0.050)},
        "novel":    {"mp_raw": sb(0.999, 0.995), "two_stage": sb(0.998, 0.994),
                     "mp_rank_norm": sb(0.999, 0.980), "trivial": sb(0.750, 0.050)},
        "fleet_aggregate": {"mp_raw": sb(0.900, 0.600), "two_stage": sb(0.459, 0.300),
                            "trivial_blocklist": sb(0.459, 0.300), "trivial": sb(0.459, 0.300)},
        "fleet_residual": {
            "mp_raw":              fb(0.980, 0.750, 180, 0.918),
            "two_stage":           fb(0.457, 0.200,   0, 0.0),
            "mp_rank_norm":        fb(0.975, 0.600, 150, 0.850),
            "two_stage_rank_norm": fb(0.460, 0.200,   0, 0.0),
            "trivial_blocklist":   fb(0.457, 0.200,   0, 0.0),
            "two_stage_blocklist": fb(0.457, 0.200,   0, 0.0),
            "combined":            fb(0.980, 0.750, 180, 0.918),
            "trivial":             fb(0.457, 0.200,   0, 0.0),
        },
        "t8_token_similarity": {"within_feature_mean": j(0.487), "cross_feature_mean": j(0.432),
                                "within_cross_ratio": j(1.127), "collapse_detected": False},
        "verdicts": {"primary_criterion_confirmed": True,
                     "rank_norm_collapse_confirmed": True, "gate_blinds_fleet_confirmed": True},
    }


def make_smoke_npz(seed: int) -> FakeNpz:
    rng = np.random.default_rng(seed)
    n_pos, n_neg = 50, 500
    combos = [
        ("spoof_k1",       "mp_raw",       0.65, 0.25),
        ("spoof_k1",       "mp_rank_norm", 0.60, 0.25),
        ("spoof_k1",       "trivial",      0.52, 0.48),
        ("fleet_residual", "mp_raw",       0.70, 0.30),
        ("fleet_residual", "two_stage",    0.51, 0.49),
        ("fleet_residual", "trivial",      0.51, 0.49),
    ]
    data: dict = {}
    for atk, scorer, pos_m, neg_m in combos:
        pos = np.clip(rng.normal(pos_m, 0.1, n_pos), 0, 1)
        neg = np.clip(rng.normal(neg_m, 0.1, n_neg), 0, 1)
        data[f"{atk}__{scorer}__s"] = np.concatenate([pos, neg])
        data[f"{atk}__{scorer}__l"] = np.array([1] * n_pos + [0] * n_neg)
    return FakeNpz(data)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seed_results(seeds_root: Path) -> dict:
    results = {}
    for seed in ALL_SEEDS:
        p = seeds_root / f"seed_{seed}" / "h6" / "results.json"
        if p.exists():
            with open(p) as f:
                results[seed] = json.load(f)
    return results


def load_scores(seed: int, seeds_root: Path):
    p = seeds_root / f"seed_{seed}" / "h6" / "scores.npz"
    if not p.exists():
        sys.exit(
            f"ERROR: scores.npz not found for seed {seed}.\n"
            f"Run: uv run experiments/rerun/scripts/h6/h6_rerun.py --seed {seed}"
        )
    return np.load(p)


def agg_metric(seed_results: dict, path: list) -> tuple:
    """Walk `path` into each seed result; return (mean, std) of collected floats."""
    vals = []
    for res in seed_results.values():
        node = res
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, (int, float)) and not (isinstance(node, float) and node != node):
            vals.append(float(node))
    if not vals:
        return float("nan"), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


# ---------------------------------------------------------------------------
# Top-k precision helper (mirrors top_pct_metrics from h6_rerun.py)
# ---------------------------------------------------------------------------

def top_k_precision(scores: np.ndarray, labels: np.ndarray, k_pct: float) -> float:
    n_flag = max(1, int(len(scores) * k_pct / 100.0))
    threshold = np.sort(scores)[-n_flag]
    flagged = scores >= threshold
    tp = int(np.sum(flagged & (labels == 1)))
    fp = int(np.sum(flagged & (labels == 0)))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _npz_pair(npz, atk: str, scorer: str):
    """Return (scores, labels) arrays or (None, None) if keys absent."""
    ks = f"{atk}__{scorer}__s"
    kl = f"{atk}__{scorer}__l"
    if ks not in npz.files or kl not in npz.files:
        return None, None
    return npz[ks], npz[kl]


def _apply_style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _seed_note(ax, n: int):
    ax.text(0.97, 0.03, f"n={n} seed{'s' if n != 1 else ''}",
            transform=ax.transAxes, ha="right", fontsize=8, color="#666666")


# ---------------------------------------------------------------------------
# Figure 1 — C2: PR curves, spoof k=1
# ---------------------------------------------------------------------------

def fig_c2_pr_curves(npz, seed_results: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    for scorer in ("mp_raw", "mp_rank_norm", "trivial"):
        sc, lab = _npz_pair(npz, "spoof_k1", scorer)
        if sc is None or lab is None or len(np.unique(lab)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(lab, sc)
        mean_pr, _ = agg_metric(seed_results, ["spoof_k1", scorer, "pr_auc"])
        lbl = f"{DISPLAY_NAMES.get(scorer, scorer)}  (PR-AUC={mean_pr:.3f})"
        ax.plot(rec, prec, color=COLORS.get(scorer, "#333333"), label=lbl, lw=1.8)

    _, lab0 = _npz_pair(npz, "spoof_k1", "mp_raw")
    if lab0 is not None:
        pos_rate = float(np.mean(lab0 == 1))
        ax.axhline(pos_rate, color="black", lw=0.8, ls="--",
                   label=f"No-skill (pos. rate={pos_rate:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("C2: Precision-Recall — Spoof k=1 (1:100 imbalance)", fontsize=TITLE_FS)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
    ax.legend(loc="upper right", fontsize=8)
    _apply_style(ax)
    _seed_note(ax, len(seed_results))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 2 — C2: ROC curves, spoof k=1
# ---------------------------------------------------------------------------

def fig_c2_roc_curves(npz, seed_results: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    for scorer in ("mp_raw", "mp_rank_norm", "trivial"):
        sc, lab = _npz_pair(npz, "spoof_k1", scorer)
        if sc is None or lab is None or len(np.unique(lab)) < 2:
            continue
        fpr, tpr, _ = roc_curve(lab, sc)
        mean_roc, _ = agg_metric(seed_results, ["spoof_k1", scorer, "roc_auc"])
        lbl = f"{DISPLAY_NAMES.get(scorer, scorer)}  (ROC-AUC={mean_roc:.3f})"
        ax.plot(fpr, tpr, color=COLORS.get(scorer, "#333333"), label=lbl, lw=1.8)

    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("C2: ROC — Spoof k=1 (ROC-AUC masks rank-norm collapse)", fontsize=TITLE_FS)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
    ax.legend(loc="lower right", fontsize=8)
    _apply_style(ax)
    _seed_note(ax, len(seed_results))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 3 — C2: Score distribution histograms
# ---------------------------------------------------------------------------

def fig_c2_score_dist(npz, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    panels = [
        ("spoof_k1", "mp_raw",      "Raw cosine distance (mp_raw)"),
        ("spoof_k1", "mp_rank_norm", "Rank-normalized score (mp_rank_norm)"),
    ]
    for ax, (atk, scorer, title) in zip(axes, panels):
        sc, lab = _npz_pair(npz, atk, scorer)
        if sc is None:
            ax.set_title(f"{title}\n(data unavailable)", fontsize=TITLE_FS)
            continue
        neg_sc = sc[lab == 0]
        pos_sc = sc[lab == 1]
        lo = float(np.percentile(sc, 0.5))
        hi = float(np.percentile(sc, 99.5))
        bins = np.linspace(lo, hi, 60)
        ax.hist(neg_sc, bins=bins, alpha=0.55, color="#878787",
                label=f"Benign (n={len(neg_sc):,})", density=True)
        ax.hist(pos_sc, bins=bins, alpha=0.75, color="#d6604d",
                label=f"Attack (n={len(pos_sc):,})", density=True)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=TITLE_FS)
        ax.legend(fontsize=LEGEND_FS)
        _apply_style(ax)

    fig.suptitle("C2: Score Distributions — CDF Compression Mechanism (Spoof k=1)", y=1.01, fontsize=TITLE_FS)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 4 — C3: Population decomposition bar chart
# ---------------------------------------------------------------------------

def fig_c3_population_decomp(seed_results: dict, out: Path) -> None:
    n_fleet_vals, n_pre_vals, n_post_vals = [], [], []
    for res in seed_results.values():
        cfg = res.get("config", {})
        if "n_fleet_accounts" in cfg:
            n_fleet_vals.append(cfg["n_fleet_accounts"])
            n_pre_vals.append(cfg["n_pre_lag"])
            n_post_vals.append(cfg["n_post_lag"])

    if not n_fleet_vals:
        print(f"  SKIP {out.name} — config.n_fleet_accounts absent; re-run h6_rerun.py")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Re-run h6_rerun.py to populate fleet counts",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return

    n_fleet = int(round(np.mean(n_fleet_vals)))
    n_pre   = int(round(np.mean(n_pre_vals)))
    n_post  = int(round(np.mean(n_post_vals)))

    fig, ax = plt.subplots(figsize=(7, 4))
    categories = ["Fleet\n(total)", "Pre-lag\n(model-served)", "Post-lag\n(blocklist)"]
    counts     = [n_fleet, n_pre, n_post]
    bar_colors = ["#4393c3", "#2166ac", "#878787"]

    bars = ax.bar(categories, counts, color=bar_colors, width=0.45, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + n_fleet * 0.01,
                f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Number of fleet accounts")
    ax.set_title(
        "C3: Fleet Population Decomposition\n"
        "pre-lag = model-served (gate failure population); "
        "post-lag = blocklist-handled",
        fontsize=TITLE_FS,
    )
    ax.set_ylim(0, n_fleet * 1.2)
    _apply_style(ax)
    _seed_note(ax, len(seed_results))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 5 — C3: Top-k precision chart, fleet residual
# ---------------------------------------------------------------------------

def fig_c3_topk_precision(npz, out: Path) -> None:
    k_pcts = [1, 2, 3, 5, 7, 10]

    fig, ax = plt.subplots(figsize=(7, 5))

    for scorer in ("mp_raw", "two_stage", "trivial"):
        sc, lab = _npz_pair(npz, "fleet_residual", scorer)
        if sc is None or lab is None:
            continue
        precisions = [top_k_precision(sc, lab, k) for k in k_pcts]
        ax.plot(k_pcts, precisions, marker="o", markersize=5, lw=1.8,
                color=COLORS.get(scorer, "#333333"),
                label=DISPLAY_NAMES.get(scorer, scorer))

    ax.set_xlabel("Top-k threshold (%)")
    ax.set_ylabel("Precision at top-k%")
    ax.set_title("C3: Top-k Precision — Fleet Residual (pre-lag cold-start)\n"
                 "two_stage and trivial produce zero TPs at all thresholds",
                 fontsize=TITLE_FS)
    ax.set_xticks(k_pcts)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", fontsize=LEGEND_FS)
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 6 — C3: PR curves, fleet residual
# ---------------------------------------------------------------------------

def fig_c3_pr_curves(npz, seed_results: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    for scorer in ("mp_raw", "two_stage", "trivial"):
        sc, lab = _npz_pair(npz, "fleet_residual", scorer)
        if sc is None or lab is None or len(np.unique(lab)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(lab, sc)
        mean_pr, _ = agg_metric(seed_results, ["fleet_residual", scorer, "pr_auc"])
        lbl = f"{DISPLAY_NAMES.get(scorer, scorer)}  (PR-AUC={mean_pr:.3f})"
        ax.plot(rec, prec, color=COLORS.get(scorer, "#333333"), label=lbl, lw=1.8)

    _, lab0 = _npz_pair(npz, "fleet_residual", "mp_raw")
    if lab0 is not None:
        pos_rate = float(np.mean(lab0 == 1))
        ax.axhline(pos_rate, color="black", lw=0.8, ls="--",
                   label=f"No-skill (pos. rate={pos_rate:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("C3: Precision-Recall — Fleet Residual (pre-lag cold-start)", fontsize=TITLE_FS)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
    ax.legend(loc="upper right", fontsize=8)
    _apply_style(ax)
    _seed_note(ax, len(seed_results))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 7 — C4: Spoof k gradient (PR-AUC at k=1/2/3)
# ---------------------------------------------------------------------------

def fig_c4_spoof_gradient(seed_results: dict, out: Path) -> None:
    atk_keys = ["spoof_k1", "spoof_k2", "spoof_k3"]
    ks = [1, 2, 3]

    means, stds = [], []
    for atk in atk_keys:
        m, s = agg_metric(seed_results, [atk, "mp_raw", "pr_auc"])
        means.append(m)
        stds.append(s)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ks))
    show_err = len(seed_results) > 1
    bars = ax.bar(
        x, means,
        yerr=stds if show_err else None,
        color=COLORS["mp_raw"], width=0.45, capsize=5, edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Spoof k={k}" for k in ks])
    ax.set_xlabel("k = number of device features changed by attacker\n"
                  "(k=1 hardest to detect; k=3 easiest)")
    ax.set_ylabel("PR-AUC (mp_raw, mean ± std)")
    ax.set_title("C4: Spoof-k Gradient — PR-AUC by attack difficulty\n"
                 "k=1→k=2 robust on 5/5 seeds; k=2 vs k=3 within one std at saturation",
                 fontsize=TITLE_FS)
    ax.set_ylim(0, 1.12)
    _apply_style(ax)
    _seed_note(ax, len(seed_results))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H6 publication figures")
    p.add_argument("--seeds-root", type=Path, default=SEEDS_ROOT)
    p.add_argument("--rep-seed", type=int, default=42,
                   help="Seed for representative raw score arrays (default: 42)")
    p.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--smoke", action="store_true",
                   help="Run full code path and assert all output files created")
    return p.parse_args()


def main() -> None:
    import shutil
    args = parse_args()

    print("=" * 60)
    print("H6 Figures")
    print("=" * 60)

    if args.smoke:
        seed_results = {s: make_smoke_seed_result(s) for s in [42, 123, 456]}
        npz          = make_smoke_npz(42)
        tmp_root     = Path(tempfile.mkdtemp())
        out          = tmp_root / "h6_figures_smoke"
    else:
        print("\n[1/2] Loading data ...")
        seed_results = load_seed_results(args.seeds_root)
        if not seed_results:
            sys.exit(
                "ERROR: No seed results found.\n"
                f"Expected: {args.seeds_root}/seed_{{42,...}}/h6/results.json\n"
                "Run: uv run experiments/rerun/scripts/h6/h6_rerun.py --seed 42"
            )
        npz      = load_scores(args.rep_seed, args.seeds_root)
        tmp_root = None
        out      = args.out_dir

    out.mkdir(parents=True, exist_ok=True)
    print(f"  {len(seed_results)} seed(s) loaded: {sorted(seed_results)}")
    print(f"  scores: {len(npz.files)} arrays")
    print(f"\n[2/2] Generating figures → {out}")

    fig_c2_pr_curves(npz, seed_results, out / "h6_c2_pr_curves.png")
    fig_c2_roc_curves(npz, seed_results, out / "h6_c2_roc_curves.png")
    fig_c2_score_dist(npz, out / "h6_c2_score_dist.png")
    fig_c3_population_decomp(seed_results, out / "h6_c3_population_decomp.png")
    fig_c3_topk_precision(npz, out / "h6_c3_topk_precision.png")
    fig_c3_pr_curves(npz, seed_results, out / "h6_c3_pr_curves.png")
    fig_c4_spoof_gradient(seed_results, out / "h6_c4_spoof_gradient.png")

    if args.smoke:
        missing = [f for f in EXPECTED_FIGURES if not (out / f).exists()]
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)
        if missing:
            for m in missing:
                print(f"ASSERTION FAILED: {m} not created", file=sys.stderr)
            sys.exit(1)
        print("\nSMOKE PASS — all 7 figures created.")
    else:
        print(f"\nDone. {len(EXPECTED_FIGURES)} figures in {out}")


if __name__ == "__main__":
    main()
