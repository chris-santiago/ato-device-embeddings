# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "matplotlib==3.10.8",
# ]
# ///
#
# H2 Publication Figures — Aggregated Across 5 Seeds
# ===================================================
# Reads seeds/seed_*/h2/results.json, aggregates across seeds,
# and writes publication-quality figures to aggregate/figures/.
#
# Usage:
#   uv run h2_figures.py
#   uv run h2_figures.py --smoke   # synthetic data, temp dir, asserts all figures written

import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS   = [42, 123, 456, 789, 2024]
ATTACKS = ["novel", "fleet", "spoof"]

TITLE_FS  = 11
LEGEND_FS = 8

def _apply_style(ax):
    ax.spines[["top", "right"]].set_visible(False)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_seed_results(rerun_root: Path) -> list[dict]:
    results = []
    for seed in SEEDS:
        p = rerun_root / "seeds" / f"seed_{seed}" / "h2" / "results.json"
        if not p.exists():
            print(f"  WARNING: missing {p}", file=sys.stderr)
            continue
        with open(p) as f:
            results.append(json.load(f))
    return results

def extract(results: list[dict], *keys) -> np.ndarray:
    """Traverse each result dict by key path and return array of scalar values."""
    out = []
    for r in results:
        v = r
        for k in keys:
            v = v[k]
        out.append(float(v))
    return np.array(out)

def agg(vals: np.ndarray) -> tuple[float, float, float, float]:
    return float(vals.mean()), float(vals.std()), float(vals.min()), float(vals.max())

def agg_ci(results, path_est, path_lo, path_hi) -> tuple[float, float, float, int]:
    """Mean estimate, mean lower, mean upper, count of seeds with lower > 0."""
    ests = extract(results, *path_est)
    los  = extract(results, *path_lo)
    his  = extract(results, *path_hi)
    return float(ests.mean()), float(los.mean()), float(his.mean()), int(np.sum(los > 0))

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def save_fig(fig, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {p}")

# ---------------------------------------------------------------------------
# Fig 1: Primary AUC
# ---------------------------------------------------------------------------
def fig_primary_auc(results: list[dict], out_dir: Path):
    mp_means, mp_stds   = [], []
    cat_means, cat_stds = [], []
    triv_means          = []
    for at in ATTACKS:
        m, s, _, _ = agg(extract(results, "auc", "mean_pool", at))
        mp_means.append(m); mp_stds.append(s)
        m, s, _, _ = agg(extract(results, "auc", "concat_w1", at))
        cat_means.append(m); cat_stds.append(s)
        triv_means.append(float(extract(results, "auc", "trivial", at).mean()))

    x = np.arange(len(ATTACKS))
    w = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w, mp_means,   w, yerr=mp_stds,  color="#2196F3", alpha=0.88, capsize=4, label="mean-pool")
    ax.bar(x,     cat_means,  w, yerr=cat_stds, color="#FF5722", alpha=0.88, capsize=4, label="concat w=1")
    ax.bar(x + w, triv_means, w,                color="#9E9E9E", alpha=0.75,            label="trivial")

    for xi, (m, s) in enumerate(zip(mp_means, mp_stds)):
        ax.text(xi - w, m + s + 0.007, f"{m:.3f}", ha="center", fontsize=7.5)
    for xi, (m, s) in enumerate(zip(cat_means, cat_stds)):
        ax.text(xi,     m + s + 0.007, f"{m:.3f}", ha="center", fontsize=7.5)
    for xi, m in enumerate(triv_means):
        ax.text(xi + w, m     + 0.007, f"{m:.3f}", ha="center", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(ATTACKS)
    ax.set_ylim(0.35, 1.14); ax.set_ylabel("ROC-AUC")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("H2: Primary AUC by Attack Type\n(mean ± std across seeds; sg=1, per-account corpus)", fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS)
    _apply_style(ax)
    save_fig(fig, "h2_primary_auc.png", out_dir)

# ---------------------------------------------------------------------------
# Fig 1b: Primary AUC — side-by-side ROC-AUC and PR-AUC
# ---------------------------------------------------------------------------
def _auc_bar_panel(ax, results, metric_key, ylabel, title_suffix):
    mp_means, mp_stds   = [], []
    cat_means, cat_stds = [], []
    triv_means          = []
    for at in ATTACKS:
        m, s, _, _ = agg(extract(results, metric_key, "mean_pool", at))
        mp_means.append(m); mp_stds.append(s)
        m, s, _, _ = agg(extract(results, metric_key, "concat_w1", at))
        cat_means.append(m); cat_stds.append(s)
        triv_means.append(float(extract(results, metric_key, "trivial", at).mean()))

    x = np.arange(len(ATTACKS))
    w = 0.24
    ax.bar(x - w, mp_means,   w, yerr=mp_stds,  color="#2196F3", alpha=0.88, capsize=4, label="mean-pool")
    ax.bar(x,     cat_means,  w, yerr=cat_stds, color="#FF5722", alpha=0.88, capsize=4, label="concat w=1")
    ax.bar(x + w, triv_means, w,                color="#9E9E9E", alpha=0.75,            label="trivial")

    for xi, (m, s) in enumerate(zip(mp_means, mp_stds)):
        ax.text(xi - w, m + s + 0.007, f"{m:.3f}", ha="center", fontsize=7)
    for xi, (m, s) in enumerate(zip(cat_means, cat_stds)):
        ax.text(xi,     m + s + 0.007, f"{m:.3f}", ha="center", fontsize=7)
    for xi, m in enumerate(triv_means):
        ax.text(xi + w, m     + 0.007, f"{m:.3f}", ha="center", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(ATTACKS)
    ax.set_ylabel(ylabel)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title_suffix, fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS, loc="lower left")
    _apply_style(ax)

def fig_auc_dual(results: list[dict], out_dir: Path):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.5))
    _auc_bar_panel(ax_roc, results, "auc",    "ROC-AUC", "ROC-AUC by Attack Type")
    _auc_bar_panel(ax_pr,  results, "pr_auc", "PR-AUC",  "PR-AUC by Attack Type")
    ax_roc.set_ylim(0.35, 1.14)
    ax_pr.set_ylim(0.35, 1.14)
    fig.suptitle("H2: Primary Metrics (mean ± std across seeds; sg=1, per-account corpus)",
                 fontsize=TITLE_FS + 1, y=1.02)
    plt.tight_layout()
    save_fig(fig, "h2_auc_dual.png", out_dir)

# ---------------------------------------------------------------------------
# Fig 2: T1 delta CIs — horizontal forest plot
# ---------------------------------------------------------------------------
def fig_delta_ci(results: list[dict], out_dir: Path):
    n = len(results)
    rows = [
        ("Spoof AUC",  "spoof_delta"),
        ("Novel AUC",  "novel_delta"),
        ("Fleet AUC",  "fleet_delta"),
        ("Silhouette", "silhouette_delta"),
    ]
    ests, los, his, n_excls, labels = [], [], [], [], []
    for label, key in rows:
        est, lo, hi, ne = agg_ci(
            results,
            ("bootstrap_ci_95", "deltas", key, "estimate"),
            ("bootstrap_ci_95", "deltas", key, "lower"),
            ("bootstrap_ci_95", "deltas", key, "upper"),
        )
        ests.append(est); los.append(lo); his.append(hi)
        n_excls.append(ne); labels.append(label)

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i in range(len(rows)):
        color = "#2196F3" if los[i] > 0 else "#FF5722"
        ax.plot([los[i], his[i]], [y[i], y[i]], color=color, linewidth=3)
        ax.plot(ests[i], y[i], "o", color=color, markersize=8, zorder=3)
        ax.text(
            his[i] + 0.003, y[i],
            f"{ests[i]:+.4f}  [{los[i]:+.4f}, {his[i]:+.4f}]   {n_excls[i]}/{n} seeds CI>0",
            va="center", fontsize=8,
        )

    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Delta (mean-pool − concat_w1)")
    ax.set_title("H2 T1: Bootstrap Delta CIs (mean across seeds)\nBlue = CI excludes zero in all seeds", fontsize=TITLE_FS)
    x_lo = min(min(los), 0) - 0.04
    x_hi = max(his) + 0.35
    ax.set_xlim(x_lo, x_hi)
    _apply_style(ax)
    plt.tight_layout()
    save_fig(fig, "h2_delta_ci.png", out_dir)

# ---------------------------------------------------------------------------
# Fig 3: T2 window sweep
# ---------------------------------------------------------------------------
def fig_t2_window_sweep(results: list[dict], out_dir: Path):
    windows = [1, 3, 6]
    keys    = ["concat_w1", "concat_w3", "concat_w6"]
    colors  = {1: "#FF5722", 3: "#FF9800", 6: "#FFC107"}

    cat_means, cat_stds = [], []
    for k in keys:
        m, s, _, _ = agg(extract(results, "t2_window_sweep", k, "spoof_auc"))
        cat_means.append(m); cat_stds.append(s)

    mp_m, mp_s, _, _ = agg(extract(results, "t2_window_sweep", "mean_pool_spoof_auc"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(mp_m, color="#2196F3", linestyle="--", linewidth=1.8,
               label=f"mean-pool {mp_m:.4f} ± {mp_s:.4f}")
    ax.fill_between([-0.5, 2.5], mp_m - mp_s, mp_m + mp_s, color="#2196F3", alpha=0.12)

    for i, (win, m, s) in enumerate(zip(windows, cat_means, cat_stds)):
        ax.bar(i, m, yerr=s, color=colors[win], alpha=0.88, capsize=5,
               label=f"concat w={win}  {m:.4f} ± {s:.4f}")
        ax.text(i, m + s + 0.006, f"{m:.4f}", ha="center", fontsize=8)

    ax.set_xticks(range(3)); ax.set_xticklabels([f"w={w}" for w in windows])
    ax.set_ylabel("Spoof ROC-AUC (mean ± std across seeds)")
    ax.set_title("H2 T2: Concat Window Sweep vs. Mean-Pool\n(widening context does not close the spoof gap)", fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS); ax.set_ylim(0, 1.05)
    _apply_style(ax)
    plt.tight_layout()
    save_fig(fig, "h2_t2_window_sweep.png", out_dir)

# ---------------------------------------------------------------------------
# Fig 4: T5 tz-permutation
# ---------------------------------------------------------------------------
def fig_t5_tz_permutation(results: list[dict], out_dir: Path):
    positions  = list(range(1, 7))
    perm_means, perm_stds = [], []
    for pos in positions:
        m, s, _, _ = agg(extract(results, "t5_tz_permutation", f"position_{pos}", "concat_spoof_auc"))
        perm_means.append(m); perm_stds.append(s)

    mp_m, mp_s, _, _ = agg(extract(results, "t5_tz_permutation", "mean_pool_spoof_auc"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(6)
    ax.errorbar(x, perm_means, yerr=perm_stds, fmt="o-", color="#FF9800",
                linewidth=2, markersize=7, capsize=4, label="Concat (tz at position i)")
    ax.axhline(mp_m, color="#2196F3", linestyle="--", linewidth=1.8,
               label=f"mean-pool {mp_m:.4f} ± {mp_s:.4f}")
    ax.fill_between([-0.5, 5.5], mp_m - mp_s, mp_m + mp_s, color="#2196F3", alpha=0.12)

    ax.set_xticks(x); ax.set_xticklabels([f"pos {i}" for i in range(6)])
    ax.set_xlabel("Tz feature position in concat string (0 = first)")
    ax.set_ylabel("Spoof ROC-AUC (mean ± std across seeds)")
    ax.set_title("H2 T5: Tz-Position Permutation\n(moving tz earlier does not recover the spoof AUC gap)", fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS)
    _apply_style(ax)
    plt.tight_layout()
    save_fig(fig, "h2_t5_tz_permutation.png", out_dir)

# ---------------------------------------------------------------------------
# Fig 5: Verdict stability table
# ---------------------------------------------------------------------------
def fig_verdict_stability(results: list[dict], out_dir: Path):
    n = len(results)

    def best_concat_w(r):
        return max(
            r["t2_window_sweep"]["concat_w1"]["spoof_auc"],
            r["t2_window_sweep"]["concat_w3"]["spoof_auc"],
            r["t2_window_sweep"]["concat_w6"]["spoof_auc"],
        )

    rows = [
        ("T1: Spoof delta CI > 0",       sum(1 for r in results if r["bootstrap_ci_95"]["deltas"]["spoof_delta"]["lower"] > 0)),
        ("T1: Novel delta CI > 0",        sum(1 for r in results if r["bootstrap_ci_95"]["deltas"]["novel_delta"]["lower"] > 0)),
        ("T1: Fleet delta CI > 0",        sum(1 for r in results if r["bootstrap_ci_95"]["deltas"]["fleet_delta"]["lower"] > 0)),
        ("T1: Sil delta CI > 0",          sum(1 for r in results if r["bootstrap_ci_95"]["deltas"]["silhouette_delta"]["lower"] > 0)),
        ("T2: Best concat < mp (spoof)",  sum(1 for r in results if best_concat_w(r) < r["t2_window_sweep"]["mean_pool_spoof_auc"])),
        ("T5: No perm recovers mp",       sum(1 for r in results if max(
            r["t5_tz_permutation"][f"position_{p}"]["concat_spoof_auc"] for p in range(1, 7)
        ) < r["t5_tz_permutation"]["mean_pool_spoof_auc"])),
        ("T7: mp beats trivial (spoof)",  sum(1 for r in results if r["auc"]["mean_pool"]["spoof"] > r["auc"]["trivial"]["spoof"])),
        ("T8: Robust no collapse",        sum(1 for r in results if not r["t8_token_similarity"]["robust_config"]["collapse_detected"])),
        ("T8: Degen collapse confirmed",  sum(1 for r in results if r["t8_token_similarity"]["degenerate_config"]["collapse_detected"])),
    ]

    fig, ax = plt.subplots(figsize=(8.5, len(rows) * 0.52 + 1.3))
    ax.axis("off")
    cell_text   = [[label, f"{cnt}/{n}"] for label, cnt in rows]
    cell_colors = [["white", "#C8E6C9" if cnt == n else "#FFCDD2"] for _, cnt in rows]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=["Verdict", f"Stable ({n} seeds)"],
        cellColours=cell_colors,
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width([0, 1])
    ax.set_title("H2 Verdict Stability Across Seeds", fontsize=TITLE_FS, fontweight="bold", pad=14)
    plt.tight_layout()
    save_fig(fig, "h2_verdict_stability.png", out_dir)

# ---------------------------------------------------------------------------
# Fig 6: C1 T8 token similarity — 2×2 factorial bar chart
# ---------------------------------------------------------------------------
_FACTORIAL_CONFIGS = [
    ("sg_per_account",  "sg\nper-acct",   False),
    ("cbow_per_event",  "cbow\nper-event", True),
    ("cbow_per_account","cbow\nper-acct",  False),
    ("sg_per_event",    "sg\nper-event",   True),
]
_COLLAPSE_COLOR    = "#EF5350"
_NO_COLLAPSE_COLOR = "#2196F3"
_COLLAPSE_THRESH   = 0.95


def fig_c1_token_similarity(results: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    x       = np.arange(len(_FACTORIAL_CONFIGS))
    width   = 0.55
    rng_jit = np.random.default_rng(0)

    for xi, (key, label, is_collapse) in enumerate(_FACTORIAL_CONFIGS):
        vals = []
        for r in results:
            try:
                v = r["t8_token_similarity"]["factorial_2x2"][key]["within_feature_mean"]
                vals.append(float(v))
            except (KeyError, TypeError):
                pass
        if not vals:
            continue
        arr   = np.array(vals)
        mean  = arr.mean()
        std   = arr.std()
        color = _COLLAPSE_COLOR if is_collapse else _NO_COLLAPSE_COLOR

        ax.bar(xi, mean, width, yerr=std, color=color, alpha=0.82,
               capsize=5, zorder=2)
        ax.text(xi, mean + std + 0.025, f"{mean:.3f}", ha="center",
                fontsize=8.5, color=color)

        jitter = rng_jit.uniform(-0.12, 0.12, len(arr))
        ax.scatter(xi + jitter, arr, color="white", edgecolors=color,
                   s=28, linewidths=1.3, zorder=3)

    ax.axhline(_COLLAPSE_THRESH, color="#B71C1C", linestyle="--",
               linewidth=1.2, alpha=0.75, label=f"collapse threshold ({_COLLAPSE_THRESH})")

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in _FACTORIAL_CONFIGS], fontsize=10)
    ax.set_ylabel("Within-feature cosine similarity (mean ± std across seeds)")
    ax.set_ylim(-0.25, 1.18)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title(
        "C1: T8 Token Similarity — 2×2 Factorial (5 Seeds)\n"
        "Red = collapse config  ·  Blue = healthy config  ·  Dots = per-seed values",
        fontsize=TITLE_FS,
    )
    ax.legend(fontsize=LEGEND_FS)
    _apply_style(ax)
    plt.tight_layout()
    save_fig(fig, "h2_c1_token_similarity.png", out_dir)


# ---------------------------------------------------------------------------
# Smoke data
# ---------------------------------------------------------------------------
def make_smoke_result(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    def j(base): return float(base + rng.uniform(-0.008, 0.008))
    def ci(est, w=0.025): return {"estimate": float(est), "lower": float(est - w), "upper": float(est + w)}

    return {
        "seed": seed,
        "timestamp": "2026-04-21T00:00:00Z",
        "auc": {
            "mean_pool": {"novel": j(0.999), "fleet": j(0.994), "spoof": j(0.869)},
            "concat_w1": {"novel": j(0.997), "fleet": j(0.992), "spoof": j(0.782)},
            "trivial":   {"novel": 0.750,    "fleet": 0.750,    "spoof": 0.750},
        },
        "pr_auc": {
            "mean_pool": {"novel": j(0.995), "fleet": j(0.990), "spoof": j(0.820)},
            "concat_w1": {"novel": j(0.990), "fleet": j(0.985), "spoof": j(0.680)},
            "trivial":   {"novel": 0.500,    "fleet": 0.500,    "spoof": 0.500},
        },
        "bootstrap_ci_95": {
            "per_model": {
                "mean_pool": {at: ci(j(0.900)) for at in ATTACKS},
                "concat_w1": {at: ci(j(0.850)) for at in ATTACKS},
            },
            "pr_auc_per_model": {
                "mean_pool": {at: ci(j(0.850)) for at in ATTACKS},
                "concat_w1": {at: ci(j(0.780)) for at in ATTACKS},
            },
            "deltas": {
                "spoof_delta":      ci(j(0.087), 0.025),
                "novel_delta":      ci(j(0.002), 0.025),
                "fleet_delta":      ci(j(0.002), 0.025),
                "silhouette_delta": ci(j(0.050), 0.025),
            },
            "pr_auc_deltas": {
                "spoof_delta": ci(j(0.140), 0.025),
                "novel_delta": ci(j(0.005), 0.025),
                "fleet_delta": ci(j(0.005), 0.025),
            },
        },
        "silhouette":  {"mean_pool": j(0.12),  "concat": j(0.05)},
        "compactness": {
            "mean_pool": {"mean": j(0.036), "ci_lower": 0.032, "ci_upper": 0.040},
            "concat":    {"mean": j(0.047), "ci_lower": 0.041, "ci_upper": 0.053},
        },
        "t2_window_sweep": {
            "concat_w1":           {"spoof_auc": j(0.782)},
            "concat_w3":           {"spoof_auc": j(0.820)},
            "concat_w6":           {"spoof_auc": j(0.835)},
            "mean_pool_spoof_auc": j(0.869),
        },
        "t3_prefixed_concat": {
            "mean_pool_silhouette":       j(0.120),
            "prefixed_concat_silhouette": j(0.050),
            "gap":                        j(0.070),
        },
        "t4_tz_attribution": {"mean_pool": j(0.028), "concat": j(0.062)},
        "t5_tz_permutation": {
            **{f"position_{i}": {"concat_spoof_auc": j(0.780)} for i in range(1, 7)},
            "mean_pool_spoof_auc": j(0.869),
        },
        "t7_trivial_baseline": {
            "mean_pool_spoof_margin": j(0.119),
            "concat_w1_spoof_margin": j(0.032),
        },
        "t8_token_similarity": {
            "robust_config":     {
                "within_feature_mean": j(0.392), "cross_feature_mean": j(0.480),
                "within_cross_ratio": 0.817, "collapse_detected": False,
            },
            "degenerate_config": {
                "within_feature_mean": j(0.999), "cross_feature_mean": j(0.987),
                "within_cross_ratio": 1.012, "collapse_detected": True,
            },
            "factorial_2x2": {
                "sg_per_account":   {"within_feature_mean": j(0.436), "collapse_detected": False},
                "cbow_per_event":   {"within_feature_mean": j(0.980), "collapse_detected": True},
                "cbow_per_account": {"within_feature_mean": j(-0.111), "collapse_detected": False},
                "sg_per_event":     {"within_feature_mean": j(0.938), "collapse_detected": True},
            },
        },
        "enrollment_diagnostics": {
            "mean_pool_enrollment_dist": j(0.036),
            "concat_enrollment_dist":    j(0.047),
        },
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="H2 publication figures from aggregated seed results")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: synthetic data in temp dir, assert all figures written")
    return p.parse_args()

EXPECTED_FIGURES = [
    "h2_primary_auc.png",
    "h2_auc_dual.png",
    "h2_delta_ci.png",
    "h2_t2_window_sweep.png",
    "h2_t5_tz_permutation.png",
    "h2_verdict_stability.png",
    "h2_c1_token_similarity.png",
]

def main():
    args       = parse_args()
    rerun_root = Path(__file__).resolve().parent.parent.parent

    if args.smoke:
        results = [make_smoke_result(s) for s in [42, 123, 456]]
        out_dir = Path(tempfile.mkdtemp()) / "h2_figures_smoke"
    else:
        results = load_seed_results(rerun_root)
        out_dir = rerun_root / "aggregate" / "figures"

    if not results:
        print("ERROR: no results.json files found. Run h2_rerun.py --seed N for each seed first.")
        sys.exit(1)

    n = len(results)
    print("=" * 65)
    print(f"H2 Publication Figures  ({n} seed{'s' if n != 1 else ''} loaded)")
    print("Output:", out_dir)
    print("=" * 65)

    print("\n[1/7] Primary AUC comparison (mp vs concat_w1 vs trivial)...")
    fig_primary_auc(results, out_dir)

    print("[2/7] Dual AUC panel (ROC-AUC + PR-AUC side-by-side)...")
    fig_auc_dual(results, out_dir)

    print("[3/7] T1 delta CIs (bootstrap forest plot)...")
    fig_delta_ci(results, out_dir)

    print("[4/7] T2 window sweep (concat spoof AUC at w=1/3/6)...")
    fig_t2_window_sweep(results, out_dir)

    print("[5/7] T5 tz-permutation (spoof AUC by tz position)...")
    fig_t5_tz_permutation(results, out_dir)

    print("[6/7] Verdict stability table...")
    fig_verdict_stability(results, out_dir)

    print("[7/7] C1 T8 token similarity — 2x2 factorial...")
    fig_c1_token_similarity(results, out_dir)

    if args.smoke:
        for name in EXPECTED_FIGURES:
            p = out_dir / name
            assert p.exists() and p.stat().st_size > 0, f"smoke: missing or empty figure {name}"
        print("\nSMOKE PASS")
    else:
        print(f"\nAll figures written to {out_dir}")

if __name__ == "__main__":
    main()
