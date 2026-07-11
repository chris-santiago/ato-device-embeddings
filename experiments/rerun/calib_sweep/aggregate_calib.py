# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
# ]
# ///
#
# Aggregate the calibration-window ablation across seeds.
# Reads results/calib_{K}/seed_{S}/results.json for every cell and produces
# mean +/- std per calib_size for each metric, for both scorers, plus the
# (mp_raw - mp_rank_norm) PR-AUC gap (the rank-norm penalty controlling for
# embedding quality). Writes aggregate/calib_sweep_summary.{json,csv}.

import csv
import json
from pathlib import Path

import numpy as np

ROOT        = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
AGG_DIR     = ROOT / "aggregate"
ATTACK      = "spoof_k1"
SCORERS     = ["mp_raw", "mp_rank_norm"]
# (results-key, margins?) for each metric we aggregate
METRICS = [
    ("pr_auc", False),
    ("roc_auc", False),
    ("contamination_rate", True),
    ("robust_margin", True),
]


def _summ(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "n_seeds": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "min": float(a.min()),
        "max": float(a.max()),
        "values": [float(x) for x in a],
    }


def _metric_value(block: dict, key: str, is_margin: bool) -> float:
    return block["margins"][key] if is_margin else block[key]


def main() -> None:
    calib_dirs = sorted(RESULTS_DIR.glob("calib_*"), key=lambda p: int(p.name.split("_")[1]))
    if not calib_dirs:
        raise SystemExit(f"No results under {RESULTS_DIR}. Run calib_sweep.py first.")

    summary: dict = {"attack": ATTACK, "calib_sizes": [], "by_calib": {}}
    csv_rows: list[dict] = []

    for cdir in calib_dirs:
        calib = int(cdir.name.split("_")[1])
        summary["calib_sizes"].append(calib)
        cells = sorted(cdir.glob("seed_*/results.json"))
        loaded = [json.loads(p.read_text()) for p in cells]
        seeds = [c["seed"] for c in loaded]

        entry: dict = {"seeds": seeds, "n_train": 40 + calib, "scorers": {}}
        for scorer in SCORERS:
            entry["scorers"][scorer] = {}
            for key, is_margin in METRICS:
                vals = [_metric_value(c[ATTACK][scorer], key, is_margin) for c in loaded]
                s = _summ(vals)
                entry["scorers"][scorer][key] = s
                csv_rows.append({
                    "calib_size": calib, "n_train": 40 + calib, "scorer": scorer,
                    "metric": key, "n_seeds": s["n_seeds"],
                    "mean": s["mean"], "std": s["std"], "min": s["min"], "max": s["max"],
                })

        # Embedding-quality control: PR-AUC gap = mp_raw - mp_rank_norm, per seed.
        raw_pr = [c[ATTACK]["mp_raw"]["pr_auc"] for c in loaded]
        rn_pr  = [c[ATTACK]["mp_rank_norm"]["pr_auc"] for c in loaded]
        gap = _summ([r - n for r, n in zip(raw_pr, rn_pr)])
        entry["gap_pr_auc"] = gap
        # Pure-quantization prediction: a 1/len(calib)-quantized rank floor.
        entry["quantization_floor_pred"] = 1.0 / calib
        csv_rows.append({
            "calib_size": calib, "n_train": 40 + calib, "scorer": "gap(mp_raw-mp_rank_norm)",
            "metric": "pr_auc", "n_seeds": gap["n_seeds"],
            "mean": gap["mean"], "std": gap["std"], "min": gap["min"], "max": gap["max"],
        })
        summary["by_calib"][str(calib)] = entry

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    (AGG_DIR / "calib_sweep_summary.json").write_text(json.dumps(summary, indent=2))
    with open(AGG_DIR / "calib_sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["calib_size", "n_train", "scorer", "metric",
                                          "n_seeds", "mean", "std", "min", "max"])
        w.writeheader()
        w.writerows(csv_rows)

    # Console table
    print(f"\n{'calib':>6} {'N_TR':>5} | {'mp_raw PR':>18} {'rank_norm PR':>18} "
          f"{'gap':>14} {'contam(rn)':>16} {'1/calib':>8}")
    print("-" * 96)
    for calib in summary["calib_sizes"]:
        e = summary["by_calib"][str(calib)]
        raw = e["scorers"]["mp_raw"]["pr_auc"]
        rn  = e["scorers"]["mp_rank_norm"]["pr_auc"]
        con = e["scorers"]["mp_rank_norm"]["contamination_rate"]
        g   = e["gap_pr_auc"]
        print(f"{calib:>6} {e['n_train']:>5} | "
              f"{raw['mean']:.3f}+/-{raw['std']:.3f}      "
              f"{rn['mean']:.3f}+/-{rn['std']:.3f}      "
              f"{g['mean']:.3f}+/-{g['std']:.3f}  "
              f"{con['mean']:.4f}+/-{con['std']:.4f}  {1.0/calib:.4f}")
    print(f"\nWrote {AGG_DIR/'calib_sweep_summary.json'}")
    print(f"Wrote {AGG_DIR/'calib_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
