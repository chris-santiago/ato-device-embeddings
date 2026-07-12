# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
# ]
# ///
#
# Deterministic aggregation for the phase-1 critique ablations
# (plan/12-CRITIQUE_ABLATIONS.md): reads per-seed results JSONs from results/
# and writes aggregate/ablations_summary.json + .csv. Cross-seed std uses
# ddof=0 (aggregate.py convention). Verdicts are reported as per-seed booleans
# and counts only — no confirmation-rate interval statistics (plan 12).
#
# Usage:
#   uv run experiments/rerun/ablations/aggregate_ablations.py

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 123, 456, 789, 2024]
ATTACKS = ["novel", "fleet", "spoof"]
H6_FEATURES = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]
H6_CELLS = ["sg_per_event", "sg_per_account", "cbow_per_event", "cbow_per_account"]


def load(pattern: str) -> dict:
    out = {}
    for seed in SEEDS:
        path = ROOT / "results" / pattern.format(seed=seed)
        if path.exists():
            with open(path) as f:
                out[seed] = json.load(f)
    return out


def mean_std(values) -> dict:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "per_seed": list(values)}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "per_seed": list(values)}


def aggregate_baseline(runs: dict) -> dict:
    seeds = sorted(runs)
    scorer_names = sorted({name for r in runs.values() for name in r["scorers"]})
    scorers = {}
    for name in scorer_names:
        scorers[name] = {}
        for at in ATTACKS:
            for metric in ["roc_auc", "pr_auc"]:
                vals = [runs[s]["scorers"].get(name, {}).get(at, {}).get(metric) for s in seeds]
                scorers[name][f"{at}_{metric}"] = mean_std(vals)
    deltas = {}
    for dname in ["mp_minus_random", "mp_minus_lik_best"]:
        deltas[dname] = {}
        for metric in ["roc", "pr"]:
            pts = [runs[s]["deltas"][dname]["spoof"][metric][0] for s in seeds]
            los = [runs[s]["deltas"][dname]["spoof"][metric][1] for s in seeds]
            deltas[dname][f"spoof_{metric}"] = {
                "point": mean_std(pts),
                "ci_lower_min": float(np.min(los)),
                "ci_lower_per_seed": los,
            }
    verdict_names = sorted({v for r in runs.values() for v in r["verdicts"]})
    verdicts = {
        v: {
            "per_seed": {s: runs[s]["verdicts"][v] for s in seeds},
            "confirmed": sum(runs[s]["verdicts"][v] for s in seeds),
            "n_seeds": len(seeds),
        }
        for v in verdict_names
    }
    best_variants = {s: runs[s]["descriptive"]["best_likelihood_variant"] for s in seeds}
    return {"seeds": seeds, "scorers": scorers, "spoof_deltas": deltas,
            "verdicts": verdicts, "best_likelihood_variant": best_variants}


def aggregate_h6(runs: dict) -> dict:
    seeds = sorted(runs)
    cells = {}
    for cell in H6_CELLS:
        cells[cell] = {
            "pooled_within_cos_wv": mean_std(
                [runs[s]["cells"][cell]["pooled_within_cos_wv"] for s in seeds]
            ),
        }
        for f in H6_FEATURES:
            for metric in ["within_cos_wv", "within_cos_vocab"]:
                vals = [runs[s]["cells"][cell][metric].get(f) for s in seeds]
                cells[cell][f"{f}_{metric}"] = mean_std(vals)
    jsd = {}
    for shape in ["per_event", "per_account"]:
        jsd[shape] = {
            f: mean_std([runs[s]["jsd"][shape].get(f) for s in seeds]) for f in H6_FEATURES
        }
    verdict_names = sorted({v for r in runs.values() for v in r["verdicts"]})
    verdicts = {
        v: {
            "per_seed": {s: runs[s]["verdicts"][v] for s in seeds},
            "confirmed": sum(runs[s]["verdicts"][v] for s in seeds),
            "n_seeds": len(seeds),
        }
        for v in verdict_names
    }
    return {"seeds": seeds, "cells": cells, "jsd": jsd, "verdicts": verdicts}


def aggregate_a4(runs: dict) -> dict:
    seeds = sorted(runs)
    scorer_names = sorted({name for r in runs.values() for name in r["scorers"]})
    scorers = {}
    for name in scorer_names:
        scorers[name] = {}
        for at in ATTACKS:
            for metric in ["roc_auc", "pr_auc"]:
                vals = [runs[s]["scorers"].get(name, {}).get(at, {}).get(metric) for s in seeds]
                scorers[name][f"{at}_{metric}"] = mean_std(vals)
    deltas = {}
    for dname in ["mp_trained_minus_nosub", "mp_nosub_minus_random"]:
        deltas[dname] = {}
        for metric in ["roc", "pr"]:
            pts = [runs[s]["deltas"][dname]["spoof"][metric][0] for s in seeds]
            los = [runs[s]["deltas"][dname]["spoof"][metric][1] for s in seeds]
            his = [runs[s]["deltas"][dname]["spoof"][metric][2] for s in seeds]
            deltas[dname][f"spoof_{metric}"] = {
                "point": mean_std(pts),
                "ci_lower_min": float(np.min(los)),
                "ci_upper_max": float(np.max(his)),
                "ci_lower_per_seed": los,
            }
    verdict_names = sorted({v for r in runs.values() for v in r["verdicts"]})
    verdicts = {
        v: {
            "per_seed": {s: runs[s]["verdicts"][v] for s in seeds},
            "confirmed": sum(runs[s]["verdicts"][v] for s in seeds),
            "n_seeds": len(seeds),
        }
        for v in verdict_names
    }
    descriptive = {
        "cat_nosub_oov_rate_spoof": mean_std(
            [runs[s]["descriptive"]["cat_nosub_oov_rate"]["spoof"] for s in seeds]),
        "within_feature_cos_mp_trained": mean_std(
            [runs[s]["descriptive"]["within_feature_cos_mp_trained"] for s in seeds]),
        "within_feature_cos_mp_nosub": mean_std(
            [runs[s]["descriptive"]["within_feature_cos_mp_nosub"] for s in seeds]),
    }
    return {"seeds": seeds, "scorers": scorers, "spoof_deltas": deltas,
            "verdicts": verdicts, "descriptive": descriptive}


def aggregate_h6_likelihood(runs: dict) -> dict:
    seeds = sorted(runs)
    populations = ["spoof_k1", "spoof_k2", "spoof_k3", "novel", "fleet_residual"]
    scorer_names = sorted({name for r in runs.values()
                           for name in r["metrics"]["spoof_k1"]})
    metrics = {}
    for pop in populations:
        metrics[pop] = {}
        for name in scorer_names:
            for key in ["roc_auc", "pr_auc", "top1pct_tp", "top1pct_precision"]:
                vals = [(runs[s]["metrics"][pop].get(name) or {}).get(key) for s in seeds]
                metrics[pop][f"{name}_{key}"] = mean_std(vals)
    delta = {}
    for metric in ["roc", "pr"]:
        pts = [runs[s]["spoof_k1_delta_mp_minus_lik_best"][metric][0] for s in seeds]
        los = [runs[s]["spoof_k1_delta_mp_minus_lik_best"][metric][1] for s in seeds]
        delta[f"spoof_k1_{metric}"] = {
            "point": mean_std(pts),
            "ci_lower_min": float(np.min(los)),
            "ci_lower_per_seed": los,
        }
    verdict_names = sorted({v for r in runs.values() for v in r["verdicts"]})
    verdicts = {
        v: {
            "per_seed": {s: runs[s]["verdicts"][v] for s in seeds},
            "confirmed": sum(runs[s]["verdicts"][v] for s in seeds),
            "n_seeds": len(seeds),
        }
        for v in verdict_names
    }
    best_variants = {s: runs[s]["best_likelihood_variant"] for s in seeds}
    return {"seeds": seeds, "metrics": metrics, "spoof_k1_delta": delta,
            "verdicts": verdicts, "best_likelihood_variant": best_variants}


def aggregate_oov(runs: dict) -> dict:
    """Cross-seed aggregation for the OOV-regime sweep (oov_regime.py).

    Reports per (arm, level, population) mean±std for every scorer, a per-seed-selected
    incumbent line (lik_best), the FastText−incumbent spoof_k1 delta, and the subword-
    attribution quantity gap(morphological) − gap(arbitrary) at matched levels.
    """
    seeds = sorted(runs)
    cfg = runs[seeds[0]]["config"]
    arms = cfg["arms"]
    levels = [f"{p:.2f}" for p in cfg["levels"]]
    pops = cfg["populations"]
    best_variant = {s: runs[s]["best_likelihood_variant"] for s in seeds}

    def cell(arm, lvl):
        return {s: runs[s]["results"][arm][lvl] for s in seeds}

    metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for lvl in levels:
            metrics[arm][lvl] = {}
            for pop in pops:
                names = sorted(runs[seeds[0]]["results"][arm][lvl]["metrics"][pop])
                cm = {}
                for name in names:
                    for key in ["roc_auc", "pr_auc", "top1pct_tp", "top1pct_precision"]:
                        vals = [(runs[s]["results"][arm][lvl]["metrics"][pop].get(name) or {}).get(key)
                                for s in seeds]
                        cm[f"{name}_{key}"] = mean_std(vals)
                # incumbent line: per-seed best-lambda variant
                for key in ["roc_auc", "pr_auc", "top1pct_tp", "top1pct_precision"]:
                    vals = [(runs[s]["results"][arm][lvl]["metrics"][pop].get(best_variant[s]) or {}).get(key)
                            for s in seeds]
                    cm[f"lik_best_{key}"] = mean_std(vals)
                metrics[arm][lvl][pop] = cm

    deltas = {}
    for arm in arms:
        deltas[arm] = {}
        for lvl in levels:
            d = {}
            for metric in ["roc", "pr"]:
                trips = [runs[s]["results"][arm][lvl]["delta_mp_minus_likbest_spoof_k1"][metric]
                         for s in seeds]
                d[metric] = {
                    "point": mean_std([t[0] for t in trips]),
                    "ci_lower_min": float(np.min([t[1] for t in trips])),
                    "ci_upper_max": float(np.max([t[2] for t in trips])),
                    "per_seed_point": [t[0] for t in trips],
                }
            deltas[arm][lvl] = d

    # Attribution = gap(morph arm) − gap(arbitrary arm) at matched levels. cross_feature is
    # confounded with feature choice; same_feature_region isolates morphology on one feature.
    attr_pairs = [("cross_feature", "morphological", "arbitrary"),
                  ("same_feature_region", "region_morph", "region_arb")]
    subword = {}
    for pname, marm, aarm in attr_pairs:
        if marm not in arms or aarm not in arms:
            continue
        subword[pname] = {}
        for lvl in levels:
            s_att = {}
            for metric in ["roc", "pr"]:
                gaps = [
                    runs[s]["results"][marm][lvl]["delta_mp_minus_likbest_spoof_k1"][metric][0]
                    - runs[s]["results"][aarm][lvl]["delta_mp_minus_likbest_spoof_k1"][metric][0]
                    for s in seeds
                ]
                s_att[metric] = mean_std(gaps)
            subword[pname][lvl] = s_att

    obs = {
        arm: {lvl: mean_std([cell(arm, lvl)[s]["obs_oov_rate"]["spoof_k1"] for s in seeds])
              for lvl in levels}
        for arm in arms
    }
    return {"seeds": seeds, "arms": arms, "levels": levels, "populations": pops,
            "neg_ratio": cfg.get("neg_ratio"),
            "metrics": metrics, "deltas": deltas, "subword_attribution": subword,
            "obs_oov_rate": obs, "best_likelihood_variant": best_variant}


def write_csv(summary: dict, path: Path):
    rows = []
    base = summary.get("baseline_controls")
    if base:
        for name, metrics in base["scorers"].items():
            for key, stat in metrics.items():
                rows.append(["baseline_controls", name, key, stat["mean"], stat["std"]])
        for dname, metrics in base["spoof_deltas"].items():
            for key, stat in metrics.items():
                rows.append(["baseline_controls", dname, f"{key}_point",
                             stat["point"]["mean"], stat["point"]["std"]])
                rows.append(["baseline_controls", dname, f"{key}_ci_lower_min",
                             stat["ci_lower_min"], ""])
        for v, stat in base["verdicts"].items():
            rows.append(["baseline_controls", "verdict", v,
                         f"{stat['confirmed']}/{stat['n_seeds']}", ""])
    h6 = summary.get("h6_perevent")
    if h6:
        for cell, metrics in h6["cells"].items():
            for key, stat in metrics.items():
                rows.append(["h6_perevent", cell, key, stat["mean"], stat["std"]])
        for shape, feats in h6["jsd"].items():
            for f, stat in feats.items():
                rows.append(["h6_perevent", f"jsd_{shape}", f, stat["mean"], stat["std"]])
        for v, stat in h6["verdicts"].items():
            rows.append(["h6_perevent", "verdict", v,
                         f"{stat['confirmed']}/{stat['n_seeds']}", ""])
    lik = summary.get("h6_likelihood")
    if lik:
        for pop, metrics in lik["metrics"].items():
            for key, stat in metrics.items():
                rows.append(["h6_likelihood", pop, key, stat["mean"], stat["std"]])
        for key, stat in lik["spoof_k1_delta"].items():
            rows.append(["h6_likelihood", "delta_mp_minus_lik_best", f"{key}_point",
                         stat["point"]["mean"], stat["point"]["std"]])
            rows.append(["h6_likelihood", "delta_mp_minus_lik_best", f"{key}_ci_lower_min",
                         stat["ci_lower_min"], ""])
        for v, stat in lik["verdicts"].items():
            rows.append(["h6_likelihood", "verdict", v,
                         f"{stat['confirmed']}/{stat['n_seeds']}", ""])
    a4 = summary.get("a4_nosub")
    if a4:
        for name, metrics in a4["scorers"].items():
            for key, stat in metrics.items():
                rows.append(["a4_nosub", name, key, stat["mean"], stat["std"]])
        for dname, metrics in a4["spoof_deltas"].items():
            for key, stat in metrics.items():
                rows.append(["a4_nosub", dname, f"{key}_point",
                             stat["point"]["mean"], stat["point"]["std"]])
        for v, stat in a4["verdicts"].items():
            rows.append(["a4_nosub", "verdict", v,
                         f"{stat['confirmed']}/{stat['n_seeds']}", ""])
    oov = summary.get("oov_regime")
    if oov:
        for arm in oov["arms"]:
            for lvl in oov["levels"]:
                for pop in oov["populations"]:
                    for key, stat in oov["metrics"][arm][lvl][pop].items():
                        rows.append(["oov_regime", f"{arm}|p={lvl}|{pop}", key,
                                     stat["mean"], stat["std"]])
                for metric in ["roc", "pr"]:
                    d = oov["deltas"][arm][lvl][metric]
                    rows.append(["oov_regime", f"{arm}|p={lvl}", f"delta_mp_minus_likbest_{metric}_point",
                                 d["point"]["mean"], d["point"]["std"]])
                    rows.append(["oov_regime", f"{arm}|p={lvl}", f"delta_mp_minus_likbest_{metric}_ci_lower_min",
                                 d["ci_lower_min"], ""])
        for pname, per_lvl in oov["subword_attribution"].items():
            for lvl, s_att in per_lvl.items():
                for metric, stat in s_att.items():
                    rows.append(["oov_regime", f"attribution|{pname}|p={lvl}", metric,
                                 stat["mean"], stat["std"]])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "group", "metric", "mean", "std"])
        writer.writerows(rows)


def main():
    baseline_runs = load("baseline_controls_seed_{seed}.json")
    h6_runs = load("h6_perevent_seed_{seed}.json")
    lik_runs = load("h6_likelihood_seed_{seed}.json")
    a4_runs = load("a4_nosub_seed_{seed}.json")
    oov_runs = load("oov_regime_seed_{seed}.json")
    summary = {"schema_version": 1, "plan": "plan/12-CRITIQUE_ABLATIONS.md"}
    if baseline_runs:
        summary["baseline_controls"] = aggregate_baseline(baseline_runs)
    if h6_runs:
        summary["h6_perevent"] = aggregate_h6(h6_runs)
    if lik_runs:
        summary["h6_likelihood"] = aggregate_h6_likelihood(lik_runs)
    if a4_runs:
        summary["a4_nosub"] = aggregate_a4(a4_runs)
    if oov_runs:
        summary["oov_regime"] = aggregate_oov(oov_runs)

    out_dir = ROOT / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablations_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(summary, out_dir / "ablations_summary.csv")
    print(f"Wrote {out_dir / 'ablations_summary.json'}")
    print(f"Wrote {out_dir / 'ablations_summary.csv'}")

    if "baseline_controls" in summary:
        b = summary["baseline_controls"]
        print(f"\n=== A1/A2a (seeds present: {b['seeds']}) ===")
        for name, m in b["scorers"].items():
            s = m["spoof_roc_auc"]
            p = m["spoof_pr_auc"]
            print(f"  {name:<12} spoof ROC={s['mean']:.4f}±{s['std']:.4f}  PR={p['mean']:.4f}±{p['std']:.4f}")
        for v, stat in b["verdicts"].items():
            print(f"  {v}: {stat['confirmed']}/{stat['n_seeds']}")
    if "h6_perevent" in summary:
        h = summary["h6_perevent"]
        print(f"\n=== A3p (seeds present: {h['seeds']}) ===")
        for cell in H6_CELLS:
            pooled = h["cells"][cell]["pooled_within_cos_wv"]
            print(f"  {cell:<18} pooled_within={pooled['mean']:.4f}±{pooled['std']:.4f}")
        print("  per-feature within-cos (sg_per_event): " + "  ".join(
            f"{f}={h['cells']['sg_per_event'][f + '_within_cos_wv']['mean']:.3f}"
            for f in H6_FEATURES))
        for v, stat in h["verdicts"].items():
            print(f"  {v}: {stat['confirmed']}/{stat['n_seeds']}")
    if "h6_likelihood" in summary:
        k = summary["h6_likelihood"]
        print(f"\n=== A2b (seeds present: {k['seeds']}) ===")
        fr = k["metrics"]["fleet_residual"]
        for name in ["mp_raw", "two_stage", "lik_lam1.0", "lik_lam0.9", "lik_lam0.5"]:
            tp = fr.get(f"{name}_top1pct_tp")
            pr = fr.get(f"{name}_top1pct_precision")
            roc = fr.get(f"{name}_roc_auc")
            if tp:
                print(f"  fleet_residual {name:<12} TP={tp['mean']:.1f}±{tp['std']:.1f}  "
                      f"prec={pr['mean']:.3f}±{pr['std']:.3f}  ROC={roc['mean']:.3f}")
        sk1 = k["metrics"]["spoof_k1"]
        for name in ["mp_raw", "lik_lam1.0", "lik_lam0.9", "lik_lam0.5"]:
            roc = sk1.get(f"{name}_roc_auc")
            pr = sk1.get(f"{name}_pr_auc")
            if roc:
                print(f"  spoof_k1 {name:<12} ROC={roc['mean']:.4f}±{roc['std']:.4f}  "
                      f"PR={pr['mean']:.4f}±{pr['std']:.4f}")
        for v, stat in k["verdicts"].items():
            print(f"  {v}: {stat['confirmed']}/{stat['n_seeds']}")
    if "a4_nosub" in summary:
        a = summary["a4_nosub"]
        print(f"\n=== A4 (seeds present: {a['seeds']}) ===")
        for name, m in a["scorers"].items():
            s, p = m["spoof_roc_auc"], m["spoof_pr_auc"]
            print(f"  {name:<12} spoof ROC={s['mean']:.4f}±{s['std']:.4f}  PR={p['mean']:.4f}±{p['std']:.4f}")
        for dname, metrics in a["spoof_deltas"].items():
            r = metrics["spoof_roc"]
            print(f"  {dname}: dROC={r['point']['mean']:+.4f} "
                  f"(CI lower min {r['ci_lower_min']:+.4f}, upper max {r['ci_upper_max']:+.4f})")
        for v, stat in a["verdicts"].items():
            print(f"  {v}: {stat['confirmed']}/{stat['n_seeds']}")
    if "oov_regime" in summary:
        o = summary["oov_regime"]
        print(f"\n=== OOV regime (seeds present: {o['seeds']}) ===")
        for arm in o["arms"]:
            print(f"  [{arm}] spoof_k1 ROC-AUC  (mp_raw / lik_best / trivial)  +  dROC(mp-lik)")
            for lvl in o["levels"]:
                m = o["metrics"][arm][lvl]["spoof_k1"]
                mp, lik, triv = m["mp_raw_roc_auc"], m["lik_best_roc_auc"], m["trivial_roc_auc"]
                d = o["deltas"][arm][lvl]["roc"]["point"]
                print(f"    p={lvl}  mp={mp['mean']:.3f}±{mp['std']:.3f}  "
                      f"lik={lik['mean']:.3f}±{lik['std']:.3f}  "
                      f"triv={triv['mean']:.3f}±{triv['std']:.3f}  "
                      f"dROC={d['mean']:+.3f}±{d['std']:.3f}")
        for pname, per_lvl in o["subword_attribution"].items():
            print(f"  attribution [{pname}]  gap(morph)-gap(arb)  dPR (dROC):")
            for lvl, s_att in per_lvl.items():
                pr, rc = s_att["pr"], s_att["roc"]
                print(f"    p={lvl}  dPR={pr['mean']:+.4f}±{pr['std']:.4f}  "
                      f"dROC={rc['mean']:+.4f}±{rc['std']:.4f}")


if __name__ == "__main__":
    main()
