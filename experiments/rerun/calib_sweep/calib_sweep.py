# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim==4.4.0",
#   "numpy==2.2.6",
#   "scikit-learn==1.7.2",
#   "matplotlib==3.10.8",
# ]
# ///
#
# Calibration-window ablation for the rank-normalization PR-AUC collapse
# =====================================================================
# Follow-up to the C4 finding (per-user CDF rank-normalization collapses spoof-k1
# PR-AUC from ~0.888 to ~0.224 at 1:100 imbalance while ROC-AUC barely moves).
#
# Open question: is the collapse a small-sample quantization artifact of the
# 20-event calibration window, or fundamental to rank-normalization under imbalance?
# The rank floor is 1/len(calib)-quantized, so if it is a small-sample effect it
# should shrink as the calibration window grows.
#
# Design (Option A -- grow total login history):
#   N_CENTROID stays fixed at 40; the calibration slice is train[N_CENTROID:], so
#   growing the window means growing N_TRAIN = 40 + calib_size. This also grows the
#   FastText corpus (calib events feed build_corpus) -- an intended property of this
#   variant. mp_raw PR-AUC is tracked as the embedding-quality CONTROL: it never
#   touches the calibration window, so its trend across calib_size isolates any
#   embedding shift, and the (mp_raw - mp_rank_norm) gap isolates the rank-norm
#   penalty controlling for it.
#
# Reuse policy: this script does NOT reimplement data generation, corpus building,
# FastText training, scoring, or bootstrapping. It imports h6_rerun and drives its
# unmodified functions, changing only the module-level N_TRAIN per cell.
#
# Usage:
#   uv run experiments/rerun/calib_sweep/calib_sweep.py --smoke        # fast wiring check
#   uv run experiments/rerun/calib_sweep/calib_sweep.py --seed 42 --calib 20   # one cell
#   uv run experiments/rerun/calib_sweep/calib_sweep.py               # full 5x5 sweep

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- Reuse the H6 rerun pipeline as a library (import is side-effect-free) ---
H6_DIR = Path(__file__).resolve().parent.parent / "scripts" / "h6"
sys.path.insert(0, str(H6_DIR))
import h6_rerun as H  # noqa: E402
from h6_score_margin import compute_margins  # noqa: E402

CALIB_SIZES = [20, 50, 100, 200, 500]
SEEDS       = [42, 123, 456, 789, 2024]
ATTACK      = "spoof_k1"          # hardest gradient point; the C4 metric
SCORERS     = ["mp_raw", "mp_rank_norm"]
OUT_ROOT    = Path(__file__).resolve().parent / "results"


def run_cell(marginals, seed: int, calib: int, neg_ratio: int, smoke: bool) -> dict:
    """Run one (seed, calib_size) cell and return the results schema dict.

    The centroid window is held at N_CENTROID=40; the calibration slice
    train[N_CENTROID:] therefore has exactly `calib` events when N_TRAIN=40+calib.
    """
    H.N_TRAIN = H.N_CENTROID + calib          # the only data-gen "change" -- one constant
    assert H.N_TRAIN - H.N_CENTROID == calib

    n_accounts   = H.N_ACCOUNTS_SMOKE if smoke else H.N_ACCOUNTS_FULL
    n_bootstrap  = 200 if smoke else H.N_BOOTSTRAP
    n_enroll_neg = H.N_ENROLL_NEG * neg_ratio

    # Fresh rng per cell so calib=20 reproduces the published seed run exactly.
    rng = np.random.default_rng(seed)

    t0 = time.time()
    accounts, _fleet = H.generate_accounts(marginals, n_accounts, rng, n_enroll_neg=n_enroll_neg)
    corpus = H.build_corpus(accounts)
    vocab_size = len({tok for sent in corpus for tok in sent})
    model = H.train_model(corpus, seed=seed)
    t8 = H.compute_t8(model)
    centroids = H.compute_centroids(accounts, model)
    score_dict = H.collect_scores_for_attack(accounts, centroids, model, ATTACK)
    corpus_tokens = sum(len(sent) for sent in corpus)

    spoof_block: dict = {}
    for scorer in SCORERS:
        sc, lab = score_dict[scorer]
        sc = np.asarray(sc, dtype=float)
        lab = np.asarray(lab, dtype=int)
        # bootstrap_auc: point estimates are rng-independent (roc_auc_score /
        # average_precision_score computed before the resample loop); only the CI
        # bounds use rng. Same percentile method / N_BOOTSTRAP as the H6 pipeline.
        roc, roc_lo, roc_hi, pr, pr_lo, pr_hi = H.bootstrap_auc(sc, lab, n_bootstrap, rng)
        block = H.build_scorer_block(roc, pr, pr_lo, pr_hi)   # {roc_auc,pr_auc,ci_lower,ci_upper}
        block["margins"] = compute_margins(sc, lab)           # contamination_rate, robust_margin, ...
        spoof_block[scorer] = block

    elapsed = time.time() - t0
    return {
        "seed": seed,
        "calib_size": calib,
        "n_train": H.N_TRAIN,
        "n_centroid": H.N_CENTROID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 2),
        "config": {
            "n_accounts": n_accounts,
            "vocab_size": vocab_size,
            "corpus_tokens": corpus_tokens,
            "imbalance_ratio": neg_ratio,
            "neg_ratio": neg_ratio,
            "n_bootstrap": n_bootstrap,
            "attack_type": ATTACK,
            "smoke": smoke,
        },
        "t8_token_similarity": t8,
        ATTACK: spoof_block,
    }


def write_cell(result: dict, out_root: Path) -> Path:
    out_dir = out_root / f"calib_{result['calib_size']}" / f"seed_{result['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Calibration-window ablation (C4 follow-up)")
    p.add_argument("--seed", type=int, default=None, help="Single seed (default: all 5)")
    p.add_argument("--calib", type=int, default=None, help="Single calib size (default: all)")
    p.add_argument("--neg-ratio", type=int, default=100, help="Enroll negatives per attack (1:N)")
    p.add_argument("--smoke", action="store_true", help="50 accounts, 200 bootstrap resamples")
    p.add_argument("--out", type=str, default=None, help="Output root (default: ./results)")
    args = p.parse_args()

    if not H.MARGINALS_PATH.exists():
        sys.exit(
            f"ERROR: {H.MARGINALS_PATH} not found.\n"
            "Run: uv run experiments/rerun/scripts/h6/data_prep.py"
        )

    seeds  = [args.seed] if args.seed is not None else SEEDS
    calibs = [args.calib] if args.calib is not None else CALIB_SIZES
    out_root = Path(args.out) if args.out else OUT_ROOT
    marginals = H.Marginals(H.MARGINALS_PATH)   # seed-invariant; load once

    print("=" * 68)
    print(f"Calib-window ablation  seeds={seeds}  calib={calibs}  "
          f"neg_ratio={args.neg_ratio}:1" + ("  SMOKE" if args.smoke else ""))
    print("=" * 68)

    for calib in calibs:
        for seed in seeds:
            r = run_cell(marginals, seed, calib, args.neg_ratio, args.smoke)
            write_cell(r, out_root)
            raw = r[ATTACK]["mp_raw"]
            rn  = r[ATTACK]["mp_rank_norm"]
            print(
                f"  calib={calib:<3d} seed={seed:<4d} N_TRAIN={r['n_train']:<3d} "
                f"| mp_raw PR={raw['pr_auc']:.4f} ROC={raw['roc_auc']:.4f} "
                f"| rank_norm PR={rn['pr_auc']:.4f} ROC={rn['roc_auc']:.4f} "
                f"| contam={rn['margins']['contamination_rate']:.4f} "
                f"margin={rn['margins']['robust_margin']:+.4f} "
                f"| t8={t8_flag(r)} ({r['elapsed_sec']:.0f}s)"
            )
    print("\nDone.")


def t8_flag(r: dict) -> str:
    t8 = r["t8_token_similarity"]
    return "COLLAPSE!" if t8.get("collapse_detected") else f"ok({t8['within_feature_mean']:.2f})"


if __name__ == "__main__":
    main()
