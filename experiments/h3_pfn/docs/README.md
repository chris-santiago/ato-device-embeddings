# H3: Per-Feature Normalized (PFN) Scoring

**Verdict: NOT CONFIRMED.** No PFN variant improves top-1% precision and recall
simultaneously over the fair mean-pool baseline (mp-raw-split). mp-raw-split achieves
precision=0.0098, recall=0.444 (4/9 ATO events caught); pfn-mean achieves
precision=0.0074, recall=0.333 (3/9). See `h3_pfn/docs/REPORT.md` for full analysis.

## Hypothesis

Per-feature anomaly decomposition — computing per-feature cosine distances
independently, rank-normalizing each against a per-user calibration distribution, and
aggregating (uniform mean, stability-weighted, or max) — will improve top-1% precision
and recall compared to the mean-pool rank-norm baseline on the DAS Group RBA dataset.

**Motivation:** The current mean-pool approach dilutes single-feature anomalies. A
timezone-only mismatch moves only 1/7 of the mean-pool vector; PFN computes 7 independent
distances so a single-feature anomaly produces a full-signal contribution to the max or
mean aggregation.

## Quickstart

```bash
# v2 experiment (recommended — fixes F1/F2/F6 from critic review)
uv run h3_pfn/experiments/pfn_experiment2.py

# v1 experiment (original, unfixed)
uv run h3_pfn/experiments/pfn_experiment.py           # default (50k benign + all ATO users)
uv run h3_pfn/experiments/pfn_experiment.py --smoke   # fast sanity check on 5k users
uv run h3_pfn/experiments/pfn_experiment.py --full    # all eligible users
```

Requires: `data/rba/rba.parquet` — run `uv run h2_rba/experiments/data_prep.py` first.

## Pipeline (v2)

```
data/rba/rba.parquet
  → load_users(): 50/50 temporal split, drop users with <5 training events
  → split_calib() once per user: last max(2, n//3) events as calibration; rest as centroid
    (shared split — all variants use identical centroid_evts and calib_evts)
  → FastText (single model: sg=1, vector_size=64, window=6, per-account corpus)
  → 7 scoring variants:
      mp-raw         — mean-pool cosine distance, all training events
      mp-raw-split   — mean-pool cosine distance, centroid_evts only (fair comparison)
      mp-rank-norm   — mean-pool, rank-normalized per-user
      pfn-mean       — per-feature rank-norm, uniform mean
      pfn-stab       — per-feature rank-norm, stability-weighted
      pfn-max        — per-feature rank-norm, max over features
      trivial        — exact set-membership (device key in training)
  → tie-breaking perturbation: +uniform(0, 1e-9) before threshold computation (F1 fix)
  → stratified bootstrap metrics: ROC-AUC, PR-AUC, top-1% precision/recall (F6 fix)
  → figures: pfn_v2_summary_auc.png, pfn_v2_top1pct.png, pfn_v2_pr_curves.png,
             pfn_v2_feature_importance.png
  → metrics: h3_pfn/figures/pfn_v2_metrics.json
```

## Output (v2, default run, 50/50 split)

| Model | ROC-AUC | [95% CI] | PR-AUC | Top-1% Precision | Top-1% Recall |
|---|---|---|---|---|---|
| mp-raw | 0.852 | [0.708, 0.965] | 0.031 | 0.0098 | 0.444 |
| **mp-raw-split** | **0.856** | **[0.699, 0.978]** | **0.021** | **0.0098** | **0.444** |
| mp-rank-norm | 0.810 | [0.744, 0.844] | 0.001 | 0.000 | 0.000 |
| pfn-mean | 0.807 | [0.649, 0.939] | 0.027 | 0.0074 | 0.333 |
| pfn-stab | 0.782 | [0.647, 0.901] | 0.017 | 0.0074 | 0.333 |
| pfn-max | 0.699 | [0.608, 0.777] | 0.000 | 0.000 | 0.000 |
| trivial | 0.659 | [0.546, 0.717] | 0.000 | 0.000 | 0.000 |

Note: mp-rank-norm, pfn-max, and trivial top-1% metrics are zero because all their ATO
test events score at 1.0 (maximum rank-normalized value); tie-breaking perturbation
distributes them across the top-1% flag window randomly, yielding 0 TPs by chance at
point estimate. See per-feature AUC analysis for why rtt_bucket (AUC=0.461) degrades
pfn-mean vs mp-raw-split.

## Per-Feature AUC

| Feature | AUC |
|---|---|
| country | 0.829 |
| asn_bucket | 0.805 |
| device_type | 0.691 |
| os | 0.655 |
| region | 0.632 |
| browser | 0.606 |
| rtt_bucket | 0.461 (below chance) |

rtt_bucket actively degrades pfn-mean's uniform aggregate. Mean-pool implicitly
downweights uninformative features; PFN's uniform mean cannot.

## Known Limitations / Explicit Scope Exclusions

1. **Discrete score support in rank-normalized scorers (partially mitigated):** The
   proportional calibration split produces median 2 calibration events per user. With 2
   calibration events, rank-normalized scores take values in {0.0, 0.5, 1.0}. v2 applies
   tie-breaking perturbation (`+uniform(0, 1e-9)`) before threshold computation to ensure
   exactly top-1% of events are flagged, but the underlying discrete support issue
   persists. This is a fundamental dataset constraint: no ATO user survives
   MIN_TRAIN_EVENTS >= 30 on the RBA dataset.

2. **Only n=9 ATO users in test window** (50/50 split). Top-1% recall changes by 0.111
   per TP. Bootstrap CIs are wide (recall CI spans 0.000-0.778 for mp-raw). Results are
   directionally consistent but individual metric differences are not statistically reliable.

3. **IDF-style cross-user feature entropy weighting** (optional 4th variant) not
   implemented. Weighted aggregation by per-feature AUC is the recommended follow-on.

4. **No learned weights** — no labeled ATOs at sufficient granularity for training.

5. **Single FastText model** shared across all scoring variants.

6. **Stratified bootstrap** implemented in v2; v1 used simple percentile bootstrap
   (slightly optimistic CIs).
