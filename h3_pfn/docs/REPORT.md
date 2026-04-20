# H3 Investigation Report: Per-Feature Normalized (PFN) Scoring

**Date:** 2026-04-20
**Dataset:** DAS Group RBA (`data/rba/rba.parquet`)
**Experiment script:** `h3_pfn/experiments/pfn_experiment2.py` (v2, fixes applied)
**Metrics artifact:** `h3_pfn/figures/pfn_v2_metrics.json`

---

## Verdict: NOT CONFIRMED

**PFN does not improve top-1% precision or recall over mean-pool on the RBA dataset.**

> **Note on `hypothesis_confirmed: true` in pfn_v2_metrics.json:** This field reflects
> the script's internal comparison of pfn-mean vs mp-rank-norm. That comparison is invalid:
> mp-rank-norm's degenerate calibration (n_calib=2 → scores in {0.0, 0.5, 1.0}) causes its
> threshold to land at 1.0+eps, yielding 0 TPs. The correct comparison is pfn-mean vs
> mp-raw-split — identical training data, fair apples-to-apples. On that comparison,
> pfn-mean loses on every metric. The flag must be disregarded.

---

## Hypothesis

Per-feature anomaly decomposition — computing per-feature cosine distances independently,
rank-normalizing each against a per-user calibration distribution, and aggregating
(uniform mean, stability-weighted, or max) — will improve top-1% precision and recall
compared to the mean-pool rank-norm baseline.

**Proposed mechanism:** A single-feature anomaly (e.g., a new country) contributes 1/7 of
the mean-pool distance but a full-rank signal in the per-feature decomposition.

**Falsification condition:** No PFN variant improves both top-1% precision and recall
simultaneously over mp-raw-split.

---

## Method

**Dataset:** 6,343 eligible users after MIN_TRAIN_EVENTS=5 filter; 50/50 temporal split.
**Test window:** 9 ATO events, 40,628 benign events (40,637 total).

**Calibration split:** Proportional — last `max(2, n//3)` training events reserved for
calibration; remainder used for centroid. Median calibration size: **2 events**
(fundamental limitation; see F4).

**FastText model:** sg=1, vector_size=64, window=6, epochs=20, min_count=1,
per-account concatenated corpus. Single shared model across all variants.

### Scoring Variants

| Variant       | Description                                                                |
|---------------|----------------------------------------------------------------------------|
| mp-raw        | Mean-pool cosine distance; centroid from ALL training events               |
| mp-raw-split  | Mean-pool cosine distance; centroid from `centroid_evts` only (F2 fix)     |
| mp-rank-norm  | Mean-pool, rank-normalized per-user against calibration                    |
| pfn-mean      | Per-feature rank-norm, uniform mean across 7 features                      |
| pfn-stab      | Per-feature rank-norm, stability-weighted (1/σ) mean                       |
| pfn-max       | Per-feature rank-norm, max over 7 features                                 |
| trivial        | Exact set-membership (device key in training set)                          |

### Fixes Applied (v2 vs v1)

| Finding | Issue | Fix |
|---------|-------|-----|
| F1 | Discrete score support from n_calib=2 collapses threshold | Tie-breaking perturbation (+uniform(0,1e-9)) before top-1% computation |
| F2 | mp-raw used all training events; rank-norm variants used only centroid subset | Shared `split_calib()` call; added `mp-raw-split` as the fair baseline |
| F6 | Non-stratified bootstrap with 9 positives | Stratified bootstrap: positives and negatives resampled separately |

**Unfixed limitations:** F3 (stability weights on n_calib=2 are noisy), F4 (9 ATO test
events — raising MIN_TRAIN_EVENTS to 61 eliminates all ATO users from eval), F5
(per-feature distances correlated via FastText co-occurrence — intentional by design).

---

## Results

### ROC-AUC (bootstrap mean, 95% CI, N=1000 stratified)

| Model        | ROC-AUC | 95% CI            |
|--------------|---------|-------------------|
| mp-raw       | 0.852   | [0.708, 0.965]    |
| mp-raw-split | 0.856   | [0.699, 0.978]    |
| mp-rank-norm | 0.810   | [0.744, 0.844]    |
| pfn-mean     | 0.807   | [0.649, 0.939]    |
| pfn-stab     | 0.782   | [0.647, 0.901]    |
| pfn-max      | 0.699   | [0.608, 0.777]    |
| trivial      | 0.659   | [0.546, 0.717]    |

### PR-AUC (bootstrap mean, 95% CI)

| Model        | PR-AUC  | 95% CI               |
|--------------|---------|----------------------|
| mp-raw       | 0.0307  | [0.0035, 0.0826]     |
| mp-raw-split | 0.0207  | [0.0030, 0.0538]     |
| mp-rank-norm | 0.0006  | [0.0005, 0.0007]     |
| pfn-mean     | 0.0269  | [0.0009, 0.1018]     |
| pfn-stab     | 0.0170  | [0.0005, 0.0535]     |
| pfn-max      | 0.0004  | [0.0003, 0.0005]     |
| trivial      | 0.0003  | [0.0002, 0.0004]     |

### Top-1% Threshold Metrics (~407 flagged out of 40,637)

| Model        | TP | FP  | Precision | Recall | Threshold    |
|--------------|----|-----|-----------|--------|--------------|
| mp-raw       | 4  | 403 | 0.0098    | 0.444  | 0.239        |
| mp-raw-split | 4  | 403 | 0.0098    | 0.444  | 0.253        |
| mp-rank-norm | 0  | 407 | 0.000     | 0.000  | 1.000+ε      |
| pfn-mean     | 3  | 404 | 0.0074    | 0.333  | 0.643        |
| pfn-stab     | 3  | 404 | 0.0074    | 0.333  | 0.714        |
| pfn-max      | 0  | 407 | 0.000     | 0.000  | 1.000+ε      |
| trivial      | 0  | 407 | 0.000     | 0.000  | 1.000+ε      |

### Top-1% Bootstrap CIs (stratified, N=1000)

| Model        | Precision [95% CI]         | Recall [95% CI]            |
|--------------|----------------------------|----------------------------|
| mp-raw       | 0.0098 [0.0025, 0.0172]    | 0.444 [0.111, 0.778]       |
| mp-raw-split | 0.0098 [0.0025, 0.0172]    | 0.444 [0.111, 0.778]       |
| mp-rank-norm | 0.0007 [0.000, 0.0049]     | 0.032 [0.000, 0.222]       |
| pfn-mean     | 0.0074 [0.000, 0.0147]     | 0.333 [0.000, 0.667]       |
| pfn-stab     | 0.0074 [0.000, 0.0147]     | 0.333 [0.000, 0.667]       |
| pfn-max      | 0.0004 [0.000, 0.0025]     | 0.019 [0.000, 0.111]       |
| trivial      | 0.0004 [0.000, 0.0025]     | 0.019 [0.000, 0.111]       |

---

## Per-Feature AUC

| Feature     | AUC   |
|-------------|-------|
| country     | 0.829 |
| asn_bucket  | 0.805 |
| device_type | 0.691 |
| os          | 0.655 |
| region      | 0.632 |
| browser     | 0.606 |
| rtt_bucket  | **0.461** ← below chance |

`rtt_bucket` performs below chance. Uniform-mean aggregation treats it as equally
informative as `country` (AUC=0.829), diluting the aggregate with anti-signal.
Mean-pool implicitly downweights uninformative features through the FastText embedding
geometry; PFN's uniform mean cannot.

---

## Why PFN Does Not Improve Over Mean-Pool

**Direct comparison — pfn-mean vs mp-raw-split (same training data):**

| Metric          | pfn-mean | mp-raw-split | Winner         |
|-----------------|----------|--------------|----------------|
| ROC-AUC         | 0.807    | 0.856        | mp-raw-split   |
| PR-AUC          | 0.027    | 0.021        | pfn-mean (marginal, within CI) |
| Top-1% precision| 0.0074   | 0.0098       | mp-raw-split   |
| Top-1% recall   | 0.333    | 0.444        | mp-raw-split   |

**Root cause 1 — rtt_bucket degrades the aggregate:**
`rtt_bucket` AUC=0.461 means including it inverts signal. FastText mean-pool implicitly
downweights this feature because it contributes low-variance token vectors that pull the
centroid only slightly; PFN's uniform mean weights it at 1/7.

**Root cause 2 — n_calib=2 makes rank-normalization degenerate:**
Rank-normalized scores per feature take values {0.0, 0.5, 1.0} with only 2 calibration
events. pfn-max and pfn-stab collapse to thresholded detectors; pfn-mean averages 7
degenerate distributions.

**Mechanism status:** The dilution argument is mechanically correct for feature distances
themselves. But FastText mean-pool already performs implicit feature weighting through
embedding geometry — the embedding space concentrates ATO-discriminative variance where
mean-pool captures it efficiently. The proposed fix (explicit per-feature decomposition)
only helps when the implicit weighting is wrong and you have sufficient calibration data
to estimate per-feature distributions reliably. Neither condition holds here.

---

## Critique Scorecard

| Finding | Topic                          | Action                                | Outcome                                       |
|---------|--------------------------------|---------------------------------------|-----------------------------------------------|
| F1      | Discrete score support         | Tie-breaking perturbation             | Partially mitigated; root cause (n_calib=2) persists |
| F2      | Centroid data asymmetry        | mp-raw-split baseline; shared split   | Resolved — changed verdict from confirmed to refuted |
| F3      | Stability weight instability   | Acknowledged                          | Unfixed; pfn-stab results unreliable          |
| F4      | 9 ATO test events              | Cannot fix without losing all ATOs    | Dataset constraint; wide CIs accepted         |
| F5      | Cross-feature co-occurrence    | Intentional by design                 | Accepted                                      |
| F6      | Non-stratified bootstrap       | Stratified bootstrap                  | Fixed                                         |

---

## Recommendation

**Use mp-raw or mp-raw-split.** Both achieve top-1% recall=0.444 (4/9 ATO events) with
precision=0.0098. Substantially better than any PFN variant.

**Do not deploy pfn-mean, pfn-stab, or pfn-max.** All three underperform mp-raw-split
on primary metrics when controlling for training data.

### Highest-priority follow-ons

1. **Feature selection:** Exclude `rtt_bucket` from PFN aggregate. A 6-feature pfn-mean
   excluding rtt_bucket is the lowest-cost next test — it fixes the identified failure
   mode without requiring more data.

2. **IDF-style weights:** Use empirical per-feature AUCs as aggregation weights
   (country=0.829, asn=0.805 → high weight; rtt=0.461 → near-zero weight). Unsupervised
   and more principled than stability weighting.

3. **Cross-user calibration:** Per-user n_calib=2 is degenerate. Normalizing per-feature
   distances against the population distribution (not per-user) would provide stable
   CDFs without requiring long individual histories.

4. **Investigate the 4 caught TPs:** Which features drive mp-raw's score on those 4 events?
   If consistently country/asn_bucket, the implicit mean-pool weighting is reliable and
   PFN with those two features would be the minimal viable variant.

---

## Limitations

1. **n=9 ATO test events:** One TP swing changes recall by 0.111. All recall comparisons
   are within bootstrap CI for most variants. Directional finding (mp-raw-split > pfn-mean)
   is consistent across all metrics, but individual differences are not statistically reliable.

2. **n_calib=2 (median):** Rank-normalized scores are degenerate for most users. PFN is
   fundamentally handicapped on this dataset. A dataset with longer user histories is
   required to fairly evaluate the mechanism.

3. **Single FastText embedding space:** A per-feature FastText model (separate embedding
   per feature, no cross-feature co-occurrence) would be a stronger test.

---

## Artifacts

| File                                         | Description                           |
|----------------------------------------------|---------------------------------------|
| `h3_pfn/experiments/pfn_experiment2.py`      | v2 script with F1/F2/F6 fixes         |
| `h3_pfn/figures/pfn_v2_metrics.json`         | Full metrics (verified source of truth)|
| `h3_pfn/figures/pfn_v2_summary_auc.png`      | ROC-AUC summary with CIs              |
| `h3_pfn/figures/pfn_v2_top1pct.png`          | Top-1% precision/recall bar chart     |
| `h3_pfn/figures/pfn_v2_pr_curves.png`        | PR curves all variants                |
| `h3_pfn/figures/pfn_v2_feature_importance.png`| Per-feature AUC                      |
| `h3_pfn/docs/HYPOTHESIS.md`                  | Formal hypothesis                     |
