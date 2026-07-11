## Hypothesis — Cycle 1

**Claim:** Per-feature normalized (PFN) scoring — computing per-feature cosine distances
independently, rank-normalizing each against a per-user calibration distribution, and
aggregating via uniform mean, stability-weighted mean, or max — will improve top-1%
precision and recall compared to the mean-pool rank-norm baseline on the RBA dataset.

**Mechanism:** Current mean-pool embeds all 7 features into one 64-dim vector before
computing centroid distance. A timezone-only mismatch moves 1/7 of the mean-pool vector,
diluted by 6 matching features. PFN computes 7 independent per-feature distances, so a
single-feature anomaly produces a full-rank signal in that feature's dimension rather than
a 1/7-diluted signal in the aggregate. Rank-normalization within each feature accounts for
different distance scales across embedding regions.

**Signal:** Per-feature cosine distances, rank-normalized per-user per-feature, then
aggregated (uniform mean / stability-weighted / max) across 7 features. Baselines for
comparison: trivial set-membership, mp-raw, mp-rank-norm — all computed on the same RBA
dataset with the same FastText model and temporal split.

**Expected observable:** At the top-1% threshold (flag highest 1% of anomaly scores
across all test events):
- PFN (at least one aggregation variant) achieves higher precision than mp-rank-norm
- PFN (at least one aggregation variant) achieves higher recall than mp-rank-norm
- Secondary: PFN AUC >= mp-rank-norm AUC on spoof/ATO detection

**Null hypothesis (falsification condition):** No PFN variant improves both top-1%
precision and recall simultaneously over mp-rank-norm; or all PFN variants degrade AUC
relative to mp-rank-norm.

## Evaluation Metrics

**Primary:**
- Top-1% precision: fraction of flagged events (highest 1% of scores) that are true ATOs.
  Critical for operational deployment — each flag is a manual review burden.
- Top-1% recall: fraction of all ATO test events captured in top-1% flags.
  Operationally: how many attacks are we catching?

Both metrics must improve simultaneously for the hypothesis to be confirmed — an
improvement in one at the expense of the other is a partial finding only.

**Secondary:**
- ROC-AUC (spoof/ATO detection): for comparison across scoring variants.
- PR-AUC: average precision across all thresholds, captures overall ranking quality.
- TP, FP counts at top-1%: essential for interpreting rate metrics given the
  small positive class (≈34 ATO test events at 50/50 split).

**Domain:** pfn (per-feature-normalized)

## Evaluation Design Note

The evaluation is membership-based by design: known devices appear in training (25% fleet
injection in the synthetic experiment; the RBA dataset uses real training history). This
tests proximity detection — how far a test event's device profile is from the user's
centroid. This is intentional, not data leakage.

The mp-rank-norm baseline is computed fresh in this experiment on the RBA dataset
(the published "0.716" figure is from the synthetic variable-spoof experiment, not from
the RBA dataset directly). Both baseline and PFN variants share the same FastText model,
same temporal split, and same calibration-event allocation.
