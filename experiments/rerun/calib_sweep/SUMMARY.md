# Calibration-Window Ablation — Rank-Normalization PR-AUC Collapse

**Question.** Is the spoof-k1 PR-AUC collapse under per-user CDF rank-normalization a
fundamental property of rank-normalization under imbalance, or an artifact of the specific
20-event calibration window? If the rank floor is a small-sample quantization effect it
should shrink as the calibration window grows.

**Design.** Vary the calibration window ∈ {20, 50, 100, 200, 500} events per account,
holding everything else fixed (1:100 attack:benign imbalance, spoof k=1, open-vocabulary
marginals, 7 features, 5 seeds `[42, 123, 456, 789, 2024]`). The centroid window stays at
40 events; the calibration slice is `train[40:]`, so the window is grown by growing total
login history (`N_TRAIN = 40 + calib`). This also grows the FastText corpus, so **`mp_raw`
PR-AUC is reported as an embedding-quality control** — it never touches the calibration
window, so its trend isolates any embedding shift, and the `mp_raw − mp_rank_norm` gap
isolates the rank-normalization penalty. All numbers reuse the H6 rerun pipeline unchanged
(`h6_rerun.py`); the `calib=20` cell reproduces the published baseline to 6 decimals
(mp_raw PR 0.931779, mp_rank_norm PR 0.235769 at seed 42).

## Result: the collapse shrinks strongly — it does not persist

| Calib | N_TRAIN | mp_rank_norm PR-AUC | mp_raw PR-AUC (control) | PR gap | mp_rank_norm ROC-AUC | contamination | 1/calib |
|------:|--------:|--------------------:|------------------------:|-------:|---------------------:|--------------:|--------:|
|    20 |      60 | **0.224 ± 0.011**   | 0.888 ± 0.026           | 0.664  | 0.974 ± 0.002        | 0.0498 ± 0.0013 | 0.0500 |
|    50 |      90 | 0.511 ± 0.022       | 0.910 ± 0.030           | 0.399  | 0.988 ± 0.002        | 0.0218 ± 0.0069 | 0.0200 |
|   100 |     140 | 0.669 ± 0.027       | 0.861 ± 0.020           | 0.192  | 0.990 ± 0.001        | 0.0183 ± 0.0036 | 0.0100 |
|   200 |     240 | 0.778 ± 0.044       | 0.879 ± 0.041           | 0.101  | 0.993 ± 0.002        | 0.0150 ± 0.0070 | 0.0050 |
|   500 |     540 | **0.830 ± 0.028**   | 0.872 ± 0.036           | 0.042  | 0.995 ± 0.001        | 0.0089 ± 0.0026 | 0.0020 |

Mean ± std across 5 seeds. Contamination = fraction of benign events scoring above the
attacks' 10th percentile (`mp_rank_norm`).

**Rank-normalized PR-AUC recovers from 0.224 to 0.830** as the calibration window grows
20 → 500, closing 94% of the gap to raw cosine (0.664 → 0.042). The collapse is therefore
**scoped to thin calibration windows, not fundamental** to rank-normalization under
imbalance.

## The recovery is not a corpus/embedding artifact

`mp_raw` PR-AUC is **flat across the sweep** — 0.888, 0.910, 0.861, 0.879, 0.872 — with no
monotonic trend, despite the corpus growing 9× from calib=20 to calib=500. The embeddings
did not systematically improve, so the `mp_rank_norm` recovery is attributable to the
calibration window itself, not to the larger training corpus. The Option-A confound (window
size entangled with embedding quality) is empirically ruled out by the control.

## Mechanism: quantization dominates at thin windows, with a residual structural floor

Contamination at **calib=20 is 0.0498 vs. the pure-quantization prediction 1/20 = 0.0500** —
the ~5% rank floor at the deployed window is almost entirely a small-sample quantization
effect (the CDF rank `mean(baseline < raw)` is quantized in steps of `1/len(calib)`).

As the window grows, contamination keeps falling but **plateaus above `1/calib`**: at
calib=500 it is 0.0089 vs. a predicted 0.0020 (≈4× higher). This residual is genuine
structural overlap — benign events that truly rank near attacks regardless of quantization
granularity — and it is why the PR gap does not close completely (0.042 remains at
calib=500) even though ROC-AUC converges to raw (0.995).

## Verdict

**Shrinks (strongly), does not vanish.** The catastrophic collapse (PR-AUC 0.224) is a
thin-calibration-window phenomenon: at 20 events it is a near-pure `1/N` quantization
artifact, and it largely resolves by 200–500 events (PR-AUC 0.78–0.83). A small residual
rank-normalization penalty persists at every window size (gap ≥ 0.04), reflecting structural
overlap beneath the quantization floor.

The defensible claim is therefore **scoped, not universal**: per-user CDF rank-normalization
is unsafe specifically under *thin* per-user calibration windows combined with heavy class
imbalance. This remains operationally important because thin per-user history is the common
case for new accounts and low-frequency users — exactly the population where calibration
windows are smallest.

## Reproduce

```bash
uv run experiments/rerun/calib_sweep/calib_sweep.py          # full 5×5 sweep -> results/
uv run experiments/rerun/calib_sweep/aggregate_calib.py      # -> aggregate/calib_sweep_summary.{json,csv}
uv run experiments/rerun/calib_sweep/plot_calib_sweep.py     # -> aggregate/figures/calib_sweep_pr_auc.png
```
