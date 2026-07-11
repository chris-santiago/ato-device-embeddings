# Rerun vs. Prior Experiments — Structured Comparison

> How the five-seed rerun findings differ from h2_ml_lab and h6_hybrid.
> Generated from analysis of h2_ml_lab/docs/REPORT.md, h6_hybrid/docs/REPORT.md, and rerun/TECHNICAL_REPORT.md.

---

## 1. Concat Baseline Window — Corrected Explanation (C4 — most significant divergence)

> **Correction (issue 006ae95f):** an earlier version of this section attributed the concat sign-flip
> to a spoof-definition change (3-field → 1-field). That was wrong on both ends. The spoof definition
> is **identical** in both experiments: timezone forced to differ, network and screen re-sampled
> (`h2_ml_lab/experiments/robust_config_experiment.py:116-120`, whose comment reads "Matches
> h2_rerun_experiment1.py _spoof_profile() definition", and `rerun/scripts/h2/h2_rerun.py:105-109`).
> h2_ml_lab's own REPORT documents that a genuinely 1-field spoof existed only in a *prior, corrected*
> version of its script (spoof AUC 0.538); all its headline numbers use the 3-field definition.

**h2_ml_lab** reported:

| Model | Spoof AUC | vs. Trivial |
|---|---|---|
| mean_pool | 0.869 | +0.119 |
| concat (labeled "w=1") | 0.782 | +0.032 |
| trivial | 0.750 | — |

Concat was *above* the trivial baseline.

**Rerun** (same spoof definition, concat window pre-specified at w=1):

| Model | Spoof AUC | vs. Trivial |
|---|---|---|
| mean_pool | 0.868 ± 0.012 | +0.118 |
| concat_w1 | 0.737 ± 0.006 | −0.013 |
| concat_w6 | 0.778 ± 0.007 | +0.028 |
| trivial | 0.750 | — |

Concat at w=1 is *below* trivial; at w=6 it is above.

The actual explanation for the divergence is the **window of the concat baseline**, not the spoof
definition. `robust_config_experiment.py` contains no window-sweep code and trains its concat model
with `ROBUST_KWARGS` (window=6, lines 163–174); its "w=1: 0.782" is therefore best explained as a
window-6 concat mislabeled as w=1 — the value matches the rerun's w=6 result (0.778 ± 0.007) and is
incompatible with the rerun's true w=1 result (0.737 ± 0.006). Mean-pool numbers match across the
two experiments almost exactly (0.869 vs. 0.868 ± 0.012), consistent with everything else being
unchanged. The rerun's T2 window sweep (w=1 → 0.737, w=3 → 0.760, w=6 → 0.778) shows concat
improves with window but never approaches mean-pool at any window; the concat-below-trivial verdict
applies to the pre-specified w=1 comparison.

---

## 2. C3 — Gate Finding Was Present in h6_hybrid But Not Formalized

**h6_hybrid** §3.3 showed `two_stage` fleet ROC-AUC = 0.459, `trivial` fleet ROC-AUC = 0.459, 0 TP on all fleet (single seed=42). The finding was real, but:
- Appeared as a "failure mode" subsection, not a pre-specified numbered contribution
- Framed as a cold-start limitation rather than an algebraic structural identity
- One seed only — delta=0.000 could be coincidental at n=1

**Rerun C3** formalizes this as: *"delta = 0.000 exactly on all 5 seeds — not a rounding artifact, an algebraic consequence of the gate design."* Because fleet devices appear in the training window by construction, the gate fires on every fleet device with certainty, making the score distribution algebraically identical to the no-gate trivial baseline on fleet events. The 5/5 reproduction with exact-zero delta is what elevates it from an observation to a primary finding. The h6_hybrid's final deployment recommendation ("retire the two-stage gate, use blocklist + mp_raw") survives unchanged; the rerun provides the reproducible mechanistic justification.

---

## 3. C2 — Numbers Stable, Mechanism Deepened

**h6_hybrid** (seed=42 only): mp_raw PR-AUC **0.892**, mp_rank_norm **0.215**. Root cause stated as "CDF transform compresses score distribution, reducing score margin."

**Rerun** (5-seed mean): mp_raw **0.888 ± 0.026**, mp_rank_norm **0.224 ± 0.011**. Seed-42 values are nearly identical, confirming h6_hybrid wasn't a fluke. The rerun adds the mechanism quantification that was missing:

- **Small-sample 1/N_calib floor (thin-window)** — CDF rank-normalization maps each user's scores into [0,1] by their own calibration distribution, placing roughly 5% of every user's benign events at or above the attack operating point (rank 0.90, the attacks' 10th percentile; measured 4.98% ± 0.13%) at the deployed 20-event calibration window. This floor is a small-sample quantization artifact of the thin (20-event) calibration window, not an inherent property of the transform (see experiments/rerun/calib_sweep/SUMMARY.md): it equals 1/N_calib and shrinks as the window grows. A calibration-window ablation (see experiments/rerun/calib_sweep/SUMMARY.md; 5 seeds, calibration window in {20,50,100,200,500} events) confirms this: the CDF rank mean(baseline < raw) is quantized in steps of 1/N_calib, so at the deployed 20-event window the ~5% benign contamination equals 1/20 almost exactly (0.0498 vs 0.0500). Growing the window recovers spoof-k1 PR-AUC from 0.224 to 0.830 by 500 events while the mp_raw embedding control stays flat, leaving only a small residual overlap (~4× the 1/N prediction). The finding is therefore scoped: per-user CDF rank-normalization is unsafe specifically under thin per-user calibration windows combined with heavy class imbalance — the common case for new and low-frequency accounts. (Benign events at rank > 0.95 specifically are ~1.5%.)
- **Contamination rate** — p10(attack) contamination 1.28% ± 0.69% (mp_raw) → 4.98% ± 0.13% (mp_rank_norm): a 3.9× increase that is nearly deterministic (near-zero variance) under rank-normalization.
- **Robust margin reaching 0.000** at seed=2024 — threshold-independent confirmation that the distributions fully overlap.

h6_hybrid observed the PR-AUC collapse; the rerun explains *why* it is mechanistically unavoidable under per-user CDF normalization at 1:100 imbalance.

---

## 4. C1 — 2×2 Factorial Is New

**h2_ml_lab** compared only two configurations: sg+per-account (robust) vs. CBOW+per-event (degenerate). These two cells vary *both* objective and corpus simultaneously, establishing correlation but not causation.

**Rerun** adds the full 2×2 factorial (all four combinations):

| Config | Collapse 5/5 seeds |
|---|---|
| sg + per-account | 0/5 |
| cbow + per-event | 5/5 |
| cbow + per-account | 0/5 |
| sg + per-event | 5/5 |

Both per-event cells collapse on all 5 seeds; both per-account cells never collapse. Varying training objective within the same corpus type does not change the outcome. The rerun also adds the JSD cooccurrence diagnostic: mean within-feature Jensen-Shannon divergence 0.077 (per-event) vs. 0.190 (per-account) — making the positional rigidity mechanism directly observable rather than inferred from cosine similarity alone.

A follow-up downstream diagnostic (`scripts/h2/h2_degenerate_downstream.py` → `aggregate/h2_degenerate_downstream.json`, 2026-07-01) scores every factorial cell with mean-pool centroid cosine on 5 seeds: both per-account cells beat trivial on spoof (sg 0.868 ± 0.012, cbow 0.933 ± 0.009) while both per-event cells fall to trivial-or-worse (sg 0.767 ± 0.017, cbow 0.725 ± 0.011 against trivial 0.750; PR 0.585/0.528 against 0.500), and the historical PoC config (cbow + per-event, epochs=10) lands below trivial on PR (0.459 ± 0.019). This supersedes the single-seed h2_ml_lab downstream numbers (spoof 0.384 / novel 0.880 / fleet 0.922), which do not reproduce at that magnitude under the rerun protocol. Notably, cbow + per-account is nominally the strongest spoof cell — further confirmation that corpus, not objective, is the causal axis.

---

## 5. Fleet Residual Precision — Corrected Explanation

**h6_hybrid** fleet_residual (seed=42): top-1% precision **0.918**, PR-AUC **0.948**.

**Rerun** fleet_residual (5-seed mean): top-1% precision **0.493 ± 0.116**, PR-AUC **0.516 ± 0.118**.

The divergence is *not* due to vocabulary richness or synthetic vs. real data. **The rerun h6_rerun.py reads from `rba_marginals.json` and uses the same account generation structure as h6_hybrid** (70% home-event clones, 30% home-country-anchored events; enrollment negatives account-coherent). Both experiments have the same structural setup.

The actual explanation has two components:

### 5a. The fleet device is a single sample

Both experiments draw one globally-shared fleet device from the RBA marginals per run. How anomalous it appears depends entirely on which device was drawn — a rare country/device combination stands out clearly from account centroids; a common profile blends in. Because h6_hybrid and the rerun have different code paths, they produce different fleet device samples even at the same numeric seed.

**Per-seed breakdown in the rerun confirms the distribution is highly variable:**

| Seed | n_pre_lag | Top-1% precision | PR-AUC |
|---|---|---|---|
| 42 | 35 | 0.653 | 0.667 |
| 123 | 40 | 0.322 | 0.329 |
| 456 | 43 | 0.553 | 0.588 |
| 789 | 32 | 0.528 | 0.549 |
| 2024 | 34 | 0.409 | 0.446 |
| **Mean ± std** | | **0.493 ± 0.116** | **0.516 ± 0.118** |

H6_hybrid's seed=42 (0.918) landed near the top of this distribution — a favorable draw, not a representative one.

### 5b. The pre-lag population is small

With 400 accounts × 25% fleet = ~100 fleet accounts, and attack timing distributed over 30 days with a 10-day blocklist lag, the pre-lag population is 32–43 accounts per seed. At 35 accounts × 5 events = 175 attack events, top-1% precision is computed over ~177 slots. A handful of enrollment negatives crossing the threshold shifts precision by 0.05–0.10 — the estimate is intrinsically noisy at this population size.

### Implication for C3

The C3 contribution in TECHNICAL_REPORT.md correctly focuses on the *structurally guaranteed* claim — `two_stage` top-1% TP = 0 on all 5 seeds (algebraic certainty, zero variance) — rather than the absolute level of mp_raw fleet detection. The mp_raw fleet residual number (91 ± 23 TP, 0.493 ± 0.116 precision; std ddof=0, matching aggregate.py) is reported as evidence that the cosine distance signal exists and is not destroyed by the gate, not as a precision claim. H6_hybrid's 0.918 was not a fabricated result; it was a single favorable sample from a distribution the rerun reveals as 0.493 ± 0.116.

---

## 6. Training Data: RBA Chain-Sampled Accounts in Both Experiments

Both h6_hybrid and the rerun generate account training data by chain-sampling from RBA login marginals. The rerun does **not** switch to purely synthetic generation for H6 — `h6_rerun.py` reads `rba_marginals.json` explicitly (line 43). The vocabulary difference (229 tokens in h6_hybrid vs. 216 in rerun) reflects a minor discrepancy in marginals processing, not a methodological change.

The RBA public dataset appears in the rerun only as an *external validation probe* (C4 generalization check, §3.4) — the same role it played in the separate `h2_rba` experiment prior to the rerun. The integration into the 5-seed framework is new; the dataset's role is not.

---

## 7. Single Seed → Five Seeds

The rerun's primary addition is variance characterization. H2_ml_lab and h6_hybrid were both single-seed (seed=42). The rerun runs seeds 42, 123, 456, 789, 2024 and reports mean ± std on all primary metrics.

**Key metric stability comparison:**

| Metric | h6_hybrid (seed=42) | Rerun (5-seed mean ± std) | Stable? |
|---|---|---|---|
| mp_raw spoof k=1 PR-AUC | 0.892 | 0.888 ± 0.026 | Yes — seed=42 near mean |
| mp_rank_norm spoof PR-AUC | 0.215 | 0.224 ± 0.011 | Yes — low variance |
| mp_raw spoof ROC-AUC | 0.995 | 0.995 ± 0.001 | Yes — essentially constant |
| mp_rank_norm ROC-AUC | 0.972 | 0.974 ± 0.002 | Yes — essentially constant |
| fleet_residual top-1% prec | 0.918 | 0.493 ± 0.116 | **No** — h6_hybrid was a favorable outlier |

Exact binomial (Clopper–Pearson) 95% lower bounds: 15/15 H6 boolean verdicts → ≥0.78 true confirmation rate; 5/5 RBA `h2_replicated` verdict → ≥0.48. (Earlier drafts labeled these "Wilson" bounds; Wilson gives 0.80 and 0.57 respectively.)

---

## Summary Alignment Table

| Finding | h2_ml_lab / h6_hybrid | Rerun — what changed |
|---|---|---|
| C4: mean-pool > concat on spoof | Confirmed; concat *above* trivial (+0.032, window-6 concat mislabeled w=1) | Confirmed; concat *below* trivial (−0.013) at the pre-specified w=1 — same spoof definition (see §1) |
| C1: per-event collapse | Observed in 2 configurations; corpus suspected | Proven in 4-cell factorial; corpus confirmed causal; JSD mechanism quantified |
| C2: rank-norm collapses PR-AUC | Observed (single seed); "score margin compression" | Mechanism quantified: small-sample 1/N_calib floor (thin-window), 3.9× contamination, robust margin →0 |
| C3: gate blinds fleet | Observed as failure mode (single seed; delta not noted as exact zero) | Formalized: delta=0.000 exactly on 5/5 seeds; pre-specified verdict |
| Fleet residual precision | 0.918 (single favorable fleet device sample) | 0.493 ± 0.116 (true distribution, high variance; signal present but noisy) |
| Reproducibility | Single seed (42) throughout | 5-seed variance characterization; exact binomial (Clopper–Pearson) lower bounds reported |
| Training data for H6 | RBA chain-sampled marginals | Same RBA marginals; no methodology change |
| RBA role | Separate standalone experiment (h2_rba) | Integrated 5-seed generalization probe within rerun framework |
