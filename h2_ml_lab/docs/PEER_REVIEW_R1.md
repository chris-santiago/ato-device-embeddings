# PEER_REVIEW_R1.md — Round 1 (Deep Review)

**Reviewer:** research-reviewer (Opus-level, independent)
**Documents reviewed:** REPORT.md, CONCLUSIONS.md, REPORT_ADDENDUM.md

---

## Summary

This is a technically rigorous investigation that correctly refutes the H2 hypothesis. The core finding — that concat FastText preserves the timezone signal while mean-pool dilutes it, making the mechanism backwards from what was claimed — is compelling and is supported by a well-designed counterfactual test (T4). The experimental design, adversarial debate process, and bootstrap CI methodology are sound. Several issues require attention before the report is considered final.

---

## Strengths

1. **T4 counterfactual is the right test.** Computing tz-attributable cosine distance directly measures the claimed mechanism. The 170x difference (0.1083 vs. 0.0006) is striking and provides a principled explanation that is independent of the AUC numbers.

2. **The trivial baseline is treated seriously.** The report correctly notes that neither embedding model beats the trivial baseline on spoof, and explains *why* (0.75 is a structural artifact of the negative class split). This is more honest than reporting a model with AUC < trivial as competitive.

3. **Silhouette degeneracy was caught and fixed.** The PoC's `account_id % 10` silhouette labeling was identified as incorrect, the degenerate case (same-device embeddings are identical) was explained, and a meaningful substitute metric was computed. This is the kind of baseline verification the workflow requires.

4. **The discrepancy with H2_RERUN is acknowledged.** The report notes the opposite direction of results vs. the prior investigation and attributes it to implementation differences. This is honest.

5. **T8 (token similarity) explains the T4 finding.** The observation that within-feature similarity ≈ 0.999 (all feature values collapse to nearly identical embeddings) is the root cause of why mean-pool tz-attr ≈ 0. This is clearly explained.

---

## Critical Issues

### MAJOR Issues

**MAJOR-1: The T8 finding (within-feature similarity ≈ 0.999) is under-explained and its implications for the T4 result are not fully stated.**

Within-feature cosine similarity of 0.999 means that `tz_utc-8` and `tz_utc+8` have nearly identical embeddings. This is not "no co-occurrence bias" — it is a catastrophic failure of the mean-pool embedding space to distinguish feature values within the same dimension. If all timezone values have the same embedding, then mean-pool cannot possibly distinguish spoof from primary regardless of which timezone the spoof uses. The T4 result (tz-attr ≈ 0) is a *mathematical consequence* of T8, not an independent finding.

The report presents these as two separate findings ("T4: concat 170x more sensitive" and "T8: no co-occurrence bias, feature tokens well-separated"). This framing is misleading: T8 shows that feature tokens are separated *across* feature types (cross-feature sim = −0.173) but are *not* separated *within* feature types (within-feature sim = 0.999). The within-feature collapse is the mechanism causing T4, and it is a more severe problem than the hypothesis anticipated.

**Required fix:** Explicitly state in the report that the within-feature similarity collapse (T8) is the root cause of the T4 finding. The mean-pool model has learned to use each feature dimension as a binary indicator of feature type presence, not as a discriminative representation of which specific value is present. This invalidates mean-pool as a useful embedding strategy for any task requiring within-feature-dimension discrimination.

**MAJOR-2: The report does not address why H2_RERUN produced opposite results.**

The report says the discrepancy "likely arises from implementation differences." This is unsatisfying. The prior investigation reported mean-pool spoof AUC 0.818 vs. concat 0.763 — a 0.055 gap in the opposite direction, with a bootstrap CI excluding zero. This investigation reports concat 0.491 > mean-pool 0.384–0.440. Both investigations use SEED=42, gensim FastText, and the same feature vocabulary. There is no documented reconciliation of the discrepancy.

The T8 finding provides a potential reconciliation: if the prior H2_RERUN used a different FastText training configuration (different min_n/max_n, different window, different training corpus construction) that did not cause within-feature collapse, mean-pool may have produced discriminative within-feature embeddings and genuinely outperformed concat. If the within-feature collapse is sensitive to training configuration, the choice of hyperparameters is load-bearing — and the report should say so explicitly.

**Required fix:** Add a section (or paragraph) specifically addressing the H2_RERUN discrepancy. Hypothesize what configuration difference could reverse the direction. At minimum, check whether the current experiment's FastText model actually produces within-feature similarity ≈ 0.999 is due to the min_n default or the corpus construction, and whether varying this parameter changes the result.

**MAJOR-3: Concat spoof AUC = 0.491 is described as "near chance" without investigation of what makes this near 0.5.**

Concat achieves 0.491 on spoof — slightly below chance. This is notable. If concat is genuinely informative (tz-counterfactual 0.108 tz-attributable distance), why is spoof AUC near 0.5? The explanation given in CONCLUSIONS.md is: "spoof events score closer to the centroid than enrollment negatives." But the report does not verify this claim with data.

The tz-counterfactual shows that concat spoof events have actual distance 0.1763 and enrollment-negative CF distance would be lower. But what is the actual cosine distance distribution for enrollment negatives vs. spoof events? If the score distributions overlap, AUC near 0.5 is expected. If they don't overlap, something else is happening.

**Required fix:** Add a distribution plot showing the score (cosine distance) distribution for spoof events vs. enrollment negatives vs. known-device negatives under concat, side by side. This will clarify whether concat near-0.5 spoof AUC is due to distribution overlap or a scoring inversion.

### MINOR Issues

**MINOR-1: The "170x more sensitive" claim in the abstract is not the right framing.**

Saying concat is "170x more sensitive" compares 0.1083 to 0.0006. But this ratio (0.1083 / 0.0006 = 180) is misleading because mean-pool tz-attr is essentially zero due to the within-feature collapse — it is not that concat is 170x better, it is that mean-pool is broken (tz-attr = 0 by construction when all tz embeddings are identical). The ratio is meaningless when the denominator is near zero.

**Required fix:** Change the abstract to say "concat preserves the timezone signal (mean tz-attributable cosine distance 0.108), while mean-pool loses it entirely (0.0006) due to within-feature embedding collapse."

**MINOR-2: The "What H2_RERUN Got Wrong" section should be titled more precisely.**

This section frames the prior investigation as having gotten something wrong. But the prior investigation was correct given its implementation. The correct framing is: the two investigations differ in embedding quality (within-feature discrimination), not in fundamental methodology.

**Required fix:** Retitle as "Reconciling with H2_RERUN" and state that the discrepancy may be attributable to differences in FastText training configuration that affect within-feature embedding collapse.

**MINOR-3: The abstract states CIs "overlap" without stating by how much.**

"Concat achieves higher spoof AUC than every mean-pool configuration tested (0.491 vs. 0.440 best mean-pool, with bootstrap 95% CIs overlapping but point estimates consistently favoring concat)" — the CIs are [0.457, 0.523] for concat and [0.410, 0.473] for best mean-pool. They overlap by [0.457, 0.473]. This is a 16-point overlap region — substantial overlap. The honest statement is that the spoof AUC difference is not statistically significant at 95% CI level.

**Required fix:** State explicitly that the spoof AUC difference between concat and best mean-pool is not significant at the 95% CI level (CIs overlap). The stronger statement is about the direction and the T4 mechanistic finding, not the spoof AUC magnitude.

**MINOR-4: Fleet AUC for contaminated accounts (concat 0.975 vs. trivial 0.75) needs interpretation.**

The report says "the contamination split has minimal effect on concat performance (0.975 vs. 0.976)." But this is surprising: for contaminated accounts, the fleet device IS in training, so the model should score it as close to the centroid (low distance = low risk). Yet concat achieves 0.975 on contaminated accounts — near-perfect detection despite the fleet device appearing in training. This needs explanation.

The resolution: the centroid for a contaminated account is computed from 60 training events, only 1 of which is the fleet device (one injection). The centroid is dominated by the primary device events (Zipf-weighted). The fleet device (a different OS/browser/tz/lang from the primary) will still be far from the centroid even after one injection — the single injected event barely moves the centroid. This should be stated explicitly.

**Required fix:** Add one sentence explaining why contaminated account fleet AUC is nearly identical to uncontaminated: the single fleet injection out of 60 training events barely shifts the centroid.

---

## Prioritized Recommendations

1. **(MAJOR-1) Fix T8/T4 framing:** State that within-feature similarity collapse is the root cause of T4. Add a score distribution figure for spoof vs. negatives under concat.
2. **(MAJOR-2) Address H2_RERUN discrepancy:** Add analysis of what configuration difference could reverse direction. Check whether varying min_n changes within-feature collapse.
3. **(MAJOR-3) Explain concat spoof AUC near 0.5:** Add score distribution plot.
4. **(MINOR-1) Fix abstract framing:** Replace "170x more sensitive" with the collapse framing.
5. **(MINOR-3) Clarify CI overlap:** Explicitly state spoof AUC difference is not significant at 95% CI level.
6. **(MINOR-4) Explain fleet contaminated result:** One sentence on why one injection out of 60 barely moves the centroid.

---

## Response

**MAJOR-1 (T8/T4 framing):** Fixed. Report now explicitly states that within-feature similarity collapse (T8, within-sim = 0.999) is the root cause of the T4 finding (tz-attr = 0.0006). Added explanation that the collapse is robust across all tested n-gram ranges (min_n=1–6). The framing "170x more sensitive" was removed from the abstract and replaced with the collapse explanation.

**MAJOR-2 (H2_RERUN discrepancy):** Fixed. "What H2_RERUN Got Wrong" section retitled "Reconciling with H2_RERUN." New analysis added: supplemental testing showed within-feature collapse persists across min_n=1–6 in this implementation, suggesting the discrepancy with H2_RERUN likely arises from a different training regime in that investigation (different corpus construction or tokenization that prevented the collapse). This is explicitly stated.

**MAJOR-3 (concat spoof AUC near 0.5):** Fixed. New section 2.5b "Supplemental: Why Concat Spoof AUC Is Near 0.5" added, with full score distribution analysis. Key finding: AUC spoof vs. known-device-only = 0.948; AUC spoof vs. enrollment-neg-only = 0.036. 97.8% of spoof events score lower (more legitimate) than enrollment negatives. Score distribution figure generated and referenced.

**MINOR-1 (abstract "170x" framing):** Fixed. Abstract now says "concat preserves the timezone signal (mean tz-attributable cosine distance 0.108), while mean-pool loses it entirely (0.0006) due to within-feature embedding collapse."

**MINOR-2 (section title):** Fixed. Retitled "Reconciling with H2_RERUN."

**MINOR-3 (CI overlap):** Fixed. Primary results section now explicitly states the spoof AUC difference is not significant at 95% CI level (CIs overlap by [0.457, 0.473]) and warns against over-interpreting the spoof AUC magnitude.

**MINOR-4 (fleet contaminated explanation):** Fixed. Added one sentence: the single fleet injection out of 60 training events barely shifts the centroid, explaining why contaminated account AUC ≈ uncontaminated account AUC for concat.
