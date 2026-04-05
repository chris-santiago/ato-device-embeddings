# DEFENSE.md — Defense of Original H2 Design

**Reviewer persona:** ML engineer defending the original implementation. Will concede valid critique points, rebut invalid ones, and mark genuinely open questions as empirically testable.

---

## Response to Root Cause 1: N-gram behavior

### C1.1 — Cross-boundary n-gram contamination may be minimal

**Partial concession.** The hypothesis asserts the contamination effect without measuring it. However, the PoC does not need to measure n-gram weights to test the hypothesis — the hypothesis is falsified or supported at the level of AUC and silhouette, regardless of the mechanism. That said, the ml-critic is right that the mechanism claim (as opposed to the performance claim) requires n-gram frequency analysis. 

**However:** The prior investigation (H2_RERUN) ran a timezone-permutation test where every reordering of the tz feature position in the concat string made spoof AUC worse. This is direct evidence that boundary effects are not negligible. Boundary-spanning n-grams at the tz boundary specifically appear to suppress the tz signal in the concat embedding. This empirical result from the prior investigation supports the mechanism claim even without weight inspection.

**Status:** Partially rebutted. The mechanism claim is plausible and has supporting evidence from the prior investigation. However, C1.1 remains an open empirical question for this investigation: we should inspect n-gram frequency distribution.

### C1.2 — Mean-pooling creates its own cross-feature correlations

**Concession.** This is a valid critique. Mean-pooling over co-occurring tokens will encode feature co-occurrence in the individual token embeddings (via the skip-gram objective). However, the key distinction is: in mean-pool, the co-occurrence affects the embedding of each feature token, not the final device vector. Two devices that share OS but differ on all other features will have similar OS-token embeddings — but the device vector (mean of all 6 tokens) will differ because the other 5 tokens differ. In concat, a device string with 5 matching features and 1 different one may have very high cosine similarity to the original because the n-grams from the 5 matching features dominate the embedding. The co-occurrence structure in mean-pool is additive and decomposes cleanly; in concat it is multiplicative and entangled.

**Status:** Valid nuance, but does not invalidate the hypothesis. The mechanism may be weaker than claimed, but the direction should still hold. Empirically testable via the feature-ablation experiment (C6.1).

### C1.3 — Window=6 in mean-pool reintroduces inter-feature context

**Concession.** This is correct. With `window=6` and 6-token sequences, every feature token sees every other feature token as context during training. This means the skip-gram objective will push co-occurring feature pairs together. The ml-critic is right that `window=1` would eliminate this effect. **This is a real design flaw that should be corrected.**

However, even with window=6, the mean-pool model must still compute separate embeddings for each feature token and average them. The concat model embeds the entire device string as one token. These are structurally different operations, and window=6 in mean-pool does not collapse them to be equivalent.

**Status: Conceded as valid experimental gap.** A window sweep (window=1, 3, 6) should be run in the experiment. This is empirically testable.

---

## Response to Root Cause 2: Evaluation design

### C2.1 — Spoof AUC near 0.5 is expected for embedding models

**Partial concession.** The ml-critic is correct that centroid-based scoring will naturally score spoof events as legitimate when 5/6 features match — the cosine distance will be small. This is a fundamental challenge for the evaluation design.

**Rebuttal:** But this is precisely the point of the H2 hypothesis. If mean-pooling makes the tz dimension more separable (higher magnitude in the final vector, or less contaminated by the 5 matching dimensions), then the spoof device's mean-pool vector should differ more from the centroid than the spoof device's concat vector. Even a small improvement in spoof AUC above 0.5 is meaningful. The hypothesis does not claim embedding models will "solve" spoof detection — it claims mean-pool will do relatively better than concat on this hard case.

The PoC's result (mean-pool 0.41 vs. concat 0.49) shows the *opposite* direction, which is genuinely surprising. This motivates investigation, not dismissal.

**Status:** The critique motivates richer evaluation (feature-dimension salience analysis), but does not invalidate the evaluation design. Empirically testable.

### C2.2 — Trivial baseline at 0.75 is a structural artifact, not a meaningful comparison

**Concession.** The ml-critic is correct. The trivial baseline's 0.75 spoof AUC is a mechanical consequence of the negative class split: known_device negatives score 0 (correctly legitimate), enrollment negatives score 1 (incorrectly flagged), and all spoof attacks score 1 (correctly flagged). The AUC of 0.75 follows from perfect attack recall and 50% false positive rate on negatives — it is not a meaningful comparison point for embedding models.

**Implication:** The hypothesis's expected observable ("trivial baseline will be beaten on spoof attacks by both embedding strategies") is wrong as stated. Beating 0.75 would require the embedding model to simultaneously (a) score spoof attacks higher than enrollment negatives (hard, because spoofs share 5/6 features with primary, enrollment negatives share 4/6), and (b) correctly accept enrollment negatives. This is a tighter requirement than the hypothesis acknowledged.

**Status: Conceded. The expected observable for spoof vs. trivial baseline should be revised.** The correct framing is: embedding models may struggle to beat the trivial baseline on spoof because of the fundamental proximity of spoof embeddings to the account centroid.

### C2.3 — Fleet AUC contaminated by training injection

**Concession with qualification.** The ml-critic is correct that fleet AUC pools contaminated accounts (where the fleet device appears in training) with uncontaminated accounts. For contaminated accounts, the fleet device will score as low-distance (close to centroid) — correctly, because it IS in the account's history. This is not a flaw in the model; it is the intended behavior (the evaluation design is membership-based per the project memory). The question is whether the AUC for fleet attacks on *uncontaminated* accounts is meaningful.

**Rebuttal:** From the evaluation perspective, the fleet attacker appearing in training for some accounts is a feature, not a bug. In production, an attacker device that has been seen before (for this account) should score as low-risk — that is correct behavior. The fleet AUC tests whether the model can flag the fleet device in accounts where it has NOT appeared in training while accepting it in accounts where it HAS appeared. Pooling them together computes the overall fleet AUC, which is meaningful.

**Status:** Partially rebutted. The overall fleet AUC is a valid metric. Stratified analysis (contaminated vs. uncontaminated) is informative but not required to test the H2 hypothesis. Include as an informative analysis.

---

## Response to Root Cause 3: Silhouette score

### C3.1 — Silhouette labels `account_id % 10` are incorrect

**Full concession.** This is a genuine bug in the PoC. The PoC comment acknowledges it as a speed trade-off, but the resulting metric is not interpretable as per-device cluster quality. The experiment must use true device labels (the actual {OS, browser, tz, lang, network, screen} tuples).

**Status: Conceded as PoC bug. The experiment must fix this.**

### C3.2 — Negative silhouette scores

**Rebuttal.** The negative silhouette scores are expected in a high-dimensional space with many similar devices (many accounts share the same OS, browser, etc.). The hypothesis predicts mean-pool silhouette > concat silhouette — both can be negative while still satisfying the prediction. The PoC's result (mean-pool -0.089 < concat -0.030) is the opposite direction, but this is under the flawed labeling scheme.

**Status:** Consequence of C3.1. Must be re-evaluated with correct labels.

### C3.3 — Silhouette sensitivity to cluster count

**Concession.** True. With 800-1600 unique device profiles across 400 accounts, silhouette computed at the device level will be very different from silhouette at 10 bins. **Use true device labels in the experiment.**

**Status:** Addressed by fixing C3.1.

---

## Response to Root Cause 4: FastText model choice

### C4.1 — FastText's subword model may be inappropriate

**Partial concession.** FastText was chosen because the prior investigation (H2_RERUN) used it. The question of whether word2vec would show the same effect is interesting but peripheral to H2. H2 asks specifically about FastText mean-pool vs. FastText concat — not about the best embedding method overall.

**Rebuttal:** The hypothesis is about a specific implementation choice (mean-pool vs. concat with FastText). Testing word2vec would answer a different question: "is the effect due to n-grams or due to joint vs. separate embedding spaces?" That is a valid research question but it is H3, not H2. The experiment should focus on FastText to remain falsifiable against the stated H2 claim.

**Status:** Valid scientific question but out of scope for H2. Flag as future work.

### C4.2 — Two mechanisms conflated

**Concession.** The ml-critic is correct that the hypothesis conflates two mechanisms: (a) cross-boundary n-gram contamination and (b) joint vs. separate embedding spaces. Both mechanisms could explain a mean-pool advantage. Testing n-gram-disabled FastText (or word2vec) would isolate them.

**Status:** Valid but addressable within the current experiment. A comparison with n-gram min/max size = 0 or a word2vec variant is empirically testable and should be added.

---

## Response to Root Cause 5: Corpus construction

### C5.1 — Context richness mismatch

**Full concession.** This is the most serious critique. Concat trains on 1-token sentences (no context), mean-pool trains on 6-token sentences (full context). FastText's skip-gram objective trains better with more context. Any performance advantage of mean-pool could be entirely due to context richness, not n-gram contamination elimination.

**Mitigation already in design:** FastText uses its *own character n-grams* for the token embedding, regardless of context. In concat training, the concat string `ios_safari_utc-5_en_us_wifi_small` is one token, and its embedding is learned from all the character n-grams within it — including boundary-spanning ones. In mean-pool training, the token `os_ios` is one token, and its embedding is learned from character n-grams within `os_ios` only. The difference in context training is a real confound, but it cannot explain the n-gram contamination effect, which is intrinsic to the token representation.

**Status:** Partially rebutted, but the confound is real. The experiment should control for this. One approach: evaluate the models with `min_count=1, workers=1, epochs=10` and verify that mean-pool token embeddings are not simply better because they were trained on more sentences with context. Empirically testable via the window sweep (C1.3).

### C5.2 — Vocabulary mismatch

**Partial concession.** The concat vocabulary can have up to ~14,400 tokens, but in practice the training data generates far fewer distinct device strings (400 accounts × 2-4 devices = 800-1600 unique strings). The mean-pool vocabulary has exactly 30 tokens. The relative frequency of each mean-pool token will be much higher than any concat token — this means mean-pool token embeddings will be trained on more examples and will be more stable. This is a structural advantage for mean-pool that is independent of the n-gram contamination mechanism.

**Status:** Valid confound. Acknowledge in the experiment write-up.

---

## Response to Root Cause 6: Evaluation protocol

### C6.1 — No ablation over number of matching features

**Concession.** An ablation with 1, 2, 3, 4, 5 matching features would directly test the mechanism. This is a high-priority empirical test.

**Status: Conceded. Add to experiment design.**

### C6.2 — No per-feature attribution

**Partial concession.** Per-feature attribution (which feature token's embedding contributes most to cosine distance for spoof attacks) is a valuable diagnostic. However, in mean-pool, the device vector is a sum of individual feature token embeddings, and the cosine distance to the centroid depends on which feature token differs. For spoof attacks (tz differs), the tz token embedding will pull the device vector away from the centroid proportionally to its magnitude. This can be estimated by computing the counterfactual: replace the spoof tz token with the primary tz token and measure the change in cosine distance. Not implemented in PoC.

**Status:** Valid diagnostic. Add to experiment design as an informative analysis (not a primary test).

---

## Summary: Concessions, Rebuttals, and Open Empirical Questions

| ID   | Verdict | Action |
|------|---------|--------|
| C1.1 | Partially rebutted (prior evidence exists) | Inspect n-gram frequencies in experiment |
| C1.2 | Valid nuance, does not invalidate hypothesis | Note in discussion |
| C1.3 | **Conceded** — window=6 is a real design gap | Run window sweep (w=1, 2, 3, 6) in experiment |
| C2.1 | Partially rebutted | Add feature-by-feature ablation |
| C2.2 | **Conceded** — expected observable wrong | Revise: don't expect to beat trivial on spoof |
| C2.3 | Partially rebutted (membership-based eval is correct) | Stratify fleet analysis as informative |
| C3.1 | **Conceded** — PoC bug | Fix: use true device labels in experiment |
| C3.2 | Consequence of C3.1 | Fix via C3.1 |
| C3.3 | **Conceded** | Fix via C3.1 |
| C4.1 | Rebutted (out of scope for H2) | Flag as future work |
| C4.2 | **Conceded** — mechanisms conflated | Add n-gram ablation (word2vec or n-gram disabled) |
| C5.1 | Partially rebutted but confound is real | Window sweep partially controls for this |
| C5.2 | **Conceded** — vocabulary imbalance is a structural advantage | Acknowledge in discussion |
| C6.1 | **Conceded** | Add matching-feature ablation to experiment |
| C6.2 | Partially conceded | Add tz-counterfactual analysis |
