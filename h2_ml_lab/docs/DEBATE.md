# DEBATE.md — Multi-Turn Argument to Resolution

## Contested Points (Initial)

After comparing `CRITIQUE.md` and `DEFENSE.md`, the following points remain contested (defense did not fully concede, or critique did not fully concede):

| ID   | Topic | Status after initial exchange |
|------|-------|-------------------------------|
| C1.1 | Boundary n-gram contamination magnitude | Partially rebutted — still contested |
| C1.2 | Mean-pool introduces own cross-feature correlations | Valid nuance noted but not resolved |
| C1.3 | Window=6 in mean-pool reintroduces context | Conceded by defense |
| C2.1 | Spoof near-0.5 is expected; mechanism not shown | Contested |
| C2.2 | Trivial baseline comparison misleading on spoof | Conceded by defense |
| C2.3 | Fleet AUC contamination | Partially rebutted |
| C3.1 | Silhouette labels are wrong in PoC | Conceded by defense |
| C4.2 | Two mechanisms conflated | Conceded by defense |
| C5.1 | Context richness mismatch | Partially rebutted |
| C6.1 | No matching-feature ablation | Conceded by defense |

Fully resolved before debate (both sides agree): C3.2, C3.3 (consequence of C3.1), C4.1 (out of scope).

---

## Debate — Round 1

### ml-critic Round 1

**C1.1 — Sharpening: boundary n-gram frequencies can be estimated analytically**

The defense cites the H2_RERUN timezone-permutation test as evidence that boundary effects are non-negligible. However, the permutation test showed that *any* reordering of tz within the concat string made spoof AUC worse — not that *boundary n-grams* are the cause. A simpler explanation: reordering the tz position changes which other features are adjacent to tz, altering the joint context signal the model receives. This is evidence that context matters, not that boundary n-grams specifically matter. I am not conceding C1.1.

**Proposed empirical test for C1.1:** Train concat model with FastText `min_n=0, max_n=0` (disabling all character n-grams, using only full-token embeddings). Compare AUC to standard concat. If AUC is unchanged, n-grams are irrelevant. If AUC changes, n-grams matter. However, note that gensim FastText does not support `min_n=0` directly — the test would require a word2vec model trained on the concat tokens.

**C2.1 — Sharpening: the defense's rebuttal is circular**

The defense says "if mean-pooling makes the tz dimension more separable, the spoof cosine distance should be larger." But this is exactly what has not been shown. The mechanism is the hypothesis, not a rebuttal. The PoC shows the *opposite* direction (mean-pool 0.41 < concat 0.49 on spoof AUC), which is inconsistent with the hypothesis. The defense says "the PoC result is genuinely surprising" — but surprising results require explanation, not confidence in the prior hypothesis.

**Proposed empirical test for C2.1:** Compute per-account, per-feature cosine distance decomposition for spoof attacks. For a spoof event with tz=X and primary tz=Y: compute the mean-pool device vector using tz=X and using tz=Y (counterfactual). The difference in cosine distance to the centroid is the tz-attributable component. If mean-pool has higher tz-attributable distance than concat, the mechanism is supported.

**C5.1 — Not withdrawing:** Context richness is a real confound and the defense's partial rebuttal does not address it. The defense argues that "the n-gram contamination effect is intrinsic to the token representation, independent of context." This is correct but incomplete: even if n-gram contamination is real in concat, the performance advantage observed (if any) could be *amplified* by context richness in mean-pool. The two effects are not separable without controlling for context.

**Proposed empirical test for C5.1:** Train mean-pool FastText with `window=1` (each feature token sees no other feature tokens as context). If mean-pool window=1 still outperforms concat, the advantage cannot be attributed to context richness.

---

### ml-defender Round 1

**C1.1 — Maintaining partial rebuttal, but agreeing to empirical test**

The ml-critic's alternative explanation (permutation changes context, not boundary n-grams) is plausible. However, the defense argues: in the concat model, each device string is a *single token*. The word2vec skip-gram objective trains this single token against its *sentence-level context* — but concat sentences are single-token sentences (no sentence-level context). The character n-gram model is what gives the concat embeddings their structure, not sentence-level context. Therefore, the permutation test is evidence about n-gram behavior, not context.

The proposed word2vec ablation is accepted as a valid empirical test.

**Resolution for C1.1:** Empirical test agreed — train a word2vec model on concat tokens (no character n-grams) and compare AUC. If word2vec-concat matches FastText-concat performance, n-grams are not the mechanism.

**C2.1 — Accepting the counterfactual test**

The defense accepts the tz-counterfactual analysis as a valid test of the mechanism. If mean-pool produces larger tz-attributable cosine distance for spoof events, the mechanism is confirmed. This test should be run in the experiment.

**Resolution for C2.1:** Empirical test agreed — run tz-counterfactual analysis comparing mean-pool and concat on spoof events.

**C5.1 — Conceding the window confound, accepting the test**

The defense concedes that context richness is a real confound. The proposed `window=1` test isolates the context effect. The defense agrees to run this test.

**Resolution for C5.1:** Empirical test agreed — compare mean-pool with window=1 vs. mean-pool with window=6. If window=1 still outperforms concat, context richness is not driving the advantage.

---

## Debate — Round 2

### ml-critic Round 2

**C1.2 — Sharpening: feature co-occurrence in mean-pool is under-addressed**

The defense argued that co-occurrence in mean-pool "affects individual token embeddings, not the final device vector." But this is exactly backward: when `os_ios` and `browser_safari` frequently co-occur in training sequences, their embeddings will be pushed together in the embedding space. The mean-pool device vector for a device with `os_ios, browser_safari, tz_utc-5` will be similar to one with `os_ios, browser_safari, tz_utc+1` not because the devices are similar, but because the `os_ios` and `browser_safari` components dominate by sheer training frequency. This is co-occurrence bias in the token embeddings themselves, and it propagates directly into the mean-pool device vector.

**Proposed empirical test for C1.2:** Compute the cosine similarity matrix between all 30 feature token embeddings in the trained mean-pool model. If `os_ios` and `browser_safari` have high cosine similarity (despite being different feature types), co-occurrence bias is present. Compare this to the expected structure (within-feature similarity > cross-feature similarity).

**C2.3 — Sharpening: fleet contamination changes the hypothesis**

The defense argued that fleet devices appearing in training is "intentional" (membership-based eval). But the issue is not whether it's intentional — it's whether pooling contaminated and uncontaminated accounts in fleet AUC conflates two very different scenarios: "can you detect a never-seen attacker device?" (uncontaminated accounts) and "should you accept a previously-seen device?" (contaminated accounts). Pooling them means fleet AUC reflects a mix of both, and neither pure-pass nor pure-fail on either sub-population is clearly interpretable.

**Proposed empirical test for C2.3:** Report fleet AUC separately for contaminated accounts (fleet device in training) vs. uncontaminated accounts (fleet device never seen).

---

### ml-defender Round 2

**C1.2 — Accepting the empirical test, sharpening the rebuttal**

The ml-critic's point about token embedding co-occurrence bias is valid and the defense accepts the proposed cosine similarity matrix test. However, note that the *same* co-occurrence bias affects concat: if `ios_safari_utc-5_en_us_wifi_small` and `ios_safari_utc+1_en_us_wifi_small` both appear frequently in training, their concat embeddings will be similar — via shared character n-grams (not via skip-gram context). So both models have co-occurrence bias; the question is whether mean-pool's version (via token embedding proximity) is more or less severe than concat's version (via shared n-grams).

**Resolution for C1.2:** Empirical test agreed — measure cross-feature token cosine similarity in mean-pool model AND compare spoof events' cosine distance distribution between mean-pool and concat models.

**C2.3 — Conceding the stratified analysis**

The defense concedes that reporting fleet AUC separately for contaminated vs. uncontaminated accounts is informative. In the uncontaminated case (fleet device never seen), the model should flag it — and AUC there tests whether it does. In the contaminated case (fleet device in training), high AUC would indicate incorrect behavior. Stratifying clarifies the interpretation.

**Resolution for C2.3:** Empirical test agreed — stratify fleet AUC by whether fleet device appears in training.

---

## Debate — Round 3: Remaining Contested Points

After two rounds, the following points need final resolution:

**C6.1 (matching-feature ablation)** — Both sides conceded in initial exchange. Empirical test agreed.

**Summary of all resolutions:**

---

## Final Resolution Table

| ID   | Topic | Resolution | Test condition (if empirical) |
|------|-------|------------|-------------------------------|
| C1.1 | Boundary n-gram contamination magnitude | **Empirical test agreed** | Train word2vec on concat tokens (no char n-grams). If word2vec-concat ≈ FastText-concat AUC → n-grams don't matter. If word2vec-concat << FastText-concat AUC → n-grams matter. |
| C1.2 | Mean-pool co-occurrence bias | **Empirical test agreed** | Compute cosine similarity matrix of 30 feature tokens in mean-pool. Compare cross-feature vs. within-feature similarity. |
| C1.3 | Window=6 reintroduces context | **Empirical test agreed** | Run mean-pool with window=1, 2, 3, 6. Report AUC vs. concat at each window. If window=1 still outperforms → context richness doesn't explain the gap. |
| C2.1 | Mechanism not shown for spoof | **Empirical test agreed** | Tz-counterfactual: compute cosine distance change when tz token is swapped for primary tz. Compare mean-pool vs. concat attribution. |
| C2.2 | Trivial baseline misleading on spoof | **Defense conceded** — expected observable should not claim to beat trivial on spoof | No test needed — correct the hypothesis prediction. |
| C2.3 | Fleet AUC contaminated | **Empirical test agreed** | Stratify fleet AUC: contaminated (fleet in training) vs. uncontaminated (fleet never seen). |
| C3.1 | Silhouette labels wrong | **Defense conceded** — fix in experiment | Use true device labels (6-tuple) in silhouette computation. |
| C4.2 | Two mechanisms conflated | **Empirical test agreed** — same as C1.1 (word2vec ablation) | Covered by C1.1 test. |
| C5.1 | Context richness confound | **Empirical test agreed** — covered by C1.3 (window sweep) | Covered by C1.3 test. |
| C6.1 | No matching-feature ablation | **Empirical test agreed** | Generate test events with 0, 1, 2, 3, 4, 5 feature matches to primary. Plot AUC vs. number of matching features for mean-pool and concat. |

---

## Empirical Test List (for Step 6)

### Test T1 — Window sweep (resolves C1.3, C5.1)
Train mean-pool FastText with window ∈ {1, 2, 3, 6}. Compare spoof AUC vs. concat at each window.
- **Critique right if:** Window=1 still underperforms concat on spoof → mean-pool cannot find the tz signal regardless of context.
- **Defense right if:** Window=1 outperforms concat on spoof while window=6 does not → context richness was suppressing the signal, not enabling it.
- **Ambiguous if:** All mean-pool windows underperform concat with no clear trend.

### Test T2 — N-gram ablation via word2vec (resolves C1.1, C4.2)
Train gensim Word2Vec (no subword n-grams) on the concat token corpus. Compare AUC to FastText-concat.
- **Critique right if:** Word2Vec-concat ≈ FastText-concat → n-gram contamination is not the mechanism.
- **Defense right if:** Word2Vec-concat << FastText-concat on novel/fleet but ≈ on spoof → n-grams help on easy cases but hurt on spoof (contamination).
- **Ambiguous if:** Word2Vec-concat < FastText-concat on all attack types → n-grams uniformly help, and concat simply underperforms mean-pool for other reasons.

### Test T3 — Matching-feature ablation (resolves C6.1)
Generate test events with k ∈ {0, 1, 2, 3, 4, 5} features matching the primary device (remaining features randomized). Plot AUC vs. k for mean-pool and concat.
- **Critique right if:** AUC gap (mean-pool − concat) is flat or random across k → mechanism does not scale with feature overlap.
- **Defense right if:** AUC gap increases with k → mean-pool advantage grows as attacker matches more features (mechanism confirmed).
- **Ambiguous if:** Gap peaks at some k < 5 and then reverses.

### Test T4 — Tz-counterfactual attribution (resolves C2.1)
For spoof test events: compute cosine distance to centroid, then replace tz token (mean-pool) / rebuild concat string with primary tz (concat) and recompute. Report the tz-attributable distance component.
- **Critique right if:** Tz-attributable distance in mean-pool ≤ concat → mean-pool is not more sensitive to tz differences.
- **Defense right if:** Tz-attributable distance in mean-pool > concat → mean-pool isolates the tz signal better.
- **Ambiguous if:** Values are similar and both near zero.

### Test T5 — Fleet AUC stratified (resolves C2.3)
Report fleet AUC separately for contaminated accounts (fleet device in training ≥ 1 event) vs. uncontaminated.
- **Critique right if:** Uncontaminated fleet AUC is high for both models, contaminated is near 0 for both → the pooled fleet AUC is misleading.
- **Defense right if:** Both sub-populations show mean-pool ≥ concat → result is robust to contamination split.
- **Ambiguous if:** Mean-pool and concat flip relative ordering depending on contamination status.

### Test T6 — Corrected silhouette (resolves C3.1)
Compute silhouette score using true device labels (6-tuple) on a random sample of training events.
- **Critique right if:** Correct silhouette still shows concat ≥ mean-pool → H2 silhouette prediction is wrong.
- **Defense right if:** Correct silhouette shows mean-pool > concat → H2 silhouette prediction holds.
- **Ambiguous if:** Both are so close to zero that the difference is within bootstrap CI.

### Test T7 — Trivial baseline (non-negotiable)
Exact set-membership check (6/6 feature match). Already in PoC.
- Present alongside all embedding results. If trivial baseline beats both embeddings on any attack type, explain why.

### Test T8 — Co-occurrence bias in mean-pool token space (resolves C1.2)
Compute pairwise cosine similarity for all 30 feature tokens in trained mean-pool model. Check whether cross-feature pairs (e.g., os_ios vs. browser_safari) have higher similarity than expected from a semantics-free baseline.
- **Critique right if:** High cross-feature similarity (e.g., mean cross-feature cosine > 0.5) → co-occurrence bias dominates mean-pool representations.
- **Defense right if:** Cross-feature similarity is low relative to within-feature → feature tokens are well-separated.
- **Ambiguous if:** Mixed results.
