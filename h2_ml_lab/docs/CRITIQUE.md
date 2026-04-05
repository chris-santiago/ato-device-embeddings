# CRITIQUE.md — Adversarial Review of H2

**Reviewer persona:** Skeptical ML engineer with applied mathematics and NLP background. Tasked with finding every implicit claim the PoC makes but has not tested. Claims are organized by root cause, not by severity.

---

## Root Cause 1: The mechanism claim rests on an untested assumption about FastText's n-gram behavior

**C1.1 — Cross-boundary n-gram contamination may be minimal in practice**

The hypothesis claims that character n-grams spanning feature boundaries (e.g., the bigram `_s` at the boundary of `utc-5_en`) inject "spurious signal." This is an assertion, not a measurement. FastText's n-gram vocabulary is shared across all training examples. Whether boundary-spanning n-grams actually acquire large weights — or whether they are effectively washed out by the much more frequent within-feature n-grams — has not been measured.

*What was not tested:* The frequency distribution of boundary-spanning n-grams vs. within-feature n-grams in the concat corpus. If boundary n-grams are rare and the model learns near-zero weights for them, the contamination effect is negligible.

**C1.2 — Mean-pooling creates its own cross-feature correlations**

When you mean-pool 6 token embeddings, the resulting vector is a sum of contributions from each feature. If two features are correlated in the training distribution (e.g., `ios` always appears with `safari`), the mean-pool vector will encode that correlation via the joint distribution of training sentences. This is the same type of spurious correlation the hypothesis ascribes only to concat. The hypothesis asserts that separating features eliminates the problem; it does not.

*What was not tested:* Whether mean-pool embeddings exhibit inter-feature correlation in the vector space (e.g., cosine similarity between `os_ios` and `browser_safari` embeddings due to co-occurrence in training sequences).

**C1.3 — The `window` parameter controls n-gram context in mean-pool too**

In the mean-pool corpus, each training sentence is a 6-token sequence. With `window=6`, every token sees every other token as context. The FastText skip-gram (CBOW) objective will push co-occurring tokens' embeddings together. This means `os_ios` and `browser_safari` will be embedded similarly if they frequently co-occur — reintroducing the same correlation mean-pooling was supposed to eliminate. The `window` parameter was set to 6 in both models without justification.

*What was not tested:* How AUC and silhouette vary with window size in the mean-pool model. A window of 1 would eliminate inter-feature context; a window of 6 creates it.

---

## Root Cause 2: The spoof attack evaluation design creates a structural floor for the trivial baseline

**C2.1 — Spoof AUC near 0.5 is the expected outcome for embedding models, not a failure**

The spoof device matches the primary device on 5/6 features (OS, browser, language, network, screen — wait, the hypothesis says the attacker matches OS/browser/language and differs on timezone; network and screen are implicitly matched). If the embedding model has learned that these 5 features are strongly predictive of device identity, the spoof device's embedding will be very close to the account centroid. The cosine distance will be small, and the model will score the spoof event as legitimate. This is not a failure of mean-pooling vs. concat — it is a fundamental property of any centroid-based method when 5/6 features match.

*The real question:* Not "which embedding strategy scores spoof better" but "does mean-pooling make the timezone feature dimension more salient relative to the other 5, allowing the centroid to detect the single differing dimension?" The PoC does not measure feature-dimension salience.

**C2.2 — The trivial baseline achieves 0.75 on spoof by construction, not because it is good**

The negative class contains two sub-types: known devices (score 0 from trivial baseline) and enrollment events (score 1 from trivial baseline, because enrollment events differ on network/screen). All spoof attacks score 1.0 from trivial baseline (different timezone = not an exact match). So the trivial baseline correctly flags all spoofs AND incorrectly flags all enrollment negatives. With equal numbers of known and enrollment negatives, the split in the negative class produces the observed 0.75 AUC mechanically — it does not mean the trivial baseline is "better."

This structural feature of the evaluation design means comparisons against the trivial baseline on spoof are misleading. The expected observable in `HYPOTHESIS.md` states that embedding strategies should "beat the trivial baseline on spoof attacks" — but beating a baseline that flags 100% of spoofs AND 50% of negatives requires an embedding model that either (a) scores spoofs very high AND known negatives low (impossible — known devices have been seen in training, so cosine distance is small) or (b) accepts enrollment negatives while still flagging spoofs (hard when enrollment negatives share 4 of 6 features with the primary).

*What was not tested:* AUC computed separately over (spoof vs. known_device_negatives) and (spoof vs. enrollment_negatives). These two sub-tasks have very different difficulty profiles.

**C2.3 — Fleet attack evaluation is contaminated by training injection**

The fleet device appears in 25% of accounts' training events. For those accounts, the centroid is pulled toward the fleet device. This means the fleet attack test event will be scored as low-cosine-distance (close to centroid) for the 25% of accounts that have the fleet device in training — making fleet AUC artificially low for those accounts. The PoC pools all accounts regardless, mixing contaminated and uncontaminated accounts in the AUC calculation.

*What was not tested:* Fleet AUC stratified by whether the fleet device appeared in the account's training set (contaminated vs. uncontaminated accounts).

---

## Root Cause 3: Silhouette score is not measuring what the hypothesis claims

**C3.1 — The silhouette labels in the PoC are `account_id % 10`, not device identity**

The PoC comment says "group by modulo to keep tractable" — but this is not a proxy for device identity. Grouping 400 accounts into 10 bins means each bin contains ~40 accounts, each with possibly different device profiles. The silhouette score measures within-bin vs. cross-bin embedding distances, which measures account-level clustering at a coarse granularity, not per-device coherence.

*What this means:* The silhouette score in the PoC is not measuring what the hypothesis predicts. A valid silhouette score would assign each training event to its actual device label (the specific {OS, browser, tz, lang, network, screen} tuple). With 400 accounts × 2-4 devices each = 800-1600 unique device profiles, the silhouette score over those labels would be a meaningful measure of embedding space organization.

**C3.2 — Negative silhouette scores do not distinguish the two models**

Both mean-pool (-0.089) and concat (-0.030) have negative silhouette scores in the PoC, meaning within-cluster distances exceed cross-cluster distances. The hypothesis predicts mean-pool should produce more compact clusters. Instead, the PoC shows concat has *better* (less negative) silhouette than mean-pool under the (flawed) labeling scheme. This is the opposite of the predicted direction.

**C3.3 — Silhouette is sensitive to the number of clusters**

The `account_id % 10` scheme creates 10 clusters out of 400 accounts × 60 events = 24,000 training events. Silhouette at 10 clusters may reflect something entirely different than silhouette at 800-1600 clusters (one per true device). The experiment must use true device labels.

---

## Root Cause 4: FastText is not the natural choice for this task

**C4.1 — FastText's character n-gram subword model is designed for morphological generalization**

FastText was developed to handle out-of-vocabulary words in morphologically rich languages (e.g., German inflections). The tokens here (`os_ios`, `browser_safari`) are not natural language words. They have fixed structure, no morphology, and a very small vocabulary. FastText's subword machinery adds complexity without the theoretical justification that motivated it.

*The alternative not tested:* Plain word2vec (skipgram without subword n-grams) over the same token sequences. If the mechanism is purely about whether features are embedded jointly vs. separately (not about character n-grams at all), word2vec would isolate the mean-pool vs. concat effect from the n-gram effect.

**C4.2 — The hypothesis conflates two distinct mechanisms**

The hypothesis claims that cross-boundary n-grams cause concat to be worse. But there is a second, independent mechanism: even without n-grams, jointly training on concatenated strings creates a single embedding space where different combinations of features compete for representation, while mean-pooling separates them into six independent embedding spaces that are then combined linearly. The PoC cannot distinguish which mechanism is responsible for any observed difference.

*What was not tested:* FastText on concat with n-gram size n=0 (characters only within tokens, no cross-boundary), or word2vec on concat to isolate the n-gram contamination effect from the joint-representation effect.

---

## Root Cause 5: The corpus sizes are mismatched

**C5.1 — Concat model trains on single-token sentences; mean-pool model trains on 6-token sentences**

The concat corpus has 400 × 60 = 24,000 one-token sentences. The mean-pool corpus has 24,000 six-token sentences. FastText is a context-window model: in mean-pool training, each token sees 5 other feature tokens as context in every sentence. In concat training, each token has no context at all (it is the only token in its sentence). This means the mean-pool model is trained on dramatically richer context signal than the concat model, and the performance difference (if any) may be entirely attributable to context richness, not to n-gram contamination.

*What was not tested:* Using only the target token's own embedding from the mean-pool model (no context influence) vs. the full mean-pool. Or training the concat model with multiple examples concatenated into a single sentence to equalize context exposure.

**C5.2 — The effective vocabulary is very different**

The concat model's vocabulary consists of at most 5×5×6×6×4×4 = 14,400 unique device strings. The mean-pool model's vocabulary consists of 5+5+6+6+4+4 = 30 unique feature tokens. FastText's n-gram subword model will fragment both vocabularies into character n-grams — but the number of distinct n-grams, their frequencies, and the training dynamics will differ substantially. No analysis of vocabulary coverage or embedding quality per feature dimension was performed.

---

## Root Cause 6: The evaluation protocol does not isolate the claimed mechanism

**C6.1 — No ablation over the number of matching features**

The hypothesis predicts that mean-pool's advantage should increase as the number of matching features increases (because more matching features amplify the spurious similarity in concat). The PoC tests only two extremes: novel attacks (few or no matches) and spoof attacks (5/6 matches). An ablation with 1, 2, 3, 4, 5 matching features would directly validate the mechanism.

**C6.2 — No per-feature-dimension attribution**

If mean-pool does outperform concat on spoof attacks, it could be because (a) the tz dimension is more salient in mean-pool (as claimed), or (b) some other structural difference. An attribution analysis (e.g., which feature token's embedding contributes most to the cosine distance for spoof attacks) would test the mechanism directly. The PoC produces no such attribution.

---

## Summary of Untested Claims

| ID   | Claim implicit in PoC | Root cause | Empirically testable? |
|------|----------------------|------------|----------------------|
| C1.1 | Boundary n-grams are large in concat | N-gram behavior | Yes — inspect n-gram frequencies |
| C1.2 | Mean-pool eliminates cross-feature correlations | N-gram behavior | Yes — measure inter-feature cosine similarity |
| C1.3 | Window=6 is appropriate for mean-pool | N-gram behavior | Yes — window sweep |
| C2.1 | Spoof near-0.5 means embeddings fail | Evaluation design | Yes — feature-by-feature ablation |
| C2.2 | Trivial baseline comparison is meaningful on spoof | Evaluation design | Yes — decompose negative class |
| C2.3 | Fleet AUC is uncontaminated | Evaluation design | Yes — stratify fleet by training presence |
| C3.1 | Silhouette measures per-device compactness | Silhouette labels | Yes — use true device labels |
| C4.1 | FastText is appropriate for this vocabulary | Model choice | Yes — compare to word2vec |
| C5.1 | Context richness is equalized | Corpus construction | Yes — measure context exposure |
| C6.1 | Mechanism scales with number of matching features | Mechanism isolation | Yes — ablation by number of matching features |
| C6.2 | Tz dimension is more salient in mean-pool | Mechanism attribution | Yes — per-feature distance decomposition |
