# H2 Rerun — Defense

**Defender role:** Rebut the critique where the critique is wrong; concede where it is right;
propose empirical tests where the argument is genuinely unresolved.

---

## D-C1 — Single seed / no CIs

**Concede partially.** The fleet delta (+0.006) and novel delta (+0.012) are small and single-seed.
The fleet delta should not be treated as reliable. The silhouette gap (+0.119) and spoof delta
(+0.054) are more robust — the magnitude suggests they are unlikely to be pure noise.

**However:** The critique's test is fair and necessary. Bootstrap CIs should be run. The defense
predicts the spoof and silhouette results survive CI analysis; the fleet and novel deltas may not.

**Proposed test:** Bootstrap N=1000 with replacement on all AUC and silhouette results.
If spoof CI [lo, hi] for mean_pool strictly exceeds concat CI [lo, hi], defense wins C1 for spoof.
If all CIs overlap, critique wins C1.

---

## D-C2 — Window mismatch

**Concede the framing but not the conclusion.** Window=6 for mean_pool is not arbitrary — it is
the natural window for a 6-token login event, ensuring each feature token co-occurs with all
other features in the same event. Window=1 for concat is similarly principled — each login is
one token and the adjacent events in the session are the natural context.

**The critique's test is still valuable.** A concat model at window=3 or window=6 gives each
concatenated token access to 3–6 adjacent login events, providing session-level context. If
this closes the gap, it reveals that the mean_pool advantage is partly a contextual richness
artifact, not purely a within-event n-gram structure effect.

**Prediction:** Concat at window=3 will not close the silhouette gap (silhouette is driven by
embedding geometry of the device tokens themselves, not session context). Concat at window=6
may marginally improve spoof AUC but will not exceed mean_pool because the n-gram boundary
problem persists regardless of context window.

**Proposed test:** Train concat at window ∈ {1, 3, 6}. Compare AUC and silhouette vs mean_pool.
If window=6 concat matches or exceeds mean_pool on spoof AUC, critique wins C2.

---

## D-C3 — Worst-case encoding

**Concede the theoretical point.** The underscore separator does amplify cross-boundary n-gram
noise relative to a non-overlapping delimiter. The PoC measures the worst-case encoding.

**Dispute the conclusion.** Even with a prefixed format (os:ios|browser:safari|...), the FastText
n-gram slicer will produce cross-boundary character sequences spanning the key:val separator
(e.g., "ios|b", "ios|br"). The non-overlapping delimiter reduces but does not eliminate the noise.
The defense predicts a silhouette gap will persist even with prefixed-concat.

**Proposed test:** Train prefixed-concat FastText (key:val|key:val|...) and compare silhouette
and spoof AUC to mean_pool and plain concat. If prefixed-concat silhouette >= mean_pool silhouette
and prefixed-concat spoof AUC >= mean_pool spoof AUC, critique wins C3. If a gap persists,
defense wins C3.

---

## D-C4 — Fleet contamination

**Dispute the critique.** Fleet injection affects both signals identically — the same training
events are added to the same accounts. The relative gap between mean_pool and concat centroids
is set by the embedding geometry, not by the fleet injection rate. Fleet injection compresses
scores symmetrically; it does not preferentially harm one signal over the other.

**The contamination does create a real confound for the fleet AUC numbers specifically** — but
this affects the absolute AUC, not the relative ordering of the two signals. The delta between
mean_pool and concat should be preserved.

**Proposed test:** Stratify fleet AUC by targeted (fleet-injected) vs. non-targeted accounts.
If the mean_pool/concat delta changes sign across strata, critique wins C4. If the delta is
preserved across strata, defense wins C4.

---

## D-C5 — Trivial baseline absent

**Concede entirely.** The PoC lacks a trivial baseline. This is the most dangerous omission.

**Defense prediction:** A set-membership baseline (does the exact profile appear in training?)
will score very high on novel attacks (novel profiles are constructed to differ on OS, tz, and
lang — guaranteed miss) but will struggle on spoof attacks (spoof profiles share OS/browser/lang
and only differ on tz — if any spoof profile happens to match a training device, it is a false
negative). Fleet attacks depend on whether the fleet device appears in training for a given account.

The defense predicts mean_pool FastText will outperform set-membership on spoof attacks because
FastText captures partial similarity — a device that differs only on tz should score closer to
the centroid than a device that differs on all features. Set-membership is binary and cannot
express this graded similarity.

**Proposed test:** Implement exact-profile set membership baseline. Compare AUC on spoof and
fleet attack types. If set-membership >= mean_pool on spoof, critique wins C5. If mean_pool
exceeds set-membership on spoof, defense wins C5.

---

## D-C6 — Mechanism attribution (n-gram density vs. positional weighting)

**Concede the mechanistic ambiguity.** The hypothesis attributes the advantage to two mechanisms:
(a) cross-boundary n-gram noise, and (b) positional weighting. The PoC cannot distinguish these.

**However:** The defense argues the distinction is less important than the practical result.
If prefixed-concat does not close the gap, then both mechanisms are real and mean-pool is
superior for any encoding. If prefixed-concat closes the gap, then (a) is the dominant
mechanism and the fix is to use a non-overlapping delimiter — mean-pool is still preferable
but the mechanism is partially attributed.

**This is already addressed in C3/C6 combined.** The prefixed-concat test resolves both.

---

## D-C7 — Positional mechanism / permutation

**Dispute the critique's predicted outcome.** The positional-weighting mechanism predicts that
moving tz to position 0 (first) in the concat string should reduce the penalty for tz mismatch —
but the prediction is more subtle. Moving tz to position 0 in a concat string means the
n-grams produced by a tz mismatch affect only subsequent features, not preceding ones. Since
OS and browser (positions 1, 2) are high-signal features for spoof detection, moving tz first
may *not* recover the gap because the OS and browser n-grams are now corrupted by the tz prefix.

The defense predicts the permutation study will show that no ordering of features in the concat
string recovers the full mean_pool spoof advantage, because the positional weighting problem
is symmetric — whichever feature differs will corrupt n-grams for all features that follow it
in the concatenated string.

**Proposed test:** Test all 6 orderings that place tz at each of positions 0–5.
If any ordering matches mean_pool spoof AUC within 0.01, critique wins C7 for that ordering.
If no ordering closes the gap to within 0.01, defense wins C7.

---

## Summary

| Point | Concede? | Prediction |
|-------|----------|------------|
| C1 | Partially — spoof and silhouette will survive CI; fleet may not | Empirical test required |
| C2 | Partially — window sweep is fair | Predict gap persists at all windows |
| C3 | Yes (mechanism point) | Predict gap persists with prefixed-concat |
| C4 | No — contamination is symmetric | Predict delta preserved across strata |
| C5 | Yes entirely | Predict mean_pool > set-membership on spoof |
| C6 | Concede ambiguity | Resolved by C3 prefixed-concat test |
| C7 | No — permutation won't fully recover gap | Predict gap persists across all orderings |
