# H2 DEFENSE — Rebuttal of the Critique

Point-by-point response to H2_CRITIQUE.md.
Concessions are stated plainly. Contested points sharpen the disagreement.

---

### On the two foundational choices

**F1 — Window size.** Conceded as a genuine confound. The window=6/window=1 asymmetry was
chosen to make event-level context roughly equivalent (one full login event), not to give
mean-pool a training advantage. But "one event worth of context" is not a precise equivalence
when the token granularity differs — mean-pool gets 6 simultaneous cross-feature signals from
one event's context; concat gets a single opaque token. This is a real confound that the
experiment must resolve.

**F2 — Prefix loss.** Partially conceded. The concat format does omit the structured key prefixes
(`os_`, `browser_`) that the report identifies as mean-pool's n-gram advantage. However, the
claim is not simply that "prefixes are missing." The prefixes in mean-pool tokens serve two
functions: (a) they disambiguate feature dimensions at the n-gram level, and (b) they provide
structure for n-gram averaging over unseen token values. The concat string produces its own
internal structure from the feature value ordering. Whether the loss of explicit prefix structure
or the gain of cross-value n-gram noise dominates is precisely the empirical question. The
bundling of effects is real but does not invalidate the hypothesis — it means the experiment
must disentangle them.

---

### Issue 1 — Window size is not equivalent

**Defense concedes the factual claim; contests the conclusion.**

The window asymmetry is a confound. But the hypothesis is not that cross-boundary n-gram noise
is the *only* mechanism by which concat underperforms — the hypothesis is that concat
underperforms, and one mechanism is cross-boundary n-gram noise. A window equalization
experiment that shows mean-pool still outperforms concat at matched window sizes would
*strengthen* H2, not undermine it. A window equalization experiment that *closes the gap* would
show the confound is load-bearing and the n-gram noise story is overstated.

The defense position: run the window equalization test. If mean-pool still outperforms at
`concat window=6`, H2 is confirmed independently of the window confound.

---

### Issue 2 — Prefix structure absent in concat format

**Conceded. This is a genuine confound.**

The structured prefixes in mean-pool tokens are a real difference from the concat format.
The critique is correct that the PoC cannot separate "cross-boundary n-gram noise" from
"prefix structure loss." A third format — key-prefixed values with a non-underscore delimiter —
would isolate the prefix effect.

**Contest:** The values-only concat format is intentional (confirmed by the user) and reflects
what a naive implementation would actually produce. The hypothesis was always being tested on
this specific format, not on a hypothetical "best possible concat" design. The question "does
the concat format as described in the report underperform mean-pool?" is answerable in this
PoC regardless of which mechanism causes it. The mechanism question requires the third format
test — which is a valid addition to the experiment.

**Verdict:** Concede the confound; add the third-format test to isolate the prefix effect.

---

### Issue 3 — Underscore-in-value amplifies boundary noise

**Defense position: intentional, but a valid quantification point.**

The user confirmed raw feature values as-is. The `en_us`, `en_gb` values with underscores
produce more crossing n-grams than "clean" values would. This is not a bug — it reflects
what production data looks like (lang codes are typically BCP-47 format: `en-US`, `zh-CN`).
If anything, the amplification effect is realistic: real device fingerprints contain
structured sub-values (locale codes, timezone identifiers) that would generate similar
crossing n-grams.

**Contest:** The critique is correct that this makes the boundary noise effect *larger* than
the simple 6-field model predicts. But larger-than-expected does not mean wrong — it means
the effect is more severe in practice, which strengthens the case against concat. The claim
is not "cross-boundary noise is exactly N n-grams" but "cross-boundary noise degrades
concat." That claim is more true, not less, with underscore-containing lang values.

---

### Issue 4 — Both silhouette scores are negative

**Conceded in part; contested in its implication.**

Both negative silhouette scores are expected given the bounded shared feature vocabulary.
The critique is correct that the silhouette metric does not cleanly diagnose n-gram noise
or positional weighting — it measures cluster separation, which is inherently constrained
when all accounts draw from the same ~30 feature tokens.

**Contest:** The silhouette *gap* between mean-pool (−0.049) and concat (−0.165) is still
informative: concat's n-gram decomposition over the concatenated string produces *worse*
cluster separation than mean-pool's individual token embeddings, despite both operating on
the same underlying feature space. The absolute negativity of both scores is a property of
the evaluation design; the relative gap is a property of the two embedding methods. The
critique's proposed "theoretical ceiling" calculation would confirm whether the gap is
meaningful within the constrained space, and is worth computing.

---

### Issue 5 — The concat window penalizes cross-event context

**Defense position: window=1 is a reasonable default; window sensitivity is empirically testable.**

The critique is correct that there is no obviously correct window mapping between the two
training schemas. The defense of `window=1` is not that it is equivalent — it is that it
is the natural starting point: one event, one token, one context step. Whether `window=3`
or `window=6` improves concat is a real empirical question. If it does, the window choice
was underselling concat. If it doesn't, the model is not context-starved and the gap
reflects the embedding structure, not the training configuration.

**Defense position on the broader claim:** Even if window tuning narrows the gap, the
concat format still discards the structured prefix n-grams (`os_`, `browser_`) that the
report identifies as mean-pool's n-gram advantage. Window tuning can compensate for
insufficient event context but cannot restore the missing prefix structure. So some gap
is expected to persist even at optimal window.

---

### Issue 6 — Spoof gap may reflect distinct token learning, not positional weighting

**Conceded. The spoof result cannot currently distinguish the two mechanisms.**

The critique is correct: the larger spoof gap (0.073) could reflect (a) positional weighting
diluting the tz mismatch in the concat string, or (b) mean-pool's distinct `tz_utc-5` /
`tz_utc+5` tokens receiving stronger explicit embeddings trained on account-specific
co-occurrence. A permutation test — put tz at position 1 in the concat string — would settle
this. If spoof AUC improves significantly, positional weighting is the mechanism. If not,
distinct token embedding is.

**Contest:** The permutation test result is informative either way. If positional weighting
is confirmed, it proves the front-loading claim in H2. If distinct token embedding is the
mechanism, it proves the co-occurrence learning claim in H2 (mean-pool's third advantage:
explicit cross-feature co-occurrence training). Both outcomes support H2 — they just name
different mechanisms. The critique correctly identifies that the current PoC cannot distinguish
them, which is a gap in the mechanistic account rather than a gap in the directional claim.

---

### Issue 7 — OOV token behavior is not tested

**Conceded. The PoC is explicitly in-vocabulary only.**

The scope exclusion is documented. The critique is correct that the concat format's OOV
degradation profile is unknown relative to mean-pool. If concat degrades less on OOV tokens
(because the concatenated n-grams still encode some of the feature structure), the production
ranking could differ from the in-vocabulary ranking.

**Defense position:** The in-vocabulary comparison is the correct starting point. Experiment 3
established that mean-pool FastText handles OOV feature tokens gracefully via structured prefixes.
The question is whether concat handles OOV tokens as well or better. This is a genuine open
question that belongs in the experiment but is not a flaw in testing H2 in the in-vocabulary
case first.

---

## Summary of concessions vs. contested points

| Issue | Verdict |
|-------|---------|
| Window size confound (Issue 1) | Conceded as confound; window equalization test needed |
| Prefix structure confound (Issue 2) | Conceded; third-format test needed to isolate |
| Underscore amplification (Issue 3) | Contested — amplification is realistic, strengthens H2 |
| Negative silhouette (Issue 4) | Partially conceded; gap is still informative; ceiling calc useful |
| Window penalizes cross-event context (Issue 5) | Contested; window sweep needed |
| Spoof mechanism ambiguity (Issue 6) | Conceded; tz-position permutation test needed |
| OOV behavior untested (Issue 7) | Conceded; production scope; add OOV injection test |
