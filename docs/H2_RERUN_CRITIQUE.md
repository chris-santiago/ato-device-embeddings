# H2 Rerun — Adversarial Critique

**Critic role:** Find every reason the PoC result could be wrong, misleading, or unactionable.

---

## C1 — Single seed, no confidence intervals

The PoC runs on SEED=42 only. AUC differences of +0.006 (fleet) and +0.012 (novel) are within
typical bootstrap variance for N=80 evaluation events. A single-seed result cannot be distinguished
from sampling noise. The silhouette gap (+0.119) is larger but silhouette has high variance
on small subsamples.

**What critique-wins looks like:** Bootstrap 95% CI for mean_pool AUC overlaps with concat AUC.
**What defense-wins looks like:** CIs are non-overlapping across all three attack types.

---

## C2 — Non-equivalent window parameters

mean_pool uses window=6 (covers one full 6-token login event); concat uses window=1 (one adjacent
event in context). This is not an apples-to-apples comparison. Window controls how many
neighbouring tokens provide training signal. mean_pool's window=6 gives each feature token
access to all other tokens in the same login event, which is a richer training signal
independent of the mean-pooling mechanism. Concat at window=6 would give each concatenated
token access to 6 adjacent login events, potentially equalising the contextual signal.

**What critique-wins looks like:** Concat at window=3 or window=6 closes the AUC gap materially
(>50% of the delta on any attack type).
**What defense-wins looks like:** Concat at all windows remains below mean_pool on silhouette
and AUC; the gap persists regardless of window.

---

## C3 — Synthetic data over-represents the proposed mechanism

Feature values include underscores in values (en_us, es_mx, utc-5, utc+8). In the concat string
"ios_safari_utc-5_en_us_wifi_small", the separator and internal underscores are identical,
so the n-gram slicer cannot distinguish feature boundaries from intra-value structure. This
amplifies cross-boundary noise exactly as the hypothesis predicts — but a real implementation
would choose a non-underscore delimiter (pipe, hash, colon) or key-prefix the values.
The PoC measures the worst-case concat encoding, not the average real-world one.

**What critique-wins looks like:** Prefixed-concat format (os:ios|browser:safari|...) closes
the silhouette and AUC gap by >50% compared to plain underscore concat.
**What defense-wins looks like:** Even with a non-overlapping delimiter, cross-boundary n-grams
still degrade silhouette and AUC compared to mean-pooling.

---

## C4 — Fleet attack contamination inflates training centroids

25% of accounts have fleet devices injected into their training history (8 fleet events
per targeted account, out of ~60 total events). This shifts centroids toward fleet device
vectors for targeted accounts. At eval time, fleet attack events score as *closer* to
those centroids than they should, compressing the score distribution and potentially
suppressing AUC. This affects both signals equally — but the *degree* to which it
suppresses each signal depends on the embedding geometry. If mean_pool centroids
are tighter (higher within-account coherence), the centroid shift from fleet injection
may be proportionally less harmful. The PoC does not isolate this confound.

**What critique-wins looks like:** Fleet AUC for both signals is materially lower on
accounts *without* fleet injection versus accounts *with* injection.
**What defense-wins looks like:** Fleet contamination affects both signals equally;
the relative gap between mean_pool and concat is preserved after stratification.

---

## C5 — Trivial baseline absent

The PoC does not include a trivial baseline (e.g., a frequency-weighted n-gram overlap
counter or a simple per-account feature-value set membership check). A claim that
mean-pool FastText is superior to concat FastText is not useful if both are inferior
to a two-line heuristic. The AUC numbers (0.76–0.99) are high, but without a baseline
we cannot tell whether the FastText training adds anything over a simpler representation.

**What critique-wins looks like:** A simple set-membership baseline (does the new device
profile appear anywhere in the account's training history?) achieves AUC >= mean_pool on
any attack type.
**What defense-wins looks like:** mean_pool FastText outperforms set-membership baseline
on at least spoof and fleet attacks (novel attacks are easy and any method should solve them).

---

## C6 — Spoof detection may be a fluke of n-gram geometry, not the proposed mechanism

The hypothesis attributes mean_pool's spoof advantage to equal weighting of all six
features. But FastText's subword n-grams on individual tokens (e.g., "tz_utc-5") may
simply learn a more compact representation of timezone values than the concat n-grams
that must span across "safari_utc-5_en_us". The advantage could be purely a function
of n-gram vocabulary density, not positional weighting. If the mechanism is n-gram
vocabulary (not positional equality), then prefixed-concat (which separates features
with a non-overlapping delimiter) should perform similarly to mean-pool.

**What critique-wins looks like:** Prefixed-concat AUC on spoof is within 0.01 of mean_pool.
**What defense-wins looks like:** Prefixed-concat AUC on spoof is still materially below
mean_pool, indicating positional weighting / mean-pooling mechanism is the cause.

---

## C7 — Feature ordering permutation not tested

The hypothesis claims front-loaded positional weighting is a causal mechanism. This is
untestable without a permutation study — if tz (a high-signal feature for spoof) is moved
from position 3 (middle) to position 0 (first) in the concat string, the hypothesis
predicts the spoof AUC gap should shrink. If moving tz to first position recovers most
of the gap, the positional mechanism is confirmed. If AUC is invariant to position, the
mechanism is n-gram vocabulary density, not positional weighting.

**What critique-wins looks like:** Tz at position 0 in concat recovers >50% of the
mean_pool vs. concat spoof AUC gap.
**What defense-wins looks like:** Spoof AUC gap persists regardless of tz position;
permutation does not change the ordering.

---

## Summary of contested points

| Point | Critique claim | Testable? |
|-------|---------------|-----------|
| C1 | Results are sampling noise | Yes — bootstrap CIs |
| C2 | Window mismatch confounds result | Yes — window sweep on concat |
| C3 | Worst-case encoding inflates gap | Yes — prefixed-concat variant |
| C4 | Fleet contamination confounds AUC | Yes — account stratification |
| C5 | No trivial baseline | Yes — set-membership baseline |
| C6 | Mechanism is n-gram density, not position | Yes — prefixed-concat spoof AUC |
| C7 | Positional mechanism unverified | Yes — tz-position permutation |
