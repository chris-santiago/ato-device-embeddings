# H2 DEBATE

Multi-turn argument on each contested point. Each exchange ends with a resolution
statement. The final section lists agreed empirical tests.

---

## Point 1 — Window size confound

**Critique position:** Mean-pool trains with window=6 over 6-token sequences; concat trains
with window=1 over single-token sequences. Mean-pool gets explicit cross-feature co-occurrence
signal (os_ios co-predicts browser_safari within the same event window). Concat does not.
Any AUC advantage for mean-pool may trace to this training difference rather than to n-gram
noise. Test: run concat at window=6 and compare.

**Defense position:** Window=1 for concat is the natural default (one event = one context step).
Even if window=6 improves concat, it cannot restore the structured prefix n-grams (os_, browser_)
that mean-pool tokens carry — those prefixes are absent from the values-only concat format.
Some gap should persist even at matched window.

**Critique reply:** That is a distinct claim — prefix structure — which Issue 2 addresses
separately. The window confound may explain a *large portion* of the gap independently of
prefix structure. We need to know the magnitude of the window effect to know how much work
the n-gram noise story needs to do.

**Defense reply:** Agreed. The window sweep is necessary. The defense does not dispute the
test — only the conclusion that window equalization will eliminate the gap entirely.

**Resolution:** Run concat at window=1, window=3, and window=6. If the gap between best-concat
and mean-pool narrows by more than half, the window effect is load-bearing and the n-gram
noise story is overstated. If the gap persists above 0.01 AUC on novel attacks at best-concat
window, the noise story retains explanatory power independent of window size.

---

## Point 2 — Prefix structure confound

**Critique position:** Mean-pool tokens carry `os_`, `browser_`, `tz_` prefixes. These
produce n-grams (`os_`, `bro`, `row`, `wse`, `brow`, `rows`, `rowse`) that are semantically
structured — all OS tokens share `os_` as a 3-gram. The concat format `ios_safari_utc-5...`
has none of these prefixes. The PoC cannot separate "cross-boundary n-gram noise" from
"prefix structure loss."

**Defense position:** The values-only concat format is the hypothesis as stated in the report.
The question is whether *that format* underperforms mean-pool, not whether a better-designed
concat format could close the gap. Demonstrating that the stated format fails is sufficient
to support H2. The mechanism question (noise vs prefix loss) requires the third-format test.

**Critique reply:** Agreed that demonstrating failure is sufficient for the directional claim.
But the report attributes the failure to a *specific mechanism* (cross-boundary n-gram noise
and positional weighting). If prefix loss is the actual driver, the mechanism account in the
report is wrong, even if the directional result is right. The mechanism matters for knowing
whether a better concat design could recover performance.

**Defense reply:** Correct. The mechanism question is worth testing. The third format —
full key-prefixed tokens joined with a non-underscore delimiter — would isolate the prefix
effect. If that format matches mean-pool, prefix structure was the mechanism. If it still
underperforms, n-gram noise was the mechanism.

**Resolution:** Build a third signal: `prefixed_concat_fasttext`, using tokens joined as
`os:ios|browser:safari|tz:utc-5|lang:en_us|net:wifi|screen:small` (or equivalent format
that preserves key prefixes and avoids underscore crossings). Test this signal's AUC against
mean-pool. If prefixed concat matches mean-pool (within CI), prefix loss was the mechanism
in the original values-only format. If prefixed concat still underperforms, cross-boundary
n-gram noise from positional ordering is the independent mechanism.

---

## Point 3 — Underscore amplification

**Critique position:** `en_us`, `en_gb`, `es_mx` lang values contain underscores. Joined
with `_` separator, they create compound n-gram crossings that are more severe than the
simple 6-field model predicts. This makes concat look worse than it would with clean values.

**Defense position:** Real-world feature values do contain underscores (locale codes, timezone
identifiers). The amplification reflects realistic deployment conditions, not a flaw. If
anything, it shows the effect is larger in practice.

**Critique reply:** The amplification is realistic in principle, but the lab values
(`en_us`, not `en-US`) are a synthetic design choice, not a production format. Using BCP-47
with hyphens (`en-US`) would reduce the underscore crossing. The defense cannot simultaneously
claim the format is realistic and that the exact format was chosen for the experiment.

**Defense reply:** The user confirmed raw-values-as-is. The underscore amplification is
intentional — it models what a developer would produce if they naively concatenated device
profile fields without sanitizing delimiters. The critique is correct that this inflates the
effect but incorrect that this is unrepresentative. The prefixed-concat test in Point 2 will
also use a non-underscore delimiter, which naturally controls for this.

**Resolution:** The underscore amplification is intentional and not a separate test item.
The prefixed-concat test (Point 2) implicitly controls for it by using a different delimiter.
No additional test needed.

---

## Point 4 — Negative silhouette and its diagnostic value

**Critique position:** Both silhouette scores are negative (−0.049, −0.165). This reflects
the bounded shared feature vocabulary — many accounts share tokens, so within-account
clustering cannot be tight. The silhouette gap may trace to the window confound (Issue 1)
or prefix loss (Issue 2) rather than cross-boundary n-gram noise.

**Defense position:** The gap (0.116) is still informative within the constrained space.
The absolute negativity is expected; the relative gap measures which method produces better
separation under the same constraints.

**Critique reply:** Agreed that the gap is informative if the confounds in Issues 1 and 2
are resolved. After the window sweep and prefixed-concat tests, the remaining silhouette gap
can be attributed to n-gram noise or positional weighting.

**Defense reply:** Agreed. Silhouette comparison after confound isolation is the correct
diagnostic.

**Resolution:** Report silhouette for all three conditions (values-only concat, prefixed
concat, mean-pool) at best window across the sweep. The silhouette gap after controlling for
prefix structure is the operative measure of the n-gram noise effect.

---

## Point 5 — Window sweep for concat

**Critique position:** window=1 may starve the concat model of cross-event context.
A monotonically increasing AUC with window size would indicate the model is context-starved
and the gap is a configuration artifact.

**Defense position:** window=1 is the natural baseline. Even at larger windows, mean-pool's
cross-event context is richer (6 fine-grained tokens per neighboring event vs 1 opaque token).
The sweep will show whether the gap narrows, but some gap from prefix structure and n-gram
noise should remain.

**Resolution:** Merged with Point 1. Run concat at window=1, window=3, window=6. Report
AUC at each window size. The window at which AUC plateaus defines best-concat for all
subsequent comparisons.

---

## Point 6 — Spoof mechanism: positional weighting vs. distinct token embedding

**Critique position:** The spoof attack gap (0.073 AUC) could reflect (a) positional
front-loading causing the tz mismatch deep in the concat string to be discounted, or (b)
mean-pool's distinct `tz_utc-5` / `tz_utc+5` tokens receiving stronger explicit embeddings
trained on account-specific co-occurrence. The current PoC cannot distinguish these.

**Defense position:** Either mechanism supports H2 — (a) proves the front-loading claim;
(b) proves the co-occurrence learning claim. The directional result is the same regardless.

**Critique reply:** Correct that both mechanisms support H2's directional claim. But the
report specifically attributes the gap to positional weighting ("n-gram overlap is front-loaded,
so a timezone mismatch at position 3 reduces similarity for all subsequent features even
where the devices agree"). If the mechanism is actually distinct token co-occurrence learning
rather than positional weighting, the causal claim in the report is wrong. This affects
what you'd fix if you wanted to improve the concat approach.

**Defense reply:** Conceded. The mechanism matters for the report's explanation. The
tz-position permutation test is the cleanest way to settle this.

**Round 2 — Critique:** The permutation test is clean in theory but has a subtlety. Permuting
tz to position 1 changes the entire n-gram structure of the concat string, not just the tz
position. The n-grams that cross the os/tz boundary in the permuted string (`utc-5_ios_...`)
are different from those that cross the tz/lang boundary in the original string (`en_us_wifi...`).
If AUC changes after permutation, we cannot cleanly attribute it to "tz at position 1" vs.
"different n-grams at different boundaries."

**Defense reply:** Fair. The cleanest version of the positional test is to vary the tz position
across multiple permutations and test whether the spoof AUC is monotonically related to tz
position rank (tz at position 1 → highest spoof AUC, tz at position 6 → lowest). If AUC
tracks tz position rank across permutations, positional weighting is the mechanism. If AUC
is non-monotonic, the specific n-gram combinations at each boundary are the driver.

**Critique reply:** Agreed. Testing all 6 permutations where tz occupies each position, with
all other features at fixed positions, would produce 6 data points. Monotonicity over those
6 data points is strong evidence for positional weighting. Non-monotonicity with the spoof
AUC correlating with tz boundary characteristics is evidence for specific n-gram crossing.

**Resolution:** Test 6 concat orderings where tz is placed at each of the 6 positions (other
features in fixed alphabetical order). Compute spoof AUC for each. Verdict:
- If spoof AUC is monotonically decreasing with tz position (tz first → highest, tz last →
  lowest): positional weighting confirmed.
- If spoof AUC is non-monotonic and does not track tz position: specific n-gram crossing
  structure is the driver; the front-loading claim needs revision.

---

## Point 7 — OOV token behavior

**Critique position:** The PoC only evaluates in-vocabulary profiles. If concat degrades
less on OOV tokens than mean-pool, the production ranking could differ.

**Defense position:** In-vocabulary comparison is the correct starting point. Experiment 3
established mean-pool's OOV advantage. The question is whether concat matches or exceeds it.

**Critique reply:** Experiment 3 did not test concat at all. The OOV comparison is genuinely
open.

**Defense reply:** Agreed. Add OOV injection test: one unseen OS token, one unseen tz token.
Compare AUC degradation for both signals.

**Resolution:** Inject 5% of eval events with OOV feature tokens (`os_harmonyos`, `tz_utc+9`).
Measure AUC for mean-pool and concat on these OOV events. Verdict:
- If mean-pool degrades less: mean-pool is preferred on both in-vocabulary and OOV performance.
- If concat degrades less: concat retains a production-relevant OOV advantage despite in-vocabulary weakness.
- If both degrade equally: the OOV handling mechanism doesn't change the ranking; H2 holds in production.

---

## Agreed empirical tests

The following tests are the specification for Step 6. Nothing added, nothing omitted.

| # | Test | Conditions | Pre-specified verdict |
|---|------|------------|-----------------------|
| T1 | **Window sweep** | concat window ∈ {1, 3, 6} on i.i.d. corpus | If best-concat vs mean-pool gap <0.005 AUC (novel): window is the primary confound, n-gram story overstated. If gap ≥0.010 AUC: n-gram noise has independent explanatory power. |
| T2 | **Prefixed-concat format** | Third signal: `key:val\|key:val\|...` joined with `\|`; FastText on this format | If prefixed-concat AUC ≈ mean-pool (within 0.005): prefix structure was the driver in values-only format. If prefixed-concat still lags mean-pool by >0.010: cross-boundary n-gram noise is an independent mechanism. |
| T3 | **Tz-position permutation** | 6 concat orderings placing tz at each position; compute spoof AUC for each | Monotonic decrease with tz position → positional weighting confirmed. Non-monotonic → specific n-gram crossings at boundaries. |
| T4 | **OOV injection** | 5% of eval events with `os_harmonyos` or `tz_utc+9` unseen during training | Compare AUC degradation: mean-pool vs. best-concat. If mean-pool degrades less: H2 holds in OOV setting too. If concat degrades less: concat retains OOV production advantage. |
| T5 | **Bootstrap CIs** | N=1000, percentile method, on all AUC numbers across T1–T4 | All numerical claims require overlapping/non-overlapping CIs to determine significance. |

**Trivial baseline:** The trivial baseline from Experiment 3 (binary `account_oov`) achieves
AUC 0.750 by construction under this evaluation design. Any signal that does not exceed 0.750
with non-overlapping CIs has no discriminative power beyond membership detection. All signals
in this experiment already exceed this ceiling comfortably, but the 0.750 line is plotted in
all figures as reference.
