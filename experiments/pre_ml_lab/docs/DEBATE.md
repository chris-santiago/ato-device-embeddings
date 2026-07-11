# Debate: ATO FastText PoC

Each point from `CRITIQUE.md` is argued through multiple exchanges until one side
concedes, both sides converge, or an empirical test is agreed upon. The format is:

- **C:** Critique position
- **D:** Defense position
- **Resolution:** agreed conclusion

---

## Point 1 — OOV detection vs. account clustering

**C¹:** The dominant signal is trained-token vs. OOV-token distinction, not account
membership. A novel device is OOV; its embedding is generic. That alone explains the
distance gap. Account clustering is a secondary effect at best.

**D¹:** The account centroid is account-specific. A binary OOV classifier returns the
same score regardless of which account the novel device logs into. Centroid distance
returns a different score depending on the angular relationship between the device vector
and *that specific account's* device neighborhood. These are different functions. An
attacker who re-uses a device after its first use (now in-vocab) would evade OOV
detection entirely but would still show high distance from the account's centroid — if
real account clusters exist.

**C²:** That argument holds only if in-vocab cross-account devices score HIGH against
accounts they don't belong to. If the embedding space has genuine per-account cluster
structure, a device belonging to account B should have high cosine distance to account
A's centroid. If the space is unstructured, a well-trained in-vocab device from account B
might score LOW for account A — indistinguishable from a known device — making account
clustering illusory and OOV detection the only real mechanism.

**D²:** Agreed — and this is testable. The current PoC doesn't score cross-account in-
vocab devices. Run that experiment: take known devices from account B and score them
against account A's centroid. If they score similarly to novel OOV devices, account
clusters are real. If they score similarly to known devices, the signal is OOV detection.

**C³:** Exactly. And the result would be definitive. There's no theoretical prior strong
enough to skip the empirical check here — the block-diagonal co-occurrence structure
suggests clusters should exist, but negative sampling and shared character n-grams both
work against clean separation.

> **Resolution — empirical test agreed.** Add a third test class to the evaluation:
> in-vocab devices belonging to a *different* account, scored against the target account's
> centroid. If cross-account in-vocab devices score high (similar to OOV novel devices),
> account-level clustering is driving the signal. If they score low (similar to known
> devices), OOV detection is the dominant mechanism. This single test resolves the central
> mechanistic question and should be added to the PoC before any further development.

---

## Point 2 — Subword n-grams and the shared `dev_` prefix

**C¹:** All device IDs share the `dev_` prefix. Its n-gram contributions are trained over
every device in the corpus — a global average. This creates a strong shared component in
every device vector. Novel OOV devices also get this component, plus random noise from
their unique characters. As string length increases, the random component averages toward
zero and all OOV vectors converge to approximately the same embedding — the `dev_` attractor.

**D¹:** The shared prefix appears identically in both the candidate device vector and the
account centroid. If we write `v = e_prefix + e_specific`, then both the test device and
the centroid have `e_prefix`. In the cosine dot product, the shared prefix contributes
`||e_prefix||²` to the numerator and inflates both denominators equally. For large
`||e_prefix||`, the cosine is dominated by the prefix direction and the distance between
any two devices goes to zero regardless of their specific components. This doesn't destroy
discrimination — it just compresses all distances toward zero uniformly, preserving the
ranking between known and novel devices. ROC-AUC is rank-based.

**C²:** The neutrality argument holds if `e_prefix` is identical for all tokens — but
it's not. The `dev_` n-gram vectors are trained jointly with the skip-gram objective. A
token that appears in many different account contexts will have `e_prefix` n-grams that
accumulate gradients from all of them. An in-vocab known device's `e_prefix` contribution
is shaped by its specific co-occurrence history; an OOV novel device's `e_prefix`
contribution is determined solely by the n-gram lookup table, not by any skip-gram
training. The prefix is the same string but the *trained* prefix component of an in-vocab
token is not the same as the *hash-only* prefix component of an OOV token. There is no
shared constant to cancel.

**D²:** This is a good point and a genuine correction to the defense's geometric argument.
The prefix n-grams in an in-vocab token have been updated by skip-gram gradients; in an
OOV token they haven't. But this actually strengthens the detection argument: the
divergence between trained and untrained prefix components is additional signal, not noise.
The in-vocab token's prefix has been "personalized" by its co-occurrence context; the OOV
token's prefix is generic. This is another layer of trained-vs-OOV separation, operating
at the n-gram level rather than the token level.

**C³:** So we agree the prefix doesn't cancel but disagree on whether its effect helps or
hurts. The defense says it adds signal; I'm saying it conflates the n-gram training
mechanism with account clustering. If a known device's prefix is updated by gradients from
all accounts that use tokens with similar n-grams, the prefix vectors encode global corpus
statistics, not account membership. Whether this helps or hurts depends on empirical
gradient magnitudes we haven't measured.

**D³:** Agreed. The theoretical argument can be constructed in either direction. The
practical question is: does removing the shared prefix (or removing subword n-grams
entirely, replacing FastText with Word2Vec) change the AUC materially?

> **Resolution — empirical test agreed.** Run the experiment with Word2Vec (no subword
> n-grams) to isolate the prefix effect. If AUC is similar or higher under Word2Vec, the
> subword mechanism is not contributing positively. If AUC drops substantially, FastText's
> character-level generalization is providing real signal. Note that with Word2Vec, novel
> OOV devices cannot be embedded at all — they would need a separate handling strategy
> (e.g., random vector or zero vector), making this a clean ablation on the subword
> contribution specifically.

---

## Point 3 — Sparse co-occurrence for secondary devices

**C¹:** With `window=5` and a Zipf-dominated history, devices beyond the primary account
device may never co-occur directly within any training window. Their embeddings are shaped
by co-occurrence with A alone, not with each other. The "account cluster" for secondary
devices is actually just proximity to A, not a coherent multi-device cluster.

**D¹:** Proximity to A is sufficient. All secondary devices being near A means they're
near *each other* transitively — exactly how Word2Vec produces semantic clusters in NLP
without requiring every word pair to co-occur directly. The Zipf primary device A is the
pivot that defines the account neighborhood. This is the standard distributional
similarity mechanism.

**C²:** The transitivity works in NLP because pivot contexts are semantically meaningful
— "royal" is a genuine shared context for "king" and "queen." Here, the pivot is device A
appearing near devices B and C. The "context" is just temporal adjacency in a login
sequence. There's no semantic content being transferred. The embedding of B is pulled
toward A's neighborhood, and C is also pulled toward A, so B and C are near each other —
but this holds for *any* two devices that happen to share a high-frequency co-occurrence
partner, regardless of account membership.

**D²:** The difference in NLP is degree, not kind. In NLP, many different words share a
context word, creating a dense shared-context graph. Here, the shared context is strictly
within account boundaries — negative sampling explicitly pushes cross-account devices
apart. Each positive (A→B) pair has corresponding negatives drawn globally, which include
devices from other accounts being trained to be far from A. The per-account pivot plus
cross-account negative sampling creates both an attraction force (within account) and a
repulsion force (across accounts) simultaneously.

**C³:** The negative sampling point is well-taken and I hadn't fully weighted it. Standard
negative sampling draws from the unigram frequency distribution corpus-wide. For a corpus
of 300 accounts × 50 devices average, most negatives drawn for any within-account pair
will be from other accounts. This is active cross-account repulsion. Combined with
within-account attraction via the pivot device, the mechanism is stronger than the
sparsity critique implied.

> **Resolution — defense position accepted.** The combination of within-account positive
> pairs (via the pivot device) and cross-account negative sampling creates genuine account-
> level separation pressure. The co-occurrence sparsity concern is real for accounts with
> very short histories but the underlying mechanism is sound for the parameter ranges
> tested. The follow-on study question is: at what minimum history length does the
> cluster signal degrade below useful AUC? This should be studied by stratifying AUC by
> account history length.

---

## Point 4 — The centroid and its norm

**C¹:** The centroid norm `||μ||` is a natural measure of cluster coherence — it equals 1
when all device vectors are identical and approaches 0 as they become orthogonal. This
information is discarded when computing cosine distance. A better score would be
`cosine_distance(x, μ) / (||μ|| + ε)`, penalizing comparisons against diffuse centroids.

**D¹:** The proposed correction amplifies scores for accounts with diffuse centroids —
exactly the accounts where you're least certain. A new account with 2 devices will have
a low `||μ||`, and dividing by it would push its scores toward infinity. The current
behavior, ignoring coherence, is conservative. The correction increases false positives
precisely on the accounts where the model should be least confident.

**C²:** You've misread the direction of the correction. If a centroid is diffuse
(`||μ||` small), it means we *can't* reliably distinguish near from far — all cosine
distances to a near-zero centroid are approximately equal and meaningless. The correction
says: when the centroid is diffuse, treat its signal as weak. That means you should
*reduce* confidence in high scores, not increase them. The correction should suppress
anomaly flags on diffuse accounts — but the formula `score / (||μ|| + ε)` would
*increase* scores, not decrease them. The proposed formula is wrong for the stated intent.

**D²:** Agreed — the critique's proposed formula has the wrong sign for its own stated
purpose. If the intent is to suppress scores from uncertain centroids, the correction
should multiply by `||μ||`, not divide:
`score × ||μ||`. This shrinks scores toward zero when the centroid is diffuse and
preserves them when the centroid is tight. But even this has a problem: it mixes what
the score *means* (anomaly signal) with how *reliable* it is (centroid coherence). These
should be reported as separate quantities, not combined into a single scalar.

**C³:** Agreed on the formula direction. And agreed that conflating signal strength and
confidence into a single scalar is bad design. The underlying problem — that thin
accounts produce unreliable scores that can't be distinguished from genuine anomalies —
is real regardless of whether we correct the score or separate the quantities.

> **Resolution — both concede on mechanics, agree on the real problem.** The proposed
> `score / (||μ|| + ε)` formula in the critique has the wrong sign for its stated intent.
> The defense correctly identifies the conservative behavior of the current approach.
> Both sides agree the underlying issue is that centroid reliability varies across accounts
> and the current scalar score doesn't encode this. The right solution is to output the
> score and a confidence estimate separately (e.g., `(cosine_distance, ||μ||)`) and let
> the decision layer threshold accordingly. The score function itself should be left as-is;
> the calibration layer should use `||μ||` or history length as a feature.

---

## Point 5 — Evaluation conflates OOV and clustering signals

**C¹:** The AUC cannot be cleanly attributed to account clustering without an ablation.
A binary "was this token seen in training?" detector might achieve comparable AUC. Without
running it, the PoC doesn't know what it's actually demonstrating.

**D¹:** The hypothesis as stated doesn't require attribution. It claims novel devices
score higher — and they do. Whether the mechanism is OOV detection, account clustering,
or both, the stated claim holds.

**C²:** The attribution matters for what to do next. If it's OOV detection: adding more
training data per account won't help, longer histories won't help, better sequence models
won't help — you've already hit the ceiling. If it's account clustering: all of those
improvements will increase AUC. Building the wrong next experiment because the mechanism
is unconfirmed is a real cost.

**D²:** Fair. And the OOV baseline is a two-line addition — `score = 1.0 if device not
in model.wv else 0.0`. Running it would resolve the attribution question immediately.

**C³:** Then we agree: the ablation is not optional follow-on work. It's a missing
component of the current PoC. Without it, the experiment is incomplete.

**D³:** Conceded. The binary OOV baseline should run alongside the FastText centroid
scorer in the current evaluation, not in a separate experiment. The PoC claims to
validate the FastText approach specifically — if the OOV baseline matches its AUC, that
claim is not supported.

> **Resolution — critique position accepted.** The binary OOV baseline
> (`1.0` if device not in `model.wv`, else `0.0`) must be added to the evaluation as a
> comparison. This is not future work — it's a missing control that determines whether
> the FastText model is adding value over a trivial lookup. If FastText centroid AUC
> significantly exceeds the OOV baseline, the embedding approach is validated. If not,
> the PoC has demonstrated OOV detection works — a useful but weaker finding.

---

## Point 6 — Youden's J threshold fitted on the test set

**C¹:** The threshold is selected by maximizing Youden's J on the same data used to
report precision and recall. This is threshold search on the test set — the reported
numbers are optimistic and don't generalize.

**D¹:** The ROC-AUC is unaffected. The threshold reporting is illustrative only. And
with 45 attack events, splitting further for validation would make threshold estimates
*more* variable, not less.

**C²:** The small sample argument undermines both positions equally. If the sample is too
small to split for validation, it's also too small to produce stable precision and recall
estimates at any threshold. Reporting precision=0.343 and recall=0.800 implies more
precision than the data support. With 45 positives, a 95% CI on recall of 0.800 is
roughly ±0.12 by Wilson interval. The headline number has a ±15% uncertainty that goes
unmentioned.

**D²:** That's correct and worth acknowledging. The fix is simple: either report
bootstrap confidence intervals on precision and recall, or drop the threshold metrics
entirely and report only ROC-AUC, which aggregates over all thresholds and is more
stable at this sample size.

> **Resolution — agreement.** The precision and recall reported at the Youden threshold
> should either be accompanied by bootstrap confidence intervals (to make the uncertainty
> visible) or dropped in favor of reporting ROC-AUC only. With n=45 positives, the
> single-threshold metrics imply false certainty. ROC-AUC is the appropriate headline
> statistic for this sample size.

---

## Point 7 — ROC-AUC vs. PR-AUC and class imbalance

**C¹:** The 6.7:1 imbalance in the synthetic data already understates real-world
conditions. In production, the ratio could be 1000:1. ROC-AUC is insensitive to this;
PR-AUC would reveal the operational cost. A detector with 0.836 ROC-AUC can still have
poor precision at realistic operating points.

**D¹:** ROC-AUC is the correct metric for the question "does this signal discriminate?"
That question must be answered before the operational question. A PoC that gates on PR-AUC
before confirming the signal exists is doing the steps out of order. Class imbalance is a
deployment concern.

**C²:** Agreed that ROC-AUC is the right primary metric for signal validation. The
concern is narrower: the PoC also reports precision/recall at a specific threshold, and
those numbers would collapse at realistic imbalance. A reader might take 34% precision as
a benchmark. It isn't.

**D²:** That's a documentation and framing problem, not a methodology problem. The README
should state explicitly that the threshold metrics reflect the synthetic imbalance ratio
and will not hold at production imbalance. It should also note that precision at fixed
recall (e.g., precision at 90% recall) is the operationally relevant metric.

> **Resolution — easy agreement.** ROC-AUC stays as the primary metric. The precision and
> recall numbers (already flagged in Point 6 as requiring CIs or removal) should be
> additionally caveated: they reflect the synthetic 6.7:1 imbalance, not production
> conditions. A note in the README (or evaluation output) should direct readers to
> consult the ROC curve for the full precision/recall tradeoff and to select an operating
> point based on their actual cost structure.

---

## Point 8 — Synthetic data realism

**C¹:** The i.i.d. history assumption is particularly problematic. Real device usage is
autocorrelated: users have sessions, not random draws. Autocorrelation reduces the
effective sample size for learning device embeddings — a 60-event history with session
autocorrelation provides far fewer independent training signal pairs than 60 i.i.d. draws.
The PoC may be overestimating how much signal is available per account.

**D¹:** The Zipf weighting already introduces marginal autocorrelation — a device used
70% of the time creates runs of the same device. The i.i.d. assumption is already
violated to some degree in the simulation.

**C²:** Zipf weights produce the right *marginal distribution* over devices but not the
right *transition structure*. In reality, if I use device A today I'm very likely to use
it tomorrow — the transition matrix has high diagonal values. Zipf i.i.d. draws would
produce device A 70% of the time but with random interleaving. These are statistically
distinguishable. The effective skip-gram pair count under session structure is much lower
because a run of 10 consecutive A's produces only one informative window: A transitions
to B at the boundary. Under i.i.d. Zipf, those 10 positions produce varied pairs.

**D²:** This is a genuine modeling deficiency that would affect how many training pairs
the model sees per account. But the direction of the effect is important: autocorrelation
reduces effective training pairs, making device embeddings noisier. This would reduce AUC
from the reported 0.836 — the synthetic result is an upper bound on what session-
structured data would produce, not a floor. The hypothesis could still hold; we just need
to test whether it remains useful above the noise floor with realistic session structure.

**C³:** Agreed on the direction. The question is whether the AUC stays meaningfully above
the OOV baseline (from Point 5) under session structure. If yes, the approach is robust.
If the AUC degrades to near-OOV-baseline levels, the clustering signal is fragile.

> **Resolution — agreement with a specified next experiment.** The i.i.d. assumption is
> a real simplification, and the defense correctly characterizes the synthetic result as
> an upper bound. The next data generation step should implement a simple Markov
> device-switching model (high diagonal in the transition matrix) to stress-test whether
> the account-clustering AUC holds under realistic session structure. This is a defined
> phase-2 task, not a PoC blocker — but the README should note the upper-bound
> interpretation of the current AUC.

---

## Point 9 — UMAP visualization

**C¹:** UMAP distances are not monotone with distances in the original 64D space. Account
cluster boundaries in 2D are artifacts of projection parameters. The visualization cannot
validate the hypothesis.

**D¹:** The visualization is labeled qualitative and isn't cited as evidence. The
quantitative validation is the ROC-AUC. A silhouette score computed directly in 64D would
be the correct cluster validation and would likely support the hypothesis given the block-
diagonal co-occurrence structure.

**C²:** Agreed. The UMAP is fine as a sanity check. The claim in the critique is that it
*cannot validate* the hypothesis — not that it's wrong to include. On the silhouette
score: yes, compute it. If the within-account mean cosine distance is substantially less
than the cross-account mean, the clusters are real in 64D. That single number would be
stronger evidence for the hypothesis than the UMAP.

**D²:** Completely agreed. The silhouette score (or equivalently, a direct comparison of
mean within-account distance vs. mean cross-account distance) should be added to the
evaluation. It directly tests whether account clusters exist in the embedding space,
independent of the detection AUC.

> **Resolution — full agreement, concrete addition identified.** Add a cluster validation
> step to the evaluation: compute mean within-account cosine distance vs. mean cross-
> account cosine distance (or a formal silhouette score over account-labeled embeddings
> in 64D). Report the ratio. This directly validates whether the embedding space has
> account-level structure, which is the foundational claim the hypothesis rests on.

---

## Point 10 — Incremental update and the new-device enrollment problem

**C¹:** Newly enrolled legitimate devices are OOV until the model retrains. They receive
the same generic subword embedding as attack devices. In a consumer product, users get
new phones every 12–18 months. At 10% annual device turnover, a meaningful fraction of
legitimate logins would trigger the anomaly signal even with a perfect account-clustering
model. This isn't just a productionization concern — it defines the operating false
positive floor.

**D¹:** The solution is to separate "enrollment" from "anomaly detection." A new device
triggers an enrollment flow (step-up auth, OTP, etc.); after enrollment it joins the
known-device list. The anomaly detector only fires on devices that bypass the enrollment
gate. The two flows are orthogonal.

**C²:** Enrollment separation is a product design decision that's not in scope for the
signal PoC. But the critique is about the signal itself: if the signal fires on all new
legitimate devices, its utility as a *signal* depends entirely on the enrollment
architecture being in place. Without that architecture, the raw false positive rate is
bounded below by the device turnover rate. The PoC should acknowledge this explicitly —
the signal is not "anomalous device" in general; it is "anomalous device that was not
enrolled through the normal enrollment flow."

**D²:** That's a fair reframing. The signal's operational meaning is conditioned on
enrollment infrastructure. Describing it as "novel device signal" without that context
implies broader utility than it has. The README should be explicit: this signal
*complements* an enrollment system; it is not a replacement for one. Used in isolation
it would produce high FPR from legitimate device turnover.

**C³:** Conceded on the scope — this is an operational constraint, not a theoretical
flaw in the signal. But the constraint is load-bearing and belongs in the README's
"known limitations" section, not just in a productionization discussion.

> **Resolution — defense concedes on documentation scope.** The signal's dependence on a
> separate enrollment architecture is a primary operational constraint that must be stated
> in the README's limitations section. The PoC correctly validates that enrolled known
> devices score lower than novel devices — but the system's real-world FPR floor is set
> by whatever fraction of "novel" logins are legitimate new enrollments, not by the
> model's discriminative power alone. Add this explicitly to the README.

---

## Summary of resolutions

| Point | Resolution |
|-------|-----------|
| 1. OOV vs. clustering | **Empirical test:** score cross-account in-vocab devices and compare to OOV novel devices |
| 2. Subword n-grams | **Empirical test:** run with Word2Vec (no subword) to isolate the prefix/n-gram effect on AUC |
| 3. Sparse co-occurrence | **Defense accepted:** negative sampling + pivot transitivity make mechanism sound; stratify AUC by history length |
| 4. Centroid norm | **Both concede:** output `(score, ||μ||)` as separate quantities; let the decision layer use both |
| 5. Conflated signals | **Critique accepted:** OOV binary baseline must be added to the current PoC evaluation |
| 6. Youden's J on test set | **Agreement:** add bootstrap CIs to precision/recall or drop threshold metrics; report ROC-AUC only |
| 7. ROC-AUC vs PR-AUC | **Agreement:** ROC-AUC is correct primary metric; add caveat that precision/recall reflect synthetic imbalance, not production |
| 8. Synthetic data | **Agreement:** i.i.d. is an upper-bound assumption; add Markov session model in phase 2; note this in README |
| 9. UMAP visualization | **Agreement:** add silhouette score in 64D as quantitative cluster validation |
| 10. Enrollment gap | **Defense concedes scope:** add explicit enrollment-dependence caveat to README limitations |

### Concrete additions to the PoC (not future work)

1. **Cross-account in-vocab baseline** — score known devices from other accounts against the target account centroid
2. **OOV binary baseline** — `1.0` if OOV, `0.0` if in-vocab; compare AUC directly against FastText centroid AUC
3. **Word2Vec ablation** — rerun without subword n-grams to isolate the character n-gram contribution
4. **Silhouette score** — mean within-account vs. cross-account cosine distance in 64D
5. **Bootstrap CIs** on precision/recall, or drop threshold metrics in favor of ROC-AUC only
6. **README update** — enrollment-dependence caveat; upper-bound interpretation of synthetic AUC; imbalance caveat on precision/recall
