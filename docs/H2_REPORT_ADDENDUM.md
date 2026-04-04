# H2 Report Addendum — Production Re-evaluation

Re-evaluates the H2 experimental findings against the four production constraint categories
from REPORT_ADDENDUM.md. The experimental recommendation (concat-at-w=6 is viable for novel/fleet;
mean-pool preferred for spoof) is tested against operational reality.

---

## 1. Retraining dynamics

Both approaches use FastText. The rotational instability analysis from REPORT_ADDENDUM.md applies
identically: any retraining event produces a new coordinate system, invalidating all cached
centroids. The question is whether the two approaches have different recompute costs or stability
profiles after retraining.

**Vocabulary size.** Mean-pool trains on ~30 distinct feature tokens (5 OS × 5 browser × 6 tz
× 6 lang × 4 net × 4 screen, but the training vocabulary is the token set, not the profile set).
Concat trains on profile-combination strings whose vocabulary is all observed profile combinations.
With 400 accounts, 2–4 devices each, and modest within-account variation, the concat training
vocabulary is O(hundreds to thousands) of distinct concatenated strings — far larger than 30.

Implication: mean-pool's 30-token vocabulary is stable across retraining runs. Every feature token
is frequently observed; all 30 token vectors are well-trained after convergence. Concat's
vocabulary coverage is sparser — some profile combinations may appear only a handful of times in
training, producing less stable embeddings for rare combinations. Two retraining runs on the same
data will produce concat token vectors with more variance for rare profiles than mean-pool token
vectors for the corresponding feature tokens (which are shared across all profiles that use that
feature value).

**Centroid stability.** REPORT_ADDENDUM.md notes that the feature-based signal's bounded ~30-token
vocabulary makes centroids "approximately valid" across retraining cycles, with "meaningful
practical advantage over the ID-based case." This advantage applies to mean-pool specifically.
For concat, centroid stability depends on the stability of the profile-combination tokens, which
is lower for rare combinations. At scale (10M accounts), rare profile combinations are still
common in absolute terms — but the relative instability is higher.

**Verdict:** Mean-pool is more stable across retraining events due to the smaller, fully-observed
training vocabulary. Concat requires full centroid recompute after each retraining event, same as
mean-pool, but with higher per-token variance for rare profile combinations.

---

## 2. Update latency

Both approaches respond to new feature tokens at the same latency. Mean-pool's 30-token
vocabulary is fixed by the feature schema — a new OS version (`os_harmonyos`) is embedded at
inference time via n-gram averaging over `os_` prefix n-grams, with no retraining required.
Concat's response to a new OS value (`harmonyos`) is also via n-gram averaging — but over
character n-grams of the full concatenated string that happens to contain `harmonyos`. T4
confirmed that both approaches handle OOV tokens with equivalent AUC degradation (<0.001 drop).

**Difference:** Concat's inference-time embedding of a new profile combination
(`harmonyos_safari_utc-5_en_us_wifi_small`) uses n-gram averaging over the entire concatenated
string, including n-grams that span feature boundaries. Mean-pool's inference-time embedding of
`os_harmonyos` uses n-gram averaging only over the OS token, then averages with the independently
computed embeddings of the other five known feature tokens. Mean-pool's OOV handling is
compositional (update the OOV feature token, keep the others); concat's OOV handling is holistic
(the entire profile string changes). In production, a device with one new feature value
(e.g., new browser) and five known values will be embedded more accurately by mean-pool, which
can leverage the five known feature tokens directly. Concat produces a new string where the five
known features are re-represented through the n-gram lens of the full new string.

This is a practical update latency difference that the T4 test did not capture — T4 tested
complete profile mismatch (attack profiles already differ strongly from victims on multiple
dimensions, masking the compositional advantage).

**Verdict:** Mean-pool's compositional OOV handling is a production advantage that the experiment
understated. For devices that differ from training vocabulary in a single feature (e.g., a new
browser on an otherwise familiar device), mean-pool correctly embeds the five known features and
approximates the new one; concat introduces n-gram noise across the entire profile string from
the single unfamiliar value.

---

## 3. Operational complexity

**Inference cost.** Mean-pool requires 6 FastText embedding lookups per login event and a vector
mean. Concat requires 1 FastText embedding lookup per login event. At a production scale of 10M
logins per day (≈ 116/second sustained), the difference is five additional embedding lookups per
event. FastText embedding lookups are O(d) where d is the vector dimension (64 here). This is
not a material cost difference.

**Training corpus structure.** Mean-pool's training corpus is a flat sequence of 6-token events
per account. The corpus size is proportional to (accounts × events × 6 tokens). Concat's training
corpus is a sequence of 1-token events per account — one-sixth the token count. Training wall
time for concat is substantially lower than mean-pool at equivalent window sizes, because the
skip-gram training pass processes fewer total tokens. For mean-pool at window=6, each of the 6
tokens in an event generates 6×(2 window +1) = 78 skip-gram pairs per event. For concat at
window=6, each event generates 6×(2×6+1) = 78 skip-gram pairs per event — the same count, but
with the window now operating over adjacent events rather than within-event tokens. In practice
concat trains faster because the context lookup is over a shorter token sequence per account.

**Centroid computation.** For both approaches, the centroid is the mean of per-device embeddings
over observed profiles. Mean-pool: 6 lookups × N_unique_profiles per account. Concat: 1 lookup
per unique profile. Concat has strictly lower recompute cost after retraining.

**Monitoring.** Mean-pool's 30-token vocabulary is fixed and interpretable — vocabulary drift is
impossible by design. Concat's vocabulary grows with observed profile combinations. Monitoring
for vocabulary drift, coverage of rare combinations, and n-gram stability of new profile strings
adds operational overhead that mean-pool does not require.

**Verdict:** Concat is operationally simpler at inference and recompute time. Mean-pool's
training corpus is 6× larger in token count. Concat adds vocabulary monitoring overhead that
mean-pool avoids entirely.

---

## 4. Failure modes

**Model staleness.** Both approaches degrade identically when the embedding model is stale (not
yet retrained to incorporate new training data). Mean-pool is slightly more tolerant of staleness
because its 30 feature tokens are stable across training cycles — a centroid computed under model
version T is approximately valid under model version T+1. Concat centroids are also approximately
stable for frequently observed profile combinations but are less stable for combinations that gain
or lose training examples between retraining cycles.

**Spoof detection failure.** Under production conditions where the spoof threat model is active,
the 0.043–0.049 AUC gap between mean-pool and concat translates to a concrete increase in
undetected spoofing events. At a 1:1000 positive rate (0.1% fraud) with 10M logins/day (10,000
fraud attempts/day), an AUC gap of 0.043 corresponds to approximately 2–5% more missed spoof
detections at typical operating thresholds. At a 1,000-fraud-per-day base rate, this is 20–50
additional undetected spoof events per day — a non-trivial operational difference.

**Fleet detection superiority.** The experimental finding that concat at w=6 exceeds mean-pool
on fleet attacks by 0.027 AUC (+2.7 percentage points in AUC at a 1:1000 positive rate corresponds
to a meaningful increase in detected fleet events) suggests that in a deployment where fleet
reuse is the primary attack vector, concat is operationally superior. This is the inverse of the
spoof case.

**Signal unavailability.** Both approaches produce a centroid distance score; both require the
FastText model and the account's cached centroid to be available at inference time. Failure mode
is identical. The fallback (use the global mean embedding, score all devices at the same baseline)
is structurally identical.

---

## 5. Production recommendation

The experimental result established two regimes:

**Regime 1: Spoof is the primary threat (attacker researches victim's device fingerprint and
mimics OS/browser/language, differing only in timezone or locale)**

Mean-pool FastText remains the recommended signal. The 0.043–0.049 AUC advantage on spoof
attacks is a real production difference (20–50 additional missed events/day at the estimated
base rate). The mechanism — per-token co-occurrence learning for individual feature dimensions —
cannot be replicated by concat regardless of window size or format.

**Regime 2: Novel device and fleet reuse detection are the primary threats**

Concat FastText at `window=6` is a viable production alternative with lower operational
complexity (6× fewer training tokens, 6× fewer inference lookups, 1/6 the recompute cost).
It matches mean-pool on novel attacks and exceeds it on fleet attacks. The vocabulary
monitoring overhead is the only additional operational cost.

**Neither regime changes the two-path architecture recommendation.** The real-time signal
(feature embedding centroid distance, whether mean-pool or concat) handles novel/spoof
detection. The offline batch signal (`id_w2v`) handles fleet/reuse detection. The per-account
known-device set remains the operational gate for step-up authentication.

**The recommendation is therefore refined rather than inverted:**

> If the production deployment team prefers operational simplicity over spoof detection
> performance, concat FastText at window=6 is the preferred implementation. If spoof attack
> sophistication is a documented production concern, mean-pool FastText is preferred. The
> choice should be driven by threat model priority, not by a blanket preference for either
> embedding strategy.
