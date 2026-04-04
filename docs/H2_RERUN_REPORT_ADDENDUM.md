# H2 Rerun — Production Re-Evaluation Addendum

This addendum evaluates whether the experimental recommendation (use mean-pool
FastText over concat FastText for ATO device fingerprinting) holds under
real-world production constraints.

**Short answer:** The recommendation holds. There is no production constraint
that favors concat over mean-pool. Mean-pool is actually *easier* to deploy in
several production-relevant ways that the experiment did not measure. One
constraint — cold-start on new accounts — affects both signals equally and
warrants a separate architecture decision.

---

## P1 — Inference latency

**Experiment design:** Silhouette and AUC were measured; inference time was not.

**Production concern:** Mean-pool FastText requires 6 vocabulary lookups and
one mean-pool operation per inference. Concat FastText requires 1 vocabulary
lookup. At millions of logins per day, this 6x lookup cost may be prohibitive.

**Assessment: Not a meaningful constraint in practice.**

FastText vocabulary lookup is O(1) hash table access. For VEC_DIM=64, each
lookup returns a 64-float vector. Six lookups and one mean of six 64-float
vectors requires approximately 6 × 64 multiplications + 64 additions = ~450
floating-point operations. At modern CPU throughput (~10^9 FLOP/s), this is
~450 nanoseconds per device. A full login event including network round-trip
dominates by 5–6 orders of magnitude. **Latency is not a constraint.**

If FastText is deployed as a shared vocabulary file (typical pattern), the
6 token lookups also benefit from cache locality — the feature tokens
(os_ios, browser_safari, etc.) are a small vocabulary that will reside in L1
cache after warm-up.

**Verdict:** Mean-pool does not lose to concat on latency. The recommendation
holds.

---

## P2 — Vocabulary management

**Experiment design:** A single FastText model is trained once on synthetic data.

**Production concern:** Real deployments encounter new feature values over time
(new OS versions, new browser strings, new timezone identifiers). FastText handles
OOV tokens via subword n-grams — it computes an OOV vector from character n-grams
of the unseen token. How does this OOV handling compare between mean-pool and concat?

**Mean-pool OOV:** If a new OS value `harmonyos` appears, `os_harmonyos` is an OOV
token. FastText constructs its vector from n-grams: `os_`, `os_h`, `_ha`, `har`, etc.
These n-grams overlap with similar tokens (`os_android`, `os_ios`) providing a
reasonable embedding. The other 5 feature tokens are in-vocabulary and contribute
correctly to the mean.

**Concat OOV:** If `harmonyos` appears in `harmonyos_safari_utc-5_en_us_wifi_small`,
the entire string is OOV. FastText constructs the full-string vector from n-grams,
including cross-boundary ones. The OOV vector is derived from character patterns
spanning the entire concatenated string — a much noisier basis for generalisation.

**Assessment:** Mean-pool has better OOV resilience than concat. A single new
feature value contaminates the entire concat vector but only one of six mean-pool
inputs. This is an additional advantage for mean-pool not captured in the experiments
(the experiment used a fixed vocabulary with no OOV injection, per the deliberate
exclusion noted in the PoC).

**Verdict:** Mean-pool is more robust to vocabulary drift. The recommendation is
strengthened for production use.

---

## P3 — Model retraining and embedding stability

**Experiment design:** A fresh model is trained on the full dataset; centroids are
computed from the trained model. Retraining is implicit.

**Production concern:** FastText model weights change with each retraining. When
weights change, centroid vectors computed under the old model are no longer comparable
to embeddings computed under the new model. This rotational instability problem was
identified in the prior investigation (see REPORT_ADDENDUM.md) and is a known issue
with all embedding models.

**Does this affect mean-pool and concat differently?**

Both signals use the same FastText architecture and suffer from the same rotational
instability. The issue is model-agnostic — the critical question is how fast centroids
must be invalidated when weights change.

Mean-pool centroids may be *more* stable across retraining events than concat centroids
for the following reason: mean-pool centroids are averages of 6 token vectors. After
retraining, the new embedding space is a rotation+rescaling of the old space (because
word2vec/FastText objectives are rotation-invariant). The mean-pool centroid, being
an average of vectors that individually rotate, will tend to rotate coherently with
the new space. Concat centroids (one vector per device) have no averaging to smooth
rotational variance.

**Assessment:** This is a theoretical argument, not an empirical one. The practical
mitigation (Procrustes alignment or full centroid recomputation on retraining) applies
equally to both signals. This constraint does not change the relative recommendation.

**Verdict:** Mean-pool may be marginally more robust to retraining instability, but
the constraint applies to both signals and does not change the recommendation.

---

## P4 — Cold-start on new accounts

**Experiment design:** All accounts have 60 training events before evaluation.

**Production concern:** New accounts have zero or very few login events. The centroid
is uninformative. Both signals will perform poorly.

**Assessment:** This constraint affects both signals identically — it is a function
of the centroid architecture, not the embedding approach. The cold-start problem
requires a separate architectural decision (e.g., step-up auth for all logins until
N events are observed, where N is tuned to balance security and friction).

The prior investigation recommended combining the embedding-based signal with a
per-account known-device set membership check (operational gate) for new accounts.
This recommendation is unchanged and applies regardless of whether mean-pool or
concat is used.

**Verdict:** Cold-start is a real constraint but does not differentiate mean-pool
from concat. The recommendation is unchanged.

---

## P5 — Feature availability and encoding consistency

**Experiment design:** All 6 features are always present. Feature values are clean
and canonical.

**Production concern:** In production, features may be missing (e.g., timezone
unavailable in some browsers), or values may be non-canonical (e.g., timezone
offsets with and without leading zeros, browser user-agent strings that vary by
patch version).

**Mean-pool behavior when features are missing:** If a feature is unavailable,
its token can simply be omitted from the mean-pool average. The remaining K tokens
(K < 6) are averaged. The dimensionality of the embedding is unchanged; the
centroid computation is degraded by one fewer feature but remains valid.

**Concat behavior when features are missing:** A missing feature in a concatenated
string requires a placeholder (e.g., `unknown`) or a skip of the delimiter. Either
choice creates a different token string. The `unknown` placeholder becomes a
high-frequency noise token across all missing-feature events, pulling centroids
toward a common "unknown" direction. Skipping creates a token collision (different
missing features produce the same string if the same features are present).

**Assessment:** Mean-pool is substantially more robust to missing features in
production. Concat requires a careful encoding strategy for missing values; mean-pool
handles them naturally by averaging over available features.

**Verdict:** This is an additional production advantage for mean-pool not captured
in the experiments. The recommendation is further strengthened.

---

## P6 — Model serving infrastructure

**Experiment design:** Single Python process with gensim FastText loaded in memory.

**Production concern:** Serving at scale typically uses a model server (TensorFlow
Serving, TorchServe, Triton, or a custom gRPC service). The FastText vocabulary file
must be loaded and served.

**Mean-pool serving:** The service receives a login event as 6 key-value pairs.
It looks up 6 tokens in the vocabulary and returns their mean. This is a stateless,
embarrassingly parallelisable operation. No model inference graph — just 6 hash
lookups and a vector mean.

**Concat serving:** The service receives a login event, concatenates the values into
a single string, and looks up the string in the vocabulary. If the string is OOV
(which is more likely than for any individual feature token, since the full string
space is exponentially larger), it triggers subword n-gram computation. Concat
serving has a longer OOV code path than mean-pool.

**Assessment:** Mean-pool is simpler to serve and has a more predictable latency
distribution (no OOV n-gram fallback for most lookups). Concat has a heavier tail
on inference latency due to OOV n-gram computation.

**Verdict:** Mean-pool is operationally preferable at scale.

---

## Summary

| Constraint | Affects mean-pool? | Affects concat? | Changes recommendation? |
|-----------|-------------------|----------------|------------------------|
| P1 — latency | No (negligible) | No | No — mean-pool unchanged |
| P2 — OOV / vocab drift | Less than concat | More than mean-pool | No — strengthens mean-pool |
| P3 — retraining stability | Yes (both) | Yes (both) | No — applies equally |
| P4 — cold-start | Yes (both) | Yes (both) | No — separate architecture |
| P5 — missing features | Handles gracefully | Requires placeholder strategy | No — strengthens mean-pool |
| P6 — serving complexity | Lower | Higher (OOV tail) | No — strengthens mean-pool |

**The production re-evaluation does not invert the experimental recommendation.**
It strengthens it: mean-pool FastText is easier to deploy, more robust to
vocabulary drift and missing features, and has lower inference latency variance
than concatenated-string FastText.

---

## Final recommendation (production-qualified)

**Deploy mean-pool FastText for real-time device fingerprint scoring in ATO
detection pipelines.**

Architecture:
```
Inference (per login event, < 1ms):
  device_features = {os, browser, tz, lang, net, screen}
  available_tokens = [f"{k}_{v}" for k, v in device_features.items() if v is not None]
  device_vec = mean([fasttext_vocab[t] for t in available_tokens])
  centroid = account_centroid_store.get(account_id, global_fallback)
  risk_score = cosine_distance(device_vec, centroid)
  → flag if risk_score > threshold

Training (batch, weekly or on significant traffic growth):
  sentences = [flatten(account.feature_corpus) for account in accounts]
  fasttext_model.train(sentences, window=6)
  centroids = {acct: mean([embed(p) for p in acct.observed_profiles]) for acct in accounts}

Cold-start mitigation:
  if account.login_count < N_min:
    → apply step-up auth regardless of risk score
    → update centroid after each confirmed-legitimate login

Missing feature handling:
  → embed only available features; do not include placeholder tokens
```

The AUC on spoof attacks (0.818 with enrollment in the negative class, CI:
[0.76, 0.86] estimated from bootstrap results) is the operational figure to track.
Spoof is the hardest attack type and the most realistic near-term attacker
capability. Novel attack AUC (0.993) is less informative operationally — even the
trivial baseline achieves 0.79.

**Do not deploy concatenated-string FastText.** It fails to beat the trivial baseline
on spoof attacks, produces worse cluster structure, and is harder to maintain as the
feature vocabulary evolves.
