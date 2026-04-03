# Critical Analysis: ATO FastText PoC

A systematic review of theoretical and implementation weaknesses. Issues are grouped by
root cause, not severity, because most of the implementation flaws trace back to a small
number of foundational choices.

---

## 1. The actual detection mechanism differs from the stated hypothesis

**Stated:** FastText will embed known devices near their account's centroid because they
co-occurred in the same account's history.

**What's actually happening:** The useful signal is not account-level clustering — it is
trained-token vs. untrained-token distinction. Novel devices are OOV. Their embedding is
computed purely from character n-gram vectors, which sum to something close to the global
mean of the subword embedding space (see §2). Known devices were trained on the skip-gram
objective and their vectors were pulled toward co-occurring devices. That pull exists, but
the dominant effect is simply that trained vectors ≠ untrained vectors.

This is a subtler distinction because it means the approach is detecting out-of-distribution
tokens, not account membership specifically. Consequences:

- An account with thin history (1–2 known devices appearing rarely) will have poorly
  trained embeddings for its own devices. Those devices' vectors will also be close to the
  global mean, producing high cosine distance from the centroid — false positives from
  legitimate thin-account logins.
- A sufficiently frequent device from a completely different account will have a
  well-trained vector that lands somewhere in the global space. If that space is
  unstructured (no tight account clusters), cosine distance to any given account's centroid
  is essentially arbitrary.

The PoC's 0.836 AUC is real, but it is measuring OOV detection with a thin account-
clustering wrapper, not the stronger claim that device embeddings form interpretable
per-account neighborhoods.

---

## 2. Subword n-grams actively undermine the hypothesis

FastText decomposes each token into character n-grams and represents it as the sum of
their embeddings. For a token like `dev_x7k2mq9pzrn1wyj4`:

- The prefix `dev_` appears in **every** device ID in the corpus. Its n-gram embedding
  becomes a strong global attractor shared by all devices. The "dev_" n-gram vectors will
  be trained to average over the co-occurrence context of every device ID across all
  accounts — a meaningless centroid.
- The remaining 16 characters are random alphanumeric. Two random strings of that length
  share character n-grams purely by chance. The expected n-gram overlap between any two
  random device IDs is determined by alphabet size and string length, not by account
  membership.

The practical consequence: OOV novel devices will receive embeddings that are sums of
random n-gram vectors plus the dominant `dev_` component. As string length grows, by the
central limit theorem, the random component averages out and the OOV embedding converges
toward the `dev_` n-gram contribution — a fixed vector approximately equal for *all*
novel devices. They will cluster together near the global mean, not scattered randomly.

This has two effects on the experiment:

1. **Novel devices are systematically similar to each other.** The PoC detects "this looks
   like no known device in the corpus" rather than "this device is far from this account's
   cluster." Swapping FastText for a baseline of "was this token seen during training?"
   would produce comparable or better AUC with no embeddings at all.

2. **The subword mechanism provides zero benefit for this data.** The README correctly
   notes this, but the implication is stronger: subword n-grams are not merely neutral
   here — they introduce noise into the in-vocabulary embeddings by pulling all known
   devices toward the same `dev_` n-gram attractor, reducing within-account cluster
   tightness.

---

## 3. Skip-gram co-occurrence signal is weaker than assumed

Word2Vec-style objectives learn from token co-occurrence within a sliding window. For
this to create tight per-account clusters, two conditions must hold:

- **High within-account co-occurrence:** devices from the same account appear near each
  other frequently.
- **Low cross-account co-occurrence:** devices from different accounts never appear in
  the same sentence. ✓ (guaranteed by construction)

The first condition is variable and often weak. With `window=5` and a Zipf-distributed
history where the primary device dominates:

```
A A B A A A C A A A B A A D A A A A B A ...
```

Within any window of 5, most pairs are (A, A) or (A, B). Devices C and D may never
co-occur within any window at all. Their embeddings are shaped primarily by co-occurrence
with A — not with each other — giving them a noisy centroid contribution.

With `history_len=(25, 80)` and `window=5`, the effective co-occurrence graph is sparse.
For an account with 8 devices and a 30-event history, many device pairs will never appear
as skip-gram training pairs.

---

## 4. The centroid is not a principled cluster representative

**Frequency weighting is implicit, not deliberate.** `account_centroid` receives the
full corpus history including repeats:

```python
centroid = account_centroid(model, corpus[event.account_id])
```

A device appearing 40 times contributes 40 identical vectors to the mean. This makes the
centroid a frequency-weighted mean — but the weights are login counts, not embedding
uncertainty. A device seen 3 times gets weight 3; a device seen 40 times gets weight 40.
The high-frequency device likely has a better-trained embedding (more gradient updates),
but that's incidental. The choice between frequency-weighted and equal-weight centroids
should be explicit and motivated.

**The centroid norm carries information that is discarded.** For a set of unit vectors
{v₁, …, vₙ}, their mean μ has norm

```
||μ|| = ||Σvᵢ|| / n
```

which equals 1 when all vectors are identical and approaches 0 as they become orthogonal.
This norm is a natural measure of cluster coherence. An account with tight, consistently-
used devices will have ||μ|| ≈ 0.9; an account with diverse, poorly-trained device
embeddings will have ||μ|| ≈ 0.1. Cosine distance from a new point to μ ignores this
entirely. A better anomaly score would incorporate cluster coherence:

```
score = cosine_distance(x, μ) / (||μ|| + ε)
```

This penalizes comparisons against diffuse centroids (correctly widening the confidence
interval for uncertain accounts) rather than treating all centroids equally.

**The centroid is not robust to poor embeddings.** Poorly-trained devices (seen once or
twice) will have embeddings close to random initialization scaled by a few gradient steps.
Including them in the centroid adds noise proportional to the fraction of rare devices.

---

## 5. Evaluation confounds two phenomena

The test set consists of:
- **Legit events:** known devices, trained with skip-gram in their account's context
- **Attack events:** novel devices, fully OOV, embedded via n-gram averaging

The measured distance gap between these two groups conflates:

1. The skip-gram account-clustering effect (what we want to measure)
2. The trained-vs-OOV distinction (a near-trivial baseline)

To isolate the account-clustering signal, a proper ablation would score known devices
that were *excluded from training* for their account (seen by the account but withheld).
If the AUC for those events is still high (low distance from centroid), then the model has
genuinely learned account structure. If it drops, the signal is primarily OOV detection.

This ablation was considered and rejected in this project because it conflicts with the
production scenario (where centroids are built from all known history). That's a
reasonable product decision — but it means the 0.836 AUC cannot be cleanly attributed to
the embedding model vs. a trivial "seen this token before" lookup.

---

## 6. Youden's J threshold is fit on the test set

```python
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
thresh = thresholds[best_idx]
preds = (scores >= thresh).astype(int)
```

The threshold is selected to maximize Youden's J (`tpr - fpr`) on the same set that
precision and recall are then computed against. This is equivalent to exhaustive
threshold search on the test set. The reported precision (0.343) and recall (0.800) are
therefore optimistic — there is no guarantee this threshold generalizes.

The ROC-AUC is unaffected (it's threshold-free), but any single-threshold metric should
be reported on a held-out validation split or under cross-validation.

---

## 7. Class imbalance makes ROC-AUC a misleading headline metric

300 legit events vs. 45 attack events is a 6.7:1 imbalance. ROC-AUC is insensitive to
class imbalance by construction — it integrates over all thresholds, weighting TPR and
FPR equally regardless of class frequency.

In production, the operating regime is extreme imbalance: a typical account may have
thousands of legitimate logins and one attack. Precision-Recall AUC (PR-AUC) is the
appropriate metric here. A classifier that catches 80% of attacks at 34% precision would
generate enormous false positive volume in a real system (for every real ATO caught, two
legitimate users are flagged).

---

## 8. Synthetic data is too favorable

**i. No cross-account device sharing.** In production, common devices include company
laptops, shared household devices, corporate VPNs appearing as the same IP → device
fingerprint. Devices shared across accounts will have embeddings pulled toward a global
centroid rather than any single account's centroid, making them look anomalous for all
accounts that use them — a systematic source of false positives not present in the
synthetic data.

**ii. No adversarial accounts.** A sophisticated attacker using credential stuffing will
log into many accounts from the same device fleet. Those attacker devices will eventually
accumulate co-occurrence signal with each other if any attacked account has prior
compromised history. The model provides no defense against repeated use of the same
attack device.

**iii. History is i.i.d. across time.** Each event is drawn independently from the
usage distribution. Real login sequences are autocorrelated: if you use device A today,
you will use it tomorrow. Autocorrelation means the effective number of independent
training examples is less than the sequence length, making the learned embeddings noisier
than they appear.

**iv. No device retirement or account dormancy.** A device last seen 3 years ago is
indistinguishable from one seen yesterday. Real ATO risk is higher on dormant accounts
where the centroid is stale.

---

## 9. UMAP visualization cannot validate the 64-dimensional hypothesis

The UMAP plot shows 2D projections of 64D embeddings. UMAP optimizes a 2D layout that
preserves approximate local neighborhood structure, but it is a non-linear dimensionality
reduction that necessarily distorts global geometry. Specifically:

- **Distances in UMAP space are not monotone with distances in the original space.**
  A point that looks isolated in 2D may be surrounded by neighbors in 64D and vice versa.
- **Account cluster boundaries are artifacts of projection parameters** (`n_neighbors`,
  `min_dist`, random seed). The same 64D data can produce dramatically different 2D
  layouts under different UMAP hyperparameters.

The visualization is useful for qualitative sanity-checking but should not be cited as
evidence that the hypothesis holds. The correct way to validate account-level clustering
is to compute intra- vs. inter-account cosine distances directly in 64D and compare their
distributions (e.g., a silhouette score over account-labeled devices).

---

## 10. The model is a global snapshot incompatible with production update patterns

FastText is trained once on the full corpus. In production:

- New accounts are created daily. Their devices have no embedding.
- New devices are enrolled on existing accounts. The centroid must update without
  retraining the global model.
- The model must be retrained periodically to incorporate new vocabulary.

The centroid can be updated incrementally (it's a running mean), but the embedding for
any new device relies on the subword n-gram fallback — exactly the mechanism shown above
to produce generic "global mean" vectors. This means newly enrolled legitimate devices
will score similarly to attack devices until the model is retrained, producing false
positives during the retraining gap.

---

## Summary table

| # | Issue | Root cause | Impact on AUC |
|---|-------|-----------|---------------|
| 1 | Mechanism is OOV detection, not account clustering | Token vocabulary structure | Inflates AUC; conflates two signals |
| 2 | Subword n-grams on random strings add noise, not signal | Data design | Reduces cluster tightness |
| 3 | Sparse co-occurrence for secondary devices | Window × history mismatch | Noisy embeddings for rare devices |
| 4 | Centroid norm (cluster coherence) discarded | Score design | Inflates AUC for diffuse accounts |
| 5 | No ablation separating OOV signal from clustering signal | Evaluation design | AUC unattributable |
| 6 | Threshold selected on test set | Evaluation procedure | Precision/recall optimistic |
| 7 | ROC-AUC reported instead of PR-AUC | Metric choice | Masks true operational cost |
| 8 | Synthetic data lacks adversarial realism | Data design | AUC overstated vs. real distribution |
| 9 | UMAP cited as validation of 64D structure | Visualization misuse | Does not validate hypothesis |
| 10 | No incremental update path | Architecture | Not a critique of the experiment per se; blocks productionization |

---

## What would actually validate the hypothesis

1. **Silhouette score on account-labeled device embeddings in 64D** — does within-account
   distance < cross-account distance? If yes, account clusters exist.

2. **Ablation: OOV baseline vs. embedding model.** Compare against a detector that
   simply returns 1.0 for OOV tokens and 0.0 for in-vocabulary tokens. If the FastText
   centroid distance AUC is not meaningfully higher than this trivial baseline, the
   embeddings are not adding value beyond vocabulary membership.

3. **Thin-account FPR analysis.** Compute false positive rate stratified by history
   length. If thin-account FPR is much higher than rich-account FPR, the centroid
   estimate is the bottleneck, not the embedding model.

4. **PR-AUC at realistic class imbalance** (e.g., 1000:1 legit:attack). Report
   precision at fixed recall levels (90%, 95%) rather than Youden's J threshold.

5. **Cross-account device test.** Inject a device that appears in two different accounts'
   histories. Verify it does not systematically score as anomalous on both.
