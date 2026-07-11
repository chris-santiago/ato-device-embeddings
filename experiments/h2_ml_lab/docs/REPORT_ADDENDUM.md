# REPORT_ADDENDUM.md — Production Re-Evaluation

## Overview

The recommendation from Step 8 is to deploy **mean-pool FastText** (sg=1, per-account corpus) for novel and fleet ATO attack detection. Spoof detection requires a different approach (H3). This recommendation is unchanged from the H2_RERUN investigation and is now confirmed by the supplemental experiment.

---

## 1. Retraining Dynamics

FastText retrained monthly on a 6-month rolling window of all login events per account.

**Per-account corpus construction:** All historical events per account are flattened into one training sentence. This requires maintaining per-account event history in a store (e.g., a database table keyed by account_id).

**Critical post-retraining check: Run T8 (token similarity analysis) after every retraining.** Within-feature similarity < 0.9 is required. If collapse is detected (sim >= 0.9), the model is degenerate and must not be deployed — roll back to the prior checkpoint or retrain with verified sg=1 + per-account corpus configuration.

Per-account centroids must be recomputed after every FastText retraining (embeddings are not stable across training runs).

**Warm-start not recommended.** Gensim incremental training can degrade quality. Full retraining on rolling window corpus is safest.

---

## 2. Update Latency

- **Embedding lookup:** 6 token lookups (one per feature) + mean → <1ms per event
- **Centroid update:** After each confirmed-legitimate login, update running mean centroid. Real-time pipeline achievable with message queue.
- **Batch centroid recompute after retraining:** Runs nightly after model update completes.

---

## 3. Operational Complexity

### Infrastructure

```
Nightly (or on event confirmation):
  1. Fetch confirmed legitimate logins per account (rolling 6 months)
  2. For each event: tokenize to 6 feature tokens, look up embeddings
  3. Compute per-account centroid (mean of all event embeddings)
  4. Store in key-value store (Redis/DynamoDB) keyed by account_id

At login time (real-time, <1ms):
  device_tokens = [f"{feat}_{val}" for feat, val in device_features.items()]
  device_vec = mean([fasttext_model[t] for t in device_tokens])
  centroid = account_centroid_store[account_id]
  risk_score = cosine_distance(device_vec, centroid)
  → step-up auth if risk_score > threshold

Monthly:
  Retrain FastText on 6-month rolling corpus (per-account sentences)
  Run T8: verify within-feature sim < 0.9
  Recompute all centroids from stored event embeddings
  Run embedding health check on confirmed-legitimate holdout
```

### Jobs and Cadence

| Job | Cadence | Trigger condition |
|-----|---------|------------------|
| FastText retraining | Monthly | New vocabulary coverage drops below 95% |
| T8 collapse check | After every retraining | Mandatory before deploying new model |
| Centroid bulk recompute | After every retraining | Mandatory before deploying new model |
| Centroid incremental update | Per-confirmed-login | User completes step-up auth |
| Embedding health check | Daily | Monitor mean cosine distance on legitimate holdout |

---

## 4. Failure Modes

- **Within-feature collapse:** The primary production risk unique to mean-pool. If training configuration drifts (e.g., library default changes, corpus construction bug), collapse silently destroys spoof detection while novel/fleet AUC appears healthy. T8 monitoring after every retraining is the prevention. Under the degenerate config (CBOW, per-event), within-feature sim reaches 0.999; under the robust config (sg=1, per-account), it is 0.392.
- **False negatives on spoof:** Mean-pool spoof AUC 0.869 beats the trivial baseline (0.750) but the margin is thin (+0.119). Set operational thresholds conservatively and monitor spoof false negative rate via honeypot accounts or red team events.
- **Cold-start:** Accounts with fewer than 20 confirmed events have unreliable centroids. Fallback: exact set-membership check for accounts below the threshold.
- **OOV tokens:** Mean-pool handles OOV gracefully — if a feature value is unseen, FastText falls back to character n-gram subword embedding, which degrades gracefully. Concat is more brittle: the full concatenated string is OOV, triggering more complex subword computation.

---

## 5. Revised Recommendation

The robust-config experimental recommendation stands. No ranking inversion under production analysis.

**Deploy: Mean-pool FastText (sg=1, per-account corpus, epochs=20, min_n=3, max_n=6) for novel and fleet detection.**
- Retraining: monthly, 6-month rolling window, per-account sentences
- Post-retraining: T8 collapse check mandatory (within-feature sim < 0.9)
- Centroid update: incremental after confirmed-legitimate login; bulk after retraining
- Fallback: exact set-membership (O(1)) for accounts with < 20 events or when embedding service unavailable
- Monitoring: daily health check on legitimate holdout; OOV rate alert at 5%; T8 collapse check monthly

**Do not deploy:** Any embedding-based spoof detector without first confirming that within-feature collapse has not occurred (T8 check). Both mean-pool and concat fail to beat the trivial baseline on spoof under the current data model when embeddings are degenerate; mean-pool beats it by +0.119 under the robust config.

**Spoof detection gap:** Mean-pool spoof AUC 0.869 exceeds the trivial baseline but by a narrow margin. Spoof detection requires per-feature-dimension anomaly detection (H3) for reliable production performance.

---

## 6. Open Questions

1. **Spoof detection (H3):** Per-feature anomaly scoring — one score per feature dimension, aggregated by max rather than mean distance — as the candidate approach. The T4 finding (tz-attr 0.028 for mean-pool under robust config) suggests that per-dimension scoring would improve spoof detection over centroid distance.
2. **T8 monitoring threshold:** Is within-feature sim < 0.9 the right threshold? Testing with varying degrees of intentional config degradation would calibrate this.
3. **Account history depth:** How many historical events should contribute to the centroid? Recency-weighted centroid vs flat mean is untested.
4. **Multi-device accounts:** Users with 4+ active devices have higher centroid spread. Personalized thresholds may be needed.
5. **Adaptive attackers:** If an attacker learns the scoring system, they can sample the victim's primary timezone. Mean-pool's +0.119 advantage over the trivial baseline is the operational gap available to defend.
