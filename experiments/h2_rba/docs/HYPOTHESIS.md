# H2-RBA Hypothesis

> **Note:** This document was revised post-data-inspection. The pre-registered plan used
> an 80/20 temporal split and ≥10 training events per user. After observing that all 141 ATO
> events occur before the 70th percentile, the split was changed to 50/50 and the training
> floor was lowered to ≥5 events. See REPORT.md §2 for the full rationale.


**Claim:** The core H2 finding — that FastText skip-gram mean-pool centroid scoring
outperforms a trivial set-membership baseline at detecting account takeover events —
will replicate on the real-world RBA dataset.

## Background

The H2 ml-lab round (`h2_ml_lab/experiments/robust_config_experiment.py`) established
that mean-pooling per-feature tokens (Skip-gram sg=1, per-account corpus) produces
embeddings where cosine distance from a user's centroid reliably flags spoofed logins,
beating exact device-fingerprint matching on the hardest attack type.

All H2 evidence came from fully synthetic data: 400 accounts, 6 closed-vocab categorical
features, hand-crafted novel/fleet/spoof attack labels. The winning metric (spoof AUC 0.82
vs. trivial 0.50) depended on clean label construction and a controlled feature space.

## Replication target

- **Dataset:** DAS Group RBA dataset v1.0.0 — synthesized Norwegian SSO login logs,
  31,269,264 events, 4,304,857 users, with per-login `Is Account Takeover` ground truth
  derived from real incident response data.
- **Features:** `os`, `browser`, `device_type`, `country`, `region`, `asn_bucket`,
  `rtt_bucket` (schema-driven; values are open-vocabulary from the real dataset).
- **Label:** binary `Is Account Takeover` — no novel/fleet/spoof trichotomy.
- **Split:** global chronological cutoff at 50th timestamp percentile; training requires
  ≥ 5 events per user. (Original plan used 80th percentile and ≥10 events, but all 141
  ATO events occur before the 70th percentile, so the 80/20 split would leave zero ATO
  events in the test window. The 50/50 split yields 34 ATO users with test-window events,
  of which 9 pass the training floor.)

## Success criterion

**Primary:** Mean-pool ROC-AUC > trivial set-membership ROC-AUC on ATO test events,
with non-overlapping 95% bootstrap CIs.

**Secondary:** Mean-pool PR-AUC > trivial baseline PR-AUC (PR-AUC is the more honest
metric given severe class imbalance in real ATO data).

**Null result:** If mean-pool AUC ≤ trivial baseline AUC, the H2 finding does not
transfer to real data. This is a valid and informative outcome — the synthetic H2 result
may have been an artifact of clean label construction or closed-vocabulary features.
