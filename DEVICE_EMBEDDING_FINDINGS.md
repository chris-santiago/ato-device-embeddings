# How to Embed Device Identity for ATO Detection: Key Findings

> Talking points for colleagues. Based on two phases of investigation across five experiments
> comparing embedding strategies for Account Takeover (ATO) anomaly detection.

---

## The Core Problem

When a user logs in from a device we haven't seen before, is it them enrolling a new phone — or an
attacker? The goal is to score that question at login time using only device attributes we observe
passively (OS, browser, timezone, language, network type, screen resolution).

The natural instinct is to embed device IDs directly and measure how close a new device is to the
account's historical centroid. This turns out to be wrong in a non-obvious way. Here's why, and
what works instead.

---

## What We Tested

We compared six embedding strategies across three attack types:

- **Novel attacks** — completely new attacker device, different profile from the account
- **Fleet/reuse attacks** — attacker uses a device that has been seen attacking *other* accounts
- **Spoof attacks** — attacker closely mimics the victim's device profile (same OS, browser,
  language) but differs on one or two features like timezone or network

Spoof is the hardest attack to detect and the one that matters most operationally, because it's
what a capable adversary actually does. We used **spoof AUC** as the primary verdict metric.

---

## What Doesn't Work (and Why)

### 1. FastText directly on device IDs

**Result:** Silhouette score −0.051. Devices end up *closer* to other accounts' devices than to
their own account's historical set.

**Why it fails:** FastText uses character n-grams to build embeddings. Device IDs that look like
`dev_a3f9b2c1...` all share the `dev_` prefix, and random hex suffixes produce accidental n-gram
overlaps across accounts. The n-gram mechanism — designed to generalize across morphologically
related words — actively injects cross-account noise here. FastText's most useful feature becomes
its most damaging one when applied to opaque identifiers.

### 2. Word2Vec on device IDs — works for fleet, fails for novel/spoof

**Result:** Silhouette +0.941 (beautiful per-account clusters), fleet AUC 0.891, but novel and
spoof AUC collapse structurally.

**Why it fails on novel/spoof:** Every brand-new device — whether it's a legitimate enrollment, a
novel attack, or a spoof attack — is out-of-vocabulary (OOV). All OOV tokens receive the same
global mean vector, landing at identical distance from the account centroid. This is not a threshold
problem. Structurally, under any operating point, novel attacks and spoof attacks are
indistinguishable from enrollment events. The analytical ceiling for OOV-based binary signals is
AUC 0.750.

Word2Vec on device IDs is useful, but only as an offline fleet-detection signal run in batch. It
cannot be used for real-time novel/spoof scoring.

### 3. Concatenated feature strings with FastText

**Setup:** Combine the six feature values into a single string like
`ios_safari_utc-5_en_wifi_1920x1080` and embed that as one token.

**Result:** Spoof AUC 0.763 — *below* a trivial exact-match baseline (0.791). This approach is
worse than a hash lookup.

**Why it fails:** FastText applies character n-grams across the *entire* string, spanning feature
boundaries. The substring `ri_ut` spans from `browser_safari` into `tz_utc...`. When an attacker
matches five features but differs on timezone, the five matching features produce boundary-crossing
n-grams that bleed into the timezone representation, diluting its signal. We confirmed the
mechanism experimentally: moving timezone to different positions in the string produces monotonic
AUC degradation (position 5 = AUC 0.655, position 0 = AUC 0.713). No position recovers mean-pool
performance. This is a structural property of the concatenation, not a positional artifact.

---

## What Works: Mean-Pool FastText on Structured Feature Tokens

**Setup:**
1. Represent each device as six discrete tokens: `os_ios`, `browser_safari`, `tz_utc-5`,
   `lang_en`, `net_wifi`, `screen_1920x1080`
2. Train FastText with skip-gram on a per-account corpus (all login events for one account
   flattened into one sentence)
3. At inference, embed each of the six tokens independently and **mean-pool** the six vectors
4. Compute cosine distance from the account's historical centroid

**Results:**

| Attack Type   | AUC             | Notes                              |
|---------------|-----------------|-------------------------------------|
| Novel         | 0.985           | Strong                              |
| Fleet/reuse   | 0.920           | Strong                              |
| Spoof         | 0.798           | Beats trivial baseline (0.791) ✓    |

**Why it works:**

- **Solves OOV:** A new iPhone is never OOV — `os_ios` is a known token. A new attacker in a
  different timezone lands far from the account centroid because `tz_utc-8` is a known, distinct
  token. Only opaque device IDs are OOV; their features never are.
- **Mean-pool isolates each feature's signal:** Embedding each token separately means a mismatched
  timezone contributes its full cosine distance independently. There's no adjacent feature to dilute
  it. When five features match and one doesn't, the mismatch gets its own dedicated dimension of
  signal.
- **FastText's n-gram mechanism is *beneficial* here:** `browser_chrome_v120` shares n-grams with
  `browser_chrome_v119`, so new browser versions receive positioned embeddings (close to their
  known variants) rather than falling back to the global mean. Unlike opaque device IDs, structured
  feature tokens have genuine morphological relationships that the n-gram mechanism exploits well.

---

## The Silent Failure Mode You Must Test For

This is the most operationally dangerous finding from this investigation.

**What can go wrong:** Using CBOW training objective with a per-event corpus (one 6-token sentence
per login event) causes all values within a feature dimension to converge to near-identical
embeddings.

**What "near-identical" means in practice:** Cosine similarity between `tz_utc-5` and `tz_utc-8`
reaches 0.9993. They are functionally the same vector.

**The outcome:** Spoof AUC = 0.384 (below chance). The model cannot distinguish a matching timezone
from a mismatching one.

**Why it's dangerous:** Novel AUC and fleet AUC remain healthy (0.880 and 0.922). A monitoring
dashboard that watches aggregate AUC would see nothing wrong. The failure only surfaces on spoof
attacks — the attack type you care most about stopping.

**The fix:** Use skip-gram (`sg=1`) with per-account corpus construction (all events for one
account concatenated). This exposes each feature token to diverse contexts, giving it differentiated
gradients during training.

**The required health check after every retraining:**

```python
for dim in ['os', 'browser', 'tz', 'lang', 'net', 'screen']:
    all_vectors = [model.wv[f"{dim}_{val}"] for val in known_values[dim]]
    sim = mean_pairwise_cosine_similarity(all_vectors)
    assert sim < 0.5, f"Embedding collapse in {dim}: similarity={sim:.4f}"
```

If any dimension exceeds 0.5, halt deployment. This check catches the failure mode before it ships.

---

## Configuration Decisions That Matter

| Decision | Wrong | Right | Why It Matters |
|----------|-------|-------|----------------|
| Training objective | CBOW | Skip-gram | CBOW collapses within-feature embeddings silently |
| Corpus construction | Per-event (one sentence per login) | Per-account (one sentence per account) | Per-event exposes all timezone values to identical contexts |
| Feature representation | Concatenated string | Mean-pooled tokens | Concat allows cross-boundary n-gram contamination |
| Model on device IDs | FastText | Word2Vec (or none) | FastText n-grams poison opaque-ID embeddings |

Getting any one of these wrong can change the conclusion entirely — the results from the wrong
configuration suggest the mean-pool approach *doesn't* work.

---

## Recommended Production Architecture

### Real-time scoring (novel + spoof attacks)
- **Signal:** Mean-pool FastText on feature tokens
- **Latency:** Single forward pass, ~30ms
- **Update cadence:** Retrain account centroids monthly
- **Pre-flight check:** T8 within-feature similarity assertion after every retrain

### Offline batch scoring (fleet/reuse attacks)
- **Signal:** Word2Vec on per-account device ID sequences with cross-account fleet injection
- **Why offline:** Device ID embeddings are rotationally variant across retraining runs — not stable
  for real-time centroid comparison across model versions
- **Cadence:** Weekly batch job, outputs a review queue for manual or downstream triage

### Operational gate (all devices)
- **Signal:** Per-account confirmed-device set (Redis/DynamoDB hash)
- **Purpose:** Fast-path skip for returning known-good devices; triggers step-up auth for any
  unknown device regardless of embedding score
- **Not for scoring:** Do not use presence/absence as the risk score — this is the OOV trap

---

## Real-World Validation

The synthetic findings were tested on the DAS Group RBA dataset (Wiefling et al. 2022,
ACM TOPS): ~31M synthesized Norwegian SSO login events, ~4.3M users, 141 ground-truth ATO
events (0.0005% positive rate). The same training configuration (FastText sg=1, per-account
corpus, ROBUST_KWARGS verbatim) was applied without modification to 7 open-vocabulary
categorical features.

**Result:** Mean-pool ROC-AUC 0.852 vs. trivial set-membership 0.661 — non-overlapping
bootstrap 95% CIs. The mean-pool > trivial ordering holds across three temporal split
percentiles (40/60, 50/50, 60/40). Token structure diagnostics (T6 compactness, T8
within/cross similarity ratio) are consistent with the synthetic results.

**Caveat:** Only 9 ATO test events survive the temporal split and training floor (141 total
ATO events; all occur before the 70th percentile of timestamps). The result is exploratory
and directional, not statistically definitive. A leakage audit (Opus adversarial review)
found no label leakage, temporal leakage, or known-device contamination.

**The key difference on real data:** The trivial set-membership baseline is weaker on real
open-vocabulary data (0.661 vs. 0.791 synthetic) because real users access the system from
diverse device/region combinations — the training-window known-device set is sparser than
the synthetic 2–4 known-device setup. Mean-pool tracks the harder baseline and maintains its
margin.

The T8 health check threshold (within-feature sim < 0.5) designed for closed vocabularies is
slightly exceeded on real data (0.563) without indicating collapse — the within/cross ratio
(1.66) remains healthy. For open-vocabulary production settings, monitor the ratio rather
than the absolute threshold.

---

## Bottom Line

The key insight is that **you should not embed what you want to identify; you should embed the
semantic attributes that describe it.** Opaque device IDs are the wrong unit of embedding because
new devices are always OOV. Structured feature tokens are never OOV — they form a bounded
vocabulary (~30 tokens total) that generalizes naturally to new devices through shared features.

Mean-pooling the feature token embeddings rather than concatenating them into a single string
preserves each feature's independent signal and avoids cross-boundary contamination — a subtle
architectural choice that changes whether the approach beats a trivial baseline on the hardest
attack type.

The training configuration is not a tuning knob; it determines whether the embedding space
collapses silently. Skip-gram + per-account corpus is the only configuration that produces genuine
feature differentiation. Validate it with the T8 health check on every retrain cycle.
