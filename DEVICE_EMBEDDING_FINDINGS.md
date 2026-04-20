# How to Embed Device Identity for ATO Detection: Key Findings

> Talking points for colleagues. Based on multiple phases of investigation — from initial PoC
> through H2 (robust config), H2-RBA (real-world replication), H5 (stress test), and H6
> (RBA-calibrated hybrid with realistic imbalance and temporal fleet blocklist).

**TL;DR (post-H6):** Single-stage mean-pool FastText cosine distance (`mp_raw`), fronted by a
system-level blocklist, is the recommended architecture. The two-stage per-account set-membership
gate that earlier experiments recommended has been retired — it blinds the model during the
cold-start fleet window. Rank-normalization, also previously recommended, catastrophically
degrades PR-AUC under realistic 1:100 class imbalance and should not be used operationally.

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

**Result:** Spoof AUC 0.782 — above the trivial exact-match baseline (0.750) but only by +0.032.
The narrow margin offers little practical advantage over a hash lookup.

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
| Novel         | 0.999           | Strong                              |
| Fleet/reuse   | 0.994           | Strong                              |
| Spoof         | 0.869           | Beats trivial baseline (0.750) ✓ (+0.119) |

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

## Extending Mean-Pool: Per-User Rank Normalization

Raw cosine distance to the account centroid is the base signal, but it has a scale problem: an account with 60 clean training events from a single device produces a very tight centroid (distances near zero for known devices); an account with diverse logins across multiple devices produces a looser centroid. A single global threshold over-flags low-history or high-variance accounts.

**Per-user rank normalization** solves this by converting each raw score to an empirical percentile within that account's own calibration set:
- First 40 login events → compute centroid
- Last 20 events → held-out calibration baseline
- Score = `P(calibration_dist < test_dist)` ∈ [0, 1]

This is most valuable when the raw cosine signal is weak — exactly the case for sophisticated attackers who match most features.

**Variable-K spoof results** (from `h2_ml_lab/experiments/variable_spoof_experiment.py`):

| Spoof type | Fields changed | Analog | mp-raw AUC | mp-rank-norm AUC | Trivial |
|------------|---------------|--------|:----------:|:----------------:|:-------:|
| k=1 — VPN | tz only | Sophisticated, single-field | 0.522 | **0.714** | 0.750 |
| k=2 — Datacenter VPN | tz + network | Moderate, two fields | 0.689 | **0.735** | 0.750 |
| k=3 — Emulated device | tz + net + screen | Detectable, three fields | **0.869** | 0.784 | 0.750 |

**The crossover (H2 balanced evaluation):** Rank normalization wins at k=1 and k=2 (sophisticated attackers with small raw signal). Raw scoring wins at k=3.

**The H6 reversal under realistic imbalance.** On the RBA-calibrated hybrid dataset with a 1:100 attack-to-benign ratio, rank-normalization collapses PR-AUC rather than helping. At k=1, `mp_rank_norm` scores PR-AUC 0.215 against `mp_raw` at PR-AUC 0.892 — a 4× degradation. The CDF transform compresses the score margin between positives and negatives, and at 1:100 imbalance that compression is catastrophic. ROC-AUC hides this: rank-norm ROC-AUC (0.972) looks competitive with raw (0.995), but the PR curve degrades sharply.

**The revised takeaway:** Do not use rank-normalization operationally. The earlier k=1 weakness that motivated rank-norm was largely a vocabulary-poverty artifact (30-token toy synthetic). With an RBA-calibrated 229-token vocabulary, raw `mp_raw` at k=1 reaches ROC-AUC 0.995 / PR-AUC 0.892 without any normalization. Raw cosine distance is the production scorer.

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
# Production vocabulary (H6: country/region/ASN model):
for dim in ['os', 'browser', 'country', 'region', 'asn', 'dev']:
    all_vectors = [model.wv[f"{dim}_{val}"] for val in known_values[dim]]
    sim = mean_pairwise_cosine_similarity(all_vectors)
    assert sim < 0.5, f"Embedding collapse in {dim}: similarity={sim:.4f}"
```

If any dimension exceeds 0.5, halt deployment. This check catches the failure mode before it ships.

**Open-vocabulary note:** On real RBA data the raw within-feature similarity can exceed 0.5 (observed 0.563) without indicating collapse — open vocabularies have more synonym-like tokens. Monitor the within/cross-feature ratio (target > 1.0) rather than the absolute threshold when deploying against production open-vocab features.

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

## Recommended Production Architecture (Post-H6)

> **Note:** The architecture below is grounded in the H6 hybrid experiment (RBA-calibrated
> marginals, 1:100 attack-to-benign ratio, temporal cross-account blocklist model). See
> `h6_hybrid/docs/REPORT.md` for full results.

Two decoupled layers, each evaluated on the population it actually serves:

### Layer 1 — System-level blocklist (upstream)
- **Signal:** Cross-account deny-list of confirmed fleet/abuse device keys.
- **Population:** Post-lag fleet accounts — in H6's model (10-day lag, 30-day attack window),
  this is 61% of fleet-attacked accounts.
- **Precision:** 1.0 by construction (only confirmed device keys are listed).
- **At login:** If device key is in blocklist → deny / hard decision. Event never reaches Layer 2.
- **Population source:** Customer complaints, downstream triage, threat intelligence feeds,
  optionally accelerated by Layer 2 detections flagged for review.

### Layer 2 — Real-time mean-pool scoring (single-stage, no gate)
- **Signal:** Mean-pool FastText cosine distance to account centroid (`mp_raw`).
- **No per-account set-membership gate.** The two-stage gate (known device → score=0) blinds
  the model during the cold-start fleet window: the fleet device appears in the account's
  training set (it was a normal login before the attack), so the gate scores it 0 precisely
  when the blocklist has not yet activated.
- **No rank-normalization.** Collapses PR-AUC from 0.892 → 0.215 under 1:100 imbalance (H6).
- **Population:** All events that clear Layer 1 — includes spoof (k=1/2/3), novel devices,
  and cold-start fleet events.
- **H6 performance on this population:**
  - Spoof k=1 (hardest, country-change only): PR-AUC 0.892, top-1% precision 82.3%
  - Novel device: PR-AUC 0.965, top-1% precision 92.0%
  - Fleet residual (pre-lag cold-start only): PR-AUC 0.948, top-1% precision 91.8%
- **Latency:** Single forward pass, sub-millisecond.
- **Update cadence:** Retrain account centroids monthly; retrain FastText monthly.
- **Pre-flight check:** T8 within-feature similarity assertion after every retrain.

### Offline support (feeds Layer 1)
- **Signal:** Word2Vec on per-account device ID sequences with cross-account fleet injection.
- **Why offline:** Device ID embeddings are rotationally variant across retraining runs — not
  stable for real-time centroid comparison across model versions.
- **Cadence:** Weekly batch job, outputs a review queue that can accelerate blocklist
  population.

### What was removed from prior recommendations
- **Two-stage per-account gate:** Retired. Blinds the model during cold-start fleet attacks.
  H6 fleet-residual evaluation: trivial/two-stage score 0 TP on the cold-start population;
  `mp_raw` scores 180 TP at 91.8% precision on the same events.
- **Rank-normalization:** Retired. Catastrophically degrades PR-AUC under realistic imbalance.

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
open-vocabulary data (0.661 vs. 0.750 synthetic) because real users access the system from
diverse device/region combinations — the training-window known-device set is sparser than
the synthetic 2–4 known-device setup. Mean-pool tracks the harder baseline and maintains its
margin.

The T8 health check threshold (within-feature sim < 0.5) designed for closed vocabularies is
slightly exceeded on real data (0.563) without indicating collapse — the within/cross ratio
(1.66) remains healthy. For open-vocabulary production settings, monitor the ratio rather
than the absolute threshold.

---

## H6 Summary: Class Imbalance, Vocabulary Depth, and Fleet Architecture

H6 replaced the closed 30-token synthetic vocabulary with an RBA-calibrated 229-token
vocabulary generated by chain-sampling real login marginals (11.7M clean RBA logins). It also
replaced the 1:1 attack/benign evaluation with a realistic 1:100 enrollment ratio and added a
temporal cross-account blocklist model (10-day lag from first attack to activation). Four
conclusions emerged that revise the earlier architecture guidance:

1. **k=1 country-change spoofing is detectable.** Prior H5 reported k=1 mean-pool ROC-AUC 0.530
   vs. trivial 0.750 on 30-token synthetic and raised the possibility that k=1 was a fundamental
   limit. H6 disproves this: with 229-token RBA-calibrated vocabulary, k=1 `mp_raw` reaches
   ROC-AUC 0.995 / PR-AUC 0.892 — an 8.6× PR-AUC lift over trivial. The earlier failure was
   vocabulary poverty, not an architectural ceiling.

2. **PR-AUC is the only honest metric under class imbalance.** At 1:100, trivial ROC-AUC is 0.943
   while trivial PR-AUC is 0.104 — ROC compresses into an uninformative band where all models
   appear competitive. PR-AUC correctly surfaces the 8.6× gap between `mp_raw` (0.892) and
   trivial (0.104). Report PR-AUC and top-k precision/recall for any imbalanced deployment
   evaluation.

3. **Rank-normalization is harmful under imbalance.** PR-AUC drops from 0.892 to 0.215 at k=1.
   The CDF transform compresses score margins, and compression is catastrophic at 1:100. Do not
   use rank-norm in operational scoring.

4. **Fleet detection is a two-layer composite, not a single scorer.** A realistic fleet attack
   has two populations: post-lag events (blocklist can catch with precision=1.0) and pre-lag
   cold-start events (blocklist cannot see; model must handle). The per-account set-membership
   gate (previously recommended as "two-stage") scores the fleet device as `known → 0` during
   cold-start — exactly when the model is the only available defense. Retire the gate. Use
   system-level blocklist + single-stage `mp_raw`. On the fleet-residual population (pre-lag
   only), `mp_raw` scores PR-AUC 0.948 / 91.8% precision; trivial and two-stage score 0 TP.

---

## Bottom Line

The key insight is that **you should not embed what you want to identify; you should embed the
semantic attributes that describe it.** Opaque device IDs are the wrong unit of embedding because
new devices are always OOV. Structured feature tokens are never OOV — they form a bounded
vocabulary that generalizes naturally to new devices through shared features. (In H6, that
vocabulary is 229 tokens drawn from real RBA marginals, not the original 30-token toy
vocabulary — richness here directly determines whether k=1 spoofs are detectable.)

Mean-pooling the feature token embeddings rather than concatenating them into a single string
preserves each feature's independent signal and avoids cross-boundary contamination — a subtle
architectural choice that changes whether the approach beats a trivial baseline on the hardest
attack type.

The training configuration is not a tuning knob; it determines whether the embedding space
collapses silently. Skip-gram + per-account corpus is the only configuration that produces genuine
feature differentiation. Validate it with the T8 health check on every retrain cycle.

Finally, resist the temptation to add per-account gates or rank-normalization on top of
`mp_raw`. Both were supported by earlier balanced evaluations but degrade PR-AUC under realistic
class imbalance. Keep the model simple, front it with a system-level blocklist, and measure
with PR-AUC.
