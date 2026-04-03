# Addendum: Production Deployment Constraints
## Re-evaluation of the Final Recommendation

This addendum examines two production realities that were outside the scope of the
experimental work but materially change the architectural recommendation:

1. Word2Vec and FastText embeddings are not stable across training runs — retraining
   produces a new, incompatible coordinate system.
2. The model must be retrained frequently, because the very events it is trying to detect
   (novel devices) are indistinguishable from legitimate new device enrollments until
   confirmed, creating continuous pressure to incorporate new vocabulary.

Together, these two facts create a contradiction at the center of the centroid-based
approach that the report's recommendation did not address.

---

## 1. Rotational invariance: what it means and why it matters here

### 1.1 What "rotationally invariant" means in this context

Cosine distance is rotationally invariant as a metric within a single embedding space.
If every vector in a trained model is rotated by the same orthogonal matrix **R**:

```
cos_dist(Rv₁, Rv₂) = 1 − (Rv₁)·(Rv₂) / (‖Rv₁‖ ‖Rv₂‖)
                    = 1 − v₁·(RᵀR)v₂ / (‖v₁‖ ‖v₂‖)
                    = 1 − v₁·v₂ / (‖v₁‖ ‖v₂‖)
                    = cos_dist(v₁, v₂)
```

So all pairwise cosine distances are preserved under any rotation of the entire space.
Within a single trained model, cosine distances are geometrically meaningful regardless
of the arbitrary global orientation the optimizer happened to find.

The problem is not the metric. The problem is that **two independently trained models
do not share a coordinate system**.

### 1.2 Basis instability across training runs

Word2Vec and FastText are trained by gradient descent from a random initialization.
The optimization landscape has many equivalent local minima — solutions that achieve
similar loss values but correspond to different global orientations of the embedding
space. Two runs on identical data with different random seeds will produce models
where:

```
‖v_device_ABC (run T) − v_device_ABC (run T+1)‖ > 0
```

Not because the device's relationship to its account changed, but because the coordinate
frame rotated, reflected, and scaled arbitrarily. Formally, if run T produces embeddings
**V** and run T+1 produces embeddings **V'**, there exists some orthogonal matrix **R**
and scale **s** such that:

```
V' ≈ sRV + ε
```

where ε is noise from different stochastic gradient paths. For clean data the
approximation is tight; for sparse or noisy data it is loose.

**Within a run:** `cos_dist(v_device, v_centroid)` is meaningful — both vectors live in
the same coordinate frame.

**Across runs:** `cos_dist(v_device_runT+1, v_centroid_runT)` is meaningless — the
vectors are in different frames and the angle between them reflects coordinate geometry,
not the device-account relationship.

### 1.3 Why FastText is not more stable than Word2Vec

FastText uses a deterministic hash function to map character n-grams to fixed integer
indices in the subword embedding matrix. One might expect this deterministic structure
to provide cross-run stability. It does not.

The hash indices are deterministic; the learned embedding values stored at those indices
are not. Each training run updates the subword matrix through stochastic gradient descent
starting from a different random initialization. The hash table maps the same n-gram
string to the same bucket in every run, but the value in that bucket changes between
runs. An OOV device's embedding in run T and run T+1 is the sum of values at the same
indices, but those values differ.

FastText and Word2Vec are equally unstable across training runs. The subword mechanism
provides no cross-run coordinate continuity.

### 1.4 The practical consequence: centroids are model-version-specific

An account centroid is a function of the embedding model used to compute it:

```
μ_account = mean({ model.wv[d] for d in account.device_history })
```

When the model is retrained, `model.wv[d]` changes for every device `d`. The stored
centroid is now a vector in the old coordinate frame; the model it is being compared
against operates in a new, incompatible frame. Every centroid in the system is
simultaneously invalidated the moment a new model is deployed.

This has a direct operational requirement: **centroids cannot be persisted across model
versions**. Every retraining event requires a full recompute of every account's centroid.

---

## 2. Retraining pressure: how often must the model be retrained?

### 2.1 The new-device enrollment problem

The central production pressure for retraining is device enrollment. A genuine user
who buys a new phone has a device that is OOV — it has never appeared in the training
corpus. Under the recommended Word2Vec architecture, that device receives the global
mean vector and scores anomalously against the account's centroid.

The user calls support. The device gets confirmed as legitimate. But it will remain OOV,
and continue generating false positives, until the next model retraining cycle
incorporates it into the vocabulary.

This is not a corner case. Estimating from publicly reported consumer device
replacement patterns:

| Device class | Avg. replacement cycle | Annual turnover |
|---|---|---|
| Smartphone | 2.5 years | ~40% of users/year |
| Laptop/desktop | 3–4 years | ~25–30% of users/year |
| Tablet | 3–5 years | ~20–25% of users/year |

For a consumer application with 10 million accounts, conservatively assuming 1.5 new
device enrollments per account per year across device classes:

```
~15 million new device enrollments per year
~41,000 new device enrollments per day
```

Each new enrollment generates elevated false positive rates until the next retraining.
At a weekly retraining cadence, up to 280,000 accounts have a newly enrolled device in
the OOV window at any given time — roughly 2.8% of the user base showing elevated fraud
signals from entirely legitimate behavior.

At a daily retraining cadence, the window shrinks to 41,000 accounts (~0.4%), but
the operational cost of daily retraining and full centroid recompute may be prohibitive.

### 2.2 The contradiction: retraining causes the problem it is solving

This is the key architectural tension. Retraining more frequently reduces the OOV
false positive window — but each retraining event invalidates all centroids and
requires a full recompute from stored device histories. The two pressures are coupled:

```
↑ retraining frequency → ↓ OOV false positive window
                        → ↑ centroid recompute frequency
                        → ↑ storage requirement for full device histories
                        → ↑ operational complexity
```

And there is no frequency at which the problem is fully resolved. Continuous online
learning would require the Word2Vec objective to be updated incrementally — which
gensim's `model.train()` update supports, but with documented degradation in embedding
quality compared to full retraining from scratch. Incremental training perturbs the
coordinate frame gradually rather than replacing it wholesale, introducing a different
class of centroid staleness.

### 2.3 Storage requirements

Because centroids cannot be persisted across model versions, the full device history
for every account must be stored indefinitely — not just the current centroid. At
inference time, the current model version's centroid for an account is:

```
μ = mean(model_vT[d] for d in account.full_device_history)
```

If device histories are stored as lists of device ID strings (32 bytes each) with
timestamps (8 bytes each), at 50 events per account per year and 10 million accounts:

```
10M accounts × 50 events/year × 40 bytes/event = 20 GB/year of raw history data
```

At 5 years of history retention: 100 GB, before indexing overhead. For context, storing
just the centroid vectors would be 10M × 64 dimensions × 4 bytes = 2.56 GB — fixed cost,
never growing. The history requirement imposes a linearly growing storage footprint with
no alternative if centroid recompute is required after every retraining.

---

## 3. Mitigation: Procrustes alignment

### 3.1 What it is

Given two trained models producing embedding matrices **V** (old) and **V'** (new) for
a shared set of anchor tokens, Procrustes alignment finds the orthogonal rotation matrix
**R*** that minimizes the Frobenius norm of the difference:

```
R* = argmin_R ‖RV' − V‖²_F  subject to RᵀR = I
```

This has a closed-form solution via SVD:
```
[U, Σ, Vᵀ] = SVD(VᵀV')
R* = UVᵀ
```

Applying **R*** to all new model embeddings maps them into the old model's coordinate
frame. Existing centroids remain valid; only new devices (those not present in the old
model's vocabulary) need their embeddings computed and centroids updated.

### 3.2 What it buys

If alignment succeeds, the operational cost of retraining drops from:
- Full centroid recompute for all N accounts → O(N × avg_history_length)

to:
- Compute Procrustes matrix from anchor tokens → O(anchor_set_size²)
- Update centroids for accounts with newly enrolled devices only → O(new_devices)

For an anchor set of 1,000 stable tokens and 41,000 new devices per day, this reduces
the post-retraining work by roughly 3 orders of magnitude for a 10M-account system.

### 3.3 Its failure modes

**Alignment quality degrades with vocabulary changes.** If a retraining cycle adds many
new tokens or removes many old ones, the anchor set's shared subspace shrinks. A small
shared subspace makes the Procrustes solution less stable. For a system adding 41,000
new devices per day to a vocabulary of 1.8 million tokens, the vocabulary is growing by
~2.3% per day — aggressive retraining cycles would accumulate alignment error quickly.

**Alignment error compounds.** Each retraining cycle introduces residual alignment error
ε. Over K cycles, the stored centroids are aligned to the current model through a chain
of K rotations, each with error εₖ. The total drift is bounded by ∑εₖ, but in practice
it grows faster than linearly for non-stationary data distributions.

**Architecture changes break alignment entirely.** If the embedding dimension is changed
between versions (e.g., 64 → 128 dimensions), or the vocabulary is rebuilt with
different tokenization, Procrustes alignment is undefined. This makes Procrustes
incompatible with any model architecture evolution.

Procrustes alignment is a useful mitigation for low-frequency, stable retraining
regimes. For the enrollment pressure described above — multiple retraining cycles per
week at minimum — its failure modes accumulate faster than its benefits.

---

## 4. Re-evaluation of the final recommendation

The report's recommendation was a two-signal system:

> **Signal 1 — Word2Vec centroid distance:** behavioral fit to account cluster  
> **Signal 2 — OOV binary flag:** first-appearance novelty

This recommendation was correct on the experimental evidence but assumed a stable
production system with infrequent retraining. The production analysis above reveals
that the Word2Vec centroid signal carries significant hidden operational cost:

| Property | Word2Vec centroid | OOV binary flag |
|---|---|---|
| Rotationally stable across retraining | **No** — full recompute required | **Yes** — stateless lookup |
| Storage requirement | Full device history (grows linearly) | Per-account device set (∝ distinct devices) |
| Update latency for new enrollment | Days-to-weeks (next retraining + recompute) | **Milliseconds** (add to known set) |
| Catches device reuse attack | **Yes** | No |
| Sensitive to vocabulary staleness | Yes | Less so |
| Operational complexity | High | Low |

### 4.1 The hierarchy must be inverted

The report framed the OOV binary flag as a complementary secondary signal to the
Word2Vec centroid. Given production constraints, the correct hierarchy is the reverse.

**The OOV flag is the primary real-time signal.** It is:
- Immediately updatable when a device is confirmed legitimate (no retraining required)
- Immune to coordinate frame instability
- Computable in microseconds from a database set-membership check
- AUC 0.989 — the strongest single signal measured

**The Word2Vec centroid is a secondary, batch-computed risk feature.** It should not be
in the real-time inference path. Compute it offline, update it on a slower cadence (e.g.,
weekly), store it alongside a model version tag, and treat it as one feature among many
in an upstream risk scoring model — not as a live distance computation triggered at login.

### 4.2 The per-account known-device set replaces the model vocabulary

The OOV binary flag as implemented in the experiment checks model vocabulary membership
(`device_id in model.wv.key_to_index`). In production, this lookup should not depend on
the embedding model at all. A per-account known-device set stored in a low-latency
database (Redis, DynamoDB) decouples enrollment state from model training entirely:

```
known_devices[account_id] = {device_id_1, device_id_2, ...}

score(login_event):
    if login_event.device_id not in known_devices[login_event.account_id]:
        return SUSPICIOUS  # triggers step-up auth or review
    return CLEAN
```

This set is updated immediately when:
- A new device completes step-up authentication successfully → add to set
- A device is confirmed fraudulent → remove from set, flag all recent logins
- An account enrolls a new device through the app's device management flow → add to set

No model retrain is required for any of these updates. The signal is always current.

### 4.3 When embeddings are still the right tool

The embedding model's genuine advantage — detecting reuse by a device that is in-vocabulary
but behaviorally foreign to the target account — is real but serves a different threat
model than first-appearance novelty detection. It is most relevant when:

1. An attacker has previously authenticated against some account in the system (making
   their device in-vocab for the binary baseline)
2. That device's cluster neighborhood in embedding space is meaningfully far from the
   target account's cluster

For this specific scenario, the Word2Vec centroid signal adds genuine coverage the OOV
flag cannot provide. The right framing is not "embed device IDs and score at login" but
rather: compute a batch "account behavioral fit score" for all recent device interactions
daily or weekly, flag accounts whose recent logins show a pattern of elevated centroid
distance, and route those accounts to a review queue.

This reframes the embedding signal from a real-time classifier to an offline anomaly
detector — which is architecturally consistent with its retraining constraints.

### 4.4 An alternative that resolves the root cause

Both problems — rotational instability and continuous retraining pressure — trace to
the same root: using the device ID *string* as the token. An opaque, random alphanumeric
string carries no information across model versions. It is assigned an embedding purely
by co-occurrence, which depends entirely on the training run's random initialization.

If devices were represented by **structured feature vectors** instead of opaque IDs —
OS family, browser, timezone, screen resolution, network type, language settings — the
embedding input would be stable across model versions. A device seen for the first time
is not OOV; it has features. Two devices with similar features receive similar embeddings
regardless of when the model was trained. Retraining updates the feature-to-embedding
mapping smoothly rather than replacing the entire coordinate frame.

This requires investment in device fingerprinting infrastructure that may not exist at
PoC stage, but it is the architecturally correct solution to both problems simultaneously.

---

## 5. Revised recommendations

The original six priorities from the report are restated and re-ordered below, with two
new priorities added and one (Priority 1) significantly revised.

| Priority | Action | Rationale |
|---|---|---|
| **1** | **Replace the model vocabulary OOV check with a per-account known-device database** | Decouples enrollment state from model training; updates in milliseconds; immune to retraining disruption; AUC 0.989 |
| **2** | Replace FastText with Word2Vec for embedding | 7-point AUC gain, silhouette 0.94 vs −0.05 — but treat as a batch signal, not real-time |
| **3** | **Move Word2Vec centroid scoring to an offline batch pipeline** | Centroid recompute after retraining cannot be done in real-time; design for it explicitly |
| **4** | **Evaluate Procrustes alignment to reduce centroid recompute frequency** | Between retraining cycles, alignment may allow incremental centroid updates; measure alignment error accumulation |
| **5** | Stress-test cross-account device sharing | Highest-impact unresolved confound for both signals |
| **6** | Evaluate the returning-attacker scenario | Critical gap in OOV flag coverage; primary motivation for retaining embedding signal |
| **7** | Default to Markov corpus for all future experiments | i.i.d. remains an upper bound; Markov is the correct benchmark |
| **8** | Prototype feature-based device embeddings | Resolves root cause of both rotational instability and retraining pressure; requires device fingerprinting infrastructure |
| **9** | Report PR-AUC at realistic class imbalance | ROC-AUC validated the signal; operational cost requires PR-AUC |

### Revised architecture summary

```
Real-time path (latency: <10ms):
  ┌─────────────────────────────────────────────────┐
  │  login_event.device_id                          │
  │      ↓                                          │
  │  known_devices[account_id]   ← database lookup  │
  │      ↓                                          │
  │  OOV flag (0/1)  →  step-up auth if 1           │
  └─────────────────────────────────────────────────┘

Offline path (latency: hours; updated daily/weekly):
  ┌─────────────────────────────────────────────────┐
  │  account.device_history                         │
  │      ↓                                          │
  │  Word2Vec centroid (batch recompute)            │
  │      ↓                                          │
  │  centroid_distance_score → risk feature store   │
  │      ↓                                          │
  │  upstream risk model (logistic / gradient boost)│
  │  combined with: velocity, geo, time-of-day, etc │
  │      ↓                                          │
  │  account risk score  →  review queue if high    │
  └─────────────────────────────────────────────────┘
```

The real-time path catches first-appearance attacks with sub-millisecond latency and
no model dependency. The offline path provides the behavioral clustering signal for
reuse detection and is explicitly decoupled from the login event loop, giving the system
time to recompute centroids after each retraining cycle without degrading real-time
performance.

---

## 6. What this changes in the final report

Section 8 of the report ("Conclusions and Recommendations") described the two-signal
system as a **real-time classifier combining both signals at inference**. That framing
should be replaced by the two-path architecture above.

The report's observation that "the two signals address different attack modalities" remains
correct and is actually strengthened here: they now also operate on different timescales,
with different update latencies and different operational dependencies, which is the right
architecture for a signal with these properties.

The experimental finding that Word2Vec centroid distance achieves AUC 0.982 with
silhouette 0.941 stands. What changes is the deployment contract: that AUC is achievable
as a batch feature with a retraining + recompute pipeline, not as a live inference call
that must be available at every login event in real time.

---

## 7. Experiment 3: Realistic Data Redesign and Feature Embeddings

### 7.1 What Experiment 3 tested and why

The addendum's §4.4 proposed feature-based device embeddings as the architecturally
correct solution to both rotational instability and retraining pressure. Experiment 3
(`ato_experiment3.py`) tested this proposal empirically, while simultaneously correcting
two successive flaws in the evaluation design that had made the Experiment 2 results
uninterpretable.

Experiment 3 introduced:
- A shared fraud fleet (80 attacker-profile devices injected across 25% of accounts
  during training), producing genuine cross-account device co-occurrence signal
- Feature-based Word2Vec embeddings: each device represented as a sequence of six
  feature tokens (OS, browser, timezone, language, network type, screen resolution),
  enabling feature-profile proximity scoring independent of device ID novelty
- Three distinct attack types: `novel` (new ID + foreign profile), `fleet` (reused
  attacker device ID + foreign profile), and `spoof` (new ID + victim-mimicking profile)

### 7.2 The evaluation design correction: enrollment events in the negative class

The initial Experiment 3 run revealed a second evaluation flaw following the one
corrected from Experiment 2. Without legitimate new device enrollments in the negative
class, any signal that flags OOV devices achieved AUC 1.000 by construction — there
were no OOV legitimate events to produce false positives.

The corrected evaluation adds enrollment events to the negative class at 1:1 ratio with
returning known-device events. Enrollment events have new device IDs (globally OOV) with
feature profiles consistent with the account's primary profile (same OS, browser,
timezone, language; randomised network type and screen resolution, label = 0).

This change is not a cosmetic adjustment. It determines which signals are viable. Any
binary signal that fires on all OOV devices — including both `global_oov` and
`account_oov` — is structurally limited to AUC 0.750 when 50% of the negative class
is OOV. The 0.750 value is not empirical; it is the only possible result for any binary
OOV signal under this evaluation design (see `EXP3_CONCLUSIONS.md`, Finding 1, for the
derivation).

### 7.3 Key finding: feature_w2v is the only signal with a structural enrollment false-positive advantage

With enrollment events in the negative class, five signals were evaluated across three
attack types and two corpus modes. The full results are in `EXP3_CONCLUSIONS.md`. The
critical finding for the architectural recommendation is:

![Fig E3-1 — AUC heatmap by signal and attack type](../figures/exp3_fig1_auc_heatmap.png)

**`feature_w2v` is the only signal that does not collapse under enrollment pressure.**

Enrollment devices have feature profiles consistent with the account's primary profile.
Their feature embeddings (mean of six token embeddings from a bounded-vocabulary Word2Vec
model) land close to the account's feature centroid. Novel attack devices have foreign
profiles; their embeddings land far. feature_w2v achieves AUC **0.984** (i.i.d.) and
**0.993** (Markov) on novel attacks with enrollment in the negative class.

No other signal achieves this:
- `global_oov` and `account_oov`: analytically constrained to 0.750 on novel/spoof;
  AUC 0.250 on fleet (anti-correlated — fleet devices are in-vocabulary)
- `id_w2v`: AUC ~0.691–0.696 on novel/spoof (OOV devices receive global mean vector;
  enrollment and attack devices are treated identically); AUC 0.891 on fleet
- `feature_fasttext`: AUC 0.985 (i.i.d.) / 0.988 (Markov) on novel; 0.920 / 0.901 on
  fleet; 0.798 / 0.785 on spoof — see analysis below
- `feature_novelty`: AUC ~0.800 — partially degraded but not collapsed; limited by
  enrollment events whose new (net, screen) values produce novel tuples

feature_w2v is also rotationally stable in the sense that matters operationally: its
token vocabulary is bounded (approximately 30 distinct values across six dimensions).
Retraining updates the feature-to-embedding mapping, but because the vocabulary does
not grow with the user base, the co-occurrence structure is stable across retraining
cycles. This directly addresses the concern in §1.2 that motivated the addendum.

**`feature_fasttext` achieves comparable AUC with an additional production advantage.**
FastText trained on structured feature tokens (e.g., `os_ios`, `browser_safari`,
`tz_utc-5`) matches `feature_w2v` within confidence intervals on all three attack types:
novel (0.985 vs 0.984 i.i.d.), fleet (0.920 vs 0.914 i.i.d.), spoof (0.798 vs 0.817
i.i.d.). The AUC differences are not statistically distinguishable. The critical
production distinction is OOV token handling: when a new OS version, browser family, or
timezone variant appears between retraining cycles, `feature_w2v` falls back to the
global mean vector for that unseen token. `feature_fasttext` instead averages the n-gram
embeddings of the token's structured prefix and suffix components — `os_`, `browser_`,
`tz_` prefixes are well-represented in training — producing a positioned embedding that
places the new token approximately where it belongs relative to known feature clusters.
This is the signal recommended for production deployment. `feature_w2v` remains a valid
alternative when the feature vocabulary is fully stable and slightly better spoof
detection (marginal, within CIs) is preferred over OOV resilience.

![Fig E3-2 — feature_w2v score distributions by event type](../figures/exp3_fig2_score_distributions.png)

### 7.4 How this changes the revised architecture from Section 4

The two-path architecture in §4 placed the **per-account OOV binary flag** as the
primary real-time signal. Experiment 3 demonstrates that this signal has a structural
false-positive ceiling when legitimate enrollment is present. The architecture should
be updated as follows:

**Real-time path: replace the binary OOV flag with feature-profile proximity.**

The per-account known-device set lookup (§4.2) remains valuable as a first-pass gate and
for triggering step-up authentication, but it should not be the sole or primary scoring
signal. A feature-profile proximity score — computed from `feature_fasttext` (preferred)
or `feature_w2v` (acceptable alternative) — should be the primary score. It correctly
separates enrollment (profile-consistent, low distance) from novel attack
(profile-inconsistent, high distance) without requiring a device ID lookup at all.

`feature_fasttext` is the preferred production signal because it handles OOV feature
tokens at inference time via n-gram averaging over structured token prefixes (`os_`,
`browser_`, `tz_`). This means new OS versions, browser families, or timezone variants
that appear between retraining cycles receive positioned embeddings rather than a global
mean fallback. `feature_w2v` is an acceptable alternative when the feature vocabulary
is known to be stable between retraining cycles and marginally better spoof detection
is a priority (0.817 vs 0.798 i.i.d., within confidence intervals).

**Binary OOV signals should not drive risk scoring.** Both `global_oov` and `account_oov`
are suitable for operational gates (step-up auth trigger, alert queuing) but must not be
used as inputs to a fraud risk score — their scores are determined analytically by the
evaluation design, not by the device's behavioral relationship to the account.

**id_w2v remains the correct offline signal for fleet/reuse detection.** AUC 0.856–0.891
on fleet attacks is genuine signal for cross-account device reuse. The offline batch
pipeline described in §4's architecture diagram is the right deployment for this signal.
No change to that recommendation.

### 7.5 Updated priority table

The priority table from Section 5 is extended below. Priorities 1–9 from §5 are
retained with one revision to Priority 1; three new priorities are added.

| Priority | Action | Rationale |
|---|---|---|
| **1** | **Implement feature-profile proximity (`feature_fasttext`, preferred; or `feature_w2v`) as the primary real-time signal** | Replaces binary OOV flag in real-time path; AUC 0.984–0.993 on novel attacks with enrollment in negative class; structurally separates enrollment from attack; bounded vocabulary is rotationally stable across retraining; `feature_fasttext` additionally handles unseen feature tokens (new OS versions, browsers) at inference time without OOV fallback |
| **2** | Replace FastText with Word2Vec for ID-based embedding | 7-point AUC gain for fleet detection (id_w2v); treat as batch signal, not real-time |
| **3** | Move Word2Vec ID-centroid scoring to offline batch pipeline | Handles retraining constraints; targets fleet/reuse threat model specifically |
| **4** | Evaluate Procrustes alignment to reduce centroid recompute frequency | Between retraining cycles for ID-based model; measure alignment error accumulation |
| **5** | Retain per-account known-device set as operational gate | Step-up auth trigger on OOV events; decouple from risk scoring; update in milliseconds on enrollment confirmation |
| **6** | Stress-test cross-account device sharing | Highest-impact unresolved confound; fleet injection at 25% of accounts is a starting point |
| **7** | Evaluate the returning-attacker scenario | Critical gap in OOV flag coverage; id_w2v offline signal is primary coverage |
| **8** | Default to Markov corpus for all future experiments | i.i.d. remains an upper bound; note that feature_w2v improves under Markov (bounded vocabulary) |
| **9** | Report PR-AUC at realistic class imbalance | ROC-AUC validated signals; operational cost requires PR-AUC at production fraud rates |
| **10** | Extend spoof attack coverage with higher-cardinality features | feature_w2v AUC 0.817 on spoof (timezone mismatch only); test harder spoof with more matched dimensions |
| **11** | Evaluate feature_w2v + id_w2v ensemble | Combined signal should outperform either alone; logistic regression over both targets complementary threat models |
| **12** | Test the perfect-spoof scenario | Measure minimum feature mismatches needed for detection; defines the signal's hard boundary |
