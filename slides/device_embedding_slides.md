---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 30px 52px;
    font-size: 1.0em;
  }
  h1 {
    font-size: 1.65em;
    color: #1a1a2e;
    border-bottom: 3px solid #e94560;
    padding-bottom: 6px;
    margin-bottom: 10px;
    margin-top: 0;
  }
  h3 { color: #444; margin: 2px 0 8px 0; }
  table {
    font-size: 0.8em;
    width: 100%;
    margin: 8px 0;
  }
  th {
    background: #1a1a2e;
    color: white;
    padding: 5px 8px;
  }
  td { padding: 4px 8px; }
  tr:nth-child(even) { background: #f4f4f4; }
  .hero {
    font-size: 0.9em;
    background: #f0f7ff;
    border-left: 4px solid #e94560;
    padding: 8px 14px;
    margin: 8px 0 0 0;
  }
  pre {
    font-size: 0.77em;
    margin: 6px 0;
  }
  p { margin: 5px 0; }
  ul { margin: 4px 0; padding-left: 1.4em; }
  blockquote { margin: 8px 0; }
---

<!-- Slide 1 -->

# Embedding Device Identity for ATO Detection

### Is an unfamiliar device a new phone — or an attacker?

&nbsp;

- Tested five embedding strategies across three attack types
- Re-run on RBA-calibrated marginals at realistic 1:100 imbalance (H6)
- The right signal, metric, and training config matter more than model choice

&nbsp;

> **Verdict:** FastText mean-pool on feature tokens · **PR-AUC 0.892** / **ROC-AUC 0.995** on k=1 spoof at 1:100

---

<!-- Slide 2 -->

# The Key Metric: Spoof PR-AUC at Realistic Imbalance

Attacker mimics OS, browser, language — differs only on country/network. At 1:100 attack-to-benign, **ROC-AUC misleads**.

| Attack | Difficulty | Why |
|--------|-----------|-----|
| Novel | Easy | Any "new device" signal works |
| Fleet/reuse | Two-regime | System-level blocklist (post-lag) + model (cold-start) |
| **Spoof k=1** | **Hard** | Only country changes; device features match |

<div class="hero">
<strong>ROC vs PR at 1:100:</strong> trivial set-membership reaches ROC 0.943 — looks competitive.
PR-AUC tells the truth: trivial PR=0.104 vs mp_raw PR=0.892 (<strong>8.6× lift</strong>).
PR-AUC is the operational metric.
</div>

---

<!-- Slide 3 -->

# Embedding Device IDs Directly — Both Fail

### FastText on device IDs
- Character n-grams on `dev_a3f9b2...` share substrings across accounts
- Silhouette **−0.051** — devices closer to *other* accounts than their own

### Word2Vec on device IDs
- Clean clusters (silhouette +0.941), fleet AUC **0.891**
- Every new device is OOV → all get the global mean vector
- Novel attack, spoof attack, and enrollment are **identical to the model**

<div class="hero">
OOV ceiling is structural — not tunable. Fleet detection belongs upstream of the model, in a system-level blocklist (Layer 1), not in a device-ID embedding.
</div>

---

<!-- Slide 4 -->

# Concatenated Feature String — Structurally Weaker Than Mean-Pool

**Token:** `ios_safari_utc-5_en_wifi_1920x1080` → one FastText embedding

- Cross-boundary n-grams (`ari_ut`, `i_utc`) bleed across feature boundaries
- A single mismatched feature gets diluted by five matching features around it

| Evidence | Result |
|----------|--------|
| Shift timezone to different string positions | AUC 0.655–0.713 at all positions |
| Widen window (w=1 → w=6) | Recovers only 43.6% of mean-pool gap |
| Bootstrap CI vs. mean-pool | [+0.034, +0.077] — gap is structural |

Mean-pool preserves per-feature independence; concat does not. Retired as a candidate.

---

<!-- Slide 5 -->

# Mean-Pool FastText on Feature Tokens ✅ (H6, 1:100 Imbalance)

Feature tokens · embed independently · average the vectors · cosine distance to account centroid.

| Attack | ROC-AUC | **PR-AUC** | vs. Trivial PR |
|--------|:-------:|:----------:|:--------------:|
| Novel | 0.999 | 0.965 | +0.862 |
| Fleet residual (cold-start) | 0.997 | 0.948 | +0.938 |
| **Spoof k=1** | **0.995** | **0.892** | **+0.788 (8.6×)** |

- **No OOV problem** — `os_ios` is always a known token
- **Independent signal** — one mismatched feature isn't diluted by matching features
- **k=1 now works** — 229-token RBA-calibrated vocabulary vs. 30-token H5 synthetic

<div class="hero">
H5 failed k=1 at ROC 0.530 on a 30-token vocabulary. H6 resolves it: vocabulary richness, not a fundamental limit of mean-pool embeddings.
</div>

---

<!-- Slide 6 -->

# ⚠️ Two Silent Failure Modes

**(1) Config collapse:** CBOW + per-event corpus converges all values of a feature to one vector.

| Config | Within-feature sim | Spoof signal |
|--------|:------------------:|:------------:|
| CBOW + per-event corpus | 0.9993 — **collapse** | Destroyed |
| **Skip-gram + per-account corpus** | **0.392** | **PR-AUC 0.892** |

**(2) Rank-normalization under imbalance:** Previously recommended for cross-account calibration. At 1:100 it collapses PR-AUC **0.892 → 0.215** while preserving ROC-AUC — a metric-shaped trap.

<div class="hero">
Both failures are silent on ROC-AUC dashboards. Run two checks after every retrain:
within-feature cosine sim &lt; 0.5, and PR-AUC (not just ROC) on a held-out 1:100 eval slice.
</div>

---

<!-- Slide 7 -->

# Six Decisions That Change the Conclusion

| Decision | Wrong | Right |
|----------|-------|-------|
| Primary metric at 1:100 imbalance | ROC-AUC | **PR-AUC** |
| Score transform | Rank-normalization | **Raw cosine distance** |
| Per-account gate | Set-membership (known→0) | **No gate — mp_raw alone** |
| Training objective | CBOW | **Skip-gram** |
| Corpus construction | Per-event | **Per-account** |
| Feature representation | Concat string | **Mean-pool tokens** |

Each row determines whether the signal works, collapses silently, or is misread. They are not tuning knobs.

---

<!-- Slide 8 -->

# Production Architecture — Two-Layer Composite

```
Layer 1 — System-level blocklist (upstream, before model):
  Cross-account deny-list of confirmed fleet device keys
  Populated from customer complaints (lag ~10d from first attack)
  Precision = 1.0 by construction; covers ~61% of fleet attacks

Layer 2 — Single-stage mp_raw (no per-account gate):
  embedding  = mean(fasttext[t] for t in feature_tokens)
  risk_score = cosine_distance(embedding, account_centroid)
  → step-up auth if risk_score > threshold
  Handles: spoof k=1/2/3, novel device, cold-start fleet (PR=0.948)

Retraining (monthly):
  FastText: sg=1, per-account corpus, window=6, vector_size=64
  Health check: within-feature sim < 0.5 → halt if fails
```

<div class="hero">
<strong>REMOVED:</strong> per-account known-device gate — blinds model during cold-start fleet window.
<strong>REMOVED:</strong> rank-normalization — collapses PR-AUC (0.892 → 0.215) under realistic imbalance.
<strong>REMOVED:</strong> Word2Vec offline fleet batch — replaced by Layer 1 blocklist with certainty coverage.
</div>
