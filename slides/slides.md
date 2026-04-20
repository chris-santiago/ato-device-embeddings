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

# ATO Detection via Device Embeddings

### Can login device sequences catch account takeover in real time?

&nbsp;

**Answer:** Yes — with the right signal, metric, and training configuration.

> FastText mean-pool on structured feature tokens achieves **PR-AUC 0.892** / **ROC-AUC 0.995**
> on k=1 country-change spoofs at a realistic **1:100** attack-to-benign ratio.

&nbsp;

Three experiments · RBA-calibrated replication · Adversarial critique & debate · Production analysis

&nbsp;

`github.com/chris-santiago/ato-device-embeddings`

---

<!-- Slide 2 -->

# Which Signal Works? (H6: RBA-Calibrated, 1:100 Imbalance)

Scorers evaluated on k=1 country-change spoof at realistic attack-to-benign ratio.

| Scorer | ROC-AUC | **PR-AUC** | Top-1% prec | Verdict |
|--------|:-------:|:----------:|:-----------:|---------|
| **mp_raw** (mean-pool, raw cosine) | **0.995** | **0.892** | **0.823** | ✅ Recommended |
| two_stage (gate + mp_raw) | 0.980 | 0.890 | 0.834 | Confirms H6 hypothesis; retired for production |
| mp_rank_norm | 0.972 | 0.215 | 0.278 | ❌ Collapses under imbalance |
| Trivial set-membership | 0.943 | 0.104 | 0.119 | Baseline — misleadingly high ROC |

<div class="hero">
<strong>Key insight:</strong> At 1:100 imbalance, ROC-AUC looks close (trivial 0.943 vs mp_raw 0.995).
PR-AUC exposes the real gap — <strong>8.6× lift</strong> (0.892 vs 0.104). PR-AUC is the operative metric.
</div>

---

<!-- Slide 3 -->

# Metric Hierarchy Under Realistic Imbalance

At 1:100 attack-to-benign ratio, **ROC-AUC is not the decision-relevant metric**.

| Scorer | ROC-AUC | PR-AUC | Reading |
|--------|:-------:|:------:|---------|
| Trivial set-membership | 0.943 | 0.104 | ROC inflates because enrollment negatives are novel too |
| mp_rank_norm | 0.972 | 0.215 | CDF transform compresses score margin |
| **mp_raw** | **0.995** | **0.892** | Raw cosine preserves the margin positives need |

<div class="hero">
<strong>Why ROC misleads here:</strong> RBA chain-sampling makes enrollment negatives largely novel,
so trivial scores them 1.0 alongside spoofs — partially correct, inflating ROC past 0.94.
PR-AUC correctly penalizes the false positive mass.
</div>

**k=1 resolution:** H5 failed at k=1 (ROC 0.530, 30-token vocabulary). H6 succeeds (ROC 0.995, PR 0.892, 229-token RBA-calibrated vocabulary). k=1 failure was vocabulary poverty — not a fundamental limit of mean-pool embeddings.

---

<!-- Slide 4 -->

# ⚠️ Two Silent Failure Modes: Config Collapse & Rank-Norm

**(1) Training config collapse:** CBOW + per-event corpus converges all timezone vectors to sim=0.9993, destroying spoof signal. Novel/fleet AUC look fine. Only a within-feature similarity check catches it.

| Configuration | Within-feature sim | Spoof signal |
|--------------|:------------------:|:------------:|
| CBOW + per-event corpus | **0.9993** — collapse | Destroyed |
| **Skip-gram + per-account corpus** | **0.392** — healthy | **PR-AUC 0.892** |

**(2) Rank-normalization collapses PR-AUC under imbalance:** Previously recommended for cross-account calibration. At 1:100, it drops PR-AUC **0.892 → 0.215** while preserving ROC-AUC — exactly the metric that misleads.

<div class="hero">
<strong>Do not use rank-norm in production scoring.</strong> It is only appropriate for balanced evaluation.
Keep the within-feature similarity health check — alert if any feature's mean intra-class sim &gt; 0.5.
</div>

---

<!-- Slide 5 -->

# Recommended Production Architecture (Two-Layer Composite)

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
<strong>REMOVED:</strong> per-account known-device gate (fallback) — blinds model during cold-start fleet window.
<strong>REMOVED:</strong> rank-normalization — collapses PR-AUC (0.892 → 0.215) under realistic imbalance.
<strong>REMOVED:</strong> Word2Vec offline fleet batch — system-level blocklist handles post-lag fleet with certainty; mp_raw handles the pre-lag residual.
</div>
