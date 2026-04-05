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

**Answer:** Yes — with the right signal and training configuration.

> FastText on structured feature tokens achieves **AUC 0.985** on novel attacks
> at sub-millisecond inference latency.

&nbsp;

Three experiments · Adversarial critique & debate · Peer review · Production analysis

&nbsp;

`github.com/chris-santiago/ato-device-embeddings`

---

<!-- Slide 2 -->

# Which Signal Works?

Three experiments compared embedding strategies across attack types.

| Signal | Novel AUC | Fleet AUC | Spoof AUC | Verdict |
|--------|:---------:|:---------:|:---------:|---------|
| **Feature tokens, mean-pool** | **0.993** | **0.939** | **0.818** | ✅ Recommended |
| Feature tokens, concat string | 0.981 | 0.933 | 0.763 | ❌ Below trivial on spoof |
| Word2Vec on device IDs | — | 0.891 | — | ✅ Offline fleet detection only |
| FastText on device IDs | silhouette −0.051 | — | — | ❌ Destroys cluster structure |
| Trivial set-membership | 0.750 | 0.750 | 0.791 | Fallback only |

<div class="hero">
<strong>Key insight:</strong> Mean-pool is the <em>only</em> configuration that beats the trivial
baseline on spoof attacks — the hardest case, where the attacker matches the victim's OS,
browser, and language and differs only on timezone.
</div>

---

<!-- Slide 3 -->

# Why Mean-Pool Beats Concatenated Strings

Character n-grams in a concat string span feature boundaries (`ari_ut`, `i_utc`), injecting spurious signal that dilutes any single mismatched feature. Mean-pooling embeds each feature independently — a differing timezone contributes its full weight to the score.

| Test | Result | Verdict |
|------|--------|:-------:|
| Bootstrap CIs on all 4 deltas | All exclude zero | ✅ |
| Window sweep (w=1 to w=6) | Best concat recovers only 43.6% of gap | ✅ |
| Prefixed-concat format | Silhouette gap 0.090 > 0.05 threshold | ✅ |
| Tz-position permutation | Every concat ordering below mean-pool spoof AUC | ✅ |
| Trivial baseline comparison | Mean-pool +0.027; concat −0.028 over trivial | ✅ |

**Score: 7/7 — confirmed across three independent investigations.**

---

<!-- Slide 4 -->

# ⚠️ Critical Finding: Training Configuration Matters

An initial investigation reached the **opposite conclusion** — concat beat mean-pool on every metric. Root cause: a degenerate training configuration causes silent within-feature embedding collapse. All timezone values converge to near-identical vectors (sim = 0.9993), eliminating the signal mean-pool depends on.

| Configuration | Within-feature sim | Spoof AUC | Conclusion |
|--------------|--------------------|:---------:|------------|
| CBOW + per-event corpus | **0.9993** — collapse | 0.384 *(below chance)* | Mean-pool refuted |
| **Skip-gram + per-account corpus** | **0.392** — healthy | **0.818** | **Mean-pool confirmed** |

<div class="hero">
<strong>This failure is silent.</strong> Novel and fleet AUC look reasonable even under collapse — only a within-feature token similarity check (T8) detects it. Run after every retraining cycle; alert if similarity &gt; 0.5.
</div>

---

<!-- Slide 5 -->

# Recommended Production Architecture

```
At login (<1ms):
  tokens     = [os_X, br_X, tz_X, lang_X, net_X, sc_X]
  embedding  = mean(fasttext[t] for t in tokens)
  risk_score = cosine_distance(embedding, account_centroid)
  → step-up auth if risk_score > threshold

Monthly retraining:
  corpus: one sentence per account (all login events concatenated)
  FastText: sg=1, epochs=20, negative=10, window=6, vector_size=64
  ✓ Health check: within-feature cosine sim — halt deployment if > 0.5

Offline (daily/weekly):
  device_id_history → Word2Vec centroid → fleet/reuse review queue

Fallback (<20 events or service unavailable):
  per-account known-device hash set → step-up if unseen profile
```

**Risk to monitor:** The mean-pool spoof advantage over the trivial baseline is +0.027 — statistically significant but operationally narrow. Include spoof-specific AUC in any A/B evaluation before cutover.
