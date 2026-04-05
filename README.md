# ATO Detection via Device Embeddings

Can word embeddings trained on login device sequences detect account takeover in real time?
**Yes — with the right architecture.** FastText on structured feature tokens (OS, browser,
timezone, language) achieves AUC 0.985 on novel attacks and is fast enough for sub-millisecond
login scoring. Raw device ID embeddings do not work.

A secondary finding with operational consequences: **how you train the embeddings matters as
much as the architecture.** The wrong training configuration (CBOW + per-event corpus) silently
destroys spoof-attack detection while leaving other metrics intact.

---

## Results at a glance

| Signal | Novel AUC | Fleet AUC | Spoof AUC | Verdict |
|--------|-----------|-----------|-----------|---------|
| Feature tokens, mean-pool (recommended) | 0.993 | 0.939 | **0.818** | ✓ Deploy |
| Feature tokens, concat string | 0.981 | 0.933 | 0.763 | ✗ Below trivial baseline on spoof |
| Word2Vec on device IDs | — | 0.891 | — | ✓ Offline fleet/reuse detection only |
| FastText on device IDs | — | — | — | ✗ Destroys cluster structure (silhouette −0.051) |
| Exact set-membership (trivial baseline) | 0.750 | 0.750 | 0.791 | Fallback only |

**Mean-pool FastText is the only configuration that beats the trivial baseline on spoof attacks**
(the hardest attack type, where the attacker matches the victim's OS, browser, and language and
differs only on timezone). Concat string FastText falls below it.

---

## Recommended deployment

```
At login time (<1ms):
  tokens = [f"os_{os}", f"br_{browser}", f"tz_{tz}",
            f"lang_{lang}", f"net_{network}", f"sc_{screen}"]
  embedding  = mean([fasttext[t] for t in tokens])
  risk_score = cosine_distance(embedding, account_centroid)
  → step-up auth if risk_score > threshold

Monthly retraining:
  corpus: one sentence per account (all login events concatenated)
  FastText: sg=1, epochs=20, negative=10, min_n=3, max_n=6, window=6, vector_size=64
  Health check: within-feature cosine similarity → halt deployment if > 0.5

Offline (daily/weekly):
  device_id_history → Word2Vec centroid distance → fleet/reuse review queue

Fallback (<20 confirmed events, or embedding service down):
  per-account known-device hash set → step-up auth if unseen profile
```

See `TECHNICAL_REPORT.md` for the full deployment configuration and risk assessment.

---

## Quickstart

Scripts are self-contained with inline dependencies ([PEP 723](https://peps.python.org/pep-0723/)). No virtualenv setup required.

```bash
# Recommended signal: mean-pool feature token embeddings
uv run pre_ml_lab/experiments/ato_experiment3.py

# Mean-pool vs. concat head-to-head (robust config)
uv run pre_ml_lab/experiments/h2_rerun_poc.py

# Full H2 experiment: bootstrap CIs, window sweep, trivial baseline, tz permutation
uv run pre_ml_lab/experiments/h2_rerun_experiment1.py

# ml-lab structured investigation with config verification
uv run h2_ml_lab/experiments/robust_config_experiment.py
uv run h2_ml_lab/experiments/config_verification.py
```

---

## What we investigated

### Experiments 1–3: which signal works for ATO?

Three experiments tested FastText and Word2Vec on two device representations:
- **Opaque device IDs** (e.g., `device_8472a`) — character n-grams destroy account cluster
  structure; silhouette −0.051. Does not work.
- **Structured feature tokens** (e.g., `os_ios`, `tz_utc-5`) — FastText learns meaningful
  per-account cluster structure. AUC 0.985 on novel attacks under realistic evaluation
  (enrollment events in the negative class, cross-account fleet devices in vocabulary).

An apparent 0.989 AUC for a simple OOV binary baseline in Experiment 2 was an evaluation
artifact: every account had unique device IDs and no enrollment events, making attack detection
equivalent to flagging any globally unseen device. Under the corrected evaluation in Experiment 3,
the OOV baseline collapses to 0.750 on novel/spoof and 0.250 on fleet attacks.

### Follow-on hypothesis: mean-pool vs. concat feature embeddings

Once structured feature tokens were established as the right representation, the question
became: should you embed the features as a single concatenated string
(`ios_safari_utc-5_en_us_wifi_small`) or mean-pool six independent feature token embeddings?

**Mean-pool wins — 7/7 pre-specified tests.** The mechanism: character n-grams in the
concatenated string span feature boundaries, injecting spurious signal that dilutes the
discriminative information in any single differing feature (e.g., timezone). Mean-pooling
embeds each feature independently, so a single mismatched feature contributes its full signal
to the distance score.

This was tested three ways across independent investigations — original experiment, agent rerun,
and a structured [ml-debate-lab](https://github.com/chris-santiago/ml-debate-lab) workflow —
all reaching the same conclusion under a robust training configuration.

### Critical configuration finding

A degenerate training configuration (CBOW objective + per-event corpus) causes
**within-feature embedding collapse**: all timezone values converge to nearly identical vectors
(cosine similarity 0.9993). Under collapse, mean-pool carries no timezone signal and the
concat string wins — the opposite conclusion. This failure is silent at the novel and fleet
AUC level; only a token similarity diagnostic (T8) detects it.

| Configuration | Within-feature sim | Spoof AUC | Conclusion |
|--------------|--------------------|-----------|------------|
| CBOW, per-event corpus (broken) | 0.9993 — collapse | 0.384 (below chance) | Mean-pool refuted |
| Skip-gram, per-account corpus (correct) | 0.392 — healthy | 0.818 | **Mean-pool confirmed** |

**The skip-gram + per-account corpus configuration is not optional.** Monitor within-feature
similarity after every retraining cycle.

---

## Investigation artifacts

| Directory | Contents |
|-----------|----------|
| `pre_ml_lab/` | Experiments 1–3 and the original/rerun H2 investigation |
| `h2_ml_lab/` | Structured ml-debate-lab investigation with critic/defender agents and peer review |
| `TECHNICAL_REPORT.md` | Definitive H2 synthesis: full results, configuration sensitivity analysis, deployment recommendation |
| `archive/` | Original methodology documents that preceded the ml-debate-lab tool |

<details>
<summary>Full file inventory</summary>

### Scripts

| File | Purpose |
|------|---------|
| `pre_ml_lab/experiments/ato_fasttext_poc.py` | Original PoC: FastText on device ID sequences |
| `pre_ml_lab/experiments/ato_experiment2.py` | Experiment 2: FastText vs Word2Vec vs OOV baseline |
| `pre_ml_lab/experiments/ato_experiment3.py` | Experiment 3: fleet corpus, feature token embeddings, corrected enrollment evaluation |
| `pre_ml_lab/experiments/plot_conclusions.py` | Generates Experiment 2 figures |
| `pre_ml_lab/experiments/ato_concat_poc.py` | H2 original PoC: mean-pool vs. concat |
| `pre_ml_lab/experiments/ato_concat_experiment.py` | H2 five-test experiment |
| `pre_ml_lab/experiments/h2_rerun_poc.py` | H2 rerun PoC (independent implementation) |
| `pre_ml_lab/experiments/h2_rerun_experiment1.py` | H2 rerun: bootstrap CIs, window sweep, trivial baseline, tz permutation |
| `h2_ml_lab/experiments/ato_device_embedding_poc.py` | H2 ml-lab PoC (reveals CBOW collapse) |
| `h2_ml_lab/experiments/ato_device_embedding_experiment2.py` | H2 ml-lab experiment iteration 1 |
| `h2_ml_lab/experiments/ato_device_embedding_experiment3.py` | H2 ml-lab experiment iteration 2 |
| `h2_ml_lab/experiments/robust_config_experiment.py` | T4/T6/T8 diagnostics under robust config |
| `h2_ml_lab/experiments/config_verification.py` | T8 comparison: degenerate vs. robust config |

### Research documents

| File | Purpose |
|------|---------|
| `TECHNICAL_REPORT.md` | Definitive H2 synthesis |
| `pre_ml_lab/docs/REPORT.md` | Full report: Experiments 1–3, debate arc, production constraints |
| `pre_ml_lab/docs/CONCLUSIONS.md` | Per-finding verdicts and signal hierarchy |
| `pre_ml_lab/docs/REPORT_ADDENDUM.md` | Production deployment analysis |
| `pre_ml_lab/docs/H2_REPORT.md` | H2 original investigation report |
| `pre_ml_lab/docs/H2_RERUN_REPORT.md` | H2 rerun report |
| `h2_ml_lab/docs/REPORT.md` | H2 ml-lab investigation report |
| `h2_ml_lab/docs/CONCLUSIONS.md` | H2 ml-lab per-test verdicts |
| `h2_ml_lab/docs/PEER_REVIEW_R1.md` | Round 1 peer review (3 MAJOR issues resolved) |
| `h2_ml_lab/docs/PEER_REVIEW_R2.md` | Round 2 peer review (2 MINOR issues, no MAJOR) |

</details>
