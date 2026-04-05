# ATO Detection via Device Embeddings

A research investigation into whether word-embedding models trained on per-account
device login sequences can detect account takeover (ATO). The investigation runs from
an initial proof-of-concept through systematic critique, formal debate, three
experiments, and production deployment analysis — culminating in a full ml-lab
structured investigation of the mean-pool vs. concat architectural choice.

**The short answer:** FastText on opaque device ID strings does not work — its character
n-grams destroy per-account cluster structure (silhouette −0.051). Word2Vec on device
IDs works for offline fleet/reuse detection (AUC 0.891). FastText on structured feature
tokens (OS, browser, timezone, language) is the recommended real-time signal (AUC 0.985
on novel attacks with realistic enrollment in the negative class). Mean-pooling six
feature-token embeddings outperforms concatenated-string FastText — confirmed 7/7 tests
under a robust training configuration (sg=1, per-account corpus).

Read `pre_ml_lab/docs/REPORT.md` for the full self-contained account of Experiments 1–3.
Read `TECHNICAL_REPORT.md` for the definitive H2 synthesis.

---

## Quickstart

Each script is self-contained with inline dependencies ([PEP 723](https://peps.python.org/pep-0723/)).
No virtualenv setup required.

```bash
# Original proof-of-concept (FastText on device IDs, AUC 0.836)
uv run pre_ml_lab/experiments/ato_fasttext_poc.py

# Experiment 2: FastText vs Word2Vec vs OOV baseline, i.i.d. + Markov corpus
uv run pre_ml_lab/experiments/ato_experiment2.py

# Experiment 3: fleet corpus, feature embeddings, enrollment evaluation
uv run pre_ml_lab/experiments/ato_experiment3.py

# Regenerate Experiment 2 conclusion figures (fig1–fig6)
uv run pre_ml_lab/experiments/plot_conclusions.py
```

---

## File inventory

### Scripts

| File | Purpose |
|------|---------|
| `pre_ml_lab/experiments/ato_fasttext_poc.py` | Original PoC: FastText on device ID sequences, centroid-based scoring, ROC-AUC evaluation, UMAP visualization |
| `pre_ml_lab/experiments/ato_experiment2.py` | Experiment 2: FastText vs Word2Vec vs OOV binary baseline; two corpus modes; bootstrap CIs; silhouette; stratified AUC; centroid norm |
| `pre_ml_lab/experiments/ato_experiment3.py` | Experiment 3: shared fraud fleet corpus; feature token embeddings; 6 signals × 3 attack types × 2 corpus modes; enrollment in negative class |
| `pre_ml_lab/experiments/plot_conclusions.py` | Generates the 6 figures referenced in Experiment 2 conclusions |
| `pre_ml_lab/experiments/ato_concat_poc.py` | H2 original PoC: mean-pool vs. concat FastText, AUC + silhouette |
| `pre_ml_lab/experiments/ato_concat_experiment.py` | H2 five-test experiment: window sweep, prefix format, tz permutation, OOV injection |
| `pre_ml_lab/experiments/h2_rerun_poc.py` | H2 rerun PoC (agent-run): clean independent implementation of mean-pool vs. concat |
| `pre_ml_lab/experiments/h2_rerun_experiment1.py` | H2 rerun experiment: bootstrap CIs, window sweep, prefix format, trivial baseline, tz permutation |
| `h2_ml_lab/experiments/ato_device_embedding_poc.py` | H2 ml-lab PoC: mean-pool vs. concat, ROC-AUC + compactness (degenerate config — reveals CBOW collapse) |
| `h2_ml_lab/experiments/ato_device_embedding_experiment2.py` | H2 ml-lab experiment iteration 1: 8 debate-agreed tests |
| `h2_ml_lab/experiments/ato_device_embedding_experiment3.py` | H2 ml-lab experiment iteration 2: corrected T2 and T6 |
| `h2_ml_lab/experiments/robust_config_experiment.py` | Supplemental: T4, T6, T8 under robust config (sg=1, per-account corpus) |
| `h2_ml_lab/experiments/config_verification.py` | T8 comparison: degenerate vs. robust config — root cause of ml-lab vs. H2_RERUN divergence |

### Research documents

| File | Purpose |
|------|---------|
| `pre_ml_lab/docs/REPORT.md` | **Start here (Exps 1–3).** Self-contained report covering all three experiments, the full debate arc, production deployment constraints, and the final recommendation |
| `pre_ml_lab/docs/CONCLUSIONS.md` | Detailed findings from Experiments 2 and 3 with debate scorecard, per-finding verdicts, and signal hierarchy |
| `pre_ml_lab/docs/REPORT_ADDENDUM.md` | Production deployment analysis: rotational instability math, retraining pressure estimates, Procrustes alignment, revised architecture |
| `pre_ml_lab/docs/CRITIQUE.md` | Ten-point adversarial critique of the original PoC from first principles |
| `pre_ml_lab/docs/DEFENSE.md` | Point-by-point rebuttal of the critique |
| `pre_ml_lab/docs/DEBATE.md` | Multi-turn argument on each contested point, resolving to concession or agreed empirical test |
| `pre_ml_lab/docs/H2_REPORT.md` | H2 original investigation: full five-test experiment report with debate scorecard |
| `pre_ml_lab/docs/H2_RERUN_REPORT.md` | H2 rerun report (agent-run): self-contained account of all findings, mechanism analysis, and recommendation |
| `pre_ml_lab/docs/H2_RERUN_CONCLUSIONS.md` | H2 rerun: per-test verdicts, scorecard, and revised mechanism understanding |
| `pre_ml_lab/docs/H2_RERUN_DEBATE.md` | H2 rerun: multi-round debate between critic and defense agents |
| `pre_ml_lab/docs/H2_RERUN_CRITIQUE.md` | H2 rerun: adversarial critique of the H2 PoC |
| `pre_ml_lab/docs/H2_RERUN_DEFENSE.md` | H2 rerun: point-by-point defense of the H2 hypothesis |
| `h2_ml_lab/docs/HYPOTHESIS.md` | H2 ml-lab: canonical hypothesis and metrics |
| `h2_ml_lab/docs/CRITIQUE.md` | H2 ml-lab: adversarial critique (ml-critic agent) |
| `h2_ml_lab/docs/DEFENSE.md` | H2 ml-lab: design defense (ml-defender agent) |
| `h2_ml_lab/docs/DEBATE.md` | H2 ml-lab: multi-turn debate to 8 agreed empirical tests |
| `h2_ml_lab/docs/CONCLUSIONS.md` | H2 ml-lab: per-test verdicts, surprise findings, macro-iteration assessment |
| `h2_ml_lab/docs/REPORT.md` | H2 ml-lab: full investigation report |
| `h2_ml_lab/docs/REPORT_ADDENDUM.md` | H2 ml-lab: production re-evaluation and deployment recommendation |
| `h2_ml_lab/docs/PEER_REVIEW_R1.md` | H2 ml-lab: round 1 peer review (3 MAJOR issues identified and resolved) |
| `h2_ml_lab/docs/PEER_REVIEW_R2.md` | H2 ml-lab: round 2 peer review (2 MINOR issues, no MAJOR issues) |
| `TECHNICAL_REPORT.md` | **Definitive H2 synthesis.** Publication-ready report integrating H2_RERUN and ml-lab findings, full results tables, configuration sensitivity analysis, and deployment recommendation |
| `archive/PROCESS.md` | Prescriptive 9-step methodology for ML hypothesis investigation (general-purpose) |
| `archive/agent.md` | Agent-executable version of PROCESS.md for running this methodology on any DS/ML hypothesis |

---

## H2 Investigation — Mean-Pool vs. Concatenated-String FastText

A follow-on investigation testing whether mean-pooling six feature-token embeddings
outperforms concatenating feature values into a single FastText token.

**Hypothesis H2:** Mean-pooling six feature token embeddings will outperform concatenated-string
FastText because (a) n-gram bleed across feature boundaries contributes spurious signal
uncorrelated with any semantic dimension, and (b) a mismatch at one feature dimension dilutes
the signal from all six under concat but leaves other dimensions fully independent under
mean-pool. Both effects are measurable as lower silhouette score and lower ROC-AUC.

### H2 original investigation

```bash
# H2 original proof-of-concept
uv run pre_ml_lab/experiments/ato_concat_poc.py

# H2 five-test experiment (window sweep, prefix format, tz permutation, OOV injection)
uv run pre_ml_lab/experiments/ato_concat_experiment.py
```

The original investigation ran a five-test experiment (T1–T4) and found a **split verdict**: at
matched window sizes, concat closes most of the gap on novel and fleet attacks, but a residual
spoof-detection gap persists (+0.043 AUC). The original PoC's apparent H2 support was driven
by a window asymmetry (`window=1` for concat vs `window=6` for mean-pool). See `pre_ml_lab/docs/H2_REPORT.md`
for the full account.

### H2 rerun — conducted with the `ml-hypothesis-investigator` agent

The H2 hypothesis was re-run end-to-end using the structured
[`ml-hypothesis-investigator`](archive/agent.md) agent: independent PoC → adversarial critique →
design defense → multi-round debate → five pre-registered experiments → production evaluation
→ self-contained report.

```bash
# H2 rerun proof-of-concept (fresh, independent implementation)
uv run pre_ml_lab/experiments/h2_rerun_poc.py

# H2 rerun experiment (bootstrap CIs, window sweep, prefix format, trivial baseline, tz permutation)
uv run pre_ml_lab/experiments/h2_rerun_experiment1.py
```

**Rerun verdict: H2 confirmed — 7/7 empirical tests support mean-pool.** Unlike the original
investigation, pre-registered thresholds were applied: window equalization is credited as a
"critique wins" only if it recovers ≥50% of the AUC delta. Concat w=6 recovered only 43.6%
of the spoof delta — below that threshold. Key findings:

| Metric | Mean-pool advantage | Bootstrap 95% CI |
|--------|--------------------|--------------------|
| Silhouette | +0.119 | [+0.073, +0.133] |
| Spoof AUC | +0.055 | [+0.034, +0.077] |
| Novel AUC | +0.012 | [+0.006, +0.018] |
| Fleet AUC | +0.006 | [+0.001, +0.012] |

See `pre_ml_lab/docs/H2_RERUN_REPORT.md` for the full self-contained account.

### H2 ml-lab investigation — structured 10-step workflow with peer review

A third, independent pass using the full `ml-lab` agent workflow: critic/defender debate
agents, two macro-iterations, and two rounds of peer review. This run uncovered a critical
configuration sensitivity finding.

```bash
# H2 ml-lab proof-of-concept
uv run h2_ml_lab/experiments/ato_device_embedding_poc.py

# Full experiment (8 debate-agreed tests, two iterations)
uv run h2_ml_lab/experiments/ato_device_embedding_experiment2.py

# Supplemental: T4/T6/T8 under robust config
uv run h2_ml_lab/experiments/robust_config_experiment.py

# Config verification: degenerate vs. robust T8 comparison
uv run h2_ml_lab/experiments/config_verification.py
```

The ml-lab PoC initially found the **opposite** conclusion — concat outperformed mean-pool on
all attack types. The root cause was identified via T8 (token similarity analysis): the default
gensim configuration (CBOW, per-event corpus) causes **within-feature embedding collapse**,
driving within-feature cosine similarity to 0.9993. Under collapse, all timezone values become
nearly identical vectors, eliminating mean-pool's architectural advantage and making concat's
cross-boundary n-grams the dominant signal. Switching to skip-gram with per-account corpus
(within-feature sim = 0.392) restores the expected result.

**Final verdict under robust config: H2 confirmed — 7/7.** All tests resolve in the mean-pool
direction. Mean-pool (0.818) is the only configuration that beats the trivial set-membership
baseline on spoof (0.791); concat w=1 (0.763) falls below it.

| Configuration | Within-feature sim | Spoof AUC (mean-pool) | H2 verdict |
|--------------|--------------------|-----------------------|------------|
| ml-lab PoC (CBOW, per-event) | 0.9993 — collapse | 0.384 (below chance) | Refuted |
| Robust (sg=1, per-account) | 0.392 — no collapse | 0.818 | **Confirmed** |

**T8 is a required deployment health check.** The collapse is silent — novel and fleet AUC
remain plausible under the degenerate config. Only within-feature similarity analysis (T8)
detects it before it degrades spoof detection in production. A within-feature cosine similarity
above 0.5 after any retraining cycle should halt deployment and trigger corpus construction review.

See `TECHNICAL_REPORT.md` for the full synthesis across all three H2 investigations.

---

## Key findings

| Finding | Result |
|---------|--------|
| FastText on device IDs (silhouette) | −0.051 — no cluster structure |
| Word2Vec on device IDs (silhouette) | +0.941 — near-perfect account separation |
| Word2Vec centroid AUC (i.i.d.) | 0.982 |
| OOV binary baseline AUC (Exp 2) | 0.989 — evaluation artifact (see below) |
| feature_fasttext AUC, novel attacks | 0.985 with enrollment in negative class |
| feature_fasttext AUC, fleet/reuse | 0.920 |
| id_w2v AUC, fleet/reuse | 0.891 |
| global_oov AUC, fleet attacks | 0.250 — anti-correlated with fleet detection |
| Mean-pool spoof AUC (robust config) | 0.818 — beats trivial baseline (+0.027) |
| Concat w=1 spoof AUC (robust config) | 0.763 — below trivial baseline (−0.028) |
| Mean-pool compactness vs. concat | 3.4× tighter per-account centroid clusters |
| CBOW collapse within-feature sim | 0.9993 — eliminates timezone discriminability |
| Robust config within-feature sim | 0.392 — confirmed no collapse |

**On the 0.989 OOV baseline result:** Experiment 2's evaluation design gave every
account unique device IDs with no legitimate enrollment events in the negative class.
Under these conditions, detecting an attack was equivalent to detecting a globally
unseen device ID — a trivially separable task. Under the corrected evaluation in
Experiment 3 (cross-account fleet devices in vocabulary; legitimate enrollment in
negative class), the OOV binary baseline collapses to AUC 0.750 on novel/spoof attacks
and AUC 0.250 on fleet attacks.

---

## Recommended architecture

```
Real-time path (<1ms per event):
  tokens = [f"os_{os}", f"br_{browser}", f"tz_{tz}", f"lang_{lang}",
            f"net_{network}", f"sc_{screen}"]
  embedding = mean([fasttext[t] for t in tokens])
  risk_score = cosine_distance(embedding, account_centroid)
  → step-up auth if risk_score > threshold

Offline batch (monthly retraining):
  Build per-account corpus: flatten all confirmed login events per account into one sentence
  Train FastText: sg=1, epochs=20, negative=10, min_n=3, max_n=6, window=6, vector_size=64
  T8 health check: within-feature cosine sim → halt if > 0.5 (embedding collapse)
  Recompute all account centroids

Offline batch (daily/weekly):
  account.device_id_history → id_w2v centroid distance → fleet/reuse review queue

Fallback (service unavailable, or account < 20 confirmed events):
  login_event.device_profile → per-account known-device hash set → step-up auth if new
```

See `pre_ml_lab/docs/REPORT.md` §9 for the full recommendation with reasoning,
`pre_ml_lab/docs/REPORT_ADDENDUM.md` for the production deployment analysis, and
`TECHNICAL_REPORT.md` §5 for the H2-specific deployment configuration and risk assessment.
