# ATO Detection via Device Embeddings

A research investigation into whether word-embedding models trained on per-account
device login sequences can detect account takeover (ATO). The investigation runs from
an initial proof-of-concept through systematic critique, formal debate, three
experiments, and production deployment analysis.

**The short answer:** FastText on opaque device ID strings does not work — its character
n-grams destroy per-account cluster structure (silhouette −0.051). Word2Vec on device
IDs works for offline fleet/reuse detection (AUC 0.891). FastText on structured feature
tokens (OS, browser, timezone, language) is the recommended real-time signal (AUC 0.985
on novel attacks with realistic enrollment in the negative class).

Read `REPORT.md` for the full self-contained account.

---

## Quickstart

Each script is self-contained with inline dependencies ([PEP 723](https://peps.python.org/pep-0723/)).
No virtualenv setup required.

```bash
# Original proof-of-concept (FastText on device IDs, AUC 0.836)
uv run experiments/ato_fasttext_poc.py

# Experiment 2: FastText vs Word2Vec vs OOV baseline, i.i.d. + Markov corpus
uv run experiments/ato_experiment2.py

# Experiment 3: fleet corpus, feature embeddings, enrollment evaluation
uv run experiments/ato_experiment3.py

# Regenerate Experiment 2 conclusion figures (fig1–fig6)
uv run experiments/plot_conclusions.py
```

---

## File inventory

### Scripts

| File | Purpose |
|------|---------|
| `experiments/ato_fasttext_poc.py` | Original PoC: FastText on device ID sequences, centroid-based scoring, ROC-AUC evaluation, UMAP visualization |
| `experiments/ato_experiment2.py` | Experiment 2: FastText vs Word2Vec vs OOV binary baseline; two corpus modes; bootstrap CIs; silhouette; stratified AUC; centroid norm |
| `experiments/ato_experiment3.py` | Experiment 3: shared fraud fleet corpus; feature token embeddings; 6 signals × 3 attack types × 2 corpus modes; enrollment in negative class |
| `experiments/plot_conclusions.py` | Generates the 6 figures referenced in Experiment 2 conclusions |
| `experiments/ato_concat_poc.py` | H2 original PoC: mean-pool vs. concat FastText, AUC + silhouette |
| `experiments/ato_concat_experiment.py` | H2 five-test experiment: window sweep, prefix format, tz permutation, OOV injection |
| `experiments/h2_rerun_poc.py` | H2 rerun PoC (agent-run): clean independent implementation of mean-pool vs. concat |
| `experiments/h2_rerun_experiment1.py` | H2 rerun experiment: bootstrap CIs, window sweep, prefix format, trivial baseline, tz permutation |

### Research documents

| File | Purpose |
|------|---------|
| `docs/REPORT.md` | **Start here.** Self-contained report covering all three experiments, the full debate arc, production deployment constraints, and the final recommendation |
| `docs/CONCLUSIONS.md` | Detailed findings from Experiments 2 and 3 with debate scorecard, per-finding verdicts, and signal hierarchy |
| `docs/REPORT_ADDENDUM.md` | Production deployment analysis: rotational instability math, retraining pressure estimates, Procrustes alignment, revised architecture |
| `docs/CRITIQUE.md` | Ten-point adversarial critique of the original PoC from first principles |
| `docs/DEFENSE.md` | Point-by-point rebuttal of the critique |
| `docs/DEBATE.md` | Multi-turn argument on each contested point, resolving to concession or agreed empirical test |
| `docs/H2_REPORT.md` | H2 original investigation: full five-test experiment report with debate scorecard |
| `docs/H2_RERUN_REPORT.md` | H2 rerun report (agent-run): self-contained account of all findings, mechanism analysis, and recommendation |
| `docs/H2_RERUN_CONCLUSIONS.md` | H2 rerun: per-test verdicts, scorecard, and revised mechanism understanding |
| `docs/H2_RERUN_DEBATE.md` | H2 rerun: multi-round debate between critic and defense agents |
| `docs/H2_RERUN_CRITIQUE.md` | H2 rerun: adversarial critique of the H2 PoC |
| `docs/H2_RERUN_DEFENSE.md` | H2 rerun: point-by-point defense of the H2 hypothesis |
| `PROCESS.md` | Prescriptive 9-step methodology for ML hypothesis investigation (general-purpose) |
| `agent.md` | Agent-executable version of PROCESS.md for running this methodology on any DS/ML hypothesis |

---

## H2 Investigation — Mean-Pool vs. Concatenated-String FastText

A follow-on investigation testing whether mean-pooling six feature-token embeddings
outperforms concatenating feature values into a single FastText token.

**Hypothesis H2:** Mean-pooling six feature token embeddings will outperform concatenated-string
FastText because (a) n-gram bleed across feature boundaries contributes spurious signal
uncorrelated with any semantic dimension, and (b) front-loaded positional weighting means a
mismatch at feature N penalizes n-gram overlap for all features that follow. Both effects are
measurable as lower silhouette score and lower ROC-AUC.

### H2 original investigation

```bash
# H2 original proof-of-concept
uv run experiments/ato_concat_poc.py

# H2 five-test experiment (window sweep, prefix format, tz permutation, OOV injection)
uv run experiments/ato_concat_experiment.py
```

The original investigation ran a five-test experiment (T1–T4) and found a **split verdict**: at
matched window sizes, concat closes most of the gap on novel and fleet attacks, but a residual
spoof-detection gap persists (+0.043 AUC). The original PoC's apparent H2 support was driven
by a window asymmetry (`window=1` for concat vs `window=6` for mean-pool). See `docs/H2_REPORT.md`
for the full account.

### H2 rerun — conducted with the `ml-hypothesis-investigator` agent

The H2 hypothesis was re-run end-to-end using the structured
[`ml-hypothesis-investigator`](agent.md) agent: independent PoC → adversarial critique →
design defense → multi-round debate → five pre-registered experiments → production evaluation
→ self-contained report.

```bash
# H2 rerun proof-of-concept (fresh, independent implementation)
uv run experiments/h2_rerun_poc.py

# H2 rerun experiment (bootstrap CIs, window sweep, prefix format, trivial baseline, tz permutation)
uv run experiments/h2_rerun_experiment1.py
```

**Rerun verdict: H2 confirmed — 7/7 empirical tests support mean-pool.** Unlike the original
investigation, pre-registered thresholds were applied: window equalization is credited as a
"critique wins" only if it recovers ≥50% of the AUC delta. Concat w=6 recovered only 43.6%
of the spoof delta — below that threshold. Key findings:

| Metric | Mean-pool advantage | Bootstrap 95% CI |
|--------|--------------------|--------------------|
| Silhouette | +0.119 | [+0.073, +0.133] |
| Spoof AUC | +0.054 | [+0.034, +0.077] |
| Novel AUC | +0.012 | [+0.005, +0.022] |
| Fleet AUC | +0.006 | [+0.000, +0.013] |

**Mechanism correction:** The original "front-loaded positional weighting" framing was wrong.
The tz-permutation test showed that moving tz *later* makes spoof AUC *worse* (tz at last
position: AUC 0.655 vs. baseline 0.763). The correct mechanism is **cumulative cross-boundary
n-gram contamination** — every feature after a mismatched feature has its n-gram distribution
corrupted. Mean-pooling eliminates this entirely by embedding each feature independently.

See `docs/H2_RERUN_REPORT.md` for the full self-contained account.

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
Real-time path (< 10ms):
  login_event.device_features → feature_fasttext centroid distance → risk score

Offline batch (daily/weekly):
  account.device_id_history → id_w2v centroid distance → fleet/reuse review queue

Operational gate (milliseconds):
  login_event.device_id → per-account known-device set → step-up auth if new
```

See `REPORT.md` §9 for the full recommendation with reasoning, and
`REPORT_ADDENDUM.md` for the production deployment analysis.
