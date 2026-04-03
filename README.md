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

### Research documents

| File | Purpose |
|------|---------|
| `docs/REPORT.md` | **Start here.** Self-contained report covering all three experiments, the full debate arc, production deployment constraints, and the final recommendation |
| `docs/CONCLUSIONS.md` | Detailed findings from Experiments 2 and 3 with debate scorecard, per-finding verdicts, and signal hierarchy |
| `docs/REPORT_ADDENDUM.md` | Production deployment analysis: rotational instability math, retraining pressure estimates, Procrustes alignment, revised architecture |
| `docs/CRITIQUE.md` | Ten-point adversarial critique of the original PoC from first principles |
| `docs/DEFENSE.md` | Point-by-point rebuttal of the critique |
| `docs/DEBATE.md` | Multi-turn argument on each contested point, resolving to concession or agreed empirical test |
| `PROCESS.md` | Prescriptive 9-step methodology for ML hypothesis investigation (general-purpose) |
| `agent.md` | Agent-executable version of PROCESS.md for running this methodology on any DS/ML hypothesis |

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
