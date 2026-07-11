# Project Chronology — ATO Device-Embedding Investigation

*A dated narrative of the entire repository, from the first proof-of-concept (2026-04-03) to today. Generated 2026-07-01 from 176 journal entries, 71 commits, and the experiment artifacts cited in each section.*

---

## Orientation

This repository investigates whether word-embedding models trained on login device feature sequences can detect account takeover (ATO). The core hypothesis is that FastText skip-gram trained on per-account concatenations of structured feature tokens — prefixed strings like `os_ios`, `tz_utc-5`, `browser_safari` — embeds each account's devices into a tight cluster, so that an attacker's device lands measurably far from the centroid. The project ran seven hypothesis tracks (H1/pre-ml-lab through H6, plus a reproducibility rerun), used an adversarial ml-lab debate framework for structured tracks, and tracked all decisions and findings in a project journal. The headline finding: FastText skip-gram mean-pool on per-feature tokens with per-account corpus training achieves spoof-attack ROC-AUC 0.868 ± 0.012 and spoof PR-AUC 0.787 ± 0.018 (5-seed aggregate), with a +0.130 ROC-AUC delta and +0.248 PR-AUC delta over concatenated-string baseline. Both reproducible across five independent random seeds.

---

## Phase 1 — Initial PoC: Device-ID Embeddings (2026-04-03)

**Dates:** 2026-04-03 08:39 (commit `9c34c29` "init experiment") through 2026-04-03 22:33 (commit `af8bb0a` "rerun with agents").

**What was done.** The earliest scripts (`ato_fasttext_poc.py`, `ato_concat_poc.py`) applied FastText directly to opaque device ID strings — random hex identifiers like `dev_a3f9c2...`. The idea was that devices appearing together in the same account's login history would cluster near each other after training. Each account's event history was treated as a sentence; each device ID was a token.

The ml-lab adversarial critique-and-debate framework was immediately applied ("rerun with agents" commit same evening), producing a structured ten-point debate (CRITIQUE.md, DEFENSE.md, DEBATE.md) before the first rigorous experiment was run.

**What was observed.** Experiment 2 (`ato_experiment2.py`, covered in `pre_ml_lab/docs/CONCLUSIONS.md`) tested six signals side by side: `global_oov`, `account_oov`, `id_w2v` (Word2Vec device-ID centroid), `feature_w2v` (Word2Vec on feature tokens), `feature_fasttext` (FastText on feature tokens), and `feature_novelty`.

Three findings were decisive:

1. **FastText on device IDs fails.** FastText's character n-gram mechanism decomposes opaque device IDs (random hex strings) into character-level n-grams. Because all device IDs share the `dev_` prefix and produce overlapping n-gram patterns by chance, the n-gram component bleeds cross-account signal into every token, destroying per-account cluster structure. FastText silhouette score on device IDs: −0.051 (no structure). Word2Vec silhouette: +0.941 (near-perfect account separation). FastText centroid AUC on novel attacks: 0.910 vs. Word2Vec 0.982.

2. **The original OOV baseline is an evaluation artifact.** The first PoC evaluation included no legitimate new-device enrollments in the negative class. Any signal that fired on all globally unseen device IDs achieved near-perfect AUC because all attack devices were globally new. Adding enrollment events (new devices with profile consistent with the account's primary profile) to the negative class collapsed all OOV binary signals to an analytically determined ceiling: AUC = 0.750 on novel and spoof attacks. On fleet attacks (known cross-account devices), `global_oov` scored 0.250 — anti-correlated.

3. **Feature tokens unlock the mechanism.** FastText on structured feature tokens (`os_ios`, `tz_utc-5`, etc.) achieves AUC 0.985 on novel attacks and 0.920 on fleet attacks with enrollment in the negative class. Because the token vocabulary is bounded (~30 values across six feature dimensions), FastText's n-gram prefixes reinforce rather than corrupt per-account cluster structure. The mechanism also handles unseen feature values at inference (e.g., `os_harmonyos`) via n-gram averaging over meaningful prefixes.

**Conclusions at the time.** The recommended architecture was a two-path system: `feature_fasttext` for real-time profile-fit scoring; ID-based Word2Vec centroid (`id_w2v`) for offline fleet-reuse detection. The trivial baseline (set-membership check) was identified as the required control — any claimed signal must beat it on spoof attacks specifically.

**What was planned next.** Implement the corrected two-signal system, adopt Markov session generation as the default corpus mode, stress-test cross-account device sharing, evaluate the returning-attacker scenario, and report PR-AUC at realistic class imbalance.

**Supporting artifacts:**
- `experiments/pre_ml_lab/experiments/ato_fasttext_poc.py`
- `experiments/pre_ml_lab/experiments/ato_concat_poc.py`
- `experiments/pre_ml_lab/experiments/ato_experiment2.py`
- `experiments/pre_ml_lab/experiments/ato_experiment3.py`
- `experiments/pre_ml_lab/docs/CONCLUSIONS.md`
- `experiments/pre_ml_lab/docs/REPORT.md`
- `experiments/pre_ml_lab/figures/fig1_mechanism.png`, `fig2_model_comparison.png`, `fig3_cluster_structure.png`, `exp3_fig1_auc_heatmap.png`, `exp3_fig3_signal_comparison.png`

---

## Phase 2 — H2 Initial CBOW Investigation: A Collapsed Conclusion (Early April 2026, pre-2026-04-05)

**Dates:** Between the April 3 experiments and the April 5 reorganization commits.

**What was done.** To formalize the feature-token mean-pool vs. concat comparison, an initial H2 investigation was run using the ml-lab framework (`h2_rerun_poc.py` in `pre_ml_lab`). The training configuration was CBOW (sg=0) with a per-event corpus: one 6-token sentence per login event, each event encoded as the six feature tokens in fixed positional order.

**What was observed.** Mean-pool performed at or below chance on spoof attacks (approximately 0.500 AUC). The investigation initially concluded that concat was competitive with mean-pool.

**Conclusions at the time** (later revised — see Phase 3). The initial summary suggested the mean-pool advantage seen in Experiment 3 might not replicate under the more controlled H2 setup, or might be specific to the Word2Vec variant. This conclusion was incorrect.

**Root cause (discovered in Phase 3).** The per-event corpus enforces a rigid 6-slot positional structure: every event sentence is `[os_value, browser_value, tz_value, lang_value, net_value, screen_value]`. Within-feature tokens (e.g., all timezone values) always appear in slot 3 and therefore always co-occur with the same context-word distribution regardless of account. CBOW trains by predicting the center word from its context, so all timezone tokens receive identical gradient updates. They collapse to within-feature cosine similarity ≈ 0.9993 — effectively a single vector regardless of timezone value. Mean-pool then becomes blind to the timezone dimension. Spoof attacks differ from the account primarily on timezone, so spoof detection collapses toward chance. The CBOW objective was initially blamed, but later shown (by 2×2 factorial) to be secondary; corpus construction is the true cause.

**Supporting artifacts:**
- `experiments/pre_ml_lab/experiments/h2_rerun_poc.py`
- `experiments/pre_ml_lab/experiments/h2_rerun_experiment1.py`

---

## Phase 3 — H2 Corrected Investigation: Skip-gram and Per-Account Corpus Confirmed (Early-Mid April 2026, pre-2026-04-05)

**Dates:** Immediately after the CBOW investigation, before the April 5 reorganization.

**What was done.** After diagnosing the within-feature collapse, the training configuration was corrected to skip-gram (sg=1) with a per-account corpus: all events for an account flattened into a single sentence (~360 tokens). This breaks the rigid positional structure, exposing each timezone token to diverse cross-event and cross-feature context, allowing the model to learn distinct embeddings for `tz_utc-5` vs. `tz_utc+8`. The formalized investigation ran in the ml-lab debate framework as `robust_config_experiment.py` in `h2_ml_lab`.

A pre-registered three-field spoof definition was established: the spoof device has the target account's OS, browser, and language, but differs on timezone (forced), network (re-sampled), and screen (re-sampled). This matches `h2_rerun_experiment1.py` in `pre_ml_lab`.

Supplementary diagnostic tests were built:
- **T2:** Window sweep (concat at w=1, w=3, w=6) — tests whether wider windows mitigate the concat n-gram contamination
- **T3:** Prefixed-concat silhouette comparison — separates cross-boundary n-gram contamination from the joint-vs-separate embedding-space effect
- **T5:** Timezone-position permutation — moves tz to each of the six feature positions; a true signal should survive permutation
- **T8:** Token similarity analysis — checks whether within-feature cosine similarity collapses under the degenerate configuration

**What was observed.** H2 confirmed under the robust configuration:

| Condition | Novel AUC | Fleet AUC | Spoof AUC |
|---|---|---|---|
| Mean-pool (sg=1, per-account) | 0.999 | 0.994 | 0.869 |
| Concat (robust, labeled "w=1") | 0.997 | 0.998 | 0.782 |
| Trivial set-membership | — | — | 0.750 |

Spoof delta +0.087, bootstrap CI excludes zero. Silhouette delta [+0.073, +0.133]. T8 confirmed the mechanism: robust config within-feature similarity 0.392 (healthy); degenerate CBOW+per-event config 0.9993 (collapsed).

**Conclusions at the time.** Mean-pool sg=1 with per-account corpus is the canonical configuration. T8 (within-feature token similarity check) is a required pre-flight guard for any deployment of the mean-pool architecture.

**Note on the concat w=1 label (later revised — see Phase 15 / COMPARISON.md).** The h2_ml_lab REPORT.md labeled the concat baseline as "w=1: 0.782." The 5-seed rerun established that `robust_config_experiment.py` actually trained its concat model with `ROBUST_KWARGS` (window=6), not w=1 — the script contained no window-sweep code. The 0.782 value matches the rerun's w=6 result (0.778 ± 0.007), not the rerun's true w=1 result (0.737 ± 0.006). Mean-pool numbers matched across both experiments almost exactly (0.869 vs. 0.868 ± 0.012).

**What was planned next.** RBA dataset replication, and then a broader multi-hypothesis sweep (H3–H6) to probe the limits of the H2 result.

**Supporting artifacts:**
- `experiments/h2_ml_lab/experiments/robust_config_experiment.py`
- `experiments/h2_ml_lab/experiments/normalized_score_experiment.py`
- `experiments/h2_ml_lab/experiments/variable_spoof_experiment.py`
- `experiments/h2_ml_lab/docs/REPORT.md`
- `experiments/h2_ml_lab/docs/REPORT_ADDENDUM.md`
- `experiments/h2_ml_lab/docs/CONCLUSIONS.md`
- `experiments/h2_ml_lab/figures/robust_summary_auc.png`, `robust_t8_token_similarity.png`, `robust_t6_compactness.png`

---

## Phase 4 — Repository Reorganization and Public Cleanup (2026-04-05)

**Dates:** 2026-04-05 08:44 through 2026-04-05 13:22 (commits `486b925` through `6f01c1f`).

**What was done.** All prior experiment directories were reorganized into `pre_ml_lab/` (covering the device-ID PoC and Experiments 2–3 plus the initial H2 CBOW work) and `h2_ml_lab/` (covering the formal H2 structured investigation). The repository was prepared for public readability: README rewritten, archive directory added with provenance note, slides created, post-mortem organized.

**Supporting artifacts:**
- `experiments/pre_ml_lab/` (reorganized from flat root)
- `experiments/h2_ml_lab/` (reorganized)
- `archive/` (older process docs)
- `slides/device_embedding_slides.md`, `slides/device_embedding_slides.pdf`

---

## Phase 5 — Journal Initialization and H2 Synthesis (2026-04-09 to 2026-04-18)

**Dates:** Journal first entry 2026-04-09T17:44 (entry `7758f1b5`). Committed with other infrastructure on 2026-04-19 as part of commit `b848c18`.

**What was done.** Project journal (`.project-log/journal.jsonl`) initialized using the ml-journal plugin. `CLAUDE.md` written documenting the PEP 723 run pattern, FastText config gotchas, and membership-based eval design. A checkpoint captured the current state: no active implementation task, core findings established, FastText sg=1 + per-account corpus confirmed as the only viable config.

An issue was also logged around this period (`adb86a9c`): Phase 5 of some planning scripts wrote `INVESTIGATION_LOG.jsonl` to the repo root instead of the experiment subdirectory. Resolved by `44feca78`: changed `Path('INVESTIGATION_LOG.jsonl')` to `Path(__file__).parent / 'INVESTIGATION_LOG.jsonl'`.

**Open threads at checkpoint.** Commit `.project-log/` and `CLAUDE.md` to repo.

**Supporting artifacts:**
- `.project-log/journal.jsonl`
- `CLAUDE.md`

---

## Phase 6 — RBA Dataset Replication (2026-04-19)

**Dates:** 2026-04-19T18:47 (journal entries `b1769df5`, `64e85b60`) through 2026-04-19 16:27 (commit `efa97d3`).

**What was done.** Hypothesis H2-RBA: replicate the H2 FastText skip-gram mean-pool finding on a real-world dataset — the DAS Group RBA (Risk-Based Authentication) dataset v1.0.0, a synthesized dataset based on real SSO login behavior from 3.3 million users. A data preparation script (`data_prep.py`) and replication experiment (`rba_rerun.py`) were written. Primary evaluation: 50/50 temporal train/test split.

An Opus coherence audit immediately after the first commit (`025909a`) caught three critical numerical errors in synthesis documents: synthetic T6 compactness (0.033 → 0.047), T8 within/cross ratio (1.60 → 1.14), and synthetic AUCs (0.985/0.920/0.798 → 0.993/0.939/0.818). These were corrected.

**What was observed.** H2-RBA **confirmed** (journal entry `b1769df5`, verdict: "confirmed"): mean-pool ROC-AUC 0.852 at the 50/50 split, with n_ato_test_events = 9.

Top-1% sensitivity analysis across three temporal splits (journal entry `986cc050`, verdict: "inconclusive"): ROC-AUC holds (0.852–0.933 across splits) but threshold-based recall collapses at 60/40 due to only three positive events in test. Precision near-zero across all splits (~0.00005% base rate).

A research note was filed (`RESEARCH_NOTE_2026-04-19.md`) capturing the session.

**Note on dataset description (later corrected — see Phase 13).** Early versions of the rerun reports incorrectly described the RBA dataset as a "leak-credential dataset." Corrected to "synthesized dataset with real-world feature distributions."

**What was planned next.** Add top-1% threshold metrics, run sensitivity splits, then proceed to hypotheses H3–H6.

**Supporting artifacts:**
- `experiments/h2_rba/experiments/data_prep.py`
- `experiments/h2_rba/experiments/rba_rerun.py`
- `experiments/h2_rba/docs/HYPOTHESIS.md`
- `experiments/h2_rba/docs/REPORT.md`
- `experiments/h2_rba/figures/rba_metrics.json`, `rba_summary_auc.png`, `rba_pr_curve.png`, `rba_t6_compactness.png`
- `research-notes/RESEARCH_NOTE_2026-04-19.md`

---

## Phase 7 — H2 Rank-Normalization and Variable-k Spoof Analysis (2026-04-20 morning)

**Dates:** 2026-04-20T00:54–03:10 (journal entries `986cc050`, `3e8811e1`, `df9e6eee`, commit `5fcb7fa`).

**Context.** An issue was found (`ce3a6063`): `robust_config_experiment.py`'s spoof AUC (0.538) differed from REPORT.md's claim of 0.818. Root cause: the REPORT value came from `pre_ml_lab/h2_rerun_experiment1.py` (3-field spoof) while `robust_config_experiment.py` then used 1-field spoof (tz only). The 3-field spoof definition was restored and per-user rank-normalized scoring was added.

**What was observed.**

Per-user rank-normalized scoring experiment (journal entry `3e8811e1`, verdict: "confirmed"):
- mp-raw spoof AUC: 0.538 [0.504, 0.571]
- mp-rank-norm spoof AUC: 0.716 [0.687, 0.744]
- trivial: 0.750
- Rank-norm is the only embedding approach that beats trivial on the 1-field tz-only spoof. However, novel/fleet AUC drops ~0.04 vs mp-raw.

Variable-k spoof experiment (journal entry `df9e6eee`, verdict: "confirmed"):
- k=1 (tz only): rank-norm 0.714, raw 0.522, trivial 0.750 — rank-norm best but below trivial
- k=2 (tz+net): rank-norm 0.735, raw 0.689, trivial 0.750 — rank-norm still below trivial
- k=3 (tz+net+screen, the primary 3-field spoof): raw 0.869, rank-norm 0.784, trivial 0.750 — raw best, both beat trivial
- Crossover: rank-normalization wins at k=1 and k=2; raw distance wins at k=3.

**Conclusions at the time.** Rank-normalization is the right default for single-field spoofs (sophisticated attackers who mimic most features); raw distance handles multi-field spoofs. H2's headline spoof result (0.869) is implicitly k=3. (Later nuance — see Phase 8/H6 and Phase 11: at realistic 1:100 class imbalance, rank-normalization collapses PR-AUC and raw cosine is strictly better once the vocabulary is rich enough.)

**Supporting artifacts:**
- `experiments/h2_ml_lab/experiments/normalized_score_experiment.py`
- `experiments/h2_ml_lab/experiments/variable_spoof_experiment.py`
- `experiments/h2_ml_lab/docs/REPORT_ADDENDUM.md`
- `experiments/h2_ml_lab/figures/normalized_score_auc.png`, `variable_spoof_auc.png`

---

## Phase 8 — H3 (PFN), H4 (GRU), H5 (Stress Test), H6 (Hybrid) — Breadth Investigations (2026-04-20)

**Dates:** 2026-04-20T09:32 through 2026-04-20T20:13 (journal entries `ca3b378b` through `25d03c18`). Committed as part of commit `934f68b` "add H6 hybrid experiment; revise docs and slides for final architecture."

### H3 — Per-Feature Normalized (PFN) Scoring

**Hypothesis.** Computing a separate rank-normalized score for each feature dimension individually, then aggregating, should outperform mean-pool by down-weighting uninformative features.

**Observed** (journal entry `b5d37632`, verdict: "refuted"): pfn-mean ROC-AUC 0.807 vs mp-raw-split 0.856 on RBA; top-1% recall 0.333 (3/9 TPs) vs 0.444 (4/9 TPs). Two confounds: (1) `rtt_bucket` (round-trip time) has per-feature AUC 0.461, below chance, actively degrading the uniform aggregate; (2) median per-user calibration set size n_calib=2, making rank normalization degenerate (scores collapse to {0, 0.5, 1}). After ablating `rtt_bucket` (journal entry `21e3c9a3`): pfn-mean-6f ROC-AUC 0.818, still below mp-raw-split 0.856. **Hypothesis refuted.** Mean-pool implicitly downweights uninformative features; per-feature uniform aggregation cannot.

A bug was also found: `hypothesis_confirmed:true` in the output JSON was a script error — the correct comparison was pfn-mean vs mp-raw-split, not vs a degenerate baseline.

**Lessons logged:** (1) `rtt_bucket` is below-chance for ATO detection on RBA and should be excluded from future feature decompositions. (2) Short-history users (n_calib=2) make rank-normalization degenerate.

### H4 — GRU Temporal Sequence Modeling

**Hypothesis.** A 1-layer GRU trained on benign event sequences (hidden=64, cosine prediction loss) should capture temporal patterns missed by static mean-pool centroid scoring.

**Observed on RBA** (journal entry `06f409ac`, verdict: "inconclusive"): gru-predict ROC-AUC 0.879 vs mp-raw-split 0.856 (+0.023, overlapping CIs). Top-1% identical (4/9 ATOs caught). PR-AUC worse (0.016 vs 0.021). Recency-mean (0.841) weaker than static mean-pool, suggesting ATO histories are not recency-concentrated.

**Observed on i.i.d. synthetic** (journal entry `46575c48`, verdict: "refuted"): gru-predict loses to mp-raw-split on all spoof levels (k=1: 0.485 vs 0.519, k=2: 0.615 vs 0.687, k=3: 0.764 vs 0.866). Root causes: 30-token vocabulary too small for GRU to learn meaningful temporal transitions; val_loss=0.068 (insufficient convergence vs. RBA 0.043).

**Observed on temporally-structured synthetic** (Markov sessions, p_stay=0.70; device drift p=0.25 every 20 events; travel episodes p=0.20; journal entry `a70a0019`, verdict: "refuted"): GRU converged better (val_loss=0.043) but still loses on k=2 (0.551 vs 0.663) and k=3 (0.612 vs 0.814). Travel episodes baked tz variation into training, making k=1 spoof detection collapse toward chance (0.497). Root cause: vocabulary too small.

**Two-stage architecture** (journal entry `9300a2ca`, verdict: "inconclusive"): Sequential gate (known device → score=0) followed by embedding scoring shows GRU winning k=1 at +0.074 over mean-pool after gating. Inconclusive because single-stage trivial (0.750) still dominates.

### H5 — Stress Test of H2 Configuration

**Hypothesis.** Does H2 mean-pool (sg=1, per-account) continue to beat trivial at k=1 and k=2 spoof levels in the same i.i.d. synthetic setup?

**Observed** (journal entry `f064fc0b`, verdict: "inconclusive"): k=3 beats trivial (ROC-AUC 0.848 vs 0.750, PR-AUC 0.742 vs 0.500). k=1 fails decisively (ROC-AUC 0.530, PR-AUC 0.312, both below trivial). k=2 also fails. Two-stage gate provides negligible lift (+0.01–0.02) on i.i.d. synthetic vs. +0.10–0.18 on temporal synthetic. **H2's confirmed result (spoof 0.869) is implicitly k=3 only.** The i.i.d. 30-token vocabulary is too small to distinguish single-field spoofs.

### H6 — Hybrid RBA-Calibrated Dataset

**Hypothesis.** On a vocabulary-rich dataset calibrated from real RBA marginals (224+ tokens drawn from 11.7M clean RBA events via chain sampling), does mean-pool FastText beat trivial at k=1?

**Observed** (journal entry `1b7227f9`, verdict: "confirmed"): mp-raw k=1 ROC-AUC 0.997 vs trivial 0.956 (+0.037, non-overlapping CIs). Vocabulary richness is the primary driver — H5's 30-token vocabulary failed at k=1; the RBA-calibrated 224-token vocabulary succeeds. **Vocabulary poverty, not a fundamental embedding limit, explains H5's k=1 failure.**

Pre-registered contingency triggered: trivial AUC=0.956 (not ~0.750) because Markov generator (p_stay=0.70) fills the negative class with known-device events, enriching the trivial baseline.

**With 1:100 class imbalance** (journal entry `62a79a83`, verdict: "confirmed"): mp-raw k=1 ROC-AUC 0.997, PR-AUC 0.943; trivial PR-AUC 0.107 (8.8x worse). Rank-norm PR-AUC collapses to 0.240 under imbalance despite decent ROC-AUC — raw cosine distance is strictly better than rank-normalization when embeddings are well-calibrated at 224+ tokens.

**With temporal cross-account fleet blocklist** (journal entry `25d03c18`, verdict: "confirmed"): Two-stage embedding beats trivial at k=1 (ROC 0.980 vs 0.943, PR 0.892 vs 0.104, 8.6x lift). Fleet blocklist model (lag=10d, window=30d) catches 61% of fleet accounts post-lag; fleet_residual (pre-lag accounts only) shows mp_raw precision 91.8% at top-1%.

**Note on fleet_residual top-1% precision (later revised — see Phase 11).** The single-seed (42) fleet_residual precision of 0.918 appears prominently in the h6_hybrid REPORT. The 5-seed rerun showed this was a favorable draw from a high-variance distribution: cross-seed mean 0.493 ± 0.116, per-seed values 0.653/0.322/0.553/0.528/0.409. See Phase 11.

**Final architecture recommendation (from H6 REPORT, 2026-04-20).** Retire the two-stage gate (it is algebraically identical to the trivial baseline on fleet events by design), use blocklist + single-stage mp_raw.

**Supporting artifacts:**
- `experiments/h3_pfn/experiments/pfn_experiment2.py`
- `experiments/h3_pfn/docs/REPORT.md`
- `experiments/h4_gru/` (all scripts and docs)
- `experiments/h5_stress/` (all scripts and docs)
- `experiments/h6_hybrid/experiments/hybrid_experiment.py`
- `experiments/h6_hybrid/docs/REPORT.md`
- `experiments/h6_hybrid/figures/h6_metrics.json`
- `DEVICE_EMBEDDING_FINDINGS.md`

---

## Phase 9 — Repository Restructuring and Rerun Planning (2026-04-20T20:13 to 2026-04-21T01:10)

**Dates:** Commits `4fc74d5` through `ae3ec0d`.

**What was done.** All experiment directories were moved under a single `experiments/` parent (`4fc74d5` "move all experiment directories under experiments/"). A 5-seed reproducibility rerun was planned: `RERUN_PLAN.md` added, then split into 11 section files under `experiments/rerun/plan/`. The rerun covers H2, H6, and RBA phases across seeds 42, 123, 456, 789, 2024.

**Pre-flight audit.** A systematic pre-flight audit of all three existing scripts against the rerun plan surfaced 19 issues (journal entries `4906a795` through `c4d6de92`), covering:

- **Seed parameterization (critical, cross-cutting):** All three scripts hardcoded SEED=42 with no `--seed` CLI argument. H6's FastText was being seeded correctly via dict merge semantics but the audit initially flagged it incorrectly (later resolved as a false positive).
- **Missing scripts:** T2 (window sweep), T3 (prefixed-concat), T5 (tz-permutation) had no seed-parameterized scripts; results.json output was missing from H2 entirely.
- **Schema gaps:** H6 used `roc_lo/roc_hi` instead of `ci_lower/ci_upper`; H6 verdict fields did not match the plan schema; RBA `h2_replicated` used a point-estimate criterion instead of CI-lower-bound criterion.
- **T8 threshold conflict:** Script used >0.99, plan used >0.5. Resolved by decision `a2b73375`: canonical threshold >0.9 for all phases, grounded empirically (robust config produces 0.392, degenerate 0.9993).

**Key decisions logged:**

| ID | Decision |
|---|---|
| `a2b73375` | T8 collapse threshold = 0.9 for all phases (H2, H6, RBA) |
| `dd39e09c` | All H6 seeds must run with `--neg-ratio 100` (1:100 imbalance) |
| `06cb2acc` | `two_stage_rank_norm` included in H6 schema |
| `8420e9e8` | H2 spoof definition is 3-field (tz forced + net/screen re-sampled) |
| `6197ad37` | `run_all.sh` accepts `--seeds` and `--phases`, runs consistency checks, fails fast |
| `9511d90f` | RBA `h2_replicated` criterion: `mp_ci_lower > triv_roc` (not point-estimate) |
| `65209b9b` | H6 `primary_criterion_confirmed` uses PR-AUC non-overlapping CIs (not ROC-AUC) |

**Supporting artifacts:**
- `experiments/rerun/plan/` (11 section files: 0-REQUIREMENTS.md through 11-SUBMISSION_CHECKLIST.md)
- `experiments/rerun/RERUN_PLAN.md` (index)

---

## Phase 10 — Rerun Script Development (2026-04-21T00:15 to 2026-04-21T05:31)

**Dates:** Commits `fac976e` ("feat(rerun/h2): add h2_rerun.py") through `6ebff76` ("chore(rerun): untrack RESEARCH_REQUIREMENTS.md").

**What was done.** Three seed-parameterized rerun scripts were built from scratch, resolving all 19 pre-flight issues:

**H2 (`h2_rerun.py`):** Incremental port from `robust_config_experiment.py`. Added `--seed` CLI argument threading through data RNG, FastText constructor, and all bootstrap RNGs; exact `==` version pins (gensim==4.4.0, numpy==2.2.6, scikit-learn==1.7.2, matplotlib==3.10.8); T2 window sweep (w=1/3/6), T3 prefixed-concat silhouette, T5 tz-permutation (6 positions); live degenerate T8 computation (CBOW+per-event) per seed; 2×2 factorial T8 (SG/CBOW × per-account/per-event) to isolate corpus vs. objective collapse; paired delta CIs (bootstrap on the difference mp − concat_w1); `write_results_json()` conforming to Section 4 schema.

**Key discovery during script development** (journal entry `2b7481b1`): The 2×2 factorial T8 showed **corpus construction — not training objective — is the causal factor**. SG+per-event collapses (within=0.9958) at the same severity as CBOW+per-event (0.9967); both CBOW+per-account (0.1880) and SG+per-account (0.4867) show no collapse. The original C1 mechanistic claim "CBOW no gradient signal" is incorrect; the correct claim is "rigid positional structure in per-event corpus enforces identical co-occurrence distributions for within-feature tokens, regardless of training objective." This revised the plan's `0-REQUIREMENTS.md` immediately.

**H6 (`h6_rerun.py`):** Seed-parameterized (confirmed that FastText seed was already correctly threaded via dict merge, resolving false-positive issue `da184294`); T8 added; verdict fields corrected to schema (`primary_criterion_confirmed`, `rank_norm_collapse_confirmed`, `gate_blinds_fleet_confirmed`); CI fields standardized to `ci_lower/ci_upper`; `two_stage_vs_trivial_roc_delta` field added to fleet blocks (C3 mechanistic check); `scores.npz` output for mechanism diagnostics.

**RBA (`rba_rerun.py`):** `--seed` required argument added; three temporal splits (40/60, 50/50, 60/40) per seed; `h2_replicated` CI-criterion as per decision `9511d90f`; full Section 4 schema including `t8_token_similarity` block; pre-flight check for `data/rba/rba.parquet`.

**H2 figure script (`h2_figures.py`):** 5 publication figures including primary AUC, delta CI forest plot, T2 window sweep, T5 tz-permutation, verdict stability.

**Orchestration:** `aggregate.py`, cross-phase `check_consistency.py`, `run_all.sh` (with `--smoke` and `--dry-run` modes).

**Notable memos:**
- `workers=1` in FastText `ROBUST_KWARGS` is intentional: with workers>1, thread scheduling makes gradient update order non-deterministic even with a fixed seed. Workers=1 is required for bit-reproducible results.

**Supporting artifacts:**
- `experiments/rerun/scripts/h2/h2_rerun.py`
- `experiments/rerun/scripts/h2/h2_figures.py`
- `experiments/rerun/scripts/h6/h6_rerun.py`
- `experiments/rerun/scripts/h6/check_consistency.py`
- `experiments/rerun/scripts/rba/rba_rerun.py`
- `experiments/rerun/scripts/rba/data_prep.py`
- `experiments/rerun/scripts/rba/rba_figures.py`
- `experiments/rerun/scripts/aggregate.py`
- `experiments/rerun/scripts/check_consistency.py`
- `experiments/rerun/run_all.sh`

---

## Phase 11 — 5-Seed Rerun Execution: All Verdicts Reproduced (2026-04-21T10:38)

**Dates:** Commit `1e08532` "feat(rerun): complete 5-seed reproducibility rerun with aggregate results and report" (2026-04-21 06:38). Journal entry `d03fab44`.

**What was done.** `run_all.sh` executed H2, H6, and RBA phases across all 5 seeds (42, 123, 456, 789, 2024). Per-seed `results.json` files written; cross-seed aggregate CSVs, JSON, and figures generated; `REPORT.md` written.

**What was observed** (journal entry `d03fab44`, verdict: "confirmed"):

**C1 — Within-Feature Embedding Collapse (H2 T8):**
- Robust config within-feature similarity: 0.424 ± 0.013 (5 seeds, no collapse)
- Degenerate (CBOW+per-event) within-feature similarity: 0.9992 ± 0.00008 (collapse detected 5/5 seeds)

**C2 — Rank-Normalization Collapse Under Class Imbalance (H6):**
- mp-raw spoof k=1 PR-AUC: 0.888 ± 0.026
- mp-rank-norm spoof k=1 PR-AUC: 0.224 ± 0.011 (~4x drop)
- All 15 H6 verdicts True (3 per seed × 5 seeds)

**C3 — Known-Device Gate Blindness on Fleet (H6):**
- `two_stage` ROC-AUC equals `trivial` ROC-AUC exactly within each fleet population (`two_stage_vs_trivial_roc_delta` = 0.0 on all 5 seeds; fleet_aggregate ≈ 0.460, fleet_residual ≈ 0.460) — mechanically exact, not a rounding artifact
- `two_stage` top-1% TP = 0 on fleet_residual, all 5 seeds

**C4 — Mean-Pool vs. Concat (H2 T1):**
- mp spoof ROC-AUC: 0.868 ± 0.012
- concat_w1 spoof ROC-AUC: 0.737 ± 0.006
- delta CI [+0.111, +0.150] entirely positive across all 5 seeds

**RBA:**
- `h2_replicated` (CI-criterion: mp_ci_lower > triv_roc): True on all 5 seeds

**Notable revision from single-seed H6 values.** The single-seed (42) `h6_hybrid` fleet_residual top-1% precision was 0.918. Across 5 seeds: 0.493 ± 0.116 (ddof=0), per-seed values 0.653/0.322/0.553/0.528/0.409. The single-seed result was a favorable draw from a high-variance distribution. TP count: 91 ± 23.

Two consistency checks were revised after seeing all 5 seeds (journal entry `ac14ff1d`): RBA-S4 (mp > concat ROC-AUC) dropped as underpowered at n_ato=9; H6-X3 relaxed from strict k1≤k2≤k3 to k1 < min(k2,k3) (k2/k3 inversions of 0.0005–0.0039 at PR ≈ 0.95+ saturation).

**Research note filed:** `RESEARCH_NOTE_2026-04-21.md`.

**Supporting artifacts:**
- `experiments/rerun/seeds/seed_42/`, `seed_123/`, `seed_456/`, `seed_789/`, `seed_2024/` (per-phase `results.json`, figures)
- `experiments/rerun/aggregate/aggregate.json`
- `experiments/rerun/aggregate/h2_aggregate.csv`, `h6_aggregate.csv`, `rba_aggregate.csv`
- `experiments/rerun/aggregate/figures/` (20 aggregate figures)
- `experiments/rerun/REPORT.md`
- `research-notes/RESEARCH_NOTE_2026-04-21.md`

---

## Phase 12 — Mechanism Diagnostics: Closing the Causal Gaps (2026-04-21T11:33 to 2026-04-21T22:11)

**Dates:** Commits `ee999ea` through `c5b9b0a`, journal entries `74c55ff6` through `89487216`.

A peer review of `TECHNICAL_REPORT.md` (journal entries `74c55ff6`, `4d385b2d`) flagged two mechanistic gaps: the C1 corpus-collapse claim was supported by correlation (factorial results) but not direct evidence of co-occurrence distribution differences; the C2 rank-normalization claim was qualitatively plausible but unquantified.

**C1 co-occurrence mechanism (journal entries `06390dbf`, `541a847a`, commit `48990c9c`):**
New diagnostic script `h2_cooccurrence.py` computed within-feature Jensen-Shannon divergence for per-event vs. per-account corpora (seed 42, 400 accounts).
- Per-event mean JSD: 0.077
- Per-account mean JSD: 0.190
- Ratio: 2.5×

Confirmed: per-event corpus forces nearly identical context-word co-occurrence distributions for within-feature tokens, leaving no gradient to separate them. Per-account concatenation allows divergent distributions. Figure `h2_c1_cooccurrence.png` written to `aggregate/figures/`.

**C2 score-margin compression mechanism (journal entries `3c5f4097`, `ab7d4a98`, commit `53571da6`):**
New diagnostic script `h6_score_margin.py` computed contamination rate (fraction of benign above p10 attack) and robust margin (p10 attack − p90 benign) for mp_raw vs. mp_rank_norm across 5 seeds.
- Raw contamination: 1.28% ± 0.69%
- Rank-norm contamination: 4.98% ± 0.13% (3.9× mean ratio; structural ~5% floor from per-user CDF normalization)
- Raw robust margin: 0.065 ± 0.007
- Rank-norm robust margin: 0.040 ± 0.020; seed 2024 margin = 0.000 (distributions touching at decision boundary)

Confirmed: CDF rank-normalization structurally locks contamination at ~5% per seed, shrinking the margin on every seed.

**Additional work in this phase:**
- `DEEP_DIVE.md` added (2037 lines): comprehensive technical reference covering every step, decision, and code-level walkthrough across H2/H6/RBA phases.
- C1 and C2 mechanism figures embedded in `TECHNICAL_REPORT.md` with explanatory prose.
- Verdict stability figure replaced with markdown table sourced from per-seed `results.json`.
- Plot style standardized across all four figure scripts (TITLE_FS=11, LEGEND_FS=8, `_apply_style` helper).
- Language corrected: "pre-registered" → "pre-specified" throughout (verdicts were committed to git before experiments, not filed with an external registry).
- RBA dataset framing corrected from "real ATO events" to "independently-structured synthesized dataset with real-world feature distributions."

**Supporting artifacts:**
- `experiments/rerun/scripts/h2/h2_cooccurrence.py`
- `experiments/rerun/scripts/h6/h6_score_margin.py`
- `experiments/rerun/aggregate/figures/h2_c1_cooccurrence.png`
- `experiments/rerun/aggregate/figures/h6_c2_score_margin.png`
- `experiments/rerun/DEEP_DIVE.md`
- `experiments/rerun/TECHNICAL_REPORT.md` (with mechanism figures embedded)

---

## Phase 13 — RBA Dataset Description Correction and COMPARISON.md (2026-04-22 and 2026-04-26)

**Dates:** Commit `ae8ce06` (2026-04-22 01:33) "docs(rerun): fix RBA dataset description"; commit `087d05c` (2026-04-26 22:39) "docs(rerun): add COMPARISON.md — rerun vs. prior experiments."

**What was done.** The incorrect label "leak-credential dataset" for the RBA dataset was replaced throughout `REPORT.md` and `TECHNICAL_REPORT.md` with accurate phrasing (the DAS Group dataset is synthesized from real SSO login behavior, not from credential leaks).

`COMPARISON.md` was added documenting how the 5-seed rerun findings differ from h2_ml_lab and h6_hybrid across C1–C4. The initial version of COMPARISON.md attributed the concat sign-flip (w=1 baseline below trivial in the rerun vs. above trivial in h2_ml_lab) to a spoof-definition change. This explanation was later found to be wrong and corrected in Phase 15.

**Supporting artifacts:**
- `experiments/rerun/COMPARISON.md`
- `experiments/rerun/REPORT.md` (updated)
- `experiments/rerun/TECHNICAL_REPORT.md` (updated)

---

## Phase 14 — PR-AUC Addition to H2 Pipeline and CLAUDE.md Rules (2026-04-30 to 2026-05-01)

**Dates:** Commit `d159efc` (2026-04-30 22:52) "feat(rerun): add PR-AUC to H2 pipeline"; commit `3e570f1` (2026-05-01 14:26) "docs: add journal consultation and commit-path rules to CLAUDE.md."

**What was done.** PR-AUC computation was added to the H2 pipeline: `bootstrap_pr_auc()`, `bootstrap_pr_auc_delta()`, dual ROC/PR-AUC panel figure (`h2_auc_dual.png`), 18 new PR-AUC metric paths in `aggregate.py`. All 5 seeds were re-run. Reports (`TECHNICAL_REPORT.md`, `REPORT.md`, `DEEP_DIVE.md`) updated with PR-AUC table and figure.

**What was observed** (journal memo `55348186`):
- mean_pool spoof PR-AUC: 0.787 ± 0.018
- concat_w1 spoof PR-AUC: 0.542 ± 0.010 (barely above trivial 0.500)
- trivial spoof PR-AUC: 0.500
- Bootstrap delta CI: [+0.210, +0.283] entirely positive on all 5 seeds

PR-AUC amplifies the mean-pool advantage: delta +0.248 (vs. +0.130 for ROC-AUC). Concat's ROC-AUC of 0.737 masks near-chance PR-AUC performance.

A CLAUDE.md update codified two project-specific rules: journal must be consulted before planning (query `--unresolved-issues` and `--list decision --since 7d`); "commit" means the ml-journal `/log-commit` skill, never bare `git commit`.

**Supporting artifacts:**
- `experiments/rerun/scripts/h2/h2_rerun.py` (updated, PR-AUC)
- `experiments/rerun/scripts/h2/h2_figures.py` (updated, dual panel)
- `experiments/rerun/scripts/aggregate.py` (updated)
- `experiments/rerun/aggregate/h2_aggregate.csv` (updated)
- `experiments/rerun/aggregate/figures/h2_auc_dual.png` (new)
- `experiments/rerun/TECHNICAL_REPORT.md` (updated)
- `experiments/rerun/REPORT.md` (updated)
- `experiments/rerun/DEEP_DIVE.md` (updated)
- `CLAUDE.md` (journal consultation rules added)

---

## Phase 15 — Consolidation, Verification, and Errata Correction (2026-07-01)

**Dates:** Journal entries `006ae95f` (2026-07-02T00:49Z), `aca8f247` (2026-07-02T00:56Z), `1c4fb9b2` (2026-07-02T01:11Z). Commit `71e0711` (2026-05-25) "commit last entry" was the last git commit before this session.

**What was done.** An independent verification pass over all `experiments/rerun` reports and scripts was run, treating the scripts and per-seed data as canonical and the report prose as suspect. All four primary verdicts and headline numbers verified clean against `aggregate/aggregate.json`, per-seed `results.json`, and raw `scores.npz` arrays. Seven prose errors were found in report documents (journal issue `006ae95f`), and five additional errors were identified during the fix pass.

**Errors found and corrected** (journal resolution `1c4fb9b2`):

1. **COMPARISON.md concat-sign-flip explanation (most significant).** The initial explanation attributed the concat w=1 result dropping below trivial in the rerun (0.737) vs. being above trivial in h2_ml_lab (0.782) to a spoof-definition change. This was wrong on both ends. The spoof definition is identical in both experiments (both 3-field). The correct explanation is the **concat baseline window**: `robust_config_experiment.py` has no window-sweep code and trained its concat baseline with `ROBUST_KWARGS` (window=6); its "w=1: 0.782" label was a mislabeled window-6 result (matches the rerun's w=6 result of 0.778 ± 0.007, not the true w=1 result of 0.737 ± 0.006). COMPARISON.md Section 1 rewritten accordingly.

2. **Fleet_residual TP and precision.** "91 ± 24 TP / 0.491 ± 0.128 precision" corrected to "91 ± 23 TP / 0.493 ± 0.116 precision" (ddof=0, matching aggregate.py) in REPORT.md, TECHNICAL_REPORT.md, COMPARISON.md.

3. **RBA h2_replicated verdict definition.** Occurrences of "both encoders beat trivial" in REPORT.md and TECHNICAL_REPORT.md corrected to the pre-specified CI-lower-bound criterion: mean-pool CI-lower > trivial ROC only (concat > trivial holds on all 5 seeds but is not the verdict).

4. **Degenerate downstream AUCs.** Paragraphs citing "spoof ~0.38, novel/fleet ~0.99" were attributed to the rerun but were never measured in it (the numbers came from single-seed h2_ml_lab, which itself reported novel 0.880 / fleet 0.922, not 0.99). Superseded by a new 5-seed diagnostic (below).

5. **T5 tz-permutation range.** "0.70–0.74" was seed-42 only. Cross-seed individual values span 0.695–0.754; per-position 5-seed means span 0.70–0.74.

6. **Wilson vs. Clopper–Pearson.** "Wilson 95% lower bound ≈ 0.78" for 15/15 verdicts is actually the Clopper–Pearson exact bound (Wilson gives 0.80). Similarly for 5/5 (Clopper–Pearson 0.48, Wilson 0.57). Corrected throughout.

7. **h6_score_margin.py docstring.** "Raw contamination ~0.15% / 33×" was seed-42 only; cross-seed is 1.28% → 4.98% (3.9×). Docstring corrected.

**Five additional errors fixed during the pass:**
- TECHNICAL_REPORT "H2 and H6 closed 30-token vocab" — H6 is open 216–240 tokens (varies by seed), not 30
- "2000 resample draws" — N_BOOTSTRAP=1000 in all three runners
- `h6_c3_topk_precision` caption claimed "5-seed aggregate" — it is a representative seed-42 figure
- REPORT.md spoof taxonomy wording tightened (timezone forced, network/screen re-sampled)
- Rank-floor phrasing corrected: ~5% of benign events land at or above the attack p10 operating point (rank 0.90); benign events at rank > 0.95 specifically are ~1.5%

**New downstream diagnostic** (journal entry `aca8f247`, verdict: "confirmed"): `h2_degenerate_downstream.py` — 5-seed downstream scoring of all four H2 2×2 factorial cells plus the historical PoC config, with per-seed cross-checks against `results.json` (max spoof deviation 0.0007; collapse flags match 5/5). Closes the gap where the "collapse → downstream damage" claim rested on single-seed h2_ml_lab numbers.

Results (5 seeds, mean ± std ddof=0):

| Cell | Spoof ROC-AUC | Spoof PR-AUC | Novel ROC-AUC | Fleet ROC-AUC |
|---|---|---|---|---|
| sg + per-account (headline) | 0.868 ± 0.012 | 0.787 ± 0.018 | 0.9996 | 0.995 |
| cbow + per-account | 0.933 ± 0.009 | 0.879 ± 0.014 | 1.000 | 0.997 |
| sg + per-event (collapsed) | 0.767 ± 0.017 | 0.585 ± 0.028 | 0.960 | 0.920 |
| cbow + per-event (collapsed) | 0.725 ± 0.011 | 0.528 ± 0.024 | 0.963 | 0.927 |
| PoC config (cbow + per-event, 10 epochs) | 0.669 ± 0.013 | 0.459 ± 0.019 | 0.873 | 0.886 |
| trivial | 0.750 | 0.500 | — | — |

The earlier single-seed spoof AUC of 0.384 under collapse does not reproduce under the rerun's standardized protocol and is superseded by these values. All collapsed (per-event) configs land at or below trivial on spoof — the historical PoC config falls below the trivial PR-AUC baseline — while their novel and fleet ROC-AUC stay comparatively high (novel 0.87–0.96, fleet 0.85–0.93 across collapsed cells). The damage is selective, concentrated in the spoof dimension, which is exactly what makes it invisible to aggregate or easy-subtype monitoring. Notably, cbow + per-account (0.933) nominally outperforms sg + per-account (0.868) on spoof, reinforcing corpus construction (not training objective) as the causal axis.

**Supporting artifacts:**
- `experiments/rerun/scripts/h2/h2_degenerate_downstream.py` (new)
- `experiments/rerun/aggregate/h2_degenerate_downstream.json` (new)
- `experiments/rerun/COMPARISON.md` (errata-corrected)
- `experiments/rerun/REPORT.md` (errata-corrected)
- `experiments/rerun/TECHNICAL_REPORT.md` (errata-corrected)
- `experiments/rerun/scripts/h6/h6_score_margin.py` (docstring corrected)

---

## State of the Repo Today (2026-07-01)

The investigation is complete. All four pre-specified contributions reproduced 5/5 seeds; all logged errata corrected; downstream collapse diagnostic added. No open unresolved issues in the journal.

**Authoritative artifacts by category:**

| Artifact | Path | Notes |
|---|---|---|
| Primary synthesis | `TECHNICAL_REPORT.md` | Root-level canonical write-up |
| Headline findings | `DEVICE_EMBEDDING_FINDINGS.md` | Summary for external audiences |
| Rerun technical report | `experiments/rerun/TECHNICAL_REPORT.md` | 5-seed rerun with mechanism diagnostics |
| Rerun summary report | `experiments/rerun/REPORT.md` | Narrative summary of rerun |
| Comprehensive reference | `experiments/rerun/DEEP_DIVE.md` | Code-level walkthrough, all design rationale |
| Comparison | `experiments/rerun/COMPARISON.md` | Rerun vs. h2_ml_lab and h6_hybrid, errata-corrected |
| H2 rerun script | `experiments/rerun/scripts/h2/h2_rerun.py` | Canonical H2 runner, --seed CLI |
| H6 rerun script | `experiments/rerun/scripts/h6/h6_rerun.py` | Canonical H6 runner, --seed CLI |
| RBA rerun script | `experiments/rerun/scripts/rba/rba_rerun.py` | Canonical RBA runner, --seed CLI |
| Downstream diagnostic | `experiments/rerun/scripts/h2/h2_degenerate_downstream.py` | 5-seed collapse→spoof-damage evidence |
| C1 mechanism | `experiments/rerun/scripts/h2/h2_cooccurrence.py` | JSD diagnostic (per-event 0.077 vs per-account 0.190) |
| C2 mechanism | `experiments/rerun/scripts/h6/h6_score_margin.py` | Score-margin compression (raw 1.28% vs rank-norm 4.98%) |
| Aggregate data | `experiments/rerun/aggregate/aggregate.json` | Cross-seed aggregates for all phases |
| Aggregate figures | `experiments/rerun/aggregate/figures/` | 20 aggregate figures |
| 5-seed results | `experiments/rerun/seeds/seed_{42,123,456,789,2024}/` | Per-seed results.json and figures |
| Journal | `.project-log/journal.jsonl` | 176 entries, authoritative timestamped record |
| Slides | `slides/device_embedding_slides.{md,pdf}`, `slides/slides.{md,pdf}`, `slides/methodology_pitch.{md,pdf}` | Presentation decks |
| RBA data | `data/rba/rba.parquet` | Required for RBA rerun (not committed) |
| Research notes | `research-notes/RESEARCH_NOTE_2026-04-19.md`, `RESEARCH_NOTE_2026-04-21.md` | Session summaries |
| Rerun plan | `experiments/rerun/plan/` (11 files) | Protocol and schema reference |
| Orchestration | `experiments/rerun/run_all.sh` | Full 5-seed rerun entry point |

**Headline numbers (5-seed aggregates, as of 2026-07-01):**

| Contribution | Finding | Numbers |
|---|---|---|
| C1 Corpus collapse | Within-feature similarity: robust vs. degenerate | 0.424 ± 0.013 vs. 0.9992 ± 0.00008; collapse 5/5 seeds |
| C1 Mechanism | Per-event vs. per-account JSD | 0.077 vs. 0.190 (2.5× ratio) |
| C1 Downstream | Collapsed configs on spoof | at/below trivial (ROC 0.669–0.767; PoC PR 0.459 < 0.500) |
| C2 Rank-norm collapse | mp-raw vs. rank-norm spoof k=1 PR-AUC | 0.888 ± 0.026 vs. 0.224 ± 0.011 |
| C2 Mechanism | Rank-norm contamination vs. raw | 4.98% ± 0.13% vs. 1.28% ± 0.69% (3.9× ratio) |
| C3 Gate blindness | two_stage vs. trivial ROC-AUC delta (fleet) | 0.000 exactly on both fleet populations, 5/5 seeds |
| C4 Mean-pool vs. concat | Spoof ROC-AUC delta (w=1) | +0.130, CI [+0.111, +0.150] |
| C4 PR-AUC | Spoof PR-AUC delta (w=1) | +0.248, CI [+0.210, +0.283] |
| RBA replication | h2_replicated (CI-criterion) | True on all 5 seeds |
