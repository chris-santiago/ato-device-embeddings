# Critique Ablations: Pre-Specified Controls for C1/C2 External Validity

**Purpose:** An external methodological review (2026-07-11) identified three missing
controls that are existential for the investigation's headline claims. This plan pre-specifies the
ablations, their verdict criteria, and the interpretation matrix **before any script
runs**, following the same discipline as the primary rerun (RERUN_PLAN.md) and the
calibration-window ablation that dispatched C4 (`calib_sweep/`, decision 6be140d5).

All standing instructions from `plan/4-METRICS_SCHEMA.md` apply: saved scripts only,
PEP 723 headers, `--smoke` paths, seeds {42, 123, 456, 789, 2024}, `workers=1`,
structured JSON per seed, deterministic aggregation.

**Location:** scripts and outputs live in `experiments/rerun/ablations/`
(precedent: `calib_sweep/`). Reuse policy: drive `scripts/h2/h2_rerun.py` and
`scripts/h6/h6_rerun.py` as libraries; do not reimplement data generation, training,
scoring, or bootstrapping.

---

## A1 — Frozen Random Embedding Control (C1)

**Question.** Does trained FastText contribute anything beyond set-overlap geometry?
With 6 features, a 30-token closed vocabulary, mean-pooling, and centroid cosine, a
single-feature change perturbs the event vector by ~1/6 regardless of training. If
frozen random vectors score comparably, C1's positive claim reduces to "set-overlap
geometry beats n-gram soup."

**Design.** Closed-vocabulary configuration (identical `generate_dataset`). Scorers:

| Scorer | Definition |
|---|---|
| `mp_trained` | Robust config (SG + per-account, epochs=20) — retrained per seed, identical to rerun |
| `mp_random` | Frozen iid N(0,1) 64-d vector per feature token, seed-derived, **no training**; same mean-pool + centroid cosine |
| `cat_random` | Frozen hash-derived random 64-d vector per concatenated device string (random device-hash control) |
| `trivial` | Exact set membership (existing `evaluate_trivial`) |

Metrics: ROC/PR-AUC per attack type (novel/fleet/spoof) with within-seed bootstrap CIs
(1,000 resamples); paired bootstrap deltas (identical resample indices) for
`mp_trained − mp_random`.

**Pre-specified verdicts (per seed):**

- `a1_trained_beats_random_roc`: paired-bootstrap 95% CI lower bound of the **spoof
  ROC-AUC** delta (mp_trained − mp_random) > 0.
- `a1_trained_beats_random_pr`: same for spoof PR-AUC.

Confirmed = 5/5 seeds. **Interpretation:** both confirmed → C1 survives with its
decisive control; either fails → C1's positive recommendation is dead as stated and
the record pivots to the pitfalls framing with this result as a lead finding.
Descriptive (no verdict): `mp_random` vs `trivial` margin — how much of mean-pool's
absolute performance is pure overlap geometry.

## A2a — Per-Feature Likelihood Incumbent (C1, closed vocabulary)

**Question.** Does the embedding approach beat the CPU-cheap, label-free industry
incumbent it never compares against — smoothed per-account per-feature categorical
likelihood (Freeman et al. 2016 style)?

**Design.** Same data and seeds as A1. Score:
`NLL(e) = −Σ_f log[ λ·P̂(v_f | acct) + (1−λ)·P̂(v_f | global) ]`, Laplace α=1 over the
feature vocabulary; per-account counts from the 60 training events, global counts from
all training events. Variants: λ ∈ {1.0, 0.9, 0.5}.

**Adverse selection rule (pre-specified):** the comparison uses the **best likelihood
variant per seed by spoof ROC-AUC point estimate** — selection bias deliberately favors
the incumbent, so the embedding claim survives only if it beats the incumbent's best case.

**Pre-specified verdicts (per seed):**

- `a2_mp_beats_likelihood_roc`: paired-bootstrap 95% CI lower bound of the spoof
  ROC-AUC delta (mp_trained − lik_best) > 0.
- `a2_mp_beats_likelihood_pr`: same for spoof PR-AUC.

Confirmed = 5/5 seeds. **Interpretation:** confirmed → the record gains its missing
incumbent baseline; failed → the embedding architecture does not earn its tier and no
positive framing survives (the likelihood scorer becomes the recommended architecture,
embeddings a negative result). Fleet/novel reported descriptively.

## A3p — Correlated-Marginals Collapse Precursor (C2)

**Question.** Is C2's per-event collapse an artifact of the closed-vocab generator
sampling features independently (`h2_rerun.py:84`)? Real login features are strongly
coupled; the H6 generator already encodes real coupling from RBA marginals
(`os::device_type` joint, `browser|os`, `region|country`, `asn|country`) while
`rtt_bucket` is sampled independently of everything — a built-in independence control
within otherwise-correlated data.

**Hypothesis under test (the critique's):** collapse under per-event corpora is driven
by context-distribution indistinguishability, which requires feature independence.
Prediction: in a per-event corpus built from H6 data, `rtt_bucket` collapses hardest
(lowest within-feature context JSD, highest within-feature cosine) while conditioned
features (`region`, `asn_bucket`, `browser`, `device_type`) resist.

**Design.** Per seed: generate H6 accounts (400, N_TRAIN=60, unchanged); build
per-event (one 7-token sentence per event) and per-account (existing `build_corpus`)
corpora from the same events; train 4 cells {SG, CBOW} × {per-event, per-account} with
`ROBUST_KWARGS` (only `sg` and the corpus vary). Measure per feature:

1. **Mechanism level:** mean pairwise Jensen–Shannon distance between within-feature
   token context distributions (window 6, per corpus shape) — the same diagnostic as
   `h2_cooccurrence.py`.
2. **Outcome level:** mean pairwise within-feature cosine, T8 convention (`wv[t]`,
   collapse threshold 0.9 per decision a2b73375). Sensitivity column: `vectors_vocab`
   (subword-free input vectors) to control for shared-prefix n-gram inflation, since
   open-vocab feature prefixes differ in length.

**Token filter (pre-specified):** only tokens with corpus frequency ≥ 5 enter the JSD
and cosine statistics (open-vocab tail tokens have unstable context distributions);
per-feature token counts reported before/after filtering.

**Pre-specified verdicts (per seed, per-event SG cell, wv metric):**

- `a3p_rtt_max_cosine`: `rtt_bucket` has the highest within-feature cosine of the 7 features.
- `a3p_rtt_min_jsd`: `rtt_bucket` has the lowest within-feature context JSD (per-event corpus).
- `a3p_conditioned_below_threshold`: all of {`region`, `asn_bucket`, `browser`,
  `device_type`} stay below the 0.9 collapse threshold.
- `a3p_collapse_all`: pooled within-feature cosine (all features) > 0.9.

**Interpretation matrix:**

| Outcome | Reading |
|---|---|
| `a3p_collapse_all` holds (everything collapses despite real coupling) | C2 generalizes; the independence critique fails; the record needs only the mechanism rewrite (context distributions, not positional rigidity) |
| rtt collapses, conditioned features resist (first three verdicts hold) | C2 severity is independence-driven; the claim must be scoped to weakly-coupled feature sets, and the full ρ-sweep (A3, phase 2) is warranted |
| Mixed / neither | Escalate to the full ρ-sweep on the closed-vocab generator before concluding |

---

## A2b — Likelihood Incumbent at 1:100 (H6, pre-specified 2026-07-11 after phase-1 results)

**Question.** Phase-1 A2a showed the likelihood incumbent beats the embedding on the
closed-vocab spoof comparison. Two things remain open on the open-vocab 1:100
configuration: does the incumbent hold its advantage on the realistic spoof gradient,
and — decisive for C3 — does it detect fleet_residual contamination? A single training
appearance gives a feature-count of ~1/40, a smooth low probability, unlike the binary
gate. If the incumbent both wins on spoof and detects fleet, C3's operational lesson
becomes "binarization is the failure, not the scoring family" and the incumbent wins
the architecture argument outright.

**Design.** Identical H6 generation (400 accounts, neg_ratio 100 → 500 enrollment
negatives per account, blocklist lag 10 d, attack window 30 d). Likelihood counts use
`centroid_events` (train[:40]) for parity with the centroid window — both scorers see
fleet contamination iff the injected event lands in the first 40. Laplace α=1 over the
global training vocabulary per feature (distinct values pooled across all accounts'
centroid windows); global backoff distribution from the same pooled window;
λ ∈ {1.0, 0.9, 0.5}; **adverse rule:** best variant per seed by spoof_k1 ROC point
estimate. Scorers: `mp_raw` (retrained, identical), `trivial`, `two_stage`,
`lik_λ*`, and `lik_two_stage` (gated likelihood — descriptive; expected 0 TP on
fleet_residual by the same definitional argument as the gate). Populations: spoof_k1,
spoof_k2, spoof_k3, novel, fleet_residual; top-1% metrics per population.

**Pre-specified verdicts (per seed):**

- `a2b_lik_detects_fleet_residual`: lik_best top-1% TP > 0 on fleet_residual.
- `a2b_mp_beats_lik_spoofk1_roc`: paired-bootstrap 95% CI lower bound of the spoof_k1
  ROC-AUC delta (mp_raw − lik_best) > 0.
- `a2b_mp_beats_lik_spoofk1_pr`: same for PR-AUC.

**Interpretation:** lik detects fleet and mp fails to beat lik → the incumbent
supersedes the embedding architecture end-to-end; C3 is rewritten as a binarization
trap independent of scoring family. Lik misses fleet (TP = 0) → centroid-cosine
retains a unique contamination-robustness capability and keeps a positive niche in the
reframed record.

---

## A4 — No-Subword Cell (pre-specified 2026-07-11, run before any draft reframe)

**Question.** How much of C1 is specifically *subword* behavior? Two sub-questions:
(a) do character n-grams add anything to the mean-pool config on a closed vocabulary
(they may even hurt — shared feature prefixes like `os_`/`tz_` pull same-feature
tokens together); (b) can the concat encoder function at all without subwords, given
that unseen device combinations are OOV for whole-string tokens?

**Design.** Closed-vocab configuration, identical `generate_dataset`. New cells, both
FastText with `max_n=0` (identical class, one knob changed — equivalent to word2vec
token training):

| Scorer | Definition |
|---|---|
| `mp_nosub` | SG + per-account, epochs=20, `max_n=0`; mean-pool + centroid cosine |
| `cat_nosub` | Same, on concatenated device strings; **OOV strings score distance 1.0** (the only operationally sensible fallback — recorded, plus the OOV rate per test population) |

Reference scorers retrained per seed for paired deltas: `mp_trained` (subwords),
`mp_random` (frozen, from A1 construction). Additional diagnostic: within-feature
cosine of `mp_nosub` vs `mp_trained` (closed-vocab check of the prefix-inflation
effect found in A3p).

**Pre-specified verdicts (per seed, spoof, paired bootstrap 95% CI):**

- `a4_subwords_help_mp_roc`: CI lower bound of (mp_trained − mp_nosub) ROC delta > 0.
- `a4_subwords_hurt_mp_roc`: CI upper bound of (mp_trained − mp_nosub) ROC delta < 0.
- `a4_training_helps_nosub_roc` / `_pr`: CI lower bound of (mp_nosub − mp_random)
  delta > 0 — the purest test of whether *training* (without subword confounds) adds
  anything over frozen random vectors.

**Interpretation:** subwords neither help nor hurt mean-pool → C1's mechanism sharpens
to "subwords poison concat and add nothing to per-feature tokens; use plain token
embeddings." Subwords hurt → stronger: the proposed architecture's own subword
machinery is a net negative on closed vocabularies. `cat_nosub` OOV rate near 100% on
attack populations documents that concat *requires* subwords (or hashing) to score
unseen devices at all — the dependency that creates the C1 contamination pathology.

---

## Phase 2 (specified, not run in this pass)

- **A3 full:** closed-vocab generator with a coupling knob ρ (latent locale ties
  tz→lang; ρ=0 reproduces current independence), ρ ∈ {0, 0.25, 0.5, 0.75, 1.0},
  full C2 chain per point.
- **A5:** C1 evaluation re-pooled at 1:100 to align PR-AUC with the record's own
  imbalance guidance.

## Execution

- `ablations/baseline_controls.py --seed S` → `ablations/results/baseline_controls_seed_S.json` (A1 + A2a share data and the trained model)
- `ablations/h6_perevent_collapse.py --seed S` → `ablations/results/h6_perevent_seed_S.json`
- `ablations/run_ablations.sh` runs both across the 5 seeds, then `ablations/aggregate_ablations.py` → `ablations/aggregate/ablations_summary.{json,csv}` and the verdict table in `ablations/SUMMARY.md`
- Verdict aggregation: report per-seed booleans and the count; no confirmation-rate
  interval statistics (seeds are not independent trials — the CP bound in the draft is
  being removed for the same reason).
