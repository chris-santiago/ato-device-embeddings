# Experiment Rerun Plan: 5-Seed Reproducibility Protocol

**Purpose:** Clean, documented rerun of H2, H6, and RBA experiments across 5 random seeds
for publication submission. Produces seed-aggregated results with variance estimates and
verdict stability checks.


---

## Standing Instructions

These rules apply to every session working within this rerun protocol. They are not
optional and take precedence over convenience.

### 1. All scripts are saved artifacts

Every Python script, shell script, or helper written during this rerun **must be saved
as a file in the repository.** One-time inline commands (heredoc `EOF` blocks, `python -c`
one-liners, notebook cells) are prohibited for anything that produces results or modifies
experiment logic. If a command is worth running, it is worth saving.

- Phase-specific scripts go in `experiments/rerun/scripts/<phase>/` (e.g., `scripts/h2/`,
  `scripts/h6/`, `scripts/rba/`); common orchestration helpers (aggregation, consistency
  checks) go in `experiments/rerun/scripts/`; `run_all.sh` stays at the `rerun/` root
- Every script must carry a `# /// script` PEP 723 inline dependency block with exact
  version pins and a `requires-python` constraint
- Run with `uv run <script.py>` — never plain `python`

### 2. Every script must be verifiable

Every experiment script must provide a fast path for confirming correct implementation
before committing to a full run. Each script must have **one of the following**:

- **A `--smoke` flag** that runs the same code path on a small synthetic subset
  (e.g., 50 accounts, 200 bootstrap resamples) and exits with a non-zero code if any
  assertion fails. The smoke run should complete in under 60 seconds.
- **An accompanying test file** (`test_<script_name>.py` in the same directory) that
  imports the script's core functions and asserts expected behaviour on small inputs.

A script without one of these cannot be trusted to be faithfully implemented. Run the
smoke test or test file immediately after any script is written or modified — before
the seed loop begins.

### 3. Original experiment artifacts are ground truth

When any plan document (including this one) conflicts with what a source script or
output artifact actually does, **the source script wins.** This plan is a specification
of intent; the scripts are the record of what was actually run and validated.

- Always read the relevant script before implementing a rerun version of it — do not
  rely on plan descriptions alone
- When a discrepancy is found between this plan and a source script, log it as an
  `issue` in the journal and resolve it explicitly (update the plan or update the script)
  before proceeding
- Original output artifacts (`figures/`, `*.json` metric files) in the experiment
  directories are the authoritative single-seed baseline; rerun results should be
  compared against them for sanity, not silently overwritten

**Source directories for each phase:**

| Phase | Source directory |
|-------|-----------------|
| H2 | `experiments/h2_ml_lab/` |
| H6 | `experiments/h6_hybrid/` |
| RBA | `experiments/h2_rba/` |

### 4. Journal everything with ml-journal

Every meaningful event during this rerun must be logged immediately using the
`ml-journal` plugin (`/ml-journal:log-entry` for experiment events, `/ml-journal:log-commit` for commits). Logging is proactive — do not batch entries or defer until
after a session ends.

**Log these event types as they happen:**

| Event | Journal type | When to log |
|-------|-------------|-------------|
| A seed run completes | `experiment` | Immediately after results.json is written |
| A consistency check fails | `issue` | The moment it is detected |
| A decision about threshold, schema, or approach | `decision` | Before implementing the change |
| An unexpected result or behavior | `discovery` | When it diverges from expectation |
| A bug found in a script | `issue` | When identified; follow with `resolution` when fixed |
| A script is modified for the rerun | `decision` | Document what changed and why |
| A seed is aborted | `issue` | Record which seed, which phase, and the abort reason |

**Rules:**
- One entry per event — do not combine multiple events into one log entry
- Include the seed number and phase (H2/H6/RBA) in every experiment entry
- Issue entries must be followed by a `resolution` entry once the issue is closed
- Do not log trivial mechanical steps (file saves, formatting) — log findings and decisions

---

## 6. Experiment Run Order

### Phase 1: H2 (Mechanism)

*experiments/h2_ml_lab*

**Purpose:** Establish mechanistic foundation — embedding comparison, T1–T8 diagnostics.

**Per seed, run in order:**

1. Generate synthetic data (400 accounts, 60 events/account, fleet injection 25%)
2. Train FastText robust config (sg=1, per-account corpus, epochs=20)
3. Train FastText degenerate config (sg=0, per-event corpus, epochs=10) — for T8 comparison
4. Compute all AUC metrics (novel, fleet, spoof) for mean-pool and concat w=1.
   **Spoof definition:** 3-field (tz differs + random network + random screen), matching
   `pre_ml_lab/h2_rerun_experiment1.py`. The current `robust_config_experiment.py` uses
   1-field spoof (tz only), producing AUC 0.538 — this must be corrected in the rerun
   script. Decision `8420e9e8`; issue `ce3a6063`.
5. Compute bootstrap CIs (N=1000 resamples) on all 4 primary deltas
6. Run T2: window sweep (w=1,3,6) for concat
7. Run T3: prefixed-concat silhouette comparison
8. Run T4: tz-counterfactual attribution
9. Run T5: tz-permutation (6 positions)
10. Run T6: compactness with CIs
11. Run T8: within-feature and cross-feature cosine similarity (robust and degenerate configs)
12. Compute all verdicts and write results.json

**Expected runtime per seed:** estimate before first full run.

**Abort condition:** if T8 within-feature sim > 0.9 under robust config for any seed,
stop and debug corpus construction before continuing.
*(Threshold: > 0.9 across all phases — decision `a2b73375`. Empirical baseline: robust=0.392, degenerate=0.9993.)*

#### Implementation gap table *(initial audit snapshot — journal is authoritative)*

| Step | Plan says | Script does | Status |
|------|-----------|-------------|--------|
| 1 | 400 accts, 60 events, 25% fleet | ✓ exact match | OK |
| 2 | Train robust config sg=1, per-account, epochs=20 | ✓ exact match | OK |
| 3 | Train degenerate config sg=0, per-event, epochs=10 | Degenerate trained for figures — verify epochs=10 | ~~Verify~~ OK — `epochs=10` confirmed in `_degenerate_kwargs` |
| 4 | AUC for novel/fleet/spoof, mean-pool + concat w=1 | Script uses 1-field spoof (tz only); REPORT.md values use 3-field spoof (tz + net + screen) from pre_ml_lab — rerun must adopt 3-field definition | ~~Bug~~ resolved → `e9c536a2` + decision `8420e9e8` |
| 5 | Bootstrap CIs on 4 **deltas** (mp − concat) | Bootstraps **per-model AUC**, not deltas; no Δ(spoof), Δ(novel), Δ(fleet), Δ(silhouette) | ~~Gap~~ resolved → `90ec5dc0` |
| 6 | T2: window sweep w=1,3,6 | Not in this script — lives in a separate experiment file | ~~Gap~~ resolved → `993f16dc` |
| 7 | T3: prefixed-concat silhouette | Not in this script | ~~Gap~~ resolved → `993f16dc` |
| 8 | T4: tz-counterfactual attribution | ✓ present | OK |
| 9 | T5: tz-permutation (6 positions) | Not in this script | ~~Gap~~ resolved → `993f16dc` |
| 10 | T6: compactness with CIs | ✓ present | OK |
| 11 | T8: within/cross-feature sim, robust+degenerate | T8 runs, but collapse threshold is **0.99** in script vs. **0.5** in abort condition | ~~Conflict~~ resolved → `a5db5a5b` (threshold 0.9) |
| 12 | Write `results.json` with T1–T8 verdict fields | Console output only; no JSON file; no structured T1–T8 verdicts | ~~Gap~~ resolved → `5971b648` |

#### Existing figures — H2 (single-seed, seed=42)

**Robust config series** (`robust_*`) — these are the valid single-seed baselines for sanity-checking rerun output:

| File | Content | Maps to |
|------|---------|---------|
| `robust_summary_auc.png` | AUC by attack type (novel/fleet/spoof), mean-pool vs concat | C4 Fig 1 |
| `robust_t4_tz_counterfactual.png` | T4 tz-attribution under robust config | C4 Fig 3 precursor |
| `robust_t6_compactness.png` | T6 per-account cosine distance distribution, mean-pool vs concat | C4 Fig 4 |
| `robust_t8_token_similarity.png` | T8 within/cross-feature similarity, robust config | C1 Fig 1 (robust half) |
| `config_verification_t8.png` | T8 side-by-side robust vs degenerate | C1 Fig 1 precursor |

**Degenerate config** — needed for the robust vs degenerate collapse contrast (C1):

| File | Content | Maps to |
|------|---------|---------|
| `finding_08_token_similarity.png` | T8 within/cross-feature similarity, degenerate config | C1 Fig 1 (degenerate half) |

**Exploratory / superseded series** (`finding_*`, `poc_*`) — produced before the robust config correction; not used as rerun sanity baselines and not required for any RESEARCH_REQUIREMENTS figure:

| File | Content |
|------|---------|
| `finding_01_window_sweep.png` | T2 window sweep (degenerate config era) |
| `finding_02_ngram_ablation.png` | N-gram ablation |
| `finding_03_matching_feature_ablation.png` | Matching feature ablation |
| `finding_04_tz_counterfactual.png` | T4 tz-attribution (degenerate config) |
| `finding_05_fleet_stratified.png` | Fleet stratified AUC |
| `finding_06_compactness.png` | T6 compactness (degenerate config) |
| `finding_06_silhouette.png` | Silhouette scores |
| `finding_spoof_score_distribution.png` | Spoof cosine score distributions |
| `poc_auc_comparison.png` | Original PoC AUC comparison |
| `normalized_score_auc.png` | AUC with normalized scores |
| `summary_all_attack_types.png` | Summary AUC all attack types |
| `summary_all_conditions_spoof.png` | Spoof AUC across all conditions |
| `variable_spoof_auc.png` | Variable k-spoof AUC |

> `robust-normalized/` subdirectory exists but is empty. All publication figures are regenerated from 5-seed aggregate — both robust and degenerate configs.

---

### Phase 2: H6 (Architecture)

*experiments/h6_hybrid*

**Purpose:** Realistic imbalance, attack taxonomy, final architecture recommendation.

**Per seed, run in order:**

1. Load RBA clean-login marginals (fixed — RBA data does not vary by seed)
2. Chain-sample 400 accounts using seed
3. Inject attacks (spoof k=1/2/3, novel, fleet) using seed
4. Train FastText on chain-sampled corpus using seed
5. Score all events with mp_raw, mp_rank_norm, two_stage, trivial
6. Apply temporal blocklist (lag=10d, window=30d)
7. Compute spoof k=1 metrics (ROC-AUC, PR-AUC with CIs) for all scorers
8. Compute fleet-residual metrics (pre-lag accounts only) for all scorers
9. Run T8 on H6 embedding
10. Compute verdicts and write results.json

**Note on RBA marginals:** the marginal extraction from RBA is deterministic (no sampling).
Only the chain-sampling of accounts varies by seed.

**Abort condition:** if mp_raw PR-AUC on spoof k=1 < trivial PR-AUC for any seed,
investigate before continuing.

#### Implementation gap table *(initial audit snapshot — journal is authoritative)*

| Step | Plan says | Script does | Status |
|------|-----------|-------------|--------|
| 1 | Load RBA marginals (fixed) | ✓ loads `rba_marginals.json` | OK |
| 2 | Chain-sample 400 accounts via seed | ✓ `--seed` CLI arg, `np.random.default_rng(args.seed)` | OK |
| 3 | Inject attacks via seed | ✓ | OK |
| 4 | Train FastText via seed | FastText `seed=42` hardcoded — does not use `args.seed` | ~~Bug~~ false positive → `0b6b9c36` |
| 5 | Score: mp_raw, mp_rank_norm, two_stage, trivial | ✓ (plus `two_stage_rank_norm` not in plan) | OK+ |
| 6 | Apply blocklist lag=10d, window=30d | ✓ | OK |
| 7 | Spoof k=1 ROC-AUC + PR-AUC with CIs | ✓ | OK |
| 8 | Fleet-residual metrics for all scorers | ✓ | OK |
| 9 | T8 on H6 embedding | **T8 absent from script entirely** | ~~Gap~~ resolved → `f4fae493` |
| 10 | Write `results.json` with plan verdict schema | Writes `h6_metrics.json` — verdict fields don't match plan | ~~Mismatch~~ resolved → `e414dab0` |

#### Existing figures — H6 (single-seed, seed=42)

**Full-run figures** (non-smoke):

| File | Content | Maps to |
|------|---------|---------|
| `h6_hybrid_auc.png` | AUC comparison across attack types and scorers | C4 Fig 1 precursor |
| `h6_roc_spoof_k1.png` | ROC curve: all scorers on spoof k=1 | C2 Fig 2 precursor |
| `h6_roc_spoof_k2.png` | ROC curve: all scorers on spoof k=2 | — |
| `h6_roc_spoof_k3.png` | ROC curve: all scorers on spoof k=3 | — |
| `h6_roc_novel.png` | ROC curve: all scorers on novel attacks | — |
| `h6_roc_fleet.png` | ROC curve: all scorers on fleet aggregate | C3 precursor |
| `h6_roc_fleet_residual.png` | ROC curve: all scorers on fleet residual (pre-lag) | C3 Fig 3 precursor |

> `h6_roc_k1.png`, `h6_roc_k2.png`, `h6_roc_k3.png` also exist and appear to be earlier naming variants of the `h6_roc_spoof_k*.png` series — verify before use.

**Missing from existing figures** (required by RESEARCH_REQUIREMENTS, not yet generated):

| Missing figure | Required for |
|----------------|-------------|
| PR curves: mp_raw vs mp_rank_norm | C2 Fig 1 (primary) |
| Score distribution histogram: raw vs rank-norm | C2 Fig 3 |
| Top-k precision chart (1%–10%): fleet residual | C3 Fig 2 |
| Population decomposition diagram | C3 Fig 1 |
| Spoof k gradient: PR-AUC at k=1/2/3 | C4 Fig 5 |

**Smoke test figures** (disposable — `*_smoke.png`): `h6_hybrid_auc_smoke.png`, `h6_roc_*_smoke.png` — generated by `--smoke` flag; not used in publication.

**Metrics artifacts in figures dir:** `h6_metrics.json` (canonical, `--neg-ratio 100`), `h6_metrics_smoke.json`.

---

### Phase 3: RBA (Distributional Realism Check)

*experiments/h2_rba*

**Purpose:** Confirm embedding behavior on open-vocabulary synthetic data with
real-world distributional structure.

**Per seed, run in order:**

1. Pre-flight: verify `data/rba/rba.parquet` exists and is readable — fail fast with a clear error if not (issue `b45320bc`)
2. Load RBA dataset (fixed)
3. Apply 50/50 temporal split (primary — split is deterministic on timestamps)
4. Train FastText on training split using seed
5. Score test events with mean-pool, concat, trivial
6. Compute ROC-AUC and PR-AUC with bootstrap CIs
7. Run T6 compactness and T8 token similarity diagnostics
8. Write `results.json` (primary, 50/50 split). `h2_replicated` must use the
   CI-lower-bound criterion: `mp_ci_lower > triv_roc` — not the simple point-estimate
   comparison `mp_roc > triv_roc`. See Section 4 RBA rationale and decision `9511d90f`.
9. Re-run steps 3–8 at 40/60 split → write `results_split40.json`
10. Re-run steps 3–8 at 60/40 split → write `results_split60.json`

All three split results must be present before aggregation. Sensitivity splits share
the same trained model from step 4 — only the test set changes.

**Note:** Because the RBA split is timestamp-based, the only variation across seeds
is FastText training stochasticity. Expect lower variance here than H2 or H6.
Decision logged: `7dffb097`.

**Baseline warning:** The existing `experiments/h2_rba/rba_metrics.json` on disk was
produced from a 40/60 split (`n_ato_test_events=12`, `roc_auc=0.9212`). The canonical
primary result (TECHNICAL_REPORT §6.1) is the 50/50 split (`n_ato=9`, `roc=0.852`).
Do **not** use the on-disk file as the seed-42 sanity baseline. Seed-42 must be run
first at 50/50 to produce the correct baseline before cross-seed comparison. Decision
logged: `81ed922b`.

**Framing reminder:** results.json field `h2_replicated` should be assessed against
the criterion: mean-pool ROC-AUC > trivial with non-overlapping CI lower bound.

#### Implementation gap table *(initial audit snapshot — journal is authoritative)*

| Step | Plan says | Script does | Status |
|------|-----------|-------------|--------|
| 1 | Load RBA dataset | ✓ loads `rba.parquet` — **external file required** | OK (note) |
| 2 | 50/50 temporal split (fixed) | ✓ quantile-based, `split_pct=0.50` | OK |
| 3 | Train FastText via seed | `np.random.seed(SEED)` hardcoded; no `--seed` CLI | ~~Gap~~ resolved → `36a0e3c0` |
| 4 | Score: mean-pool, concat, trivial | ✓ | OK |
| 5 | ROC-AUC + PR-AUC with bootstrap CIs (N=1000) | ✓ | OK |
| 6 | T6 compactness + T8 token similarity | ✓ both present | OK |
| 7 | Write `results.json` matching Section 4 schema | Writes `rba_metrics.json` — schema doesn't match plan | ~~Mismatch~~ resolved → `9da4f78d` |

#### Existing figures — RBA (single-seed, seed=42)

| File | Content | Maps to |
|------|---------|---------|
| `rba_summary_auc.png` | ROC-AUC and PR-AUC: mean-pool vs concat vs trivial | Replication check |
| `rba_pr_curve.png` | PR curve: mean-pool vs trivial | Replication check |
| `rba_t6_compactness.png` | T6 per-account cosine distance distribution | Replication check |

> **Warning:** `rba_metrics.json` in this directory was produced from a **40/60 split** (`n_ato_test_events=12`, `roc_auc=0.9212`). The canonical primary result is the 50/50 split (`n_ato=9`, `roc=0.852`). Do not use the on-disk JSON as a seed-42 sanity baseline — re-run seed-42 at 50/50 first. Decision logged: `81ed922b`.
>
> No figures are explicitly required for RBA in RESEARCH_REQUIREMENTS — these serve as internal replication checks only.

---

### run_all.sh Specification

`experiments/rerun/run_all.sh` is the canonical single entry point for the full rerun.
Decision logged: `f263d320`.

**Interface:**

```bash
./run_all.sh [--seeds "42 123 456 789 2024"] [--phases "h2 h6 rba"]
```

**Behaviour — per seed × phase:**

1. Invoke the phase script with `--seed <seed>`, writing output to
   `experiments/rerun/logs/<phase>_seed<seed>.log`
2. On non-zero exit: print the abort reason, stop the entire run (no silent continuation)
3. After the phase script exits zero: run per-seed consistency checks (Section 8)
4. On consistency check failure: treat as abort — stop and report which check failed

**Behaviour — after all seeds complete:**

5. Invoke the aggregation script (`aggregate.py` or equivalent)
6. Print a summary table: seed × phase × status (pass/fail)

**Constraints:**
- Uses `uv run` for all Python invocations — never bare `python`
- Logs are append-safe: re-running a single seed does not clobber other seeds' logs
- Exit code mirrors the first failure encountered

---

### Open Issues

All open issues, decisions, and resolutions for this rerun are tracked in the project
journal — not in this document. The gap tables above are a static audit snapshot;
the journal is the live, authoritative record.

**To view open rerun issues:** use `/ml-journal:log-list` or `/ml-journal:log-status`.

**To log a resolution once an issue is fixed:** use `/ml-journal:log-entry` with
`type: resolution` and reference the issue ID.

**Key issues from the initial audit** (check journal for current status):

| Phase | Severity | Journal ID | Summary |
|-------|----------|------------|---------|
| H2 | ~~critical~~ | `4906a795` | ~~No `--seed` CLI; SEED=42 hardcoded~~ — **resolved** → `18440a94` |
| H2 | ~~critical~~ | `548285a0` | ~~T2/T3/T5 have no seed-parameterized scripts~~ — **resolved** → `993f16dc` |
| H2 | ~~critical~~ | `dd56db0b` | ~~No `results.json` written (console-only output)~~ — **resolved** → `5971b648` |
| H2 | ~~high~~ | `8361a096` | ~~T8 threshold 0.99 in script vs. 0.5 abort condition~~ — **resolved** → `a5db5a5b` |
| H2 | ~~high~~ | `fbfd0775` | ~~Bootstrap CIs on per-model AUC, not deltas~~ — **resolved** → `90ec5dc0` |
| H6 | ~~critical~~ | `da184294` | ~~FastText seed=42 literal — data varies, embeddings don't~~ — **false positive** → `0b6b9c36` |
| H6 | ~~high~~ | `3b3a777b` | ~~T8 entirely absent from script~~ — **resolved** → `f4fae493` |
| H6 | ~~high~~ | `25f721cf` | ~~Verdict schema mismatch~~ — **resolved** → `e414dab0` |
| H6 | ~~moderate~~ | `f740cf78` | ~~--neg-ratio default is 1 not 100~~ — **resolved** → `7393a705` |
| H6 | ~~moderate~~ | `324c3676` | ~~Schema missing spoof_k2/k3/novel/fleet blocks~~ — **resolved** → `13a4ee60` |
| H6 | ~~low~~ | `be453b78` | ~~two_stage_rank_norm undocumented~~ — **resolved** → `da7d1d80`; schema+script corrected 2026-04-21 |
| RBA | ~~critical~~ | `983dfc46` | ~~No `--seed` CLI; SEED=42 hardcoded~~ — **resolved** → `36a0e3c0` |
| RBA | ~~high~~ | `b0218ada` | ~~Output schema missing seed/timestamp/split_percentile~~ — **resolved** → `9da4f78d` |
| Cross | ~~critical~~ | `ab9c0306` | ~~Seed parameterization inconsistent across all phases~~ — **resolved** → `ae77729f` |
| Cross | ~~critical~~ | `c4d6de92` | ~~results.json missing or schema-inconsistent in all phases~~ — **resolved** → `2c3dd40e` |
| Cross | ~~high~~ | `27ef710e` | ~~Dep version pins are `>=` not `==`~~ — **resolved** → `34f0291e` |
| Cross | ~~high~~ | `13ced399` | ~~T8 threshold not canonical across phases~~ — **resolved** → `0e1c3fd4` |
| Cross | ~~moderate~~ | `586a4a7c` | ~~`run_all.sh` not written~~ — **resolved** → `f263d320` |
