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

## 4. Metrics Schema

Define this schema once and use it identically across all seeds.

### H2 metrics (per seed)

```json
{
  "seed": 42,
  "timestamp": "ISO8601",
  "config": { "sg": 1, "epochs": 20, "negative": 10, "min_n": 3, "max_n": 6,
              "window": 6, "vector_size": 64, "corpus": "per_account" },
  "auc": {
    "mean_pool": { "novel": 0.0, "fleet": 0.0, "spoof": 0.0 },
    "concat_w1": { "novel": 0.0, "fleet": 0.0, "spoof": 0.0 },
    "trivial":   { "novel": 0.0, "fleet": 0.0, "spoof": 0.0 }
  },
  "bootstrap_ci_95": {
    "per_model": {
      "mean_pool": {
        "spoof": { "estimate": 0.0, "lower": 0.0, "upper": 0.0 },
        "novel": { "estimate": 0.0, "lower": 0.0, "upper": 0.0 },
        "fleet": { "estimate": 0.0, "lower": 0.0, "upper": 0.0 }
      },
      "concat_w1": {
        "spoof": { "estimate": 0.0, "lower": 0.0, "upper": 0.0 },
        "novel": { "estimate": 0.0, "lower": 0.0, "upper": 0.0 },
        "fleet": { "estimate": 0.0, "lower": 0.0, "upper": 0.0 }
      }
    },
    "deltas": {
      "spoof_delta":      { "estimate": 0.0, "lower": 0.0, "upper": 0.0 },
      "novel_delta":      { "estimate": 0.0, "lower": 0.0, "upper": 0.0 },
      "fleet_delta":      { "estimate": 0.0, "lower": 0.0, "upper": 0.0 },
      "silhouette_delta": { "estimate": 0.0, "lower": 0.0, "upper": 0.0 }
    }
  },
  "silhouette": { "mean_pool": 0.0, "concat": 0.0 },
  "compactness": {
    "mean_pool": { "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "concat":    { "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 }
  },
  "t2_window_sweep": {
    "concat_w1": { "spoof_auc": 0.0 },
    "concat_w3": { "spoof_auc": 0.0 },
    "concat_w6": { "spoof_auc": 0.0 },
    "mean_pool_spoof_auc": 0.0
  },
  "t3_prefixed_concat": {
    "mean_pool_silhouette": 0.0,
    "prefixed_concat_silhouette": 0.0,
    "gap": 0.0
  },
  "t4_tz_attribution": { "mean_pool": 0.0, "concat": 0.0 },
  "t5_tz_permutation": {
    "position_1": { "concat_spoof_auc": 0.0 },
    "position_2": { "concat_spoof_auc": 0.0 },
    "position_3": { "concat_spoof_auc": 0.0 },
    "position_4": { "concat_spoof_auc": 0.0 },
    "position_5": { "concat_spoof_auc": 0.0 },
    "position_6": { "concat_spoof_auc": 0.0 },
    "mean_pool_spoof_auc": 0.0
  },
  "t7_trivial_baseline": {
    "mean_pool_spoof_margin": 0.0,
    "concat_w1_spoof_margin": 0.0
  },
  "t8_token_similarity": {
    "robust_config": {
      "within_feature_mean": 0.0,
      "cross_feature_mean": 0.0,
      "within_cross_ratio": 0.0,
      "collapse_detected": false
    },
    "degenerate_config": {
      "within_feature_mean": 0.0,
      "cross_feature_mean": 0.0,
      "within_cross_ratio": 0.0,
      "collapse_detected": true
    },
    "factorial_2x2": {
      "params": "epochs=20, negative=10, window=6 (standardized; only sg and corpus vary)",
      "sg_per_account":   { "within_feature_mean": 0.0, "cross_feature_mean": 0.0, "within_cross_ratio": 0.0, "collapse_detected": false, "note": "reuses robust_config model" },
      "cbow_per_event":   { "within_feature_mean": 0.0, "cross_feature_mean": 0.0, "within_cross_ratio": 0.0, "collapse_detected": true },
      "cbow_per_account": { "within_feature_mean": 0.0, "cross_feature_mean": 0.0, "within_cross_ratio": 0.0, "collapse_detected": null },
      "sg_per_event":     { "within_feature_mean": 0.0, "cross_feature_mean": 0.0, "within_cross_ratio": 0.0, "collapse_detected": null }
    }
  },
  "enrollment_diagnostics": {
    "mean_pool_enrollment_dist": 0.0,
    "concat_enrollment_dist": 0.0
  }
}
```

### H6 metrics (per seed)

Blocklist variants (`trivial_blocklist`, `two_stage_blocklist`, `combined`) appear
**only in fleet blocks**. `blocklist_fires` is structurally always `False` for spoof
and novel events — including blocklist scorers there would be misleading, not a
simplification (decision `b61c5405`). Run with `--neg-ratio 100` (decision `dd39e09c`).

```json
{
  "seed": 42,
  "timestamp": "ISO8601",
  "config": { "n_accounts": 400, "vocab_size": 229, "imbalance_ratio": 100,
              "fleet_frac": 0.25, "blocklist_lag_days": 10, "attack_window_days": 30,
              "neg_ratio": 100 },
  "spoof_k1": {
    "mp_raw":              { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "two_stage":           { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "mp_rank_norm":        { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "two_stage_rank_norm": { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "trivial":             { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 }
  },
  "spoof_k2": {
    "mp_raw":              { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "two_stage":           { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "mp_rank_norm":        { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "two_stage_rank_norm": { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "trivial":             { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 }
  },
  "spoof_k3": {
    "mp_raw":              { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "two_stage":           { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "mp_rank_norm":        { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "two_stage_rank_norm": { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "trivial":             { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 }
  },
  "novel": {
    "mp_raw":              { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "two_stage":           { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "mp_rank_norm":        { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "two_stage_rank_norm": { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "trivial":             { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 }
  },
  "fleet_aggregate": {
    "mp_raw":              { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "two_stage":           { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "trivial_blocklist":   { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "trivial":             { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 }
  },
  "fleet_residual": {
    "mp_raw":              { "roc_auc": 0.0, "pr_auc": 0.0, "top1pct_tp": 0, "top1pct_prec": 0.0 },
    "two_stage":           { "roc_auc": 0.0, "pr_auc": 0.0, "top1pct_tp": 0, "top1pct_prec": 0.0 },
    "mp_rank_norm":        { "roc_auc": 0.0, "pr_auc": 0.0, "top1pct_tp": 0, "top1pct_prec": 0.0 },
    "two_stage_rank_norm": { "roc_auc": 0.0, "pr_auc": 0.0, "top1pct_tp": 0, "top1pct_prec": 0.0 },
    "trivial_blocklist":   { "roc_auc": 0.0, "pr_auc": 0.0, "top1pct_tp": 0, "top1pct_prec": 0.0 },
    "two_stage_blocklist": { "roc_auc": 0.0, "pr_auc": 0.0, "top1pct_tp": 0, "top1pct_prec": 0.0 },
    "combined":            { "roc_auc": 0.0, "pr_auc": 0.0, "top1pct_tp": 0, "top1pct_prec": 0.0 },
    "trivial":             { "roc_auc": 0.0, "pr_auc": 0.0, "top1pct_tp": 0, "top1pct_prec": 0.0 }
  },
  "t8_token_similarity": {
    "within_feature_mean": 0.0,
    "cross_feature_mean": 0.0,
    "within_cross_ratio": 0.0,
    "collapse_detected": false
  },
  "verdicts": {
    "primary_criterion_confirmed": true,
    "rank_norm_collapse_confirmed": true,
    "gate_blinds_fleet_confirmed": true
  }
}
```

### RBA metrics (per seed)

```json
{
  "seed": 42,
  "timestamp": "ISO8601",
  "split_percentile": 50,
  "n_ato_test_events": 0,
  "auc": {
    "mean_pool": { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "concat":    { "roc_auc": 0.0, "pr_auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0 },
    "trivial":   { "roc_auc": 0.0, "pr_auc": 0.0 }
  },
  "t6_compactness": { "mean_pool": 0.0, "concat": 0.0 },
  "t8_token_similarity": {
    "within_feature_mean": 0.0,
    "cross_feature_mean": 0.0,
    "within_cross_ratio": 0.0,
    "collapse_detected": false
  },
  "h2_replicated": true
}
```

### Metrics Rationale

#### H2 Metrics

##### AUC by attack type (novel, fleet, spoof) — mean_pool, concat_w1, trivial

These are the primary outcome variables for the core hypothesis. Each attack type is included for a distinct reason. Novel AUC tests the easy case — three or more features differ — and establishes that the embedding is functioning at all. Fleet AUC tests the medium case where the attacker's device may have appeared in the training window, stressing the centroid-based scorer. Spoof AUC is the hardest case and the primary pre-specified claim: only one feature differs, so the embedding must be sensitive enough to detect a single-dimension shift. All three must be reported because a model that wins on novel and fleet but fails on spoof has not confirmed H2 — that was exactly the degenerate config failure mode.

Trivial is included as the floor. A model that doesn't beat exact set-membership on spoof adds no value over a hash lookup. Without the trivial column, the AUC numbers have no interpretable reference point.

`concat_w3` and `concat_w6` are not in this block — they belong in `t2_window_sweep`, which is the designated home for the window sweep comparison. Keeping the top-level `auc` block to w=1 only makes the primary comparison table clean.

##### Bootstrap CIs — per_model and deltas

The `per_model` block records per-AUC-type CIs for mean_pool and concat_w1. The `deltas` block records the four pre-registered T1 CIs: spoof AUC delta, novel AUC delta, fleet AUC delta, and silhouette delta (each mean_pool minus concat_w1). CIs excluding zero in the `deltas` block are the defense-wins condition for T1. Both blocks are stored because `per_model` CIs are useful for per-scorer uncertainty reporting (error bars in figures), while `deltas` CIs are the statistical test of the hypothesis itself. Storing both avoids reconstructing CIs from components post-hoc. Reporting all four deltas rather than just spoof guards against the appearance of cherry-picking.

##### Silhouette — mean_pool, concat

Silhouette score measures how well-separated the per-account clusters are relative to adjacent clusters. It independently corroborates the AUC finding through a different lens — cluster quality rather than downstream detection performance. A model could achieve high AUC through a lucky threshold without having learned a compact representation; silhouette tests the representation quality directly. The delta CI on silhouette has its own pre-specified defense-wins threshold (> 0.05), making it an independently falsifiable claim.

##### Compactness — mean_pool, concat (with CI)

Compactness is the mean cosine distance of training events to their account centroid. It's the T6 metric and measures tightness of within-account clusters, which directly affects the scorer's ability to flag deviations. Mean-pool's 3.4× tighter clusters than concat is mechanistically important: it explains *why* mean-pool scores spoof events more anomalously — the centroid is a more precise representation of the account's typical behavior. Without this metric the AUC advantage is an empirical observation without a structural explanation. The CIs are included because the claim is that the compactness advantage is statistically robust, not just a point estimate.

##### T2 window_sweep — concat spoof AUC at w=1, w=3, w=6

Window sweep tests whether concat can recover from cross-boundary n-gram contamination by narrowing the training context window. If contamination is structural (baked into subword vectors at training time), no window setting should close the gap to mean-pool. All three concat windows (w=1, w=3, w=6) are required to characterize the sweep monotonically — a single-point comparison at w=1 alone does not establish whether the gap is window-dependent. `mean_pool_spoof_auc` is included as a reference so the gap is readable directly from the block without joining to the `auc` block.

##### T3 prefixed_concat — raw silhouettes and gap

Prefixed-concat uses non-overlapping feature delimiters (e.g., `os:ios browser:safari tz:utc-5`) to eliminate cross-boundary n-grams structurally. If the contamination mechanism is correct, the silhouette gap versus mean-pool should persist even with non-overlapping boundaries. Recording both raw silhouettes (`mean_pool_silhouette`, `prefixed_concat_silhouette`) alongside the gap allows a reviewer to verify the comparison is coherent — a gap scalar alone does not show whether one variant is negative while the other is positive, which would require a different interpretation than a uniform shift.

##### T4 tz_attribution — mean_pool, concat

Tz-attribution is the counterfactual cosine distance attributable specifically to the timezone feature — computed by holding all other features constant and varying only timezone. This is the mechanistic test for whether the embedding is sensitive to the feature that distinguishes spoof attacks. T4 addresses the obvious alternative explanation: maybe mean-pool beats concat on spoof for reasons unrelated to timezone discriminability. Both values being non-trivially positive under the robust config is the expected pattern — it confirms the mechanism rather than just the outcome. Mean-pool's lower tz-attr (0.028 vs. concat's 0.062) is not a contradiction: concat's cross-boundary n-grams uniquely fingerprint the timezone substring, giving it higher per-token tz signal, but that advantage is swamped by dilution from five matching feature dimensions.

##### T5 tz_permutation — concat spoof AUC by position

Tz-permutation tests whether contamination is localized to a specific feature boundary. If placing tz at position 1 instead of position 3 recovers spoof AUC, the contamination is positional and addressable by reordering. Named positions (`position_1` through `position_6`) rather than an index array make the schema self-documenting. `mean_pool_spoof_auc` is the reference value for the same reason as T2: the gap to mean-pool should be readable directly per position.

##### T7 trivial_baseline — mean_pool and concat_w1 spoof margins

Margin over the trivial baseline (exact 6/6 set-membership, AUC 0.750) is the practical pass/fail criterion: an embedding that cannot beat a two-line hash lookup on spoof is not operationally justified. Storing the margin directly makes the comparison legible without looking up the trivial AUC separately. Both mean_pool and concat_w1 margins are recorded because the paper's claim is differentiated: mean_pool clears the bar decisively (+0.119), concat_w1 does not (+0.032, real but operationally narrow).

##### T8 token_similarity — robust_config and degenerate_config sub-blocks

T8 is the prerequisite check, not an outcome metric. The split into `robust_config` and `degenerate_config` sub-blocks is necessary because the finding is a contrast: robust (sg=1, per-account) produces within-feature sim = 0.392 (no collapse); degenerate (sg=0, per-event) produces 0.9993 (full collapse). Storing both in every seed result means the contrast is verifiable without consulting a separate artifact. `within_cross_ratio` (within_feature_mean / cross_feature_mean) is included because it is what the original scripts printed and provides an at-a-glance collapse signal: under no collapse, within < cross and the ratio is < 1; under collapse, the ratio inverts dramatically. `collapse_detected` is the hard gate — if it fires under `robust_config`, all downstream metrics for that seed are invalid. Under `degenerate_config` it should always be true; a seed where it is not would indicate a data generation anomaly.

**Canonical threshold (all phases):** `collapse_detected = (within_feature_mean > 0.9)`. Empirically derived from REPORT.md: robust config (H2) produced `within_feature_mean = 0.392`; degenerate CBOW config produced `0.9993`. The intermediate regime is uncharacterized, so 0.9 is used as a conservative boundary consistent with the published finding. Applies equally to H6 and RBA. Decision logged: `a2b73375`.

##### enrollment_diagnostics — mean_pool and concat enrollment distance

Enrollment distance is the mean cosine distance between a new enrollment event's embedding and the existing account centroid. It is distinct from compactness — compactness measures within-account training-event tightness; enrollment distance measures how far a typical new device sits from the centroid, which sets the operational false-positive rate for enrollment flows. Storing both models allows comparison of whether mean-pool's compactness advantage also translates to lower enrollment distance or is confined to the training distribution. This field has no baseline from the original scripts and will be computed fresh in the rerun.

---

#### H6 Metrics

##### Spoof k=1 — ROC-AUC and PR-AUC for all four scorers

k=1 country-change spoof is the primary pre-specified criterion for H6. All four scorers (mp_raw, two_stage, mp_rank_norm, trivial) must be evaluated together because the paper's claims are comparative: mp_raw beats trivial, rank_norm collapses PR-AUC relative to mp_raw, and two_stage is roughly equivalent to mp_raw on spoof but fails on fleet residual. Without all four in the same table, the comparisons can't be made.

ROC-AUC and PR-AUC are both required because the paper's central methodological argument is that ROC-AUC hides the rank_norm collapse. Showing both — ROC-AUC 0.995 vs. 0.972, PR-AUC 0.892 vs. 0.215 — is the evidence for that claim. If you only report one, you can't make the argument.

##### Fleet residual — ROC-AUC, PR-AUC, top1pct_tp, top1pct_prec

Fleet residual is the population-conditional evaluation: pre-lag cold-start accounts only, filtered to the window where the blocklist has not yet activated. This is the population the model actually serves for fleet attacks. The pre-specified finding is that trivial and two_stage score 0 true positives at top-1% on this population while mp_raw scores 180 TP at 91.8% precision.

Top-1% TP and precision are included alongside ROC-AUC and PR-AUC because they're operationally interpretable in a way that AUC is not. A fraud analyst setting a threshold cares about "how many real attacks do I catch if I review the top 1% of flagged events, and what fraction of those are real." The zero TP finding for trivial and two_stage is the decisive result for the architecture recommendation — it's not visible in AUC.

##### T8 on H6 embedding

T8 is included in H6 for the same reason as H2: the collapse check must be run on every trained model before downstream metrics are interpreted. H6 uses a different corpus (229-token RBA-calibrated vocabulary, chain-sampled accounts) than H2, so the training dynamics are different. Confirming no collapse on H6 is a separate assertion from confirming it on H2.

##### Verdicts — primary_criterion_confirmed, rank_norm_collapse_confirmed, gate_blinds_fleet_confirmed

These map directly to the three novel contributions in Sections 7 and 8 of the paper. Making them explicit boolean fields enables the same verdict stability analysis as H2 — you need to know across 5 seeds whether all three findings hold consistently. A seed where rank_norm_collapse is not confirmed would be a serious finding requiring investigation.

---

#### RBA Metrics

##### ROC-AUC and PR-AUC — mean_pool, concat, trivial

The RBA section is a distributional realism check, not a primary experiment, but it still needs both metrics for the same reason as H6: if you only report ROC-AUC you can't make any claim about operational performance. The PR-AUC at 0.0005% base rate is very low in absolute terms (0.032 vs. 0.0003 trivial), but the relative comparison (95× trivial) is meaningful and only visible through PR-AUC.

Concat is included as a secondary comparison — RBA's primary claim is that mean-pool beats trivial, but showing concat also outperforms trivial (and mean-pool outperforms concat) maintains consistency with H2 results and gives reviewers a fuller picture.

##### T6 compactness — mean_pool, concat

T6 on RBA tests whether the within-account cluster tightness finding from H2 generalizes to an open-vocabulary setting with realistic feature distributions. If compactness under mean-pool on RBA is consistent with H2 (it is: 0.036 vs. 0.047), that's diagnostic evidence that the structural property of mean-pool embeddings isn't an artifact of the closed toy vocabulary. It's a mechanistic consistency check, not an outcome metric for the RBA section.

##### T8 on RBA embedding

Same reasoning as H2 and H6 — collapse must be checked on every trained model. RBA is particularly interesting for T8 because the open vocabulary (hundreds of country codes, OS strings) creates richer co-occurrence structure, which should in principle make skip-gram's discriminative task easier. The higher within-feature similarity on RBA (0.563 vs. 0.392 in H2) while still below the 0.5 collapse threshold is a meaningful finding: it shows the collapse boundary is vocabulary-dependent, which is exactly what you'd predict from the mechanistic argument.

##### h2_replicated boolean

This is the summary verdict for the RBA section, assessed against the pre-specified criterion: mean-pool ROC-AUC > trivial with non-overlapping CI lower bound. It's a single boolean rather than a nuanced verdict because the RBA section makes one claim (directional replication) and you need to know across 5 seeds whether that claim holds consistently. The 5/5 seed stability on this field is the honest basis for saying "the finding replicates on open-vocabulary synthetic data."
