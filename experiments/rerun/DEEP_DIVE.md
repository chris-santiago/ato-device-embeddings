# Rerun Deep Dive — Complete Technical Reference

**Purpose:** Comprehensive reference for a colleague performing a deep dive into the
5-seed reproducibility rerun of H2, H6, and RBA experiments. Covers every step, every
design decision, every parameter choice, and how to interpret all outputs and figures.

**Status of this document:** Synthesized from all plan docs (Sections 1–11) and the
final implementation of all three experiment scripts. Where the two diverge, the scripts
are authoritative. Journal decisions cited by ID throughout.

---

## Table of Contents

1. [Why This Rerun Exists](#1-why-this-rerun-exists)
2. [The Four Publishable Contributions](#2-the-four-publishable-contributions)
3. [Infrastructure and Conventions](#3-infrastructure-and-conventions)
4. [Seed Policy — What a "Seed" Controls](#4-seed-policy)
5. [Phase 1: H2 — Mechanism Experiment](#5-phase-1-h2)
   - 5.1 Synthetic Data Generation
   - 5.2 Corpus Construction — Theory and Implementation
   - 5.3 FastText Architecture — Why These Hyperparameters
   - 5.4 Model Configurations (robust, degenerate, 2×2 factorial, T2/T3/T5)
   - 5.5 Scoring (H2) — centroid loop, cosine distance theory
   - 5.6 Bootstrap CI Mechanics — paired delta vs independent per-model
   - 5.7 Tests T1–T8 in Detail
   - 5.8 H2 Execution Flow
   - 5.9 H2 Results JSON Schema
6. [Phase 2: H6 — Architecture Experiment](#6-phase-2-h6)
   - 6.0 Theoretical Motivation
   - 6.1 RBA Marginals
   - 6.2 Account and Attack Generation (fleet two-pass)
   - 6.3 Spoof Base Event Selection
   - 6.4 Why N_CENTROID=40 — centroid/calib split and C2 theory
   - 6.4b The Five Scorers
   - 6.5 The Scoring Engine — collect_scores_for_attack() walkthrough
   - 6.6 Top-1% Precision — top_pct_metrics() and tie-breaking
   - 6.7 Enrollment Negatives and 1:100 Imbalance
   - 6.8 T8 in H6
   - 6.9 Scorer Sets by Attack Type
   - 6.10 two_stage_vs_trivial_roc_delta
   - 6.11 Three Verdict Fields — Exact Criteria
   - 6.12 H6 Execution Flow
   - 6.13 H6 Output Files
7. [Phase 3: RBA — Distributional Realism Check](#7-phase-3-rba)
   - 7.1 RBA Dataset and data_prep.py
   - 7.2 load_users() — The Polars Pipeline
   - 7.3 Temporal Split — Theory and Implementation
   - 7.4 User Filtering
   - 7.5 Model Training and _run_split()
   - 7.6 Three Splits Per Seed
   - 7.7 h2_replicated — CI-Lower-Bound Criterion
   - 7.8 T6 and T8 in RBA
   - 7.9 RBA Results JSON Schema
8. [Quality Gates and Consistency Checks](#8-quality-gates-and-consistency-checks)
   - 8.1 Per-Seed Abort Conditions
   - 8.2 H6 Consistency Checker — Operational Guide (S1–S8, X1–X3)
   - 8.3 Cross-Phase Consistency
9. [Aggregation and Paper Reporting](#9-aggregation-and-paper-reporting)
10. [Reading the Outputs](#10-reading-the-outputs)
11. [Key Design Decisions and Their Rationale](#11-key-design-decisions-and-their-rationale)

---

## 1. Why This Rerun Exists

The original experiments (H2 in `experiments/h2_ml_lab/`, H6 in `experiments/h6_hybrid/`,
RBA in `experiments/h2_rba/`) were single-seed runs at seed=42. They established the
findings but cannot alone support publication-quality reproducibility claims: a single seed
proves something happened, not that it reliably happens.

The rerun produces all results across 5 independent random seeds and aggregates them into
mean ± std, so the paper can state: "all pre-specified verdicts held in 5/5 seeds, with
primary metric standard deviations of ±X." It also catches any lucky-seed artifacts in the
original.

The rerun scripts are **not** copies of the originals — they were re-written from scratch
against the originals as ground truth, fixing a set of gaps that were discovered during a
pre-rerun audit: missing `--seed` CLI args, missing `results.json` output, per-model-only
bootstrap CIs instead of delta CIs, degenerate T8 configurations, and schema mismatches.
Every gap is logged as a journal issue; every fix is a journal resolution.

---

## 2. The Four Publishable Contributions

These are the claims the rerun must corroborate. Every test in every script ultimately
serves one of these.

### C1 — Within-Feature Embedding Collapse (Corpus Construction Mechanism)

**Claim:** Per-event corpus construction causes within-feature cosine similarity → 0.9992
± 0.00008 in structured categorical token sequences (degenerate config, 5-seed aggregate).
The mechanism is deterministic: rigid positional structure enforces identical co-occurrence
distributions for within-feature tokens regardless of training objective. Both CBOW and
Skip-gram collapse on per-event corpus (within = 0.982 ± 0.003 and 0.939 ± 0.007
respectively, 5/5 seeds); both recover on per-account corpus (within = −0.113 ± 0.006 and
0.424 ± 0.013, 0/5 seeds). Training objective is not the causal factor; corpus construction
is. Downstream, the collapsed cells score at or below the trivial baseline on spoof — see
the 5-seed diagnostic `scripts/h2/h2_degenerate_downstream.py` (§5.7, T8).

**Operationalized by:** H2 T8 (token similarity, robust and degenerate configs, 2×2
factorial). The 2×2 factorial is the causal proof: it holds training objective constant
while varying corpus, and vice versa, showing corpus is the varying factor.

### C2 — Rank-Normalization Collapse Under Realistic Imbalance

**Claim:** Per-user CDF rank-normalization destroys PR-AUC from 0.888 ± 0.026 to 0.224 ± 0.011
at 1:100 attack-to-benign imbalance, while ROC-AUC appears only modestly affected
(mp_raw 0.995 ± 0.001 → mp_rank_norm 0.974 ± 0.002).
ROC-AUC actively hides the collapse; PR-AUC is the honest metric in this regime.

**Operationalized by:** H6 spoof_k1 comparison of `mp_raw` vs `mp_rank_norm` PR-AUC.
The `rank_norm_collapse_confirmed` verdict in results.json.

**Scope (corrected):** A follow-up ablation shows the collapse is a thin-calibration-window
(small-sample `1/N`) effect that resolves as the calibration window grows, not an inherent
property of rank-normalization — see the CANONICAL correction in §6.4 below.

### C3 — Known-Device Gate Blinds Fleet Detection

**Claim:** The two-stage gate (suppress alerts on known devices) scores zero true positives
at top-1% on fleet_residual events. The mechanism: fleet devices appear in training by
construction (via injection), so the gate fires on every fleet device regardless of timing.
Raw cosine distance detects the same events at top-1% precision 0.493 ± 0.116 (91 ± 23 TP,
5-seed aggregate; the single-seed h6_hybrid value of 0.918 was a favorable draw from this
high-variance distribution — per-seed values span 0.32–0.65).

**Operationalized by:** H6 fleet_residual `two_stage` scorer top-1% TP count.
`gate_blinds_fleet_confirmed` verdict. Also: `two_stage_vs_trivial_roc_delta` < 1e-4 on
both fleet blocks (mechanistic claim: two_stage is literally identical to trivial on fleet;
measured exactly 0.0 on all 5 seeds).

### C4 — Mean-Pool Independent Tokens vs. Concatenated String

**Claim:** Mean-pooling one embedding per feature token outperforms embedding the full
device string as a single concatenated token for cosine-distance anomaly detection, with
the largest advantage on spoof attacks (only one feature differs). Mechanism: FastText
cross-boundary character n-grams in the concat string inject signal uncorrelated with any
single feature, diluting the contribution of the differing feature.

**Operationalized by:** H2 T1 (AUC comparison + delta CIs), T2 (window sweep), T3
(prefixed-concat), T4 (tz-counterfactual), T5 (tz-permutation), T6 (compactness), T7
(trivial baseline). PR-AUC amplifies the advantage: spoof PR-AUC delta is +0.248 vs
+0.130 for ROC-AUC. Concat_w1 spoof PR-AUC (0.542) is barely above the trivial baseline
(0.500) — ROC-AUC at 0.737 masks this near-failure. See `aggregate/figures/h2_auc_dual.png`.

---

## 3. Infrastructure and Conventions

### PEP 723 Self-Contained Scripts

Every script opens with a `# /// script` block:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim==4.4.0",
#   "numpy==2.2.6",
#   "scikit-learn==1.7.2",
#   "matplotlib==3.10.8",
# ]
# ///
```

Run **exclusively** with `uv run <script.py>`. Never plain `python`. The `uv` tool creates
an isolated environment from this block with exact version pins — no shared virtualenv,
no lockfile, no risk of cross-script dependency pollution.

**Why exact `==` pins?** An experiment that uses numpy 2.2.6 today but implicitly upgrades
to 2.3.x next month is not reproducible. Pins guarantee that every seed, on any machine,
uses identical numerical behavior. (Resolved issue `27ef710e`, resolution `34f0291e`.)

### Smoke Testing

Every script supports `--smoke`, which runs the full code path on a small synthetic subset
(typically 50 accounts, 200 bootstrap resamples) and exits zero if all assertions pass. The
smoke run typically completes in under 60 seconds. **Always run `--smoke` after any script
modification before the seed loop begins.**

The smoke run is structurally identical to the full run — same code path, same output
schema — just smaller data. This means smoke passing implies the full run won't crash on
schema errors or assertion failures. It does NOT guarantee the full run's numerical results.

### Directory Layout

```
experiments/rerun/
├── run_all.sh                  # Orchestration entry point
├── scripts/
│   ├── aggregate.py            # Cross-seed aggregation
│   ├── h2/
│   │   ├── h2_rerun.py         # Phase 1 runner
│   │   ├── h2_figures.py
│   │   ├── h2_cooccurrence.py  # C1 mechanism diagnostic (within-feature JSD)
│   │   └── h2_degenerate_downstream.py  # C1 downstream diagnostic (per-cell spoof/novel/fleet AUC)
│   ├── h6/
│   │   ├── h6_rerun.py         # Phase 2 runner
│   │   ├── h6_figures.py
│   │   ├── h6_score_margin.py  # C2 mechanism diagnostic (contamination rate, robust margin)
│   │   ├── check_consistency.py  # Per-seed and cross-seed checks (H6)
│   │   ├── data_prep.py
│   │   └── rba_marginals.json
│   └── rba/
│       ├── rba_rerun.py        # Phase 3 runner
│       ├── rba_figures.py
│       └── data_prep.py
├── seeds/
│   ├── seed_42/
│   │   ├── h2/results.json     # All H2 metrics for seed 42
│   │   ├── h6/results.json     # All H6 metrics for seed 42
│   │   ├── h6/scores.npz       # Raw score arrays (for figure generation)
│   │   └── rba/
│   │       ├── results.json          # Primary 50/50 split
│   │       ├── results_split40.json  # Sensitivity: 40/60 split
│   │       └── results_split60.json  # Sensitivity: 60/40 split
│   ├── seed_123/ ...
│   └── seed_2024/ ...
└── aggregate/                  # Cross-seed aggregated CSVs and figures
```

Each `results.json` is fully self-contained: it includes `seed`, `timestamp`, the full
config block, and all metrics. This means you can compare results from different machines
or runs without needing to reconstruct parameters.

---

## 4. Seed Policy

### The Five Seeds

```python
SEEDS = [42, 123, 456, 789, 2024]
```

Seed 42 is retained from the original for continuity (comparison baseline). The others
are arbitrary but fixed permanently.

### What Each Seed Controls

A single `--seed` integer controls **three independent randomness sources**:

| Source | Mechanism | Controlled by |
|--------|-----------|---------------|
| Synthetic data generation | `np.random.default_rng(seed)` | `rng = np.random.default_rng(args.seed)` |
| FastText training | Negative sampling draw order, word order shuffling | `FastText(..., seed=seed, workers=1)` |
| Bootstrap CI resampling | Resample index draws | `np.random.default_rng(seed)` in bootstrap functions |

**Important: `workers=1` is mandatory for reproducibility.** gensim FastText's `seed`
parameter controls weight initialization but not the order of gradient updates across
threads. With `workers > 1`, thread scheduling is OS-dependent, making training order
non-deterministic even with a fixed seed. `workers=1` ensures bit-reproducible training
— two runs with the same seed produce identical `.wv` vectors. (Memo `def12c25`.)

### FastText Seed Override Pattern

In H6, `ROBUST_KWARGS` contains `seed=42` as a default:

```python
ROBUST_KWARGS = dict(..., seed=42, workers=1)
```

This is overridden at runtime via dict merge:

```python
model = train_model(corpus, seed=args.seed)
# Inside train_model:
return FastText(sentences=corpus, **{**ROBUST_KWARGS, "seed": seed})
```

The rightmost key wins in Python dict merge, so `args.seed` always overrides the `seed=42`
literal. This was audited and confirmed correct — the original source script and the rerun
script both use this pattern. (Resolved as false-positive issue `da184294`, resolution
`0b6b9c36`.)

In RBA, `ROBUST_KWARGS_BASE` contains no seed key at all — seed is injected entirely at
runtime:

```python
ROBUST_KWARGS_BASE = dict(vector_size=64, window=6, sg=1, ..., workers=1)
robust_kwargs = {**ROBUST_KWARGS_BASE, "seed": seed}
```

---

## 5. Phase 1: H2 — Mechanism Experiment

**Script:** `experiments/rerun/scripts/h2/h2_rerun.py`
**Source:** `experiments/h2_ml_lab/`
**Purpose:** Establish mechanistic foundation — embedding comparison across attack types,
all T1–T8 diagnostics. Supports C1 (embedding collapse) and C4 (mean-pool vs concat).

### 5.1 Synthetic Data Generation

```python
accounts = generate_dataset(seed, n_accounts=400)  # N_ACCOUNTS = 400
```

**Feature space:** 6 features with fixed closed vocabularies:

```python
FEATURES = {
    "os":      ["ios", "android", "windows", "macos", "linux"],       # 5 values
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],    # 5 values
    "tz":      ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],  # 6 values
    "lang":    ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],   # 6 values
    "network": ["wifi", "lte", "5g", "broadband"],                    # 4 values
    "screen":  ["small", "medium", "large", "xlarge"],                # 4 values
}
```

Total vocabulary: 5+5+6+6+4+4 = 30 tokens (fixed, closed). This is the key difference
from H6 and RBA which use open vocabularies.

**Per account:**
- 2–4 known device profiles (drawn uniformly)
- 60 training events drawn from known devices with Zipf weights (s=1.5), so the primary
  device dominates
- 25% chance of fleet injection: one of the 60 training events is replaced with the shared
  fleet device
- One primary device (index 0 of known_devices)

**Token format:** `make_token(f, v)` → `"os_ios"`, `"browser_chrome"`, etc. The `{feature}_`
prefix is used by T8 to identify which feature a vocabulary token belongs to.

**Attack types per account:**
- **Novel:** 3+ fields differ (OS, tz, lang must all differ from primary). "Easy" attack —
  the attacker uses a completely foreign device.
- **Fleet:** Shared fleet device injected into 25% of accounts. The same physical device
  appears across multiple accounts, simulating a coordinated attack (botnet, credential
  stuffing).
- **Spoof (3-field):** Primary device with tz changed + random network + random screen.
  This is the "hardest" attack — only one identifying field (tz) definitively differs.
  Note: An earlier version of `robust_config_experiment.py` used 1-field spoof (tz only),
  producing AUC 0.538; it was corrected before its headline numbers were produced (see the
  "Spoof definition note" in `h2_ml_lab/docs/REPORT.md`). The final `robust_config_experiment.py`
  and the rerun use the identical 3-field definition from `pre_ml_lab/h2_rerun_experiment1.py`,
  matching REPORT.md values (0.869). (Decision `8420e9e8`.)
- **Negative (neg):** Same primary device with different network/screen — a benign login
  from a slightly different context. Label=0.
- **Known:** Primary device itself. Label=0. Used alongside neg as the negative pool for AUC.

**Why Zipf weights?** Real login behavior is bursty — users overwhelmingly log in from one
or two primary devices. Zipf(s=1.5) approximates this: the primary device gets ~50% of
traffic, the second ~25%, etc.

**Fleet injection mechanics:** The `fleet_device` is a single random device sampled
once per dataset (not per account) at the top of `generate_dataset()`. For each account,
`is_fleet = rng.random() < 0.25`. If True, a random position in `train_events` is
selected (`replace_idx = rng.integers(0, 60)`) and that event is overwritten with
`fleet_device`. Because `fleet_device` is now part of the training history for 25% of
accounts, it appears in `known_devices` for those accounts. This is the construction
that makes C3 possible — the fleet device gets "known" status via injection.

Note: `known_devices` in H2 is the list of device dicts (`acc["known_devices"]`), not a
set of tuples. The trivial scorer checks `device_key(ev) in acc["known_devices"]` via a
full list scan. (H6 uses a set of tuples for O(1) lookup — a performance improvement for
the larger open-vocabulary data.)

**Novel attack generation:** The novel device must have OS, tz, and lang all different
from the primary. The code retries up to 1000 times to find a valid combination:
```python
while (novel["os"] == primary["os"] or novel["tz"] == primary["tz"]
       or novel["lang"] == primary["lang"]) and attempts < 1000:
```
With a 6-feature vocabulary where each feature has 4–6 values, finding a triple-different
device takes a few tries on average. After 1000 failed attempts (near-impossible with this
vocabulary size), the last sampled device is used anyway.

**Spoof is 3-field, not 1-field:** An earlier version of `robust_config_experiment.py`
spoofed only tz, producing spoof AUC ≈ 0.538 (barely above chance); it was corrected to
the 3-field definition (tz forced + re-sampled network + re-sampled screen) from
`pre_ml_lab/h2_rerun_experiment1.py`, which matches the REPORT.md values (spoof AUC ≈
0.869). The rerun uses this same definition — the final h2_ml_lab script and the rerun
are identical on this point (verified line-for-line: `robust_config_experiment.py:116-120`
vs `h2_rerun.py:105-109`). Decision `8420e9e8`. The network and screen fields are
re-sampled (they may coincide with the primary's values), but tz is the meaningful change —
network and screen differences are also seen in neg events, so they don't uniquely
identify attacks.

**Neg events deliberately exclude the primary's exact value:** Network is sampled from
`[n for n in FEATURES["network"] if n != primary["network"]]` and screen similarly.
This ensures neg events are distinct from the primary, making the trivial baseline work
correctly — neg events will be "known" (one of the known device profiles) in many cases,
which is the expected behavior for benign logins from slightly different contexts.

### 5.2 Corpus Construction — Theory and Implementation

**Why co-occurrence structure determines embeddings:** FastText (and Word2Vec) learns
representations by predicting context. In Skip-gram, the model is trained to maximize
`P(context | center)`: given a token at position `i`, predict the tokens at positions
`i±1, i±2, ..., i±w`. The resulting embeddings encode proximity in co-occurrence space
— tokens that appear in similar contexts get similar embeddings.

This is the key theoretical insight for C1: if every sentence is exactly
`[os_X, browser_Y, tz_Z, lang_A, network_B, screen_C]` (per-event corpus), then
`tz_utc+0` and `tz_utc+5` always appear at position 2 in every sentence. They share
identical co-occurrence neighbors at every position: always `browser_*` to the left
(within window) and always `lang_*` to the right. Since their context distributions
are identical by construction, the model assigns them identical embeddings — collapse.

This is not a bug or a training failure. It is a provable consequence of the corpus
structure: if two tokens have identical PMI (pointwise mutual information) profiles
across all context tokens, gradient descent will converge to identical embedding vectors.
The per-event corpus enforces identical PMI for all within-feature token pairs.

Two corpus types are central to the experiment:

**Per-account corpus (robust config):**
```python
def build_mp_corpus_per_account(accounts):
    for acc in accounts:
        flat = []
        for e in acc["train_events"]:
            flat.extend(device_to_tokens(e))  # [os_X, browser_Y, tz_Z, ...]
        sentences.append(flat)  # ONE sentence per account containing ALL events
```

Each sentence is the concatenation of all token sequences for an account's 60 events.
This is 60 × 6 = 360 tokens per sentence. With window=6, `tz_utc+0` from event 1 and
`tz_utc+5` from event 2 are at positions 2 and 8 — distance 6, just barely within the
window. But events 3, 4, ... push them further apart. Crucially, different tz values
now appear in different positions within the long sentence, so their co-occurrence
neighborhoods are not structurally identical — they can differ based on what OS/browser
happened to co-occur in nearby events.

**The per-account corpus produces 400 sentences, each ~360 tokens:**
- Total corpus: ~144,000 tokens
- Vocabulary: 30 tokens (closed) — every token appears hundreds of times
- The large vocabulary-to-corpus ratio means each token gets rich training signal

**Per-account corpus removes the rigid positional structure.** `tz_utc+0` might be
preceded by `screen_large` (from event 59) in one account and by `lang_en_us` (from
event 12 boundary) in another. This variation breaks the identical-context guarantee
that causes collapse.

**Per-event corpus (degenerate config):**
```python
def build_per_event_corpus(accounts):
    for acc in accounts:
        for e in acc["train_events"]:
            sentences.append(device_to_tokens(e))  # ONE sentence per event = 6 tokens
```

Each sentence is exactly 6 tokens in a fixed order: `[os_X, browser_Y, tz_Z, lang_A,
network_B, screen_C]`. Because every sentence has the same fixed 6-token structure,
tokens at the same position always co-occur with the same set of tokens. All `tz_*`
tokens always appear in position 2 flanked by a `browser_*` at position 1 and a `lang_*`
at position 3. Their co-occurrence neighborhoods are structurally identical, so FastText
assigns them nearly identical embeddings → within-feature collapse.

**Concat corpus (for C4 comparison):**
```python
def device_to_concat(d):
    return "_".join(d[f] for f in FEATURE_ORDER)  # "ios_chrome_utc+0_en_us_wifi_small"
```

One token per event representing the full device fingerprint as a single string. FastText
learns sub-word n-grams from this string (min_n=3, max_n=6), which span feature
boundaries (e.g., `"us_wi"` bridges `lang=en_us` and `network=wifi`). These cross-boundary
n-grams inject noise uncorrelated with any single feature.

**FastText n-gram theory:** FastText represents each token as the sum of its character
n-gram embeddings. For `"ios_chrome_utc+0_en_us_wifi_small"`, the n-grams include
`"ios"`, `"os_"`, `"s_c"`, `"_ch"`, `"chr"`, `"hro"`, `"rom"`, ..., `"0_e"`, `"_en"`,
`"en_"`, `"n_u"`, `"_us"`, `"us_"`, `"s_w"`, `"_wi"`, etc. The n-grams `"_us_"` and
`"us_w"` encode the boundary between `lang` and `network` fields — they will get their
own embedding vectors and contribute to the overall token embedding. For a spoof attack
that changes only tz, the concat token changes from
`"ios_chrome_utc+0_en_us_wifi_small"` to `"ios_chrome_utc+5_en_us_wifi_small"`. The
n-grams in the tz segment change, but they are diluted across the full 6-feature string.
The net embedding change is smaller than the direct tz-embedding change in mean-pool,
where the altered tz contributes 1/6 of the total directly.

### 5.3 FastText Architecture — Why These Hyperparameters

**Skip-gram (`sg=1`) vs CBOW (`sg=0`):** In CBOW, the model predicts a center token
from the bag of its context tokens (averaging context embeddings). In Skip-gram, the
model predicts each context token from the center token — one prediction per
(center, context) pair. Skip-gram is known to produce better representations for rare
or infrequent tokens (Mikolov et al. 2013) because each center token gets a gradient
signal for every context it appears in. For a 30-token closed vocabulary, rarity is
not the main concern, but Skip-gram's per-pair training signal makes it more sensitive
to the specific co-occurrence structure we rely on for the collapse experiment.

**Negative sampling (`negative=10`):** FastText uses noise-contrastive estimation —
for each positive (center, context) pair, 10 "noise" context tokens are sampled from
the vocabulary and trained to be dissimilar. More negative samples = slower but more
accurate gradients. 10 is the standard robust configuration (Mikolov 2013 recommends
5–15 for large corpora). With a 30-token vocabulary, 10 negatives means almost the
entire vocabulary is used in each update — ensuring every token pair gets a training
signal in every gradient step.

**Window size (`window=6`):** The context window defines how many positions left and
right each center word can "see." For a per-account sentence of 360 tokens, window=6
means each event's tokens influence the adjacent event's tokens (each event is 6
tokens, so window=6 reaches exactly to the start of the next event). This gives the
model account-level context while keeping computational cost manageable.

**Vector size (`vector_size=64`):** 64-dimensional embeddings for a 30-token vocabulary
is intentionally over-specified — there are only 30 tokens, so a 30-dimensional space
is the minimum to represent them independently. 64 dimensions gives the model slack to
represent complex co-occurrence relationships without forcing orthogonality. Using a
larger vector (e.g., 300) would not hurt but would slow training on this small vocab.

**`min_n=3, max_n=6`:** Character n-grams of length 3 to 6 are learned. For feature
tokens like `"tz_utc+0"`, the 3-grams include `"tz_"`, `"z_u"`, `"_ut"`, `"utc"`,
`"tc+"`, `"c+0"`. These subword components allow the model to handle OOV tokens in RBA
and H6 (open vocabulary) by decomposing them into known character patterns.

**Robust config:** `sg=1` (skip-gram), per-account corpus, `epochs=20`, `window=6`,
`negative=10`, `vector_size=64`, `min_n=3`, `max_n=6`. This is the production-recommended
configuration for ATO detection.

**Degenerate config:** `sg=0` (CBOW), per-event corpus, `epochs=10`. This reproduces the
original PoC failure mode — lower epochs to match what was originally run.

**2×2 Factorial T8 configs:** All four cells at standardized `epochs=20`, `negative=10`
(only sg and corpus vary):

| | CBOW (sg=0) | Skip-gram (sg=1) |
|-|-------------|-----------------|
| **per-event corpus** | `collapse expected` | `off-diagonal` |
| **per-account corpus** | `off-diagonal` | `no collapse` (= robust) |

The diagonal SG+per-account cell reuses the robust model. The three other cells are trained
fresh. This 2×2 design is the causal proof for C1: if corpus is the cause, both CBOW and
SG should collapse on per-event and recover on per-account, regardless of training
objective. If objective were the cause, CBOW would always collapse and SG would not.
(Factorial standardized to epochs=20, negative=10 to control confounds.)

**T3 — prefixed concat:** `"os:ios|browser:chrome|tz:utc+0|..."` format. Preserves
feature identity in the token string while eliminating cross-boundary n-grams. Window=1
enforces only adjacent-token co-occurrence. Used to test whether n-gram contamination
is the mechanism for concat's spoof AUC deficit.

### 5.5 Scoring (H2)

**The full scoring loop:**
```python
def score_all(accounts, embed_fn, model):
    result = {k: {"scores": [], "labels": []} for k in ["novel","fleet","spoof","neg","known"]}
    for acc in accounts:
        centroid = compute_centroid(acc["train_events"], embed_fn, model)
        for at, (ev, lbl) in acc["test"].items():
            result[at]["scores"].append(cosine_dist(embed_fn(ev, model), centroid))
            result[at]["labels"].append(lbl)
    return result
```

Key details:
- **Centroid uses ALL 60 training events** (not a 40/20 split — that split only exists
  in H6 for rank-norm calibration, which H2 doesn't use)
- **One score per account per attack type** — the result is a list of 400 scores (one
  per account) for each attack type
- **AUC computation:** attacks are pooled against the combined neg+known pool:
  ```python
  def compute_auc(result, attack_type):
      neg_s = result["neg"]["scores"] + result["known"]["scores"]
      neg_l = result["neg"]["labels"] + result["known"]["labels"]
      comb_s = result[attack_type]["scores"] + neg_s
      comb_l = result[attack_type]["labels"] + neg_l
      return roc_auc_score(comb_l, comb_s)
  ```
  This gives 400 positive + 800 negative examples (2× negatives: one neg + one known
  per account). The trivial scorer also uses this same pool.

**Cosine distance theory:** Cosine distance = 1 − cos(θ) where cos(θ) = (a·b)/(|a||b|).
It ranges from 0 (identical direction) to 2 (opposite direction). For unit-normalized
vectors, it equals half the squared Euclidean distance. We use cosine rather than
Euclidean because FastText vectors are not constrained in magnitude — two vectors could
have similar directions but different magnitudes, and cosine is invariant to magnitude.
An account centroid is the arithmetic mean of 60 training event embeddings; it is not
further normalized. Cosine distance to this centroid therefore captures directional
deviation from the account's "average device fingerprint."

Higher score = more anomalous = higher suspicion. Labels: attack events=1, benign=0.
AUC measures how well the cosine distance ranks attacks above benign events.

**Mean-pool embedding:**
```python
def embed_mp(event, model):
    return np.mean([model.wv[make_token(f, event[f])] for f in FEATURE_ORDER], axis=0)
```
6 feature token embeddings are averaged → one 64-dim vector per event. (Note: 6 features
in H2, not 7 — H6/RBA add `rtt_bucket`. Each `model.wv[token]` call retrieves the
FastText word vector, which is the sum of the token's character n-gram embeddings.)

**Concat embedding:**
```python
def embed_cat(event, model):
    return model.wv[device_to_concat(event)]
```
One embedding lookup for the full string token.

**Trivial baseline:** Score = 0 if `device_key(event) in acc["known_devices"]` else 1.
Pure set-membership: any previously seen (OS, browser, tz, lang, network, screen) tuple
scores zero (not anomalous). Anything new scores one. No model needed.

**AUC computation:** For each attack type, negative pool = `neg` events (slightly different
network/screen) + `known` events (primary device itself). The AUC asks: among attacks of
this type mixed with these negatives, does cosine distance rank attacks higher?

**ROC-AUC interpretation:** ROC-AUC is the probability that a randomly chosen positive
(attack) event scores higher than a randomly chosen negative (benign) event. It equals
the area under the curve of TPR vs FPR as the threshold sweeps from −∞ to +∞.
AUC=0.5 means random performance; AUC=1.0 means perfect separation.

### 5.6 Bootstrap CI Mechanics (H2)

Two distinct bootstrap functions serve different purposes:

**Independent per-model CI (`bootstrap_auc`):**
```python
def bootstrap_auc(result, attack_type, n_boot, seed):
    rng = np.random.default_rng(seed)
    comb_s = np.concatenate([att_s, neg_s])
    comb_l = np.concatenate([att_l, neg_l])
    n = len(comb_s)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample WITH replacement
        if len(np.unique(comb_l[idx])) < 2: continue
        aucs.append(roc_auc_score(comb_l[idx], comb_s[idx]))
    return mean(aucs), percentile(aucs, 2.5), percentile(aucs, 97.5)
```

This is standard non-parametric bootstrap: sample N events (with replacement), compute
AUC, repeat N_BOOT=1000 times. The 2.5th and 97.5th percentiles form the 95% CI. Samples
where only one class appears are skipped to avoid `roc_auc_score` errors.

**Paired delta CI (`bootstrap_auc_delta`):**
```python
def bootstrap_auc_delta(mp_result, cat_result, attack_type, n_boot, seed):
    # Build combined score vectors for BOTH models over the SAME examples
    yt  = np.concatenate([att_lbl, neg_lbl])  # shared label vector
    ymp = np.concatenate([att_mp,  neg_mp])   # mean-pool scores
    ycc = np.concatenate([att_cat, neg_cat])  # concat scores
    
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)      # SAME idx applied to both
        deltas.append(roc_auc_score(yt[idx], ymp[idx]) - roc_auc_score(yt[idx], ycc[idx]))
    return point, percentile(deltas, 2.5), percentile(deltas, 97.5)
```

The critical difference: the same resample index `idx` is applied to BOTH models. This
is paired bootstrap — it controls for per-account difficulty. If account 47 is inherently
hard (small cosine distance for all models), paired resampling ensures it appears in the
same ratio for both models in each bootstrap iteration. Unpaired resampling would allow
one model's bootstrap to accidentally oversample easy accounts while the other's
undersamples them, inflating variance.

**Why paired bootstrap is more powerful:** For testing H₀: AUC(mp) = AUC(cat), the
variance of the estimator ΔDelta = AUC_mp − AUC_cat is smaller under paired resampling
than under independent resampling, because the within-pair correlation (both models
scoring the same event) is subtracted out. Narrower CIs = more statistical power to
detect a real difference.

**Silhouette delta CI (`bootstrap_sil_delta`):** Similar paired approach, but applied
to per-device cluster membership. Each known device is a data point, labeled by
account index. The silhouette score measures intra-cluster compactness vs inter-cluster
separation:

```
sil(i) = (b(i) - a(i)) / max(a(i), b(i))
```

where `a(i)` = mean distance from device i to all other devices in the same account,
and `b(i)` = mean distance to devices in the nearest other account. Silhouette ∈ [−1, 1];
higher is better. Only accounts with ≥2 known devices are included (silhouette requires
at least 2 members per cluster). Bootstrap: resample known_device vectors (same indices
for both models), recompute silhouette difference.

### 5.7 Tests T1–T8 in Detail

**T1 — AUC comparison with bootstrap delta CIs:**
The primary pre-specified claim: all four delta CIs (spoof, novel, fleet, silhouette)
exclude zero, where delta = AUC(mean-pool) − AUC(concat w=1).

Bootstrap method: **paired** resampling (see §5.6 for theory). N=1000 resamples.
95th percentile CI (2.5th and 97.5th percentiles of the bootstrap distribution).
CI excludes zero ↔ effect is statistically significant at the paired bootstrap level.

PR-AUC is now computed alongside ROC-AUC via `bootstrap_pr_auc_delta`. The spoof PR-AUC
delta (+0.248 [0.210, 0.283]) is entirely positive on all 5 seeds, consistent with the
ROC-AUC delta. PR-AUC exposes concat_w1 spoof performance as near-trivial (0.542 vs
0.500 baseline), a severity that ROC-AUC (0.737) understates. Both metrics and their
deltas are stored in `results.json` under `pr_auc`, `pr_auc_per_model`, and `pr_auc_deltas`.

The results.json stores both:
- `bootstrap_ci_95.per_model.mean_pool.*` — independent per-model CIs (useful for error bars)
- `bootstrap_ci_95.deltas.*` — delta CIs (the actual T1 statistical claim)
- `pr_auc_deltas.*` — PR-AUC delta CIs (parallel to ROC-AUC deltas)

The distinction matters: per-model non-overlap is a sufficient condition for a significant
delta, but it's a weaker test than the actual delta CI. The paper must cite the delta CIs.
(Decision `6197ad37`.)

**The primary comparison is mean-pool vs concat w=1, NOT vs concat w=6.** The w=6 model
is used in the AUC summary figure for a fair head-to-head comparison (both using window=6),
but w=1 is the adversarial choice for concat — the minimum context window that still
produces reasonable embeddings. If mean-pool beats concat at its most favorable setting
(w=6), that's a strong result, but the pre-specified claim is specifically against w=1.
Cat_w6 is trained in step [3/7]; cat_w1 is trained later in step [5/7] to clearly delineate
the primary comparison.

**T2 — Concat window sweep (w=1, 3, 6):**
Tests whether increasing the concat model's context window recovers the spoof AUC gap.
If the gap is purely a context-window effect, larger windows should close it. At w=6,
the concat model can see all 6 feature values within a single sentence and should have
maximum context.

Results are stored in `t2_window_sweep.concat_w1/w3/w6.spoof_auc`. If the gap persists
at w=6, it's not a window-size artifact — it's the n-gram contamination mechanism (C4).

**T3 — Prefixed-concat silhouette comparison:**
The prefixed format (`os:ios|browser:chrome|...`) prevents n-gram contamination because
`|` is a separator that makes the `{feature}:` prefix part of every n-gram touching the
feature boundary. If the prefixed-concat silhouette is higher than plain-concat, it
confirms that cross-boundary n-grams are degrading cluster separation.

Stored in `t3_prefixed_concat.gap` = mean_pool_silhouette − prefixed_concat_silhouette.
A gap > 0.05 is the threshold for declaring the contamination effect meaningful (plan §4).

**T4 — Tz-counterfactual attribution:**
For each spoof event, compute: (a) actual cosine distance to centroid, (b) counterfactual
cosine distance after restoring tz to the primary device's tz. The difference
`d_actual − d_counterfactual` measures how much the tz change contributed to anomaly
detection. Larger values → model is actually detecting the tz change, not noise.

Stored in `t4_tz_attribution.mean_pool` and `.concat` (mean attribution across accounts).

**T5 — Tz-position permutation:**
Tests whether moving tz to position 0 (first feature in the concat string) recovers the
spoof AUC gap. If cross-boundary n-grams are the mechanism, the spoof AUC should vary
with tz position — best when tz is at one end (fewest boundary crossings) and worst when
tz is in the middle (maximum cross-boundary n-gram contamination from both sides).

**Implementation:** `remaining = [f for f in FEATURE_ORDER if f != "tz"]` gives
`["os", "browser", "lang", "network", "screen"]`. For each of 6 positions:
```python
order = remaining[:tz_pos] + ["tz"] + remaining[tz_pos:]
# tz_pos=0: ["tz", "os", "browser", "lang", "network", "screen"]
# tz_pos=2: ["os", "browser", "tz", "lang", "network", "screen"]  (original)
# tz_pos=5: ["os", "browser", "lang", "network", "screen", "tz"]
```
Each position trains a fresh model with `window=1` and scores the spoof attack.

Results: `t5_tz_permutation.position_1` through `position_6`, each with
`concat_spoof_auc`. Position 1 = tz first; position 6 = tz last. (The 1-based indexing
in the JSON matches the human-readable position; position 1 is `tz_pos=0` in the code.)

**Interpretation key:** If the gap between `best_perm` and `mp_spoof_auc` is large
and consistent across positions, the n-gram contamination is structural (not positional).
If the gap shrinks significantly at the extremes, feature ordering matters and could
theoretically be exploited. The paper reports both the best permutation value and the
mean-pool reference.

If the best permutation still underperforms mean-pool (`best_perm < mp_spoof_auc`),
concat cannot recover even with optimal feature ordering — defense wins for C4.

**T6 — Per-account centroid compactness:**
Mean cosine distance from each training event to the account centroid.
Lower compactness = tighter cluster = more consistent embedding space.

```python
centroid = mean([embed_fn(e) for e in acc["train_events"]])
compactness = mean([cosine_dist(embed_fn(e), centroid) for e in acc["train_events"]])
```

Bootstrap CI (N=1000) on the mean compactness across accounts. If mean-pool has lower
compactness than concat, it forms tighter per-account clusters → more discriminative
embeddings for anomaly detection.

**T7 — Trivial baseline margin:**
Stored in `t7_trivial_baseline`. These are the margins mean-pool and concat hold above
the trivial baseline on spoof AUC. Any model that fails to beat trivial on spoof has no
practical value as an ATO detector — trivial is a deployable zero-compute heuristic.

**T8 — Token similarity analysis:**
Core of C1. For every pair of vocabulary tokens:
- If they share the same feature (`os_X` and `os_Y`): **within-feature pair**
- Otherwise: **cross-feature pair**

Compute cosine similarity for all pairs. Report:
- `within_feature_mean`: mean cosine similarity of within-feature pairs
- `cross_feature_mean`: mean cosine similarity of cross-feature pairs
- `within_cross_ratio`: within/cross (dimensionless)
- `collapse_detected`: `within_feature_mean > 0.9`

**Threshold 0.9 is canonical across all phases** (decision `a2b73375`). At > 0.9,
within-feature tokens are so similar that any test event's tz change barely moves
the mean-pool embedding — the tz dimension is dead.

Under robust config: expected within ≈ 0.424 ± 0.013, cross ≈ 0.344 ± 0.012, ratio ≈ 1.23.
Under degenerate config: expected within ≈ 0.9992 ± 0.00008, cross ≈ −0.170 ± 0.001 (everything collapses).

The 2×2 factorial T8 is stored in `t8_token_similarity.factorial_2x2` with cells
`sg_per_account`, `cbow_per_event`, `cbow_per_account`, `sg_per_event`.

**Downstream consequence of collapse (5-seed diagnostic):**
`scripts/h2/h2_degenerate_downstream.py` (output: `aggregate/h2_degenerate_downstream.json`,
cross-checked per seed against results.json) scores every factorial cell plus the historical
PoC config with mean-pool centroid cosine. Both per-account cells beat the trivial baseline
on spoof (sg 0.868 ± 0.012, cbow 0.933 ± 0.009 ROC-AUC); both per-event cells fall to
trivial-or-worse (sg 0.767 ± 0.017, cbow 0.725 ± 0.011 against trivial 0.750; PR-AUC
0.585/0.528 against 0.500), and the PoC config (cbow + per-event, epochs=10) lands at
0.669 ± 0.013 ROC / 0.459 ± 0.019 PR — below the trivial PR baseline — while novel
(0.87–0.96) and fleet (0.85–0.93) ROC-AUC stay comparatively high under collapse. The
damage is selective, concentrated on spoof, which is what makes it invisible to
easy-subtype monitoring. (Earlier single-seed h2_ml_lab downstream values — spoof 0.384,
novel 0.880, fleet 0.922 — do not reproduce at that magnitude under the rerun protocol
and are superseded by these 5-seed values.)

**Abort condition:** If `collapse_robust = True` (within > 0.9 under robust config),
the script exits with code 2 (`sys.exit(2)`). This is a fatal misconfiguration — something
is wrong with the corpus construction, and proceeding would produce meaningless AUC results.
The T8 check runs in step [6/7], AFTER scoring and delta CI computation (step [5/7]).
This is deliberate: AUC results are already in memory, so the abort triggers only if
T8 is inconsistent with what the robust config should produce. In H6, T8 runs before
scoring ([4/5]) to abort cleanly without any partial output.

**What to do if T8 aborts:** Check whether the corpus was built correctly by printing
`len(mp_corp)` (should be 400 sentences) and `len(mp_corp[0])` (should be ~360 tokens).
If sentences are only 6 tokens long, the per-event corpus was used instead of per-account.

**Enrollment diagnostics (`enrollment_diagnostics`):**
```python
def enrollment_diagnostics(accounts, mp_model, cat_model):
    for acc in accounts:
        c_mp  = compute_centroid(acc["train_events"], embed_mp, mp_model)
        c_cat = compute_centroid(acc["train_events"], embed_cat, cat_model)
        mp_dists.append(cosine_dist(embed_mp(acc["primary"], mp_model), c_mp))
        cat_dists.append(cosine_dist(embed_cat(acc["primary"], cat_model), c_cat))
    return mean(mp_dists), mean(cat_dists)
```

This measures how close the primary device's embedding is to the account centroid.
Unlike compactness (which measures cluster spread across ALL 60 training events), this
specifically asks: "does the primary device embed close to the center of its account's
cluster?" If `mp_enrollment_dist` is low, the model has learned to embed the primary
device at the centroid — ideal for anomaly detection because attacks should deviate from
the centroid. Stored in `enrollment_diagnostics.mean_pool_enrollment_dist`.

### 5.8 H2 Execution Flow

```
[1/7] Generate synthetic data (400 accounts)
[2/7] Build per-account corpora (mp and cat)
[3/7] Train robust models: mp (sg=1, per-account) + cat_w6 (sg=1, per-account, window=6)
[4/7] Train degenerate (CBOW, per-event) + 2×2 factorial (4 models)
[5/7] Score all events, compute per-model bootstrap CIs, train cat_w1, compute delta CIs
[6/7] T4 (tz counterfactual), T6 (compactness), T8 (token similarity robust+degenerate+2×2)
[7/7] T2 (window sweep w=1,3,6), T3 (prefixed-concat), T5 (tz-permutation × 6 positions)
      → write results.json, generate figures (non-smoke mode only)
```

Note: cat_w1 is trained in [5/7] (not [3/7]) because the primary T1 comparison is
mean-pool vs concat w=1. Cat_w6 is trained first for display in the AUC summary figure.
The T2 sweep at w=1 reuses the cat_w1 model already trained; w=3 trains a fresh model;
w=6 reuses cat_model.

### 5.9 H2 Results JSON Schema

Written to `seeds/seed_{N}/h2/results.json`. Key fields:

```json
{
  "seed": 42,
  "timestamp": "ISO8601",
  "config": {"sg": 1, "epochs": 20, "corpus": "per_account", ...},
  "auc": {
    "mean_pool": {"novel": 0.9996, "fleet": 0.995, "spoof": 0.868},
    "concat_w1": {"novel": 0.996,  "fleet": 0.994, "spoof": 0.737},
    "trivial":   {"novel": 0.75,   "fleet": 0.75,  "spoof": 0.75}
  },
  "bootstrap_ci_95": {
    "per_model": {
      "mean_pool": {"novel": {"estimate":..,"lower":..,"upper":..}, "fleet":..,"spoof":..},
      "concat_w1": {...}
    },
    "deltas": {
      "spoof_delta":      {"estimate": +0.130, "lower": +0.111, "upper": +0.150},
      "novel_delta":      {"estimate": +0.004, "lower": ...,    "upper": ...},
      "fleet_delta":      {"estimate": +0.001, "lower": ...,    "upper": ...},
      "silhouette_delta": {"estimate": +0.044, "lower": +0.060, "upper": +0.090}
    }
    // NB: for silhouette_delta the bootstrap distribution sits slightly above the
    // point estimate (valid-cluster resampling bias); the claim is CI > 0, which
    // holds on all 5 seeds. Values shown are 5-seed means of per-seed quantities.
  },
  "silhouette": {"mean_pool": 0.XX, "concat": 0.XX},
  "compactness": {
    "mean_pool": {"mean": 0.XX, "ci_lower": 0.XX, "ci_upper": 0.XX},
    "concat":    {"mean": 0.XX, ...}
  },
  "t2_window_sweep": {
    "concat_w1": {"spoof_auc": 0.XX}, "concat_w3": {...}, "concat_w6": {...},
    "mean_pool_spoof_auc": 0.XX
  },
  "t3_prefixed_concat": {
    "mean_pool_silhouette": 0.XX, "prefixed_concat_silhouette": 0.XX, "gap": 0.XX
  },
  "t4_tz_attribution": {"mean_pool": 0.XX, "concat": 0.XX},
  "t5_tz_permutation": {
    "position_1": {"concat_spoof_auc": 0.XX}, ..., "position_6": {...},
    "mean_pool_spoof_auc": 0.XX
  },
  "t7_trivial_baseline": {
    "mean_pool_spoof_margin": 0.XX, "concat_w1_spoof_margin": 0.XX
  },
  "t8_token_similarity": {
    "robust_config":    {"within_feature_mean": 0.424, "cross_feature_mean": 0.344, ...},
    "degenerate_config": {"within_feature_mean": 0.9992, ..., "collapse_detected": true},
    "factorial_2x2":   {
      "params": "epochs=20, negative=10, window=6 (only sg and corpus vary)",
      "sg_per_account":  {"within_feature_mean": 0.424,  "collapse_detected": false},
      "cbow_per_event":  {"within_feature_mean": 0.982,  "collapse_detected": true},
      "cbow_per_account": {"within_feature_mean": -0.113, "collapse_detected": false},
      "sg_per_event":    {"within_feature_mean": 0.939,  "collapse_detected": true}
    }
  },
  "enrollment_diagnostics": {
    "mean_pool_enrollment_dist": 0.XX,
    "concat_enrollment_dist": 0.XX
  }
}
```

**Note:** `results.json` is only written in non-smoke mode. The smoke run validates the
code path but does not write output (to avoid polluting the seeds directory with
synthetic-data artifacts).

---

## 6. Phase 2: H6 — Architecture Experiment

**Script:** `experiments/rerun/scripts/h6/h6_rerun.py`
**Source:** `experiments/h6_hybrid/`
**Purpose:** Realistic class imbalance, attack taxonomy with k-variant spoofs, fleet
blocklist mechanics. Tests C2 (rank-norm collapse) and C3 (gate blinds fleet). Also C4
via spoof k-gradient.

### 6.0 Theoretical Motivation for H6

H2 established that mean-pool embeddings can detect spoof attacks in a controlled
synthetic setting. H6 asks three harder questions:

1. **Does rank-normalization help or hurt at realistic imbalance?** Account-level
   rank-normalization should, in theory, reduce false positives (high-distance events
   from users who always log in from diverse locations should be downranked). But
   theory also predicts that CDF transforms destroy precision-recall curve performance
   at high imbalance — a well-known result from the class-imbalance literature
   (Davis & Goadrich 2006, Saito & Rehmsmeier 2015). H6 empirically confirms which
   effect dominates at 1:100.

2. **Does a known-device gate help?** The two-stage gate is a common production heuristic.
   Its failure mode on fleet attacks is a logical consequence of its design: any device
   that appears during training is "known" — including deliberately injected fleet devices.
   H6 operationalizes this as a testable hypothesis (C3).

3. **Does spoof detection degrade as more features change?** The k=1/2/3 gradient tests
   whether mean-pool's spoof advantage is specific to minimal-perturbation attacks or
   extends to multi-feature changes. If the advantage collapses at k=2, the production
   value is limited. If it holds at k=3, the model is robust.

### 6.1 RBA Marginals — The Distributional Foundation

H6 uses synthetic data **sampled from real-world marginals** extracted from the DAS Group
RBA dataset (`rba_marginals.json`). This makes the synthetic data more realistic than H2's
fixed-vocabulary toy model.

The marginals encode:
- `os_device_joint`: joint distribution of OS × device type
- `browser_given_os`: conditional distribution of browser given OS
- `country_marginal`: country frequency distribution
- `region_given_country`: region frequency conditional on country
- `asn_given_country`: ASN (network provider) conditional on country
- `rtt_marginal`: round-trip time bucket distribution

These marginals are **deterministic** (fixed; seed-invariant). Only the chain-sampling
that uses them is seeded. This means all five seeds draw from the same distributional
envelope — variance across seeds comes from sampling randomness, not from different
underlying distributions.

**Feature space (H6):** 7 features: `os`, `browser`, `device_type`, `country`, `region`,
`asn_bucket`, `rtt_bucket`. Open vocabulary (number of unique values depends on the
seeded chain-sampling; 216–240 unique tokens across the five rerun seeds). The `rtt_bucket` is included in embeddings but
excluded from `device_key()` — it's treated as a noise feature for fingerprinting
(round-trip time varies across sessions, not characteristic of a device identity).

### 6.2 Account and Attack Generation

```python
accounts, fleet_info = generate_accounts(marginals, n_accounts=400, rng, ...)
```

**Per account:**
- One `home_event` sampled from marginals (the canonical home location+device)
- 60 training events: 70% from home event (same device, fresh RTT), 30% from marginals
  (with 80% probability anchored to home country/region)
- 5 × neg_ratio = **500 enrollment negatives** in the canonical run (`--neg-ratio 100`):
  same distribution as training events (benign test examples). The base count is
  `N_ENROLL_NEG=5`; the expansion `n_enroll_neg = N_ENROLL_NEG * args.neg_ratio` happens
  in `main()` before passing to `generate_accounts()`. This is what creates the 1:100
  imbalance at evaluation time.
- 5 spoof_k1 events (one feature differs)
- 5 spoof_k2 events (two features differ)
- 5 spoof_k3 events (three features differ)
- 5 novel events (foreign country)
- Fleet injection at 25% of accounts (below)

**Spoof attack variants:**
- **k=1 (hardest):** Only the location tuple changes — country forced to differ; region
  and ASN re-sampled conditional on the new country. Same device, same OS/browser. This is
  the canonical ATO spoof: attacker uses victim's device profile from a different location.
- **k=2:** Location tuple + re-sampled OS/device/browser. Two independent feature groups touched.
- **k=3:** Location tuple + re-sampled OS/device/browser + re-sampled RTT bucket. Three groups touched
  (re-sampled fields may occasionally coincide with the original values).

**Novel attack:** Foreign country (different from home country). All other features freely
sampled from the novel country's distributions. This is the "easy" case — strong location
signal.

**Fleet attack injection — two-pass construction:**

Pass 1 generates all 400 accounts with their normal training events, spoof variants, and
novel attacks. Fleet injection happens in Pass 2 as a separate loop:

```python
fleet_device = marginals.sample_event(rng)   # one device for ALL fleet accounts
fleet_key    = device_key(fleet_device)
for acc in accounts:
    if rng.random() < FLEET_FRAC:            # 25% chance
        replace_idx = int(rng.integers(0, N_TRAIN))   # random position in train
        acc["train"][replace_idx] = dict(fleet_device)
        acc["known_devices"].add(fleet_key)   # NOW it's known
        acc["has_fleet"] = True
        acc["fleet_atk"] = [dict(fleet_device) for _ in range(N_SPOOF_PER_ACCT)]
```

**Why two-pass?** The fleet device must be the same across all fleet accounts (it's a
coordinated attacker using one device). If it were chosen per-account, it wouldn't be
a fleet attack — just individual novel devices. Sampling all regular accounts first,
then injecting, ensures the fleet device is independent of any individual account's
feature distribution.

**Why `known_devices.add(fleet_key)` is correct:** After injection, the fleet device
IS in the account's training history. A real system would have seen this login and stored
the device fingerprint in its whitelist. The two-stage gate would suppress alerts on it
in future visits. This is the operational bug C3 documents.

**Temporal blocklist construction:**

```python
# Assign each fleet account a random attack time within the 30-day window
for acc in fleet_accounts:
    acc["fleet_attack_time"] = rng.uniform(0, attack_window=30)

t_fleet_first = min(acc["fleet_attack_time"] for acc in fleet_accounts)
t_blocklist   = t_fleet_first + blocklist_lag=10

for acc in fleet_accounts:
    acc["fleet_blocklist_active"] = (acc["fleet_attack_time"] >= t_blocklist)
```

`t_fleet_first` is the earliest attack across all fleet accounts — the moment the
security team first notices the fleet device and adds it to the blocklist. `t_blocklist`
is 10 days later (the lag before the blocklist update propagates). Accounts with
`fleet_attack_time < t_blocklist` are in the cold-start window: the fleet device is not
yet blocked. Accounts with `fleet_attack_time >= t_blocklist` have the blocklist active.

This creates two populations:
- **Pre-lag (fleet_residual):** Fleet accounts attacked before the blocklist was updated
  (cold-start phase). `blocklist_fires = False` for all of them.
- **Post-lag (fleet_aggregate ∩ post-lag):** Fleet accounts attacked after blocklist
  update. `blocklist_fires = True` for their fleet device.

**The "fleet_aggregate" block in results.json** covers ALL fleet accounts (both pre-lag
and post-lag). "fleet_residual" covers only pre-lag. The C3 claim is about fleet_residual:
in the cold-start window, the two-stage gate provides zero protection.

### 6.3 Spoof Base Event Selection

Spoof attacks in H6 are not generated from the home_event directly. The code uses:

```python
base_event = train[int(rng.integers(0, N_CENTROID))]  # random event from first 40
spoof_k1 = [marginals.sample_spoof_k1(base_event, rng) for _ in range(N_SPOOF_PER_ACCT)]
```

The base event for spoof generation is drawn from the first 40 training events (the
centroid window). This ensures the spoof is constructed around a representative device
event — not a rare marginal or the single home_event — giving the spoof a realistic
starting point. `sample_spoof_k1()` then changes the country while keeping OS/browser/
device from `base_event`:

```python
def sample_spoof_k1(self, benign_event, rng):
    new_country, new_region, new_asn = self.sample_location_tuple(rng, exclude_country=home_country)
    spoof = dict(benign_event)
    spoof["country"] = new_country
    spoof["region"]  = new_region
    spoof["asn_bucket"] = new_asn
    return spoof
```

k=2 additionally changes OS/device/browser; k=3 also changes RTT bucket.

### 6.4 Why `N_CENTROID = 40`

The centroid is computed from **the first 40** training events per account, not all 60:

```python
"centroid_events": train[:N_CENTROID]
```

The remaining 20 events (`train[N_CENTROID:]` = `calib`) form the calibration set for
rank normalization. This split is fundamental to the rank-norm scorer:

```python
baseline = compute_baseline(calib, centroid, model)
# baseline[i] = cosine_dist(calib_event[i], centroid)
rank_norm_score = np.mean(baseline < raw_score)  # CDF rank
```

The rank-norm score is the CDF position of the test event's raw distance within the
calibration distribution. This produces a score in [0,1] that is account-normalized:
accounts with generally higher raw distances (loose clusters) still produce scores on
the same [0,1] scale.

**Why this causes collapse at 1:100 imbalance (C2) — theoretical explanation:**

The CDF rank-norm score is `np.mean(baseline < raw_score)`. With 20 calibration events
per account, the baseline distribution has 20 quantiles. The maximum possible score is
1.0 (raw distance exceeds all calibration distances); the minimum is 0.0. This [0,1]
range is the same for every account, regardless of how anomalous their cluster is.

The problem is compression of the high-score tail. Suppose account A has a tight cluster
(raw distances 0.01–0.05) and account B has a loose cluster (raw distances 0.1–0.5).
Under `mp_raw`, a spoof event in account A scores 0.08 (well above the 0.01–0.05 range)
and in account B it scores 0.6. These are clearly anomalous in absolute terms. Under
rank-norm, both score around 0.95 (exceeding ~95% of baseline). The high-score region
[0.95, 1.0] is now crowded with attacks from ALL accounts, but also crowded with
high-percentile benign events.

At 1:100 imbalance with 5 attacks and 500 benign per account, you need precision ≈ 1%
just to break even (attack prevalence). Rank-normalization's compression means benign
events from accounts with naturally higher baseline variation flood the high-score tail,
destroying precision. ROC-AUC is immune because it only measures rank ordering, which
monotone transforms (like CDF) preserve.

**CANONICAL correction (scoping):** A calibration-window ablation
(`experiments/rerun/calib_sweep/SUMMARY.md`; 5 seeds, calibration window in
{20, 50, 100, 200, 500} events) shows the ~5% floor is a small-sample quantization effect,
not inherent to rank-normalization. The CDF rank is quantized in steps of `1/N_calib`, so at
the deployed 20-event window the contamination equals `1/20` almost exactly (0.0498 vs 0.0500).
Growing the window recovers spoof-k1 PR-AUC from 0.224 to 0.830 by 500 events (the `mp_raw`
embedding control stays flat), leaving only a small residual overlap. The finding is therefore
scoped to thin per-user calibration windows under heavy imbalance — common for new and
low-frequency accounts — rather than a structural property of the transform. The mechanism
above still describes what happens within a thin window; the ablation clarifies that the floor
shrinks as the window grows.

This is precisely the Saito & Rehmsmeier (2015) finding: ROC-AUC is optimistic under
class imbalance because it treats FP and FN symmetrically, while PR-AUC reflects the
true cost of false positives when negatives vastly outnumber positives.

### 6.4b The Five Scorers

All five scorers are defined globally and computed for every attack type (except blocklist
variants, which are fleet-only):

**1. `mp_raw` — Raw mean-pool cosine distance:**
```python
score = cosine_dist(embed_mp(event, model), centroid)
```
No gating. High score = anomalous.

**2. `mp_rank_norm` — Rank-normalized mean-pool:**
```python
baseline = [cosine_dist(embed_mp(e), centroid) for e in calib]
score = np.mean(baseline < raw_score)  # CDF rank ∈ [0, 1]
```
Account-normalized, 0=benign, 1=highly anomalous relative to this account's history.

**3. `trivial` — Set-membership baseline:**
```python
score = 0.0 if device_key(event) in known_devices else 1.0
```
Binary: seen this (OS, browser, device_type, country, region, ASN) tuple before → not
anomalous. Anything new → maximally anomalous. No model needed.

**4. `two_stage` — Gate + mean-pool:**
```python
if device_key(event) in known_devices:
    return 0.0  # Gate fires: suppress alert
return cosine_dist(embed_mp(event), centroid)  # Gate doesn't fire: use model
```
Intended to reduce false positives by suppressing alerts on devices that have appeared in
the training window. The claim (C3) is that this gate backfires on fleet attacks because
fleet devices appear in training by construction.

**5. `two_stage_rank_norm` — Gate + rank-normalized:**
```python
if device_key(event) in known_devices:
    return 0.0
return rank_norm_score(event, centroid, baseline, model)
```
The two-stage gate applied to the rank-norm scorer. Combines C2 and C3 failure modes.

**Blocklist variants (fleet-only):**
These three scorers use a `blocklist_fires` flag that is structurally False for all
spoof and novel events (those attack types don't involve the fleet device):

- `trivial_blocklist`: If blocklist fires → 1.0 (certain anomaly), else → trivial score
- `two_stage_blocklist`: If blocklist fires → 1.0, else → two_stage score
- `combined`: If blocklist fires → 1.0, else → mp_raw score

Blocklist variants are absent from spoof and novel scorer sets because `blocklist_fires`
is always False for those attack types — including them would be actively misleading.
(Decision `b61c5405`.)

### 6.5 The Scoring Engine — `collect_scores_for_attack()`

This is the core computation for all non-fleet attack types. Full walkthrough:

```python
def collect_scores_for_attack(accounts, centroids, model, attack_key):
    scores = {s: ([], []) for s in SCORER_NAMES}   # {scorer: (score_list, label_list)}

    for acc, centroid in zip(accounts, centroids):
        kd       = acc["known_devices"]              # set of device_key tuples
        calib    = acc["train"][N_CENTROID:]         # events 40-59 (20 held-out)
        baseline = compute_baseline(calib, centroid, model)  # 20 cosine distances

        atk_events = acc[attack_key]                 # list of 5 attack events
        pairs = [(e, 0) for e in acc["enroll_neg"]] + [(e, 1) for e in atk_events]
        # = 500 negative events (label=0) + 5 attack events (label=1) = 505 total

        for e, lbl in pairs:
            scores["mp_raw"][0].append(score_single_stage(e, centroid, model))
            scores["mp_rank_norm"][0].append(score_rank_norm(e, centroid, baseline, model))
            scores["trivial"][0].append(score_trivial(e, kd))
            scores["two_stage"][0].append(score_two_stage(e, centroid, kd, model))
            scores["two_stage_rank_norm"][0].append(
                score_rank_norm_two_stage(e, centroid, baseline, kd, model))
            for s in SCORER_NAMES:
                scores[s][1].append(lbl)    # same label for all scorers
```

**Key subtlety:** `centroid` comes from `compute_centroids(accounts, model)` which uses
`acc["centroid_events"]` = `train[:40]`. The `calib` is `acc["train"][N_CENTROID:]` =
`train[40:]`. So:
- Centroid uses events 0-39 (first 40)
- Calibration baseline uses events 40-59 (last 20)
- Test events (attack and negative) use neither — they are held out entirely

**Why split centroid and calib?** If we computed the baseline from the same events used
to build the centroid, we would be measuring how well the centroid represents its own
training data — which is always good by construction. The calib split provides an
out-of-sample baseline: "how anomalous are held-out benign events relative to the centroid
built from other benign events?" This is a more honest calibration.

**The 1:100 negative expansion:** `n_enroll_neg = N_ENROLL_NEG * neg_ratio = 5 × 100 = 500`.
So `acc["enroll_neg"]` has 500 events. With 5 attack events, each account contributes
505 score/label pairs. Across 400 accounts (all of which have spoof/novel events), the
full pool has ~200,000 negative + ~2,000 positive = ~202,000 scored events per attack type.

**`collect_fleet_scores()` vs `collect_fleet_residual_scores()`:** Two separate functions
handle fleet. The aggregate function includes all fleet accounts:

```python
for acc, centroid in zip(accounts, centroids):
    if not acc["has_fleet"]: continue
    blocklist_active = acc.get("fleet_blocklist_active", False)
    for e, lbl in pairs:
        blocklist_fires = is_fleet_device and blocklist_active
        scores["trivial_blocklist"][0].append(1.0 if blocklist_fires else triv)
        scores["two_stage_blocklist"][0].append(1.0 if blocklist_fires else ts)
        scores["combined"][0].append(1.0 if blocklist_fires else mp)
```

The residual function skips post-lag accounts (`fleet_blocklist_active=True`) and
has simplified blocklist logic — since `fleet_blocklist_active=False` for all residual
accounts, `blocklist_fires` is always False:

```python
if not acc["has_fleet"] or acc.get("fleet_blocklist_active", False): continue
# blocklist never fires → trivial_blocklist = trivial, two_stage_blocklist = two_stage
scores["trivial_blocklist"][0].append(triv)   # same as trivial
scores["two_stage_blocklist"][0].append(ts)   # same as two_stage
scores["combined"][0].append(mp)              # same as mp_raw
```

This is why the paper can claim: "in the pre-lag window, no blocklist-assisted strategy
provides any advantage — the fleet device appears identical to a known device."

### 6.6 Top-1% Precision — `top_pct_metrics()` and Tie-Breaking

```python
def top_pct_metrics(scores, labels, pct=0.01, seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    n_flag = max(1, int(len(scores) * pct))
    perturbed = scores + rng.uniform(0, 1e-9, len(scores))   # TIE-BREAKING
    threshold = np.sort(perturbed)[-n_flag]
    flagged = perturbed >= threshold
    tp = int(np.sum(flagged & (labels == 1)))
    ...
```

**The tie-breaking perturbation:** `trivial` assigns exactly 0.0 to known devices and
1.0 to unknown. If there are many unknown events, many events tie at 1.0. The threshold
at the top 1% would then be exactly 1.0, but which events above the threshold are
"flagged" is ambiguous when hundreds of events share the same score. The tiny uniform
noise (max 1e-9) breaks ties randomly but deterministically (seeded). This ensures the
`n_flag` boundary is unambiguous.

**Why `max(1, ...)` for n_flag:** With only 5 positive events per account across ~100
fleet accounts, the pool might be small enough that 1% rounds to 0. `max(1, ...)` ensures
at least one event is flagged.

**What top-1% TP = 0 for two_stage means:** On fleet_residual, `two_stage` returns 0.0
for every fleet event (because `device_key(fleet_event) in known_devices` is always True,
so the gate fires). With tie-breaking, all 0.0 events get scores in [0.0, 1e-9]. The top
1% threshold lands at 0.0 or slightly above — but all fleet attack events also score 0.0.
So the top 1% flagged events are a random sample of all 0.0-scoring events, which are
overwhelmingly benign (500:5 ratio). TP = 0 is essentially guaranteed.

### 6.7 Enrollment Negatives and the 1:100 Imbalance

The canonical evaluation uses `--neg-ratio 100` (decision `dd39e09c`).

```python
n_enroll_neg = N_ENROLL_NEG * args.neg_ratio  # 5 × 100 = 500 negatives per account
```

For each evaluation, pairs are:
```python
pairs = [(e, 0) for e in acc["enroll_neg"]] + [(e, 1) for e in atk_events]
# 500 negatives + 5 positives = 505 events per account
```

At 400 accounts with ~25% fleet participation, this produces roughly:
- ~200,000 total enrollment negatives
- ~2,000 attack events

1:100 ratio matches realistic ATO prevalence — approximately 1% of login events are
from actual attackers in a live system. PR-AUC at this imbalance reflects operational
performance; ROC-AUC does not.

### 6.8 T8 in H6

H6's T8 uses an open vocabulary (no fixed feature values), so it uses prefix matching:

```python
def _feature_for_token(token: str) -> str | None:
    for f in FEATURE_ORDER:
        if token.startswith(f + "_"):
            return f
    return None
```

This relies on the `make_token()` convention: `f"{feature}_{value}"`. Any vocabulary
token starting with `"country_"` is a country token; `"os_"` is an OS token, etc.

**Abort condition:** Same threshold as H2: `within_feature_mean > 0.9` under robust
config → `sys.exit()` with a descriptive error. T8 runs **before** any scoring (step 4/5)
so abort is clean — no partial results are written.

### 6.9 Scorer Sets by Attack Type

The full scorer sets written to results.json:

| Block | Scorers |
|-------|---------|
| `spoof_k1` | mp_raw, two_stage, mp_rank_norm, two_stage_rank_norm, trivial |
| `spoof_k2` | mp_raw, two_stage, mp_rank_norm, two_stage_rank_norm, trivial |
| `spoof_k3` | mp_raw, two_stage, mp_rank_norm, two_stage_rank_norm, trivial |
| `novel` | mp_raw, two_stage, mp_rank_norm, two_stage_rank_norm, trivial |
| `fleet_aggregate` | mp_raw, two_stage, trivial_blocklist, trivial + `two_stage_vs_trivial_roc_delta` |
| `fleet_residual` | all 8 (5 standard + trivial_blocklist + two_stage_blocklist + combined) + delta + top1% |

Decision `da7d1d80` (resolution for `be453b78`): `two_stage_rank_norm` is included in
spoof_k1/k2/k3 and novel. Decision `b61c5405`: blocklist variants excluded from spoof/novel
by logical necessity.

### 6.10 `two_stage_vs_trivial_roc_delta` — The C3 Mechanistic Field

```python
"two_stage_vs_trivial_roc_delta": abs(two_stage_roc - trivial_roc)
```

This field is written into both `fleet_aggregate` and `fleet_residual`. It captures the
mechanistic claim of C3: the two-stage gate should improve on trivial (flag everything
foreign) by using the model for novel events. If the delta is essentially zero, the gate
is not adding any model-based discrimination — it's degenerate to the trivial scorer.

Expected value: < 1e-4 on both fleet blocks (decision `363331a0`). This is verified by
consistency check S6.

**Why is two_stage ≈ trivial on fleet?** The fleet device was injected into training —
it appears in `known_devices` for every fleet account. So `device_key(fleet_event) in
known_devices` is always True → `two_stage_score = 0.0` always. The trivial scorer also
returns 0.0 for any known device. Result: both scorers assign 0.0 (no anomaly) to fleet
events, so their score distributions are identical → ROC-AUC is identical.

### 6.11 Three Verdict Fields — Exact Criteria

```python
def compute_verdicts(raw_results):
    k1 = raw_results.get("spoof_k1", {})

    # Verdict 1: primary_criterion_confirmed
    # raw_results[atk][scorer] = (roc, roc_lo, roc_hi, pr, pr_lo, pr_hi, top1_dict)
    mp_pr,  mp_pr_lo,  _  = k1["mp_raw"][3],  k1["mp_raw"][4],  k1["mp_raw"][5]
    triv_pr, _, triv_pr_hi = k1["trivial"][3], k1["trivial"][4], k1["trivial"][5]
    primary_confirmed = bool(mp_pr_lo > triv_pr_hi and mp_pr > triv_pr)

    # Verdict 2: rank_norm_collapse_confirmed
    mp_raw_pr  = k1["mp_raw"][3]
    mp_rank_pr = k1["mp_rank_norm"][3]
    rank_norm_collapse = bool(mp_raw_pr > 0.0 and mp_rank_pr < 0.5 * mp_raw_pr)

    # Verdict 3: gate_blinds_fleet_confirmed
    fr = raw_results.get("fleet_residual", {})
    ts_tp = fr["two_stage"][6]["tp"]    # index 6 = top1_dict from bootstrap_auc return
    gate_blinds = bool(ts_tp == 0)
```

**`primary_criterion_confirmed`:**
Uses the **PR-AUC CI lower bound** of `mp_raw` vs the **PR-AUC CI upper bound** of
`trivial` on `spoof_k1`. This non-overlapping CI criterion is stricter than simply
`mp_pr > triv_pr` — it requires that even the pessimistic estimate for `mp_raw` exceeds
the optimistic estimate for `trivial`. Decision `65209b9b`: PR CIs are used (not ROC)
because ROC hides the rank-norm collapse at 1:100 imbalance.

**`rank_norm_collapse_confirmed`:**
`mp_rank_norm_pr < 0.5 × mp_raw_pr` on `spoof_k1`. The 50% relative threshold is
deliberately lenient — if rank-norm is even roughly competitive with raw, the collapse
claim fails. At 1:100 imbalance, the expected collapse is from ~0.8 PR-AUC (raw) to
~0.2 (rank-norm), a 75% relative drop. The 50% threshold should be easy to satisfy if
C2 is real.

**`gate_blinds_fleet_confirmed`:**
`two_stage top-1% TP = 0` on `fleet_residual`. The `[6]` index in `fr["two_stage"][6]`
is the `top1_dict` return from `top_pct_metrics()`, which is stored as element 6 in the
raw tuple `(roc, roc_lo, roc_hi, pr, pr_lo, pr_hi, top1_dict)`. Zero true positives at
1% flag rate is the operational claim of C3: a deployed system using two_stage would
miss every fleet attack in the pre-lag window.

**`scores.npz` format:** Arrays are stored as `{atk}__{scorer}__s` (scores) and
`{atk}__{scorer}__l` (labels). Example: `spoof_k1__mp_raw__s`, `spoof_k1__mp_raw__l`.
Access via `np.load("scores.npz")["spoof_k1__mp_raw__s"]`. This file is the input to
`h6_figures.py` for generating PR curves, score distribution histograms, and top-k
precision plots.

### 6.12 H6 Execution Flow

```
[1/5] Load marginals (deterministic, seed-invariant)
[2/5] Generate 400 accounts + fleet injection + temporal blocklist assignment
[3/5] Build per-account corpus, train FastText (sg=1)
[4/5] T8 token similarity → ABORT if collapse_detected
[5/5] Score all 6 attack types × all applicable scorers
      → compute bootstrap CIs (N=1000) for each scorer on each attack type
      → compute top-1% precision metrics for fleet_residual
      → compute verdicts
      → write results.json + scores.npz
```

### 6.13 H6 Output Files

- `results.json`: Full Section 4-conformant schema (described in §6.7 above)
- `scores.npz`: Raw score arrays for every attack type × scorer (used by h6_figures.py
  to generate PR curves, score distribution histograms, etc.)

---

## 7. Phase 3: RBA — Distributional Realism Check

**Script:** `experiments/rerun/scripts/rba/rba_rerun.py`
**Source:** `experiments/h2_rba/`
**Purpose:** Confirm that the H2 embedding behavior holds on the public RBA dataset —
login data synthesized from real-world behavior, with real-world feature distributions
and device fingerprint diversity. Tests whether mean-pool > trivial generalizes beyond
the fully synthetic setting. Also supports C4 via the compactness comparison.

### 7.1 The RBA Dataset and `data_prep.py`

Requires `data/rba/rba.parquet` (run `data_prep.py` first). The script fails fast with
a clear error if the file is missing — no silent failures. The parquet check also
validates that `login_timestamp` is numeric (Int64/Float64) — `data_prep.py` converts
timestamps to Unix epoch integers:

```python
if raw["login_timestamp"].dtype not in (pl.Int64, pl.Float64, pl.Int32):
    raise RuntimeError("login_timestamp dtype is ... expected numeric. Delete and re-run data_prep.py")
```

**`data_prep.py` responsibility:** Reads the raw RBA CSV/parquet, normalizes feature
column names to match `FEATURE_ORDER`, converts timestamps to numeric, and filters to
clean rows. It also extracts the `rba_marginals.json` used by H6 — the marginals are
the empirical frequency distributions of each feature value computed from the full
RBA dataset before any train/test split. This is why the marginals are seed-invariant:
they're computed once from the full data, not from any particular split.

**Open vocabulary:** Unlike H2's fixed 30 tokens, the RBA dataset has thousands of
unique feature values (OS strings, country codes, ASN numbers). The corpus for RBA
might have 5,000+ unique tokens. FastText's sub-word n-gram model handles this via
character n-grams — even a token like `"os_WindowsNT10"` can be embedded by combining
the n-grams of its constituent characters, even if it never appeared in training.
This is the key advantage of FastText over Word2Vec for open-vocabulary domains.

### 7.2 `load_users()` — The Polars Pipeline

The data loading uses Polars for performance:

```python
raw = pl.read_parquet(parquet_path)
df = (raw
    .with_columns(pl.col("login_timestamp").cast(pl.Int64, strict=False))
    .drop_nulls(subset=["user_id", "login_timestamp"])
    .sort("login_timestamp"))                # temporal sort for split

cutoff  = df["login_timestamp"].quantile(split_pct, interpolation="nearest")
train_df = df.filter(pl.col("login_timestamp") <= cutoff)
test_df  = df.filter(pl.col("login_timestamp") >  cutoff)
```

**User construction from training records:**
```python
train_rows = (train_df
    .filter(~pl.col("is_ato"))               # exclude ATO events from training
    .with_columns([pl.col(c).cast(pl.Utf8).fill_null("missing") for c in feat_cols])
    .select(["user_id"] + feat_cols)
    .to_dicts())
train_by_user = defaultdict(list)
for row in train_rows:
    uid = row.pop("user_id")
    train_by_user[uid].append(row)

users[uid] = {
    "train": events,
    "known_devices": {device_key(e) for e in events},  # set of tuples
    "test_pos": [],   # filled later from test_df
    "test_neg": [],
}
```

Note that `known_devices` in RBA is a **set of tuples** (same as H6), not a list of
dicts (unlike H2). Users with < 5 training events are dropped. The 50,000 benign user
cap (`N_BENIGN_USERS`) keeps memory and runtime manageable without sacrificing the
ATO users (who are always retained regardless of count).

**Smoke mode differences:** In smoke mode, a random subset of 5,000 users is drawn
BEFORE the temporal split. This means smoke AUC may be NaN (if no ATO users appear
in the subset) — this is expected and documented. Smoke validates the code path, not
the numerical results.

### 7.3 Temporal Split — Theory and Implementation

**Why temporal, not random split?** ATO detection is an online problem — the system
trains on historical events and must detect future attacks. A random split would allow
test events to come from earlier periods than training events, violating the causal
direction. A random split would also overestimate performance by allowing the model to
"learn" patterns from events temporally after the test events.

**Primary split: 50/50** (the first 50% of timestamps train, the last 50% evaluate).
```python
cutoff = df["login_timestamp"].quantile(split_pct, interpolation="nearest")
```
Using `quantile(0.50, interpolation="nearest")` finds the actual timestamp value at
the 50th percentile and uses it as the exact boundary. "Nearest" interpolation means
the cutoff is always an actual timestamp in the data, not an interpolated value.

**Why 50/50?** The original script used 80/20, which placed every ATO event in the
training window. Shifting to 50/50 preserves ~34 ATO events in the raw test window while
giving each user a meaningful training history. (Decision `81ed922b`; baseline warning
documented because on-disk `rba_metrics.json` reflects a 40/60 split, not 50/50.)

**ATO event distribution:** The RBA dataset has 141 labeled ATO events, all occurring
before the 70th percentile timestamp. Raw window counts shrink to *evaluated* counts
after user-eligibility filtering (a user needs ≥5 benign training-window events and at
least one test event): 12 evaluated ATO events at the 40th-percentile split, 9 at the
primary 50/50, and 3 at the 60th-percentile split — an earlier cutoff leaves more ATO
events in the test window, a later cutoff fewer (but with longer training histories).
The sensitivity splits test whether this variation in ATO test event count affects the
replication verdict (it does not: `h2_replicated` is True on all splits, all seeds).

### 7.4 User Filtering

```python
MIN_TRAIN_EVENTS = 5
```

Users with fewer than 5 training events are dropped — they don't have enough history to
form a meaningful centroid. This threshold is lower than H2's 60 events because RBA users
have sparser login histories (some ATO users have very few events).

In default mode (not `--full`), a cap of 50,000 benign users is applied to keep runtime
manageable. ATO users are always retained.

### 7.5 Model Training and `_run_split()`

Two models are trained (once, on the primary 50/50 training split):
- `mp_model`: FastText on per-account corpus (concat of all events' tokens per user)
- `cat_model`: FastText on per-account concat corpus

Both use `ROBUST_KWARGS_BASE` merged with `{"seed": seed}`. The models are trained only
once and reused across all three temporal splits.

**`_run_split()` function — the scoring core:**
```python
def _run_split(users, mp_model, cat_model, split_pct, seed, n_bootstrap):
    mp_scores,   mp_labels   = score_all(users, embed_mp,  mp_model)
    cat_scores,  cat_labels  = score_all(users, embed_cat, cat_model)
    triv_scores, triv_labels = score_trivial(users)

    mp_roc,  mp_roc_lo, ..., mp_pr,  ... = bootstrap_metrics(mp_scores,   mp_labels, ...)
    cat_roc, ...,            cat_pr, ... = bootstrap_metrics(cat_scores,  cat_labels, ...)
    triv_roc, ...            triv_pr, .. = bootstrap_metrics(triv_scores, triv_labels, ...)

    h2_replicated = bool(mp_roc_lo > triv_roc)
```

`score_all()` in RBA computes centroid from ALL training events (no calib split —
RBA doesn't use rank-norm). `score_trivial()` uses `u["known_devices"]` built from
training events. The function asserts `mp_labels == triv_labels` to guard against
misalignment.

This same function is called three times — for the primary split users, then for
users filtered to the 40/60 test window, then 60/40. The same trained model handles
all three calls.

### 7.6 Three Splits Per Seed

```
Primary (50/50): results.json           ← canonical result
Sensitivity (40/60): results_split40.json
Sensitivity (60/40): results_split60.json
```

**Crucially: the same trained model is used for all three splits.** Only the test window
changes — the training window for model fitting is always the 50/50 training set. This
is correct: we're testing the sensitivity of the evaluation metric to the choice of split
point, holding the model fixed. If the 40/60 and 60/40 splits produce similar conclusions,
the 50/50 finding is robust to reasonable variation in split choice.

Why not re-train for each split? Re-training would confound model randomness with
evaluation randomness. Using the fixed model isolates the effect of split point on
the test-set metric.

### 7.7 `h2_replicated` — The CI-Lower-Bound Criterion

```python
h2_replicated = bool(mp_roc_lo > triv_roc)
# mp_roc_lo  = 2.5th percentile of bootstrap ROC-AUC for mean-pool
# triv_roc   = mean of bootstrap ROC-AUC for trivial (point estimate)
```

where `mp_roc_lo` is the 2.5th percentile of the mean-pool ROC-AUC bootstrap distribution
and `triv_roc` is the point estimate (bootstrap mean) of the trivial ROC-AUC.

**Bootstrap in RBA (`bootstrap_metrics`) — unpaired:**
```python
for _ in range(n_boot):
    idx = rng.integers(0, n, size=n)     # standard resampling (unpaired)
    if len(np.unique(lb[idx])) < 2: continue
    roc_aucs.append(roc_auc_score(lb[idx], sc[idx]))
    pr_aucs.append(average_precision_score(lb[idx], sc[idx]))
roc_lo, roc_hi = percentile(roc_aucs, [2.5, 97.5])
```

Unlike H2, RBA doesn't use paired bootstrap (no delta CI). The standard unpaired
bootstrap is sufficient for per-model CIs in RBA.

**Statistical power concern:** With only 9 evaluated ATO test events (positive class) at
50/50, each bootstrap resample has roughly 9 ± 3 positive events. AUC estimates on small
positive populations are noisy. This is why the CI criterion is essential: `mp_roc > triv_roc`
would be met trivially on noisy data; `mp_roc_lo > triv_roc` requires the advantage to
be robust enough to hold even in unlucky bootstrap samples.

**Why not simple point-estimate comparison?** `mp_roc > triv_roc` is almost always true
just by chance — mean-pool will nearly always score higher than trivial on at least some
test events. The CI-lower-bound criterion asks: is the mean-pool advantage large enough
that even the lower bound of the confidence interval exceeds the trivial baseline? This
is a stronger replication criterion that rules out lucky-seed artifacts. (Decision `9511d90f`.)

### 7.8 T6 and T8 in RBA

**T6 compactness:** Computed on a sample of up to 2000 users (seeded subsample to avoid
memory issues on large datasets). Same metric as H2: mean cosine distance from each
training event to the account centroid, averaged across users.

**T8 token similarity:** Open-vocabulary version using prefix matching (same approach as
H6). Large vocabulary → sampling up to 100,000 random pairs to keep runtime bounded.
The threshold is still 0.9 (same canonical threshold across all phases).

### 7.9 RBA Results JSON Schema

Three files are written per seed:

```json
{
  "seed": 42,
  "timestamp": "ISO8601",
  "split_percentile": 50,
  "n_ato_test_events": 9,
  "auc": {
    "mean_pool": {"roc_auc": 0.852, "pr_auc": 0.XX, "ci_lower": 0.XX, "ci_upper": 0.XX},
    "concat":    {"roc_auc": 0.XX,  "pr_auc": 0.XX, "ci_lower": 0.XX, "ci_upper": 0.XX},
    "trivial":   {"roc_auc": 0.XX,  "pr_auc": 0.XX}
  },
  "t6_compactness": {"mean_pool": 0.XX, "concat": 0.XX},
  "t8_token_similarity": {
    "within_feature_mean": 0.XX, "cross_feature_mean": 0.XX,
    "within_cross_ratio": 0.XX,  "collapse_detected": false
  },
  "h2_replicated": true
}
```

The `_mp_roc_lo` and `_triv_roc` private fields are used internally for the abort check
but are stripped from the final output by `_make_results()`.

---

## 8. Quality Gates and Consistency Checks

### 8.1 Per-Seed Abort Conditions

These cause a non-zero exit immediately, before writing results.json:

| Phase | Condition | Exit code |
|-------|-----------|-----------|
| H2 | T8 within_feature_mean > 0.9 under robust config | 2 |
| H6 | T8 within_feature_mean > 0.9 under robust config | sys.exit() |
| RBA | T8 within_feature_mean > 0.9 under robust config | SystemExit(1) |
| RBA | rba.parquet not found | SystemExit(1) |

If a seed aborts, the rerun plan requires investigating before continuing to the next seed.

### 8.2 H6 Consistency Checker — Operational Guide

**Invocation:**
```bash
# After any seed completes — checks all available seeds
uv run experiments/rerun/scripts/h6/check_consistency.py

# Smoke mode — runs against synthetic data, validates checker logic
uv run experiments/rerun/scripts/h6/check_consistency.py --smoke

# Custom seeds root (for debugging a different output directory)
uv run experiments/rerun/scripts/h6/check_consistency.py --seeds-root /path/to/seeds
```

**Exit codes:** 0 = all checks pass (or no data yet), 1 = any check fails. When fewer
than 5 seeds are available, the checker prints a WARNING but does not fail — cross-seed
checks are run with whatever seeds are available, and results labeled accordingly.

**Output format per check:**
```
  S1  [PASS] seed 42: schema OK
  S2  [PASS] seed 42: verdicts schema OK
  S6  [FAIL] seed 42: fleet_agg_delta=0.0003 fleet_res_delta=0.0001 (threshold=0.0001)
```

**Understanding X3 (the k-gradient check):** X3 runs **per-seed** (inside `check_seed()`,
not `check_cross_seed()`). Despite the "X" prefix, it does not require ≥2 seeds and runs
against each seed independently. The check is:

```python
mono = len(ks_typed) == 3 and ks_typed[0] < min(ks_typed[1], ks_typed[2])
# i.e.: k1_pr < k2_pr  AND  k1_pr < k3_pr
# The relative ordering of k2 vs k3 is NOT checked.
```

The claim is only that k=1 is strictly harder than both k=2 and k=3 — not that k2 ≤ k3.
This is documented in the checker header: "k=1 is hardest per seed: k1 < min(k2, k3)
(mp_raw pr_auc; k2 vs k3 not claimed)." If you see a k2 > k3 inversion in your results,
X3 will still pass as long as k1 < k2 and k1 < k3.

`experiments/rerun/scripts/h6/check_consistency.py` — run after each seed or after all
seeds complete.

**Per-seed checks (S1–S8):**

| Check | What it verifies |
|-------|-----------------|
| S1 | Required top-level keys present in results.json |
| S2 | All 3 verdict keys present (`primary_criterion_confirmed`, `rank_norm_collapse_confirmed`, `gate_blinds_fleet_confirmed`) |
| S3 | `two_stage_vs_trivial_roc_delta` field present in both fleet blocks |
| S4 | No NaN in `mp_raw` ROC-AUC or PR-AUC for primary attack types |
| S5 | `t8.collapse_detected == False` (robust config must not collapse) |
| S6 | `fleet_aggregate.two_stage_vs_trivial_roc_delta < 1e-4` AND `fleet_residual.two_stage_vs_trivial_roc_delta < 1e-4` |
| S7 | `spoof_k1.mp_raw.pr_auc > spoof_k1.trivial.pr_auc` (baseline sanity: model beats trivial) |
| S8 | `spoof_k1.mp_raw.pr_auc > spoof_k1.mp_rank_norm.pr_auc` (rank-norm collapse visible) |

**Per-seed check (runs even with a single seed):**

| Check | What it verifies |
|-------|-----------------|
| X3 | k=1 is hardest: `k1_pr < min(k2_pr, k3_pr)` (k2 vs k3 order not claimed) |

**Cross-seed checks (X1–X2, require ≥2 seeds):**

| Check | What it verifies |
|-------|-----------------|
| X1 | All 3 verdicts agree across every seed (stability) |
| X2 | Standard deviation of `spoof_k1.mp_raw.pr_auc` across seeds < 0.05 |

Check S6 is the mechanistic verification of C3: if `two_stage` and `trivial` have
identical ROC-AUC on fleet (delta < 1e-4), it confirms the gate fires on all fleet
devices (because they're all "known") and produces exactly the same scores as trivial.

### 8.3 Cross-Phase Consistency (Manual / aggregate.py)

After all seeds and phases complete:
1. All three phases used the same library versions (same `# /// script` deps)
2. Vocabulary size is stable across seeds for H2 (fixed 30-token vocab)
3. H6 vocabulary varies by seed (open vocab, but should be stable within ±5%)
4. No verdict flips — any verdict that changes across seeds requires investigation
   before submission

**Verdict stability rule:**
- 5/5 agreement: report as stated
- 4/5: report with "one seed anomaly" note and investigate
- ≤3/5: majority flip; reframe the claim
- Any flip: investigate root cause before submitting

---

## 9. Aggregation and Paper Reporting

### 9.1 Aggregation Protocol

After all 5 seeds complete, `aggregate.py` collects
all `results.json` files and computes:

```
M_mean = mean(M_s1, ..., M_s5)
M_std  = std(M_s1, ..., M_s5)
M_min  = min(...)
M_max  = max(...)
```

Report in paper as **M_mean ± M_std** (range: M_min – M_max).

**CI aggregation:** Report mean of lower bounds and mean of upper bounds across seeds.
For delta CIs, also report "Seeds with CI excluding zero: N/5" — this is the key
stability check for T1.

### 9.2 Paper Reporting Format

**Primary results table (actual 5-seed aggregates):**

| Model | Novel AUC | Fleet AUC | Spoof AUC |
|-------|-----------|-----------|-----------|
| mean-pool | 0.9996 ± 0.0003 | 0.995 ± 0.002 | 0.868 ± 0.012 |
| concat w=1 | 0.996 ± 0.002 | 0.994 ± 0.003 | 0.737 ± 0.006 |
| trivial | 0.750 | 0.750 | 0.750 |

**Bootstrap delta CI table (actual 5-seed aggregates):**

| Delta | Estimate | 95% CI (mean bounds) | Seeds excluding zero |
|-------|----------|----------------------|---------------------|
| Spoof AUC (mp − cat) | +0.130 ± 0.009 | [+0.111, +0.150] | 5/5 |
| Novel AUC (mp − cat) | +0.004 ± 0.002 | [+0.002, +0.006] | 5/5 |
| Fleet AUC (mp − cat) | +0.001 ± 0.003 | [−0.002, +0.004] | 1/5 |
| Silhouette (mp − cat)| +0.044 ± 0.008 | [+0.060, +0.090]† | 5/5 |

†The silhouette-delta bootstrap distribution sits slightly above the point estimate
(valid-cluster resampling bias); the claim is CI > 0 on all seeds.

(The fleet delta CI crosses zero — fleet is an easy attack that both models handle
nearly perfectly, so the comparison is noise-limited by design.)

### 9.3 Seed Sensitivity Statement for Paper (Limitations Section)

```
"All experiments were run across 5 random seeds (42, 123, 456, 789, 2024)
governing synthetic data generation, FastText training, and bootstrap resampling.
Primary metric standard deviations across seeds were [H2 spoof AUC std],
[H6 k=1 PR-AUC std], and [RBA ROC-AUC std] respectively. All pre-specified
verdicts were stable across all 5 seeds [or: with the following exceptions: ...]."
```

Fill in after aggregation. Observed standard deviations are small (H2 spoof AUC ±0.012,
H6 k=1 PR-AUC ±0.026, RBA ROC-AUC ±0.029) given the large sample sizes (400 accounts ×
60 events each); the small-population fleet_residual metrics are the high-variance
exception (top-1% precision ±0.116).

---

## 10. Reading the Outputs

### 10.1 H2 Key Figures

| Figure | What it shows | C-number |
|--------|---------------|----------|
| `h2_summary_auc.png` | AUC by attack type: mean-pool vs concat_w6 vs trivial | C4 |
| `h2_t4_tz_counterfactual.png` | Distribution of tz-attributable anomaly distance | C4 |
| `h2_t6_compactness.png` | Per-account centroid compactness: mean-pool vs concat | C4 |
| `h2_t8_token_similarity.png` | 2×2 factorial T8: within/cross-feature similarity distributions | C1 |
| `h2_c1_cooccurrence.png` | Within-feature JSD per-event vs. per-account corpus — C1 mechanism diagnostic | C1 |
| `h2_t2_window_sweep.png` | Spoof AUC and silhouette across concat window sizes | C4 |
| `h2_t3_prefixed_concat.png` | Spoof AUC and silhouette: mean-pool vs prefixed vs plain concat | C4 |
| `h2_t5_tz_permutation.png` | Spoof AUC as tz moves through 6 positions in concat string | C4 |

**Reading the T8 figure:** The 2×2 grid shows histograms of pairwise cosine similarities.
Within-feature pairs (same feature, different values) should be **low** for a good
embedding — tokens like `os_ios` and `os_android` should be similar within the OS
dimension but not identical. Cross-feature pairs can be anywhere. Under collapse, within
≈ 0.9992: the histogram for within-feature pairs is a spike near 1.0, meaning all OS
tokens look essentially the same to the model.

**Reading the T5 figure:** The x-axis shows tz position (0 = first feature in the concat
string). If the spoof AUC is flat across positions (all close to the w=1 baseline), moving
tz doesn't help → n-gram contamination is not position-specific but structural. The
horizontal line for mean-pool shows the gap that concat cannot close regardless of position.

### 10.2 H6 Key Figures (from scores.npz via h6_figures.py)

| Figure | What it shows | C-number |
|--------|---------------|----------|
| PR curves: mp_raw vs mp_rank_norm | Collapse in precision-recall space at 1:100 imbalance | C2 |
| Score distribution: raw vs rank-norm | CDF compression visible as histogram width narrowing | C2 |
| Top-k precision: fleet residual (1%–10%) | Zero TP at all thresholds for two_stage | C3 |
| Population decomposition diagram | Pre-lag vs post-lag fleet population structure | C3 |
| Spoof k-gradient: PR-AUC at k=1/2/3 | Detection gets easier as k increases (k=1 hardest: 0.888 vs 0.951 at k≥2) | C4 |
| `h6_c2_score_margin.png` | Contamination rate at p10(attack) and robust margin before/after rank-normalization — C2 mechanism diagnostic | C2 |

**Reading the PR curve comparison (C2):** The mp_raw curve should show high precision at
moderate recall (PR-AUC 0.888). The mp_rank_norm curve collapses toward the trivial
baseline (PR-AUC 0.224 vs trivial 0.108) — far below mp_raw, though still above raw class
prevalence (1/101 ≈ 0.01). The ROC curves for the same pair of scorers look nearly
identical — this is the diagnostic that ROC is misleading at high imbalance.

**Reading the fleet residual top-k (C3):** At every flag rate from 1% to 10%, the
two_stage line should show TP=0 while mp_raw shows many TPs. This is the operational
consequence: a system using two_stage would miss every fleet attack in the pre-lag window.

### 10.3 RBA Key Figures

| Figure | What it shows |
|--------|---------------|
| `rba_summary_auc.png` | ROC-AUC and PR-AUC: mean-pool vs concat vs trivial (primary 50/50 split) |
| `rba_pr_curve.png` | PR curve at the evaluated ATO prevalence (9 evaluated ATO events at 50/50) |
| `rba_t6_compactness.png` | Per-user centroid compactness on the RBA data |
| `rba_t8_token_similarity.png` | T8 for both mean-pool and concat models on RBA vocabulary |

**Reading `h2_replicated`:** If `h2_replicated = true` in all 5 seeds' primary results,
the H2 claim ("mean-pool beats trivial on the RBA dataset, synthesized from real-world
login behavior") is robustly replicated. The CI-lower-bound criterion ensures this isn't
a marginal effect. (Measured: True on all 5 seeds, and on both sensitivity splits.)

**Important baseline warning for seed=42:** The on-disk `rba_metrics.json` in
`experiments/h2_rba/` was produced from a 40/60 split (`n_ato_test_events=12`,
`roc_auc=0.9212`). The canonical primary result is 50/50 (`n_ato=9`, `roc=0.852`). Do not
use the on-disk file as the seed-42 sanity baseline — run seed=42 first to produce the
correct 50/50 result. (Decision `81ed922b`.)

---

## 11. Key Design Decisions and Their Rationale

This section documents the non-obvious choices that a colleague would most likely question.

### Why 400 accounts and 60 training events?

**400 accounts:** With 60 training events each, 400 accounts provide 400 spoof events and
800 negatives (neg + known) for AUC computation — large enough for stable AUC estimates
while keeping training time manageable. At 400 accounts, the 95% CI on AUC is typically
±0.02 or less. Doubling to 800 would narrow CIs by ~30% but quadruple training time.

**60 training events:** Zipf-weighted sampling over 2–4 known devices. 60 events gives
the model enough exposure to the account's primary device (≈30 events at Zipf s=1.5) and
some secondary devices, while the fleet injection (one event out of 60) is a realistic
1.7% poisoning rate — low enough to not dominate the training signal, high enough to
appear in `known_devices`.

**N_CENTROID=40 / calib=20 (H6):** The 40/20 split is somewhat arbitrary but satisfies
two constraints: (a) the calibration set must be large enough for a stable CDF estimate
(20 points → 20 quantiles), and (b) the centroid should use the majority of training data.
A 50/10 split would give a noisier CDF; a 30/30 split would weaken the centroid.

### Why 25% fleet fraction and 10-day blocklist lag?

**25% fleet fraction:** Fleet attacks in real ATO campaigns are coordinated but rare.
25% (100/400 accounts) gives enough fleet accounts for meaningful aggregate and residual
statistics while keeping fleet prevalence below 50% (where it would dominate the
narrative). The fleet device appearing in 100 accounts is still a coordinated attack
signal.

**10-day blocklist lag:** Real security operations teams typically notice novel attack
patterns within a few days (via anomaly monitoring) but need time to investigate,
confirm, and push the blocklist update to production. 10 days is a conservative estimate
of the operational lag. 30-day attack window gives the blocklist time to activate for
some fleet accounts while others remain in the cold-start period.

**The temporal structure is not calibrated to real data** — it's chosen to create a
non-trivial fraction of pre-lag accounts (~1/3 of fleet) to make the fleet_residual
block statistically interesting. The exact lag and window values are parameters, not
calibrated constants.

### Why per-account corpus instead of per-event corpus?

Per-event corpus produces within-feature embedding collapse (C1 claim). Per-account corpus
breaks the rigid positional structure: different sessions from the same account appear in
one long sentence with variable offsets, so tokens from the same feature appear in varying
co-occurrence contexts. This teaches the model that `tz_utc+0` and `tz_utc+5` are
different — they appear in different positions relative to different neighbors.

### Why spoof is the hardest attack and the primary claim

Spoof (a single guaranteed feature change) is the hardest because: (a) the attacker's
device fingerprint is nearly identical to the victim's, and (b) the cosine distance between
a minimally-changed event and the centroid is small compared to the distance between a
completely novel device and the centroid. If the model can detect 1-feature changes, it can certainly detect
2-feature or 3-feature changes. Novel attacks are the "easy" case — any model should
handle them. A model that wins on novel but fails on spoof is useless for ATO.

### Why PR-AUC is primary for H6 verdicts, not ROC-AUC

At 1:100 imbalance, ROC-AUC is dominated by true negatives — there are so many negatives
that a model can have high ROC-AUC while still failing to detect most attacks at any
practical threshold. PR-AUC focuses on the positive class: it measures whether the model
maintains precision while recovering more attacks. This is operationally meaningful —
an analyst can only review N alerts per day; the model must prioritize attacks in those
N slots. (Decision `65209b9b`.)

### Why the CI-lower-bound criterion for `h2_replicated`

Simple point comparison (`mp_roc > triv_roc`) is almost always true — mean-pool beats
trivial by at least something on virtually any test set. The CI criterion (`mp_roc_lo >
triv_roc`) requires that even the lower confidence bound exceeds trivial. This ensures
the replication claim is statistically robust, not a borderline artifact of a particular
seed. On the RBA dataset with only 3–12 evaluated ATO test events depending on split
(9 at the primary 50/50), AUC estimates are noisy, making the CI criterion especially
important. (Decision `9511d90f`.)

### Why fleet accounts have the fleet device in `known_devices`

This is the operational reality of how fleet attacks work: the attacker uses the fleet
device to log into victim accounts during the training window (the injection period). The
system would have seen this device before the alert window opens. The known-device gate
then correctly identifies it as "known" — and silently suppresses the alert. The finding
(C3) is that this supposed-safety mechanism completely defeats detection precisely because
the attack was coordinated enough to establish a history first.

### Why the sensitivity splits reuse the same trained model

The RBA sensitivity analysis asks: "Does the split point matter for the conclusion?"
Re-training for each split would make the answer confounded: different training sets →
different model quality → different AUC. The interesting question is whether the test-set
composition (how many ATO events fall in the test window) drives the conclusion. By fixing
the model and only varying the split, we isolate that effect.

### Why H2 and H6 use CBOW=epochs=10 for the degenerate config but epochs=20 for factorial

The degenerate config (`sg=0, per-event, epochs=10`) replicates the original PoC settings
exactly — it's a historical reproduction. The 2×2 factorial uses `epochs=20` for ALL
four cells to hold training budget constant while varying only `sg` and corpus. If the
degenerate cell in the factorial used `epochs=10`, any performance difference could be
attributed to fewer gradient steps rather than the collapse mechanism. By standardizing
to `epochs=20`, the factorial is a clean controlled experiment.

### Why `rtt_bucket` is in FEATURE_ORDER but excluded from `device_key()` in H6/RBA

RTT (round-trip time) varies across sessions even from the same physical device and
network location — it reflects network congestion, server load, and measurement noise,
not device identity. Including RTT in `device_key()` would cause the same device at
two different times to have different "keys," preventing recognition. It IS included in
embeddings because it provides distributional signal about where logins come from
(different country → different typical RTT ranges), but it cannot be used for
fingerprinting.

This distinction is encoded in two separate constants:
```python
FEATURE_ORDER       = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]
DEVICE_KEY_FEATURES = ["os", "browser", "device_type", "country", "region", "asn_bucket"]
```

### Why three different corpus representations in H2

T1 (mp vs concat_w1) = the primary claim.
T2 (window sweep) = tests whether w=1 is unfairly penalizing concat by limiting context.
T3 (prefixed concat) = tests whether n-gram contamination is the mechanism.

The combination of T2 and T3 provides convergent evidence: T2 shows more context doesn't
help concat; T3 shows removing contamination (by prefixing) does help. Together they
support the n-gram contamination mechanism without T1 alone being sufficient.

### Why `--smoke` doesn't write results.json

Smoke runs on 50 accounts with synthetic-data distributions that don't match the
full-run conditions (fewer accounts = less diverse corpus = different AUC values). Writing
results.json from smoke would pollute the seeds directory with non-comparable data.
The smoke run's sole purpose is code-path verification — proving the script will not crash
and that all assertions pass on a small fast run.

---

*Document generated from plan docs (Sections 1–11) and scripts as of 2026-04-21;
audited against scripts and per-seed data on 2026-07-01 (issue 006ae95f / resolution
1c4fb9b2 errata classes applied; 5-seed aggregate numbers replace stale single-seed or
pre-run illustrative values throughout). Script versions: h2_rerun.py, h6_rerun.py,
rba_rerun.py, h2_degenerate_downstream.py — all on branch `rerun`. Journal entries cited
by ID are authoritative. If this document conflicts with a journal decision entry or the
scripts, trust the journal/scripts over this document.*
