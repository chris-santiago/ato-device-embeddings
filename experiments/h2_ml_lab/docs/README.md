# H2: Mean-pool vs. Concat FastText for ATO Device Embedding

**Status: H2 Confirmed** — mean-pool FastText outperforms concat for ATO device fingerprint detection under the robust training configuration

## Hypothesis

Mean-pooling six feature-token embeddings (one per device dimension: OS, browser, timezone, language, network, screen) into a device vector will outperform directly embedding a single concatenated device string using FastText, on both silhouette score and ROC-AUC for ATO (Account Takeover) detection. The advantage is expected to be largest on spoof attacks, where the attacker exactly matches the victim's OS, browser, and language but differs only on timezone — because cross-boundary character n-grams in the concatenated string inject spurious similarity that masks the single differing dimension.

## Quickstart

```bash
uv run h2_ml_lab/experiments/ato_device_embedding_poc.py
```

Requires: Python >= 3.10, `uv` package manager. All dependencies are specified in PEP 723 inline metadata.

## Pipeline

```
Data generation
  └── 400 synthetic accounts
  └── 60 Zipf-weighted training events per account (2-4 known devices each)
  └── Fleet attacker injected into 25% of accounts' training sets
        │
        ▼
FastText training
  ├── Mean-pool corpus: each event → list of 6 prefixed tokens
  │   (e.g., ["os_ios", "browser_safari", "tz_utc-5", "lang_en_us", "network_wifi", "screen_small"])
  └── Concat corpus: each event → single concatenated string token
      (e.g., ["ios_safari_utc-5_en_us_wifi_small"])
        │
        ▼
Scoring (inference)
  ├── Per-account centroid: mean of all training event embeddings
  └── Score: cosine distance from test event embedding to centroid
        │
        ▼
Evaluation
  ├── ROC-AUC per attack type (novel, fleet, spoof)
  │   └── Attack events (label=1) vs. negative events (label=0)
  ├── Silhouette score (cosine metric, per-device embedding clusters)
  └── Trivial baseline: exact 6/6 feature set-membership check
        │
        ▼
Output: structured results table + AUC bar chart
```

## Attack Types

| Type   | Definition | Why it's diagnostic |
|--------|-----------|---------------------|
| Novel  | Foreign OS, timezone, AND language | Easy — many feature differences |
| Fleet  | Cross-account attacker device injected into 25% of training sets | Tricky — the attack device has appeared in training |
| Spoof  | Matches primary OS/browser/language, differs only on timezone | Hardest — only 1 of 6 features differs |

## Negative Class

Negative test events are new legitimate devices that share OS, browser, timezone, and language with the primary profile but differ on network and/or screen. This prevents rewarding models that simply flag any unseen device — the evaluation tests whether the model correctly accepts new devices that are plausibly from the same user.

## What the Output Looks Like

Key results under the robust config (sg=1, per-account corpus, epochs=20) from H2_RERUN:

```
--- Results (Robust Config: sg=1, per-account corpus, epochs=20) ---
Metric                            Mean-pool       Concat      Trivial
AUC (novel)                          0.993         0.981        0.791
AUC (fleet)                          0.939         0.933        0.791
AUC (spoof)                          0.818         0.763        0.791
T8 within-feature sim                0.392           N/A          N/A

Spoof AUC gap (mean-pool - concat): +0.055, CI [+0.034, +0.077]
Mean-pool spoof vs. trivial baseline: +0.027
Concat spoof vs. trivial baseline: -0.028
```

Note: H2 is confirmed under the robust config. Mean-pool beats the trivial baseline on spoof (+0.027); concat does not (-0.028). T8 within-feature similarity of 0.392 confirms no embedding collapse. Under a degenerate config (CBOW, per-event corpus), within-feature sim reaches 0.999 (collapse), H2 is refuted, and neither model beats the trivial baseline on spoof.

## Known Limitations / Explicit Scope Exclusions

1. **No bootstrap CIs**: The PoC produces single-run point estimates only. Statistical significance is tested in `ato_device_embedding_experiment2.py`.
2. **Simplified silhouette**: The PoC groups all accounts' events by `account_id % 10` for silhouette labels (for speed). The experiment uses true per-device labels.
3. **Hyperparameters not tuned**: FastText uses fixed `vector_size=64, window=6, epochs=10`. No sweep.
4. **No stratified analysis by account activity level**: High-activity vs. low-activity accounts may behave differently.
5. **Fleet attack includes training injection**: Fleet attacker events appear in some accounts' training sets, making evaluation of fleet AUC non-trivial to interpret.
6. **No temporal structure**: Training events are i.i.d.; real logins have temporal order.
7. **No cold-start evaluation**: All accounts have 60 training events. Cold-start (0-5 events) is not modeled.
8. **Within-feature embedding collapse occurs under CBOW + per-event corpus** (see `config_verification.py`); robust config (sg=1, per-account corpus) verified to avoid collapse (T8 sim = 0.392).

## File Structure

```
h2_ml_lab/experiments/
  ato_device_embedding_poc.py            # Step 1: minimal proof-of-concept
  ato_device_embedding_experiment2.py    # Step 6: full experiment with bootstrap CIs
  robust_config_experiment.py            # T4/T6/T8 diagnostics under robust config
  config_verification.py                 # T8 under both robust and degenerate configs

h2_ml_lab/docs/
  HYPOTHESIS.md        # Canonical hypothesis and metrics
  README.md            # This file
  CRITIQUE.md          # Step 3: adversarial critique
  DEFENSE.md           # Step 4: defense
  DEBATE.md            # Step 5: debate to resolution
  CONCLUSIONS.md       # Step 7: per-finding verdicts
  REPORT.md            # Step 8: full report
  REPORT_ADDENDUM.md   # Step 9: production re-evaluation
TECHNICAL_REPORT.md    # Publication-ready synthesis (repo root)

h2_ml_lab/figures/
  poc_auc_comparison.png   # PoC bar chart
  ...                      # Additional figures from experiment
```
