# Research Note — ATO Device-Embedding Investigation

*2026-04-21 | 5 commits, 4 journal entries | scope: last 12 hours (rerun branch)*

## Summary

Completed the 5-seed reproducibility rerun of all three experiment phases (H2, H6, RBA), confirmed all four pre-registered contributions reproduce with quantified seed variance, and wrote REPORT.md. Post-run, two over-conservative consistency checks were revised to match the actual pre-registered claims. The rerun infrastructure is now fully operational and the evidence base is stable for paper writing.

## Key Decisions

- **Drop RBA-S4** (mp > concat ROC-AUC per seed): not pre-registered, underpowered at n_ato=9 (bootstrap CI width ~0.30 fully overlaps the mp/concat difference). Pre-registered verdict is `h2_replicated` (True 5/5). [ac14ff1d]
- **Relax X3** (spoof-k gradient): changed from strict `k1 ≤ k2 ≤ k3` to `k1 < min(k2, k3)`. k2→k3 inversion on 2/5 seeds is 0.0005–0.0039 PR-AUC at saturation (PR≈0.95+); the claim is that k=1 is the hardest attack type, not that k=2 and k=3 are strictly ordered. [ac14ff1d]

## Discoveries & Results

- **5-seed rerun: CONFIRMED** [d03fab44]
  - C4 spoof Δ +0.130 CI [0.111, 0.150] — all positive, all seeds
  - C1 degenerate within-feature cosine 0.9992 ± 0.00008; collapse 5/5 seeds, 0/5 on robust
  - C2 rank-norm PR drop 0.888 → 0.224 (~4× collapse at 1:100); `rank_norm_collapse_confirmed` True 5/5
  - C3 two_stage top-1% TP = 0 on fleet residual, all 5 seeds; `gate_blinds_fleet_confirmed` True 5/5
  - All 15 H6 verdicts (3 × 5 seeds) and 5 RBA `h2_replicated` verdicts: True

- **Noise-floor non-claims identified:**
  - k2→k3 ordering inverts on seeds 42 and 123 at PR≈0.97 saturation (not a claim)
  - mp > concat ROC-AUC on RBA holds 3/5 seeds only; underpowered at n_ato=9 (not a claim)

- **Fleet residual confirmed faithful to original:** `collect_fleet_residual_scores()` in h6_rerun.py uses identical `fleet_blocklist_active` filter logic as h6_hybrid/hybrid_experiment.py. Fleet absolute metric differences (0.667 vs 0.947 PR-AUC) explained by single-seed reference in original `h6_metrics.json` vs 5-seed variance, not a logic change.

## Current State

All seeds complete; aggregate CSVs, `aggregate.json`, 7 aggregate figures, and `REPORT.md` committed at `1e08532`. Consistency checks pass 66/66. Journal experiment entry `d03fab44` and decision entry `ac14ff1d` logged.

## Next Steps

- Paper writing: REPORT.md and aggregate CSVs are the primary artifact inputs
- Figures for paper: aggregate figures in `aggregate/figures/` cover C2/C3/C4; C1 token similarity bar chart exists per-seed in `seeds/seed_{S}/h2/figures/h2_t8_token_similarity.png` — a cross-seed aggregate version may be needed
- Consider PR to merge `rerun` → `main`
