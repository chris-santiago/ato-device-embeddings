# Research Note — ato-device-embeddings

*2026-04-19 | 8 journal entries | scope: today*

## Summary

Completed the H2 replication on the DAS Group RBA real-world dataset (~31M login events,
141 ATO events). After fixing multiple data prep and pipeline bugs inherited from the
previous session, the full experiment ran successfully: mean-pool FastText ROC-AUC 0.852
[0.689, 0.975] vs. trivial set-membership 0.661, replicated across three temporal split
cutoffs. A post-hoc leakage audit (Opus) found no contamination but flagged n=9 positive
test events and a post-hoc split adjustment as limitations. A cross-document coherence audit
caught and corrected three critical numerical errors in prior synthesis documents. All
artifacts committed and pushed to main.

## Hypothesis

**H2-RBA** `[64e85b60]`: FastText skip-gram mean-pool centroid scoring outperforms trivial
set-membership on real-world ATO detection (DAS Group RBA dataset). Registered with 50/50
temporal split and ≥5 training event floor (revised from planned 80/20 / ≥10 after
discovering all 141 ATO events fall before the 70th percentile of timestamps).

## Discoveries & Results

- **Experiment** `[b1769df5]` — **CONFIRMED (exploratory).** Mean-pool ROC-AUC 0.852
  [0.689, 0.975] vs. trivial 0.661 at 50/50 split. PR-AUC 0.032 vs. 0.0003 (95× baseline).
  Ordering holds at 40/60 (0.921 vs. 0.699) and 60/40 (0.933 vs. 0.720) splits. T6
  compactness (0.036) and T8 within/cross ratio (1.66) consistent with synthetic H2.
- **Leakage audit (Opus)** — MINOR_CONCERNS. No label leakage, temporal leakage, or
  known-device contamination. Key caveats: n=9 positive test events; post-hoc split
  adjustment from 80/20 → 50/50; non-stratified bootstrap CIs.
- **Coherence audit (Opus)** — MAJOR_REVISION_NEEDED (resolved). Three critical numerical
  errors corrected: synthetic T6 compactness (0.033→0.047), T8 within/cross ratio
  (1.60→1.14), synthetic AUCs in DEVICE_EMBEDDING_FINDINGS (0.985/0.920/0.798→0.993/0.939/0.818).

## Artifacts Produced

- `h2_rba/experiments/data_prep.py` — download, normalize, write parquet
- `h2_rba/experiments/rba_rerun.py` — full replication pipeline with `--split-pct`, `--smoke`, `--full` flags
- `h2_rba/docs/HYPOTHESIS.md` — pre-run hypothesis (with post-hoc revision note)
- `h2_rba/docs/REPORT.md` — full report: design notes, results, audit findings, sensitivity analysis, limitations
- `h2_rba/figures/rba_metrics.json` — canonical numeric results
- `TECHNICAL_REPORT.md` §6 — real-world replication section added
- `README.md`, `DEVICE_EMBEDDING_FINDINGS.md` — updated with RBA results and corrected synthetic numbers
- `slides/methodology_pitch.md` — 14-slide Marp deck for DS and business audiences (gitignored, local only)

## Current State

All commits pushed to `main` (`da78eb6`). No open issues. The methodology pitch slides are
local-only (gitignored via `slides/.gitignore`).

## Next Steps

- If pilot data available: run `rba_rerun.py` pattern against production feature schema; compare mean-pool vs. current set-membership baseline; report PR-AUC (primary) and ROC-AUC with bootstrap CIs
- Consider stratified bootstrap as a code improvement to `rba_rerun.py` (Opus audit recommendation)
