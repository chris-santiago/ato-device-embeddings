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

## Appendix

## Why CBOW + Per-Event Corpus Collapses Within-Feature Embeddings

### The corpus structure is the problem

In the per-event setup, every login event becomes one training sentence:

["os_ios", "browser_safari", "tz_utc-5", "lang_en", "net_wifi", "screen_small"]
["os_ios", "browser_safari", "tz_utc-8", "lang_en", "net_wifi", "screen_small"]
["os_ios", "browser_safari", "tz_utc+0", "lang_en", "net_wifi", "screen_small"]

Notice what's identical across all three sentences: every token except the timezone. The
positional structure is rigid — slot 0 is always an OS token, slot 2 is always a timezone
token.

### What CBOW does with this

CBOW predicts the center token from its surrounding context. With a window of 6,
the context for tz_utc-5 is [os_ios, browser_safari, lang_en, net_wifi, screen_small] —
the same five tokens appear as context for every timezone value across every sentence.

predict(tz_utc-5)  ← context: [os_ios, browser_safari, lang_en, net_wifi, screen_small]
predict(tz_utc-8)  ← context: [os_ios, browser_safari, lang_en, net_wifi, screen_small]
predict(tz_utc+0)  ← context: [os_ios, browser_safari, lang_en, net_wifi, screen_small]

The model is being trained to predict each timezone token from an **identical context
distribution.** The gradient signal that updates each timezone embedding is therefore
identical — after enough epochs, the optimizer finds that the loss is minimized by
converging all timezone vectors toward the same point.

This is not a bug. The model is doing exactly what it's supposed to do: tokens that appear
in the same context should get similar embeddings. It's just that the corpus structure
forces all timezone values to share the same context.

### Why skip-gram + per-account corpus fixes it

**Per-account corpus** flattens all of a user's events into one long sentence:

[os_ios, browser_safari, tz_utc-5, lang_en, ...,
  os_windows, browser_chrome, tz_utc-8, lang_fr, ...,
  os_ios, browser_safari, tz_utc+0, lang_en, ...]

Now tz_utc-5 has browser_safari as a neighbor in some positions and browser_chrome
nearby in others. Different timezone values co-occur with different OS/browser/language
combinations across events, so their context distributions diverge.

**Skip-gram** inverts the prediction task: predict context tokens from the center token.
Each timezone value must now predict which OS, browser, and language tokens are likely to
appear near it — and because tz_utc-5 and tz_utc-8 appear in different account
contexts, they develop genuinely different predictive distributions and thus genuinely
different embedding vectors.

### Why it's dangerous

Novel AUC and fleet AUC stay healthy under collapse (~0.88 and ~0.92) because those attack
types differ on many features at once — even with timezone collapsed, the OS and browser
differences carry enough signal. Only spoof attacks — where the attacker matches 5 of 6
features and differs only on timezone — expose the failure. With timezone embeddings
collapsed, mean-pool has nothing to work with and AUC drops to 0.384 (below chance, because
high cosine distance now slightly anti-correlates with anomaly due to noise).

*A monitoring dashboard watching aggregate AUC sees nothing wrong.*
