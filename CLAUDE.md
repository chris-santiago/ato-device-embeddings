# CLAUDE.md

> Repo-specific context for Claude Code. Loaded every session.
> Keep this under ~120 lines. Document what Claude gets wrong, not comprehensive manuals.

---

## Project Overview

**Purpose:** Research investigation into ATO (Account Takeover) detection using word embeddings trained on login device feature sequences — primary finding is that FastText mean-pool on structured feature tokens (AUC 0.985+) beats raw device-ID embeddings.
**Stack:** Python 3.10+, gensim (FastText/Word2Vec), scikit-learn, numpy, scipy, matplotlib
**Structure:**
```
experiments/                  # All investigation tracks live here
  pre_ml_lab/                 # H1 + early H2 ad-hoc experiments
  h2_ml_lab/                  # H2 — structured ml-lab investigation
  h2_rba/                     # H2 — RBA dataset variant
  h3_pfn/                     # H3 — TabPFN
  h4_gru/                     # H4 — GRU sequence models
  h5_stress/                  # H5 — stress test of H2 config
  h6_hybrid/                  # H6 — hybrid model
  rerun/                      # Reproducibility harness (see below)
    scripts/{h2,h6,rba}/      # Per-track rerun drivers
    plan/, seeds/, aggregate/ # Plan, seed sweeps, aggregated outputs
    run_all.sh                # End-to-end rerun entry point
slides/                       # Marp slide deck
research-notes/               # Standalone research notes
data/rba/                     # RBA dataset
archive/                      # Older process docs
.project-log/                 # Journal (authoritative — see below)
TECHNICAL_REPORT.md           # Canonical write-up
DEVICE_EMBEDDING_FINDINGS.md  # Headline findings
```

Each `experiments/<track>/` (except `rerun/`) follows the same shape: `experiments/` (scripts), `figures/`, `docs/`.

---

## Environment & Commands

```bash
# Run any experiment (no virtualenv needed — PEP 723 inline deps)
uv run experiments/pre_ml_lab/experiments/ato_experiment3.py
uv run experiments/h2_ml_lab/experiments/robust_config_experiment.py

# Full reproducibility rerun (all tracks)
bash experiments/rerun/run_all.sh

# Lint
ruff check .
ruff format .
```

> Every script is self-contained with `# /// script` dependency headers.
> Never introduce a shared `pyproject.toml`, `requirements.txt`, or virtualenv — that breaks the PEP 723 pattern.

---

## Code Style

- Each script is standalone — inline `# /// script` block at the top with `requires-python` and `dependencies`
- Imports at module level after the script block (not inside functions)
- Figures saved to `{experiment_dir}/figures/` with descriptive filenames
- All data is synthetic — no external data files or connections

---

## Architecture Notes

- **Investigations are organized by hypothesis (H1–H6).** `pre_ml_lab/` holds H1 + early H2 ad-hoc work; `h2_ml_lab/` is the structured H2 ml-lab investigation; `h2_rba/` is the RBA-dataset H2 variant; `h3_pfn/`, `h4_gru/`, `h5_stress/`, `h6_hybrid/` follow. Do not conflate tracks.
- **`rerun/` is a reproducibility harness, not an experiment.** It re-executes H2/H6/RBA tracks under seed sweeps to validate headline results. Treat it as a separate concern from primary experimentation.
- **Eval design is membership-based:** known devices appear in training by design (25% fleet injection simulates unconfirmed prior attacks). This is intentional, not leakage — the PoC tests proximity detection, not generalization to fully unseen devices.
- The trivial baseline (exact set-membership) is a critical benchmark — any proposed signal must beat it on spoof attacks specifically.

---

## Journal — Consultation Before Planning

Before drafting any non-trivial plan, query the journal:
1. `python3 .project-log/journal_query.py --unresolved-issues` — surface active blockers
2. `python3 .project-log/journal_query.py --list decision --since 7d` — surface recent decisions that constrain the approach

When a plan step is informed by a journal entry, cite the short ID inline (e.g. `[→ issue 71af7634]`). A `decision` entry is a resolved constraint — retrieve full context before re-opening: `python3 .project-log/journal_query.py --entry <id>`.

## Journal — Committing

The word **"commit"** means the **ml-journal `/log-commit` skill** — not bare `git commit`, not any other commit helper. The skill stages files, synthesizes a commit message, creates the commit, and writes a `git` journal entry in one step. Never use a different commit path in this project.

## Journal — Proactive Logging

When `.project-log/journal.jsonl` exists, propose logging at natural pauses — not mid-investigation. Always ask first; full draft only after user confirms.

**Auto-propose these types:**

| Pattern | Type | When |
|---------|------|------|
| User confirms a direction | `decision` | After "I agree", "let's do X", "go with that" |
| Unexpected finding | `discovery` | When exploration changes understanding or approach |
| Bug/inconsistency found | `issue` | After identifying and explaining a problem |
| Bug fixed and verified | `resolution` | After fix confirmed working |
| Root cause understood | `lesson` | After explaining *why* something broke — ask "should I log this as a lesson?" |
| Results interpreted | `experiment` | When verdict is clear |

**Do not auto-propose:** `/checkpoint`, `/resume`, `/log-commit`, `/research-note`, `/research-report`, read skills, `hypothesis`, `post_mortem`, `memo`.

**Rules:** One proposal per event. Don't re-propose if declined. Chain issue→resolution→lesson at completion, not as three interruptions.

## Known Gotchas

- `uv run` is required — plain `python` will fail if the script's inline deps aren't already installed in the active environment
- Spoof attack AUC is the hardest and most important metric; overall AUC alone is not sufficient for declaring a signal viable
- `.ruff_cache/` is present — ruff is the linter, not flake8/pylint
