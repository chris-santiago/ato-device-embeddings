# CLAUDE.md

> Repo-specific context for Claude Code. Loaded every session.
> Keep this under ~120 lines. Document what Claude gets wrong, not comprehensive manuals.

---

## Project Overview

**Purpose:** Research investigation into ATO (Account Takeover) detection using word embeddings trained on login device feature sequences — primary finding is that FastText mean-pool on structured feature tokens (AUC 0.985+) beats raw device-ID embeddings.
**Stack:** Python 3.10+, gensim (FastText/Word2Vec), scikit-learn, numpy, scipy, matplotlib
**Structure:**
```
pre_ml_lab/       # Pre-ml-lab experiments (H1 and H2 reruns)
  experiments/    # Self-contained Python scripts (PEP 723)
  figures/        # Output plots
  docs/           # Reports, critiques, debates, conclusions
h2_ml_lab/        # ml-lab structured investigation (H2)
  experiments/    # Self-contained Python scripts
  figures/
  docs/
slides/           # Marp slide deck
archive/          # Older process docs
```

---

## Environment & Commands

```bash
# Run any experiment (no virtualenv needed — PEP 723 inline deps)
uv run pre_ml_lab/experiments/ato_experiment3.py
uv run h2_ml_lab/experiments/robust_config_experiment.py

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

- **Two investigation phases:** `pre_ml_lab/` holds earlier ad-hoc experiments; `h2_ml_lab/` holds the structured ml-lab investigation. Do not conflate them.
- **Eval design is membership-based:** known devices appear in training by design (25% fleet injection simulates unconfirmed prior attacks). This is intentional, not leakage — the PoC tests proximity detection, not generalization to fully unseen devices.
- The trivial baseline (exact set-membership) is a critical benchmark — any proposed signal must beat it on spoof attacks specifically.

---

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
- FastText config matters critically: CBOW + per-event corpus silently destroys spoof detection while leaving other metrics intact — always use Skip-gram (`sg=1`) with per-account concatenated corpus
- `.ruff_cache/` is present — ruff is the linter, not flake8/pylint
