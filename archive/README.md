# Archive

This directory contains the original methodology files that seeded the investigation in this repo.

| File | Description |
|------|-------------|
| `PROCESS.md` | Human-readable 9-step guide for ML hypothesis investigation, illustrated with prompts and outcomes from this ATO investigation |
| `agent.md` | Agent-executable version of the same methodology — a prompt document instructing a Claude agent to run the full workflow on any ML hypothesis |

## Why these are archived

These files were hand-authored at the start of this project to define a rigorous investigation process: PoC → adversarial critique → design defense → multi-round debate → pre-registered experiments → production re-evaluation → peer review.

The methodology proved effective. The `h2_ml_lab/` investigation in this repo was used as a test case to develop and validate a formal implementation of this process as a reusable Claude Code tool: **[ml-debate-lab](https://github.com/chris-santiago/ml-debate-lab)**.

`ml-debate-lab` supersedes these files. It implements the same workflow as a structured multi-agent system with dedicated critic, defender, and peer-reviewer subagents, macro-iteration support, and standardized artifact output. Use that tool for new investigations; these files are kept here for provenance.
