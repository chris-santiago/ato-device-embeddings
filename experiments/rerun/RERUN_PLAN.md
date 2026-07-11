# Experiment Rerun Plan — Index

**Purpose:** 5-seed reproducibility rerun of H2, H6, and RBA experiments for ICAIF 2026.


Each section is a standalone document in `plan/`. Every file opens with the full
Standing Instructions block so the rules are visible regardless of which section
you open first.

---

## Plan Sections

| # | File | Contents |
|---|------|----------|
| — | [Standing Instructions](plan/1-SEED_POLICY.md#standing-instructions) | Artifact hygiene, verifiability, ground truth, ml-journal protocol |
| 1 | [1-SEED_POLICY.md](plan/1-SEED_POLICY.md) | Seeds, scope of seed control, implementation pattern |
| 2 | [2-ENVIRONMENT_PINNING.md](plan/2-ENVIRONMENT_PINNING.md) | `# /// script` blocks, `uv run`, exact version pins |
| 3 | [3-DIRECTORY_STRUCTURE.md](plan/3-DIRECTORY_STRUCTURE.md) | `seeds/`, `aggregate/`, `logs/` layout |
| 4 | [4-METRICS_SCHEMA.md](plan/4-METRICS_SCHEMA.md) | Per-seed JSON schemas for H2, H6, RBA + rationale |
| 5 | [5-AGGREGATION_PROTOCOL.md](plan/5-AGGREGATION_PROTOCOL.md) | Mean ± std, CI aggregation, verdict stability |
| 6 | [6-EXPERIMENT_RUN_ORDER.md](plan/6-EXPERIMENT_RUN_ORDER.md) | Phase 1–3 run steps, gap analysis, cross-cutting issues |
| 7 | [7-RUN_DOCUMENTATION.md](plan/7-RUN_DOCUMENTATION.md) | Per-seed log requirements |
| 8 | [8-CONSISTENCY_CHECKS.md](plan/8-CONSISTENCY_CHECKS.md) | Pre-aggregation assertions |
| 9 | [9-PAPER_REPORTING.md](plan/9-PAPER_REPORTING.md) | Table formats, CI reporting, verdict stability table |
| 10 | [10-SEED_SENSITIVITY.md](plan/10-SEED_SENSITIVITY.md) | Limitations section template |
| 11 | [11-SUBMISSION_CHECKLIST.md](plan/11-SUBMISSION_CHECKLIST.md) | Final pre-submission checklist |

---

## Quick Reference

**Seeds:** `[42, 123, 456, 789, 2024]`

**Run any script:** `uv run experiments/<phase>/experiments/<script>.py --seed <N>`

**Source directories:**

| Phase | Source | Rerun output |
|-------|--------|-------------|
| H2 | `experiments/h2_ml_lab/` | `rerun/seeds/seed_N/h2/` |
| H6 | `experiments/h6_hybrid/` | `rerun/seeds/seed_N/h6/` |
| RBA | `experiments/h2_rba/` | `rerun/seeds/seed_N/rba/` |

**Open issues before starting:** see Section 6 gap tables and cross-cutting issues.
