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

## 3. Directory Structure

```
rerun/
├── run_all.sh                  # Single entry point: runs all seeds × all experiments
├── scripts/
│   ├── aggregate.py            # Aggregates seed results → {h2,h6,rba}_aggregate.csv
│   ├── check_consistency.py    # Cross-seed consistency and schema checks
│   ├── h2/
│   │   ├── h2_rerun.py         # H2 experiment runner (called per seed by run_all.sh)
│   │   └── h2_figures.py       # H2 publication figures from aggregate
│   ├── h6/
│   │   ├── h6_rerun.py         # H6 experiment runner (called per seed by run_all.sh)
│   │   └── h6_figures.py       # H6 publication figures from aggregate
│   └── rba/
│       ├── rba_rerun.py        # RBA experiment runner (called per seed by run_all.sh)
│       └── rba_figures.py      # RBA publication figures from aggregate
├── seeds/
│   ├── seed_42/
│   │   ├── h2/
│   │   │   ├── results.json    # All H2 metrics for this seed
│   │   │   └── logs/
│   │   ├── h6/
│   │   │   ├── results.json
│   │   │   └── logs/
│   │   └── rba/
│   │       ├── results.json
│   │       └── logs/
│   ├── seed_123/
│   ├── seed_456/
│   ├── seed_789/
│   └── seed_2024/
├── aggregate/
│   ├── h2_aggregate.csv        # Mean ± std across seeds for all H2 metrics
│   ├── h6_aggregate.csv
│   ├── rba_aggregate.csv
│   ├── verdict_stability.csv   # Per-test verdict count across 5 seeds
│   └── figures/                # Publication figures (generated from aggregate)
└── RERUN_PLAN.md               # Plan index
```

Each `results.json` must be self-contained: seed, timestamp, library versions, all metrics.
