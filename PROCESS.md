# ML Research Methodology: Hypothesis to Production Recommendation

A prescriptive guide for rigorously investigating an ML hypothesis — from initial
implementation through adversarial review, empirical resolution, and deployment
re-evaluation. The steps below are illustrated with verbatim prompts and outcomes from
an ATO device-embedding investigation, but the pattern applies to any applied ML
research thread.

---

## Step 1 — State the hypothesis precisely and build the minimal PoC

**What to do:** Before writing any code, state the hypothesis as a specific, falsifiable
claim. Then implement the simplest version that directly tests it. Resist scope creep —
the PoC should do one thing: provide initial evidence that the signal exists.

A well-formed hypothesis names the mechanism, the signal, and the expected observable:

> *"FastText trained on per-account device sequences will embed known devices near their
> account's centroid. Novel (takeover) devices will land far from that centroid, creating
> a detectable anomaly signal."*

The PoC should be end-to-end: synthetic data generation with controlled ground truth,
the proposed model or signal, an evaluation metric, and at least one visualization. If
it cannot be run in one command and produce a number, it is not a PoC.

**Example prompt:**

> Build a self-contained Python proof-of-concept for embedding high-cardinality device
> IDs using FastText to detect account takeover signals. [Requirements: synthetic data
> generation, FastText training, centroid-based inference, ROC curve evaluation, UMAP
> visualization.] The goal is a clear, reproducible script that validates or refutes the
> core hypothesis.

**What to watch for:** Be explicit about what the PoC is testing and what it is
deliberately leaving out. Write those exclusions down in the README from day one — they
become the scope boundary for every subsequent step.

---

## Step 2 — Clarify intent before adding to or reviewing the code

**What to do:** Before performing a code review or extending the implementation, verify
that you understand the *intent* of the existing design choices — not just what the code
does, but why. The most dangerous code review error is flagging intentional behavior as
a bug.

Use plan mode for any non-trivial review. Write down what you believe each component is
doing and why, then check against the actual code. When something looks like an error,
ask whether it might be a deliberate constraint before proposing a fix.

**Example prompt:**

> Create a readme. Clarify intent in code — review for logical errors

**What happened:** An automated review flagged known devices appearing in training as
"data leakage." The proposal was to withhold known devices from training to create a
proper held-out set. The user rejected this:

> Are you sure there are logical errors? I'm not sure all of the test set should be held
> out devices — I think that conflicts with the intent

The design was correct: this is a *membership/proximity test*, not a generalization test.
The production scenario is to compute a centroid from all known history and score
incoming devices against it. Including known devices in training is intentional.

**The rule:** When reviewing someone else's code, ask "could this be intentional?" before
"is this wrong?" — especially for evaluation design in ML systems, where the definition
of "correct" depends entirely on what the system is supposed to do in production.

---

## Step 3 — Critique the implementation from first principles

**What to do:** Assign a skeptic persona with domain expertise relevant to the
approach — mathematics, systems, statistics, or the relevant applied field. The critique
should be grounded in first principles, not just implementation nits. The goal is to
surface claims that the PoC makes implicitly but has not tested.

Good critiques are falsifiable. "The subword n-grams will interfere with account
clustering" is useful because it can be tested. "The code is complex" is not.

Organize the critique by root cause, not by severity or line number. Many surface issues
trace to a small number of foundational choices. Fixing symptoms without diagnosing root
causes produces a cleaner PoC that is still wrong.

**Example prompt:**

> You're a seasoned ML engineer with a background in applied math. Inspect the toy
> example critically, and create a new markdown file detailing any weaknesses in theory
> or implementation.

**What to produce:** A numbered list of issues, each with: the specific claim being made
(implicitly or explicitly), why that claim might be wrong, and what would constitute
evidence one way or the other. The last part — what would constitute evidence — is the
most important, because it feeds directly into the debate and experiment design.

---

## Step 4 — Defend the original approach

**What to do:** Argue for the original design against each critique. This step is not
about being right — it is about ensuring the critiques are calibrated. An overconfident
critique that calls everything a fatal flaw will produce a useless experiment that tests
the wrong things. The defense forces precision.

Some critiques will be strong and require empirical resolution. Others will be wrong, or
will identify concerns that are real but not disqualifying. The defense separates these
categories before the debate.

**Example prompt:**

> Now take the opposite view. I want you to create a new markdown file that addresses
> these critiques and supports the original hypothesis, if able.

**What to produce:** A point-by-point rebuttal. Concede clearly where the critique is
right. For contested points, state the strongest version of the defense argument and
identify precisely what empirical observation would confirm or refute it. The defense
should sharpen disagreements, not paper over them.

---

## Step 5 — Debate each point to resolution

**What to do:** Take every contested point through multiple argument turns until it
reaches one of three outcomes:

1. **Critique wins** — the original design has a real problem; the experiment must test it
2. **Defense wins** — the critique was wrong or overstated; no further testing needed
3. **Empirical test agreed** — the argument cannot be resolved theoretically; specify
   exactly what experimental condition would settle it

The debate must be genuine — each side should update when the other makes a good point.
A debate that just restates initial positions is not a debate; it is noise. The measure
of a good debate round is whether the argument has moved: the claim is now more precise,
a counter-example has been identified, or a logical consequence has been derived that
neither side originally stated.

**Example prompt:**

> Now I want you to take both the critique and the defense, and create a new markdown to
> debate each point. Take multiple turns on each point until you can arrive at an
> agreement — could be concession or plan to study empirically.

**What to produce:** A document with multiple exchange rounds per point, ending in a
clear resolution statement. The most important output is the list of empirical tests
agreed — these become the specification for the next experiment. They should be stated
as exact experimental conditions, not vague directions: not "investigate subword
n-grams" but "run with Word2Vec (no subword) and compare AUC against FastText."

---

## Step 6 — Design and run the experiment from the debate outcomes

**What to do:** Translate every "empirical test agreed" from the debate into a concrete
experimental condition with a pre-specified verdict. For each test, write down what
result would mean the critique was right, what would mean the defense was right, and what
would be ambiguous. The experiment should produce answers, not new questions.

Include the trivial baseline. The debate in this investigation agreed that a binary
OOV lookup should be compared against the embedding model. This is the most important
discipline in ML evaluation: if a two-line solution outperforms a complex model, that
is the most important finding, not a footnote.

**Example prompt:**

> Now use this debate to create and run a new experiment to address any points requiring
> empirical evidence or next steps.

**What to watch for:** Bugs in baselines are especially dangerous. A broken baseline
that appears weak makes the primary model look stronger than it is. In this investigation,
gensim's FastText `__contains__` method returned `True` for any string with valid
character n-grams — making the OOV binary baseline score every device as in-vocabulary,
producing AUC 0.500 and appearing to confirm that the baseline fails. The actual result
after fixing the bug was AUC 0.989 — the strongest single result in the entire study.
Always verify that baseline implementations are testing what you think they are testing.

**What to produce:** A script that implements all agreed experimental conditions, reports
results against pre-specified verdicts, and produces outputs that can be directly cited
in the conclusions. Bootstrap confidence intervals should be standard. Stratified
analysis (by history length, corpus mode, etc.) should be included where the debate
identified relevant subpopulations.

---

## Step 7 — Synthesize findings into a conclusions document

**What to do:** Write findings as verdicts against the pre-specified debate resolutions —
not as a summary of what you ran. Each finding should state what was agreed, what the
evidence showed, and which side was right. Findings that surprise both sides deserve
special attention: they indicate that the debate failed to anticipate a consequence, and
that consequence is often the most informative result.

Generate figures at this stage. Each figure should illustrate exactly one finding.
Prefer figures that show distributions and uncertainty over figures that show point
estimates. A score distribution histogram that shows three classes tells a more complete
story than a single AUC number.

**Example prompt:**

> Use these results and the debate to create a detailed conclusions markdown file.
> Including any plots that help illustrate points.

**The most important result in this investigation** was not anticipated by either side of
the debate: cross-account in-vocab devices scored *higher* (more anomalous) than OOV
attack devices (AUC 0.375, inverted). Both sides spent considerable effort debating
whether OOV detection or account clustering dominated the signal. Neither predicted that
the two mechanisms would be ordered the way the data showed. Findings that surprise both
sides of a well-constructed debate are strong evidence that the experiment was measuring
something real.

---

## Step 8 — Write a coherent report

**What to do:** Synthesize the full arc — hypothesis, initial results, critique, defense,
debate, experiment, conclusions — into a single document that can be read without
reference to any of the intermediate files. The report should be self-contained: someone
who reads only the report should be able to understand what was claimed, what was tested,
what the evidence showed, and what should be built next.

Structure around the intellectual arc, not the chronological sequence. The reader does
not need to know that the critique came before the defense; they need to understand why
the final recommendation differs from the initial one and what evidence drove that
revision.

**Example prompt:**

> Finally, use the original hypothesis, experiment and results, critique, defense, debate,
> new experiment and results, and conclusion to generate a coherent final report.

**What to produce:** A document with abstract, introduction, methods, results, discussion,
and recommendations — standard scientific report structure. Every quantitative claim
should be traceable to a specific experimental result. The recommendation should be
clearly motivated by the findings, not asserted.

---

## Step 9 — Re-evaluate recommendations under production constraints

**What to do:** After the experimental work is complete and a recommendation has been
made, evaluate that recommendation against production constraints that were deliberately
excluded from the PoC. The most common omissions are:

- **Retraining dynamics:** How often must the model be retrained, and what happens to
  existing state (embeddings, centroids, cached scores) when it is?
- **Update latency:** How quickly can the signal respond to new information (new
  enrollments, confirmed fraud)?
- **Operational complexity:** What infrastructure is required to deploy, monitor, and
  maintain the recommended approach?
- **Failure modes:** What happens when the model is wrong, stale, or unavailable?

Production constraints frequently invert the ranking of candidate approaches. A model
with superior experimental AUC may be architecturally expensive to operate; a simpler
signal that appeared weaker in the experiment may be more robust in deployment.

**Example prompt:**

> Now let's take a production deployment perspective. Are FastText embeddings rotationally
> invariant? What happens to existing embeddings if I have to retrain a model? I would
> expect this model would have to be retrained frequently so that new legitimate devices
> are not constantly creating false positives. Re-evaluate final recommendation.

**What this revealed:** Word2Vec embedding spaces are not stable across training runs.
Each retraining produces a new, incompatible coordinate frame. Account centroids computed
from one model version are meaningless in the next. This means every retraining event
requires a full recompute of all account centroids from stored device histories —
a batch job that must complete before the new model can be used. The retraining is itself
driven by the enrollment pressure the model is trying to address (new legitimate devices
are OOV until retrained), creating a self-reinforcing operational constraint.

The experimental recommendation (Word2Vec centroid as the primary real-time signal) was
revised: the OOV binary flag — which is immune to coordinate frame instability and
updatable in milliseconds from a database set-membership check — should be the primary
real-time signal. The Word2Vec centroid should be a secondary offline batch feature,
computed on a slower cadence and decoupled from the real-time inference path.

**The rule:** If a recommendation requires infrastructure that cannot be described
concretely, the recommendation is incomplete. "Use a centroid-based approach" is not a
recommendation; "store full device histories per account, run a nightly recompute job
after each model retraining, gate deployment on centroid recompute completion, and use
the centroid score as a feature in an upstream risk model updated daily" is a
recommendation.

---

## The general pattern

```
Hypothesis
    │
    ▼
Minimal PoC ──────────────────────────────────────────┐
    │                                                  │
    ▼                                                  │
Clarify intent ◄── user correction if needed          │
    │                                                  │
    ▼                                                  │
Adversarial critique (first principles)                │
    │                                                  │
    ▼                                                  │
Defense (calibrate critique, identify strong vs weak)  │
    │                                                  │
    ▼                                                  │
Debate (multi-turn, to concession or testable claim)   │
    │                                                  │
    ▼                                                  │
Experiment (pre-specified verdicts, trivial baselines) │
    │                                                  │
    ▼                                                  │
Conclusions (per-finding verdicts, targeted figures)   │
    │                                                  │
    ▼                                                  │
Report (full arc, self-contained)                      │
    │                                                  │
    ▼                                                  │
Production re-evaluation ◄─────────────────────────────┘
    │
    ▼
Revised recommendation
```

Each step generates a concrete artifact. The artifacts are not documentation for its own
sake — they are the audit trail that makes the final recommendation defensible. When the
production re-evaluation in the last step overturns a finding from the experimental
step, the trail shows exactly why the experiment's conclusion was correct in its stated
scope and why the production constraint changes it.

---

## What this pattern is not

**It is not a waterfall.** Corrections and reversals at any step are expected and
healthy. The user correction at Step 2 (clarify intent) prevented the entire experiment
from testing the wrong hypothesis. The bug caught at Step 6 (run experiment) changed the
most important numerical result by nearly 0.5 AUC points.

**It is not complete without the trivial baseline.** The binary OOV lookup was the
control the debate required before the experiment could be considered valid. A model that
cannot outperform a two-line baseline is not a model — it is a dressed-up lookup table.

**It is not finished at the report.** The production re-evaluation step is not an
afterthought. It is the step where experimental findings collide with operational
reality, and it frequently produces the most actionable insight of the entire sequence.
In this investigation, it inverted the signal hierarchy in the final recommendation.

---

## Artifact inventory

| Artifact | Step | Role |
|----------|------|------|
| `ato_fasttext_poc.py` | 1 | Implements hypothesis as runnable code |
| `README.md` | 2 | Documents intent, quickstart, known limitations |
| `CRITIQUE.md` | 3 | Adversarial analysis from first principles |
| `DEFENSE.md` | 4 | Calibrated rebuttal; separates strong from weak critiques |
| `DEBATE.md` | 5 | Multi-turn argument to concession or testable prediction |
| `ato_experiment2.py` | 6 | Implements all debate-agreed empirical tests |
| `CONCLUSIONS.md` | 7 | Per-finding verdicts with supporting figures |
| `REPORT.md` | 8 | Self-contained report of the full arc |
| `REPORT_ADDENDUM.md` | 9 | Production re-evaluation; revised recommendation |
| `PROCESS.md` | — | This document |
