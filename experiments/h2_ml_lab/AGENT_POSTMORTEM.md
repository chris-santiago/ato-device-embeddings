# Agent Post-Mortem: Configuration Failure in H2 ml-lab Investigation

## What Failed

The ml-lab investigation ran the H2 experiment under a training configuration (sg=0 CBOW, per-event corpus) that causes within-feature embedding collapse (within-feature cosine similarity = 0.9993). This collapse made mean-pool spoof AUC = 0.384 — below chance — and reversed every finding. The investigation concluded H2 refuted when H2 is in fact confirmed under the appropriate configuration (sg=1 skip-gram, per-account corpus, spoof AUC = 0.818).

Neither the ml-critic nor the ml-defender caught this before the experiment ran.

---

## Timeline of Missed Opportunities

### Step 1: PoC Implementation

The PoC (`ato_device_embedding_poc.py`) used gensim's default FastText constructor without specifying `sg`. The default is `sg=0` (CBOW). H2_RERUN, the prior investigation that established the original H2 hypothesis, used `sg=1` with a per-account corpus. This divergence was present from the first line of training code.

**No agent checked whether the PoC configuration matched the prior investigation.** The PoC was treated as a fresh implementation rather than a replication of H2_RERUN under a new agent framework.

---

### Step 3: Critique

The ml-critic raised several concerns in the right neighborhood but missed the failure entirely.

**C5.1 — Context richness mismatch** is the closest near-miss:

> "Concat model trains on single-token sentences; mean-pool model trains on 6-token sentences... FastText is a context-window model: in mean-pool training, each token sees 5 other feature tokens as context in every sentence. In concat training, each token has no context at all... This means the mean-pool model is trained on dramatically richer context signal than the concat model."

This correctly identifies the per-event vs. per-token corpus difference as a confound. But the critique frames it as a confound that might *amplify* a real signal, not as a mechanism that could *collapse* the embedding space. The critic proposed a window sweep to control for context richness, not an inspection of whether training dynamics had produced degenerate representations.

**C1.2 — Mean-pool co-occurrence bias** is the second near-miss:

> "Compute the cosine similarity matrix between all 30 feature token embeddings in the trained mean-pool model. If `os_ios` and `browser_safari` have high cosine similarity (despite being different feature types), co-occurrence bias is present."

The test was motivated by the question: *do cross-feature token pairs accidentally become similar?* The actual failure is the inverse: *do within-feature token pairs become identical?* `tz_utc-8` and `tz_utc+8` having cosine similarity 0.9993 is not co-occurrence bias — it is a direct consequence of CBOW's training objective when applied to a per-event corpus. The critic proposed T8 to catch the wrong phenomenon.

**What the critic did not ask:**
- What training objective does the PoC use, and does it match H2_RERUN?
- Under CBOW, what happens when all values within a feature dimension share the same co-occurring context tokens?
- Is there a class of input structure — like fixed-slot 6-token sentences — that produces degenerate CBOW embeddings?

---

### Step 4: Defense

The ml-defender's response to C5.1 compounded the miss:

> "Partially rebutted, but the confound is real. The experiment should control for this. One approach: evaluate the models with `min_count=1, workers=1, epochs=10` and verify that mean-pool token embeddings are not simply better because they were trained on more sentences with context. Empirically testable via the window sweep (C1.3)."

The window sweep does not address whether CBOW + per-event produces within-feature collapse. Varying window size (w=1, 2, 3, 6) changes how many tokens are treated as context, but does not change the training objective. Under CBOW, within a 6-token per-event sentence, all six tz values (`utc-8`, `utc-5`, `utc+0`, etc.) appear as the center token in sentences where the exact same five context tokens appear. The CBOW objective converges all tz values to the same embedding regardless of window size, because the conditional context distributions are identical by construction.

The defender had access to the H2_RERUN experiment design — the prior investigation that H2_RERUN used `sg=1, per-account corpus, epochs=20, negative=10`. The defense cited H2_RERUN results in C1.1:

> "The prior investigation (H2_RERUN) ran a timezone-permutation test where every reordering of the tz feature position in the concat string made spoof AUC worse. This is direct evidence that boundary effects are non-negligible."

But the defender used H2_RERUN as evidence for the mechanism claim (n-gram contamination), not as a specification of the training configuration that the PoC must replicate. The configuration difference was visible — H2_RERUN's sg=1 + per-account corpus was known to the defender — but was not treated as a constraint on the PoC.

---

### Step 5: Debate

The debate produced T8 as an agreed empirical test:

> "Compute pairwise cosine similarity for all 30 feature tokens in trained mean-pool model. Check whether cross-feature pairs (e.g., os_ios vs. browser_safari) have higher similarity than expected from a semantics-free baseline."

This is the correct diagnostic tool for the wrong failure mode. The test was designed to catch co-occurrence-induced *cross-feature* similarity. It happened to catch *within-feature* collapse when the experiment ran (within-feature sim = 0.9993), but only because the T8 implementation in `experiment3.py` computed both within-feature and cross-feature similarity distributions. The T8 test in the debate's resolution table did not specify that within-feature similarity should be checked — the winning condition was "cross-feature similarity is low relative to within-feature."

The T8 result (within-feature sim = 0.9993) was reported in the experiment output, and the defender identified it post-hoc as the root cause. But this was a retrospective explanation, not a prospective catch.

---

## Root Cause Analysis

### Root Cause 1: No Configuration Specification Before the PoC

Neither the hypothesis nor any pre-experiment artifact specified which training objective or corpus construction the PoC must use. `HYPOTHESIS.md` listed AUC and silhouette as success metrics but did not constrain how the model must be trained. The critic reviewed the PoC's *outputs* (AUC, silhouette) and *design choices* (window, n-gram ranges, attack construction) but did not audit the *training configuration* against H2_RERUN's known working setup.

**The gap:** An investigation that repeats a prior experiment under a new agent framework must include a configuration audit step — verifying that the new implementation matches the prior one on all parameters known to affect the result.

### Root Cause 2: The Critic Scoped T8 to the Wrong Failure Mode

The critic's C1.2 proposed measuring cross-feature token similarity to catch co-occurrence bias. This is a legitimate concern, but it is a second-order concern (feature tokens becoming similar to each other across dimensions). The first-order concern for mean-pool — that tokens within a dimension become identical — was never raised.

The CBOW within-feature collapse mechanism is not obvious from an NLP background. In NLP applications, CBOW collapse of this kind does not occur because natural language words within a syntactic category (e.g., all nouns) do not share identical context distributions. In the ATO device embedding context, the structure is reversed: by construction, `tz_utc-8` and `tz_utc+8` appear in exactly the same distributional context (the same OS, browser, language, network, screen values from the same event). The critic's NLP intuition did not extend to this structured, non-linguistic domain.

### Root Cause 3: The Defender Treated Configuration as a Confound Rather Than a Constraint

The defender's role is to argue for the implementation. But the defender accepted gensim's CBOW default as "the implementation" rather than checking whether it matched H2_RERUN's configuration. The correct defensive posture would have been:

> "The PoC must use sg=1 (skip-gram) and per-account corpus construction to match H2_RERUN, which is the prior investigation that established the expected direction of results."

Instead, the defender treated the CBOW/per-event implementation as the baseline and argued about whether its results were valid.

### Root Cause 4: The Debate Produced No Pre-Flight Checklist

The debate resolved 10 contested points and produced 8 empirical tests. None of the tests included: "verify that the training configuration does not produce within-feature embedding collapse." T8 was added to the test list, but its success condition was framed as a diagnostic of co-occurrence bias, not as a go/no-go gate on whether the mean-pool model is even functional.

A pre-flight check — "verify within-feature sim < 0.5 before proceeding with any mean-pool result interpretation" — would have immediately flagged the collapse (sim = 0.9993) and prevented the incorrect H2 refutation verdict.

---

## What Would Have Caught This

**In Step 1 (PoC):** An implementation checklist requiring that all hyperparameters match the prior reference implementation (H2_RERUN: `sg=1, per-account corpus, epochs=20, negative=10, min_n=3, max_n=6, window=6`) before the PoC is considered complete.

**In Step 3 (Critique):** A diagnostic question targeted at the training objective: "Under CBOW training with a fixed-slot positional structure, do within-feature token pairs converge to the same embedding?" This requires domain knowledge specific to how CBOW behaves on structured non-linguistic corpora — it is not in the standard NLP critique repertoire.

**In Step 5 (Debate):** A within-feature similarity go/no-go gate as a prerequisite for any mean-pool result: T8 success condition = "within-feature sim < 0.9." This condition would have been violated (sim = 0.9993), requiring the investigation to halt and debug before reporting any verdicts.

---

## Generalizable Lessons for Future ml-lab Investigations

### Lesson 1: Configuration audits are mandatory for replication investigations

If the investigation is framed as a replication or extension of a prior experiment, the PoC must include an explicit configuration audit: list every hyperparameter of the prior investigation and verify the new implementation matches on each. Divergences must be flagged as intentional or unintentional before proceeding.

**Practical form:** Add a "Configuration Baseline" section to `HYPOTHESIS.md` for any replication investigation, listing the reference configuration and requiring that the PoC explicitly justify any divergence.

### Lesson 2: T8-style prerequisite checks should precede result interpretation, not accompany it

A within-feature similarity diagnostic is not merely an informative test — it is a precondition for mean-pool results being interpretable. When the mean-pool hypothesis assumes that tokens within a dimension are differentiated, verifying that assumption must happen before claiming AUC results support or refute the hypothesis.

**Practical form:** For any mean-pool embedding investigation, require a "Prerequisite: Embedding Quality" section before the results section that reports within-feature similarity. If within-feature sim > 0.5, all mean-pool results are conditional on a degenerate embedding space and must be labeled as such.

### Lesson 3: Critics should audit structured non-linguistic corpora against CBOW collapse

The CBOW within-feature collapse failure mode generalizes beyond ATO device embeddings. Any application of CBOW to structured categorical sequences — where values within a dimension share identical co-occurrence contexts — is susceptible. User agent strings, product attribute tuples, transaction feature vectors, and similar structured sequences all have this property. The critic's standard checklist for NLP applications does not include this.

**Practical form:** Add to the critic's checklist for mean-pool + FastText/word2vec investigations: "Do all values within any feature dimension share identical co-occurrence contexts in the training corpus? If so, CBOW training will collapse within-feature embeddings regardless of vocabulary size."

### Lesson 4: The defender's role includes defending the configuration, not just the results

The defender correctly argued for the mechanism (n-gram contamination, tz-counterfactual, compactness). But defending the implementation means verifying that the implementation is sound before arguing for its results. A defender who accepts a broken implementation and argues its results are valid is not performing the defensive function.

**Practical form:** Add to the defender's Step 4 checklist: "Does the PoC implementation match the reference configuration for this hypothesis? List any divergences."

---

## Recommended Agent Changes

Each root cause maps to a change in an agent definition. These are general principles applicable to any ML experiment.

### ml-lab.md

**Step 1 (Build the PoC) — add:**

> Before writing any code, identify any reference implementation this PoC must match. If one exists, record its configuration explicitly. Framework and library defaults are never a safe assumption — they are the most common source of silent divergence from a reference. Any parameter not explicitly set is a potential source of failure.

**Step 6 (Design and Run the Experiment) — add:**

> Before interpreting any result, verify that the model satisfies the preconditions the hypothesis depends on. If the hypothesis claims a model is sensitive to a particular signal, confirm the model actually encodes that signal before treating outcome metrics as meaningful. A model can look healthy on aggregate metrics while being completely blind to the specific discriminative requirement the hypothesis targets. Failed preconditions halt result interpretation — do not report verdicts from an unverified model.

---

### ml-critic.md

**"What to critique" — add:**

> - **Silent misconfiguration:** Ask whether the implementation could be misconfigured in a way that produces plausible-looking results on easy cases while failing on the specific cases the hypothesis targets. Aggregate metrics passing is not evidence that the model is functional for the hypothesis's hardest requirement. Look for configurations — including framework defaults — that would cause the model to degrade silently on the targeted signal without producing obvious errors or metric collapse.
> - **Prerequisite assumptions:** Identify any property the model must have for the hypothesis's mechanism to operate. These are not evaluation metrics — they are preconditions. If a precondition is not verified before the experiment runs, the experiment cannot produce an interpretable verdict.

---

### ml-defender.md

**Pass 1 — add as first item:**

> **Implementation soundness check (before all other analysis):** Before defending any design choice, verify that the implementation is sound enough to produce interpretable results. Check that all parameters are explicitly set and appropriate for this problem, not inherited from defaults designed for a different use case. If the implementation has a configuration flaw that would silently invalidate the results, identify it here — defending results from a flawed implementation is not a defense of the design.

---

## Verdict on Agent Failure Mode

This failure is a **specification gap**, not a reasoning failure. Both the critic and defender produced valid, well-reasoned analyses within their respective frames. The critic correctly identified context richness as a confound and proposed a sensible diagnostic (T8). The defender correctly rebutted several weaker critiques. The debate produced a sound set of empirical tests.

The failure is that neither agent was equipped to ask: *does this implementation produce degenerate embeddings?* That question requires domain knowledge about CBOW collapse in structured corpora — knowledge that is not in the standard ML critique or defense repertoire, and that is not obviously prompted by reviewing AUC results or attack construction.

The gap is remediable by adding configuration audits and prerequisite embedding quality checks to the front of the investigation workflow, before any result interpretation begins.
