# Is FastText the Right Way to Embed Device Attributes for ATO Detection?

This repository asks one question and answers it with pre-specified, seeded,
controlled experiments: **is a FastText embedding over login device attributes
an appropriate anomaly signal for account-takeover detection? If so, how, when,
and why. If not, what wins.**

The short answer: **not as usually proposed, and never below a vocabulary-size
threshold.** The value in the architecture comes from per-feature tokenization,
per-account corpus construction, and centroid cosine scoring. It does not come
from FastText's distinguishing feature: character n-gram subwords measurably
hurt in-vocabulary discrimination and are not repaid by out-of-vocabulary
robustness. Measured directly, a per-feature fallback vector (which
field-tokenized telemetry always affords) matches or beats subword composition
for arbitrary novel values, the realistic case for device attributes whose codes
carry no informative character structure. Subwords hold only a directional,
not-robust edge under morphological drift on versioned/structured fields. On small vocabularies a smoothed counting baseline
beats every embedding configuration tested; on realistic open vocabularies at
1:100 imbalance, plain token embeddings earn their place. Every number below is
a 5-seed mean ± std from the reproducibility record in `experiments/rerun/`.

---

## When an embedding is justified at all

The decision is a memorization-to-generalization crossover, measured at two
points:

| Vocabulary regime | Winner | Spoof detection |
|---|---|---|
| Closed, ~30 tokens | Per-feature likelihood (counting) | likelihood 0.895 ± 0.006 ROC vs embedding 0.868 ± 0.012 |
| Open, 216–240 tokens, 1:100 imbalance | Plain token embedding | embedding 0.888 ± 0.026 PR vs likelihood 0.766 ± 0.011 |

While per-account value counts are dense, counting wins: thirty familiar tokens
are a lookup problem, and a distributed representation adds nothing. When the
vocabulary grows and counts thin out, the embedding's cross-feature structure
becomes the signal, and the gap is decisive on PR-AUC (invisible on ROC-AUC,
0.995 vs 0.992, which is itself a lesson about metrics at 1:100). The crossover
lies somewhere between these two points; benchmark both on your own vocabulary
before choosing.

**What wins otherwise.** The incumbent is a smoothed per-account, per-feature
categorical likelihood with global backoff:
`score(event) = −Σ_f log[λ·P̂(v_f | account) + (1−λ)·P̂(v_f | global)]`,
Laplace-smoothed. It is CPU-cheap, label-free, needs no training loop, and
detects training-contaminated attacker devices about as well as the embedding
does (top-1% precision 0.53 vs 0.49). Any embedding proposal in this tier that
does not report this baseline is unfinished.

---

## Refutations

Each row is a tempting claim about this architecture, tested and rejected. The
rules that survive are the design in the next section.

| # | Tempting claim | Verdict |
|---|---|---|
| 1 | Embed the concatenated device string | Structurally broken |
| 2 | The embedding learns device semantics | Mostly set-overlap geometry |
| 3 | Subwords add robustness for free | They cost in-vocab accuracy and aren't repaid — a per-feature fallback matches or beats them under OOV |
| 4 | Skip-gram is mandatory | The corpus is causal, the objective is not |
| 5 | Per-event corpora destroy embeddings | True only for weakly-coupled features |
| 6 | Gate alerts on known devices | Suppresses exactly the attacks that matter, in any scoring family |
| 7 | Rank-normalize per account | Collapses precision at thin calibration windows |
| 8 | It beats the baseline | Against the wrong baseline and the wrong metric |

**1 — Concatenated device strings.** Joining attributes into one token
(`ios_chrome_utc+0_...`) scores 0.737 ± 0.006 spoof ROC, below the trivial
set-membership baseline (0.750) on every seed: cross-boundary character n-grams
dilute a single changed field below detectability. The failure is
representational, not a training artifact. A frozen random vector per device
string scores at the trivial baseline (0.744), and without subwords the encoder
cannot score unseen devices at all (93% of spoof events are OOV as whole
strings) and degenerates to trivial (0.752). Concatenation depends on exactly
the mechanism that poisons it.

**2 — "It learns device semantics."** Frozen random per-token vectors, no
training, score 0.836 ± 0.014 spoof ROC: about 73% of the trained encoder's
advantage over trivial is untrained set-overlap geometry (a changed field moves
a mean-pooled vector by ~1/6 no matter what training did). Training adds real
signal, but the as-proposed FastText config beats its own random floor by only
+0.032 ROC. The honest claim is geometry plus learned account co-occurrence,
nothing stronger.

**3 — Subwords.** Disabling character n-grams (`max_n=0`, plain token
embeddings, everything else identical) *improves* spoof detection to
0.887 ± 0.007 ROC / 0.819 ± 0.011 PR. The subword-vs-none delta is negative on
4/5 seeds and positive on none, because shared feature prefixes (`os_`, `tz_`)
pull same-feature tokens together and blunt the separation the scorer needs.
Only the no-subword encoder beats the random floor on all seeds (+0.051 ROC,
+0.111 PR). The OOV robustness subwords are purchased for is now measured
directly (`oov_regime.py`) against the encoder this design actually recommends
(`max_n=0` plus a per-feature fallback vector), not against the likelihood
incumbent, and it does not favor subwords. Novel tokens are injected into both
benign and attack events at 1:100 imbalance on the `region` feature, split by
whether the novelty is morphological. Under **arbitrary** novel codes (the
realistic case for geo/network attributes, where a new region or ASN shares no
informative characters with any seen value) the fallback beats subwords by
+0.19 PR-AUC at full drift (CI-robust): FastText composes an arbitrary token
from meaningless n-gram hashes and scatters the benign class into false
positives, while the fallback maps every unseen value of a feature to one stable
vector. Under **morphological** drift (`region_viken` → `region_viken_v577`,
shared stem) subwords progressively overtake the fallback: a clean monotonic
crossover around p ≈ 0.5, reaching +0.085 PR at full drift. This is the one
regime where the subword mechanism earns its keep, though the per-seed CI at
each level still crosses zero, so it is a directional edge, not a robust margin.
Subwords start behind in-vocabulary (−0.076 PR at p=0). Net: for device
attribute codes, which have no morphology for n-grams to exploit, a fallback
vector dominates subword composition in-vocabulary and under arbitrary drift, and
`max_n=0` is the right default there. Subwords earn their cost only on genuinely
morphological fields (version strings, structured IDs), where the crossover above
gives them a directional edge over the fallback.

**4 — "Always use skip-gram."** The 2×2 factorial (objective × corpus) shows
corpus construction is the causal axis of embedding health. Both per-event
cells collapse on all seeds regardless of objective; both per-account cells are
healthy regardless of objective, and CBOW + per-account is nominally the
strongest spoof scorer of the four (0.933 ± 0.009). Per-account corpora are
mandatory. The training objective is a tuning choice.

**5 — Within-feature collapse.** One-sentence-per-event corpora make all values
of a feature converge to near-identical vectors (within-feature cosine 0.94 to
0.98), which silently destroys the hardest attack subtype while novel and fleet
detection stay high. The mechanism is context-distribution overlap, and it has
a precondition: the other features must be uninformative about the collapsed
one. On data with realistic feature coupling (browser|OS, region|country), the
pooled collapse disappears (0.758 ± 0.011, below threshold on every seed) and
only the one independently-sampled feature collapses (0.961, the outlier on 5/5
seeds). Benchmarks that sample features independently overstate this failure
mode. If you monitor within-feature cosine as a health check, use per-feature
baselines: shared-prefix n-gram inflation puts healthy features above any
sensible global threshold.

**6 — The known-device gate.** Suppressing alerts on devices seen in the
training window produces zero top-1% true positives on accounts whose training
history contains an attacker device, on every seed, with ROC exactly equal to
trivial. This is structural: an attacker admitted into the history is by
definition "known". The trap is binarization, not the scorer. Raw cosine
detects the contaminated-fleet population at top-1% precision 0.49 ± 0.12, the
likelihood incumbent at 0.53 ± 0.14, and wrapping *either* in the gate zeroes
it. Never binarize trust on training-window membership; confirmed-abuse
blocklists belong upstream of scoring and are orthogonal to it.

**7 — Per-account rank-normalization.** Converting cosine distances to
per-account percentile ranks collapses spoof PR-AUC from ~0.89 to ~0.22 at the
20-event calibration windows realistic for new and low-frequency accounts. The
rank floor is quantized at 1/N_calib, so precision recovers only once windows
reach hundreds of events (`experiments/rerun/calib_sweep/`). Score with raw
cosine distance.

**8 — Baselines and metrics.** Exact set-membership is a floor, not an
incumbent: any proposal must beat the smoothed likelihood scorer (see above) on
its actual vocabulary regime, and must report PR-AUC at deployment imbalance,
stratified by attack population. Every failure in this table is invisible to
aggregate held-out ROC-AUC: the concat failure needs spoof stratification, the
collapse preserves the easy subtypes, the gate failure lives in a population
that standard splits average away, and the regime reversal in the first section
appears only on PR.

---

## The design that survives

```
tokens      = [f"{feature}_{value}" for each attribute]   # per-feature tokens, never concatenated
corpus      = one sentence per account (all login events concatenated)
model       = FastText(vector_size=64, window=6, epochs=20, negative=10,
                       max_n=0,            # plain token embeddings; see refutation 3 before changing
                       sg=0 or 1)          # tuning choice; see refutation 4
event_vec   = mean(token vectors)
risk_score  = cosine_distance(event_vec, account_centroid)

unseen value  → per-feature fallback vector (field is always known to the tokenizer)
no known-device gate; no per-account rank-normalization
health check  → within-feature cosine vs per-feature baselines recorded at first healthy training
evaluation    → PR-AUC at deployment imbalance, stratified by attack type,
                vs trivial membership AND the smoothed likelihood incumbent
```

Deploy this only in the vocabulary regime where it beats the likelihood
incumbent on your data. Below that regime, deploy the incumbent.

**Replication check.** On the public RBA dataset (real-world feature
distributions, temporal split), the embedding beats the trivial baseline on all
seeds (ROC 0.852 ± 0.029 vs 0.679 ± 0.046). With n = 9 labeled ATO test events
this is a directional check only, stable across split-sensitivity variants but
not a powered evaluation.

---

## Quickstart

Scripts are self-contained with inline dependencies
([PEP 723](https://peps.python.org/pep-0723/)); run with `uv`, no virtualenv.

```bash
# Full 5-seed reproducibility record (H2 encoder comparison, H6 system-level, RBA replication)
bash experiments/rerun/run_all.sh

# The controls behind the refutations (random floor, likelihood incumbent,
# no-subword cell, correlated-marginals collapse)
bash experiments/rerun/ablations/run_ablations.sh

# Single pieces
uv run experiments/rerun/scripts/h2/h2_rerun.py --seed 42     # encoder comparison, one seed
uv run experiments/rerun/ablations/a4_nosub.py --smoke        # subword ablation, fast check
```

---

## Where the evidence lives

| Claim | Evidence |
|---|---|
| Refutations 1–3 (concat, random floor, subwords) | `experiments/rerun/scripts/h2/`, `experiments/rerun/ablations/` (`baseline_controls.py`, `a4_nosub.py`) |
| Subword vs fallback under OOV (refutation 3, measured) | `experiments/rerun/ablations/oov_regime.py`, `SUMMARY.md` (OOV section) |
| Refutation 4 (factorial) and 5 (collapse + scoping) | `experiments/rerun/scripts/h2/h2_rerun.py`, `experiments/rerun/ablations/h6_perevent_collapse.py` |
| Refutation 6 (gate, cross-family) | `experiments/rerun/scripts/h6/`, `experiments/rerun/ablations/h6_likelihood_incumbent.py` |
| Refutation 7 (rank-norm) | `experiments/rerun/calib_sweep/` |
| Vocabulary-regime comparison | `experiments/rerun/ablations/SUMMARY.md` |
| Aggregated numbers (5-seed) | `experiments/rerun/aggregate/`, `experiments/rerun/ablations/aggregate/` |
| RBA replication | `experiments/rerun/scripts/rba/`, `experiments/h2_rba/` |
| Full synthesis | `TECHNICAL_REPORT.md`, `DEVICE_EMBEDDING_FINDINGS.md` |
| Original investigations (historical, single-run) | `experiments/pre_ml_lab/`, `experiments/h2_ml_lab/`, `experiments/h3_pfn/`, `experiments/h4_gru/`, `experiments/h5_stress/`, `experiments/h6_hybrid/` |

The `experiments/rerun/` record is authoritative; the per-hypothesis
directories preserve the original investigations and may contain superseded
single-run numbers.
