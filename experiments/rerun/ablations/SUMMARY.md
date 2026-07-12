# Critique Ablations — Results (A1, A2a, A2b, A3p, A4, OOV)

**Plan:** `plan/12-CRITIQUE_ABLATIONS.md` (A1–A4/A3p pre-specified before their run;
the OOV section was added 2026-07-12, not part of plan 12).
**Scripts:** `baseline_controls.py`, `h6_perevent_collapse.py`,
`h6_likelihood_incumbent.py`, `a4_nosub.py`, `oov_regime.py` (+ `oov_figures.py`);
seeds {42, 123, 456, 789, 2024}; aggregates in `aggregate/ablations_summary.{json,csv}`.

## Headline

The original positive claim (C1 as stated: FastText mean-pool as the recommended
encoder) does not survive its controls — but the full ablation set supports a
*revised* positive recommendation rather than a pure negative result:

1. **A1:** Frozen random embeddings reach spoof ROC 0.836 ± 0.014 — covering ~73%
   of trained mean-pool's advantage over the trivial baseline on the closed-vocab
   config. As-proposed FastText adds only +0.032 ROC over no training at all.
2. **A4:** The reason is the subword machinery: disabling it (max_n=0, plain
   token embeddings) *improves* spoof ROC to 0.887 ± 0.007 (subwords hurt on 4/5
   seeds, help on 0/5), and no-subword training beats frozen random on **5/5
   seeds** (+0.051 ROC, +0.111 PR). Training works; subwords were masking it.
3. **A2a:** On the small closed vocabulary, the Freeman-style likelihood
   incumbent beats the trained embedding on every seed (0.895 vs 0.868 ROC).
4. **A2b:** On the realistic open-vocab 1:100 configuration the ranking
   *reverses*: mean-pool beats the incumbent's best variant on spoof_k1 on 5/5
   seeds (PR 0.888 ± 0.026 vs 0.766 ± 0.011; paired delta CI-positive), while
   the incumbent *also* detects fleet contamination (TP 100 ± 34 vs gate's 0) —
   C3's suppression is a **binarization trap that transfers across scoring
   families** (the gated likelihood also scores 0 TP).
5. **A3p:** Per-event collapse severity is independence-driven: on correlated
   RBA marginals, pooled SG collapse disappears (0.758 vs 0.939 independent),
   and the per-feature gradient tracks feature coupling exactly.
6. **OOV:** Under 1:100 imbalance, *symmetric* OOV drift does not collapse either
   model on ROC (both hold ≥0.96; the trivial baseline dies to chance), but the
   incumbent's **precision** craters on arbitrary high-cardinality OOV (PR
   0.76 → 0.26) while mean-pool holds (0.84 → 0.50) — dPR up to +0.31, a gap ROC
   understated ~15×. A same-feature disambiguation splits mean-pool's edge into
   **dilution** (+0.19 dPR, even on unrelated novelty) plus **subword recovery**
   (+0.27 dPR when the novel value is a morphological variant). This complicates
   A4: subwords hurt in-vocab discrimination but *help* under morphological OOV.

## A1 — Frozen random embedding control

| Scorer | Spoof ROC-AUC | Spoof PR-AUC |
|---|---|---|
| mp_trained | 0.868 ± 0.012 | 0.787 ± 0.018 |
| mp_random (frozen N(0,1)) | 0.836 ± 0.014 | 0.712 ± 0.025 |
| cat_random (random device-hash) | 0.744 ± 0.002 | 0.494 ± 0.009 |
| trivial | 0.750 | 0.500 |

Verdicts: `a1_trained_beats_random_pr` **5/5**; `a1_trained_beats_random_roc`
**4/5** (seed 42 ROC delta CI crosses zero: +0.008 [−0.006, +0.023]).

Reading: training is not inert — the paired PR delta is CI-positive on every
seed — but the frozen-random floor shows most of mean-pool's absolute
performance is set-overlap geometry ((0.836 − 0.750)/(0.868 − 0.750) ≈ 0.73).
Per the plan's criterion (both verdicts 5/5), **A1 fails**: C1 cannot be
presented as "FastText embedding quality" without attributing ~73% of the
effect to geometry that needs no training. `cat_random` ≈ trivial confirms the
concat pathology is representational, not a training artifact.

## A2a — Likelihood incumbent

| Scorer | Spoof ROC-AUC | Spoof PR-AUC |
|---|---|---|
| lik λ=0.5 (acct/global backoff) | **0.895 ± 0.006** | **0.841 ± 0.009** |
| lik λ=0.9 | 0.870 ± 0.007 | 0.801 ± 0.011 |
| lik λ=1.0 (pure per-account) | 0.851 ± 0.007 | 0.765 ± 0.011 |
| mp_trained | 0.868 ± 0.012 | 0.787 ± 0.018 |

Best variant per seed: λ=0.5 on all 5 seeds. Verdicts:
`a2_mp_beats_likelihood_roc` **0/5**, `a2_mp_beats_likelihood_pr` **0/5**.

Reading: the CPU-cheap, label-free, industry-standard smoothed per-feature
likelihood **beats the proposed architecture on the investigation's own primary
metric** on this closed 30-token vocabulary. See A2b: the ranking reverses on
the open-vocab realistic configuration — the regime where the embedding wins is
exactly the larger-vocabulary regime.

## A2b — Likelihood incumbent at 1:100 (H6 open vocabulary)

Best likelihood variant by spoof_k1 ROC: λ=1.0 (pure per-account) on all 5
seeds. All three pre-specified verdicts **5/5**.

| Scorer | spoof_k1 ROC | spoof_k1 PR | fleet_residual top-1% TP | precision |
|---|---|---|---|---|
| mp_raw | **0.995 ± 0.001** | **0.888 ± 0.026** | 91 ± 23 | 0.493 ± 0.116 |
| lik λ=1.0 | 0.992 ± 0.001 | 0.766 ± 0.011 | **100 ± 34** | 0.534 ± 0.144 |
| two_stage (gate) | — | — | 0 (all seeds) | 0.000 |
| lik λ=1.0 gated | — | — | 0 (all seeds) | 0.000 |

Paired spoof_k1 delta (mp_raw − lik_best): ROC +0.003 (CI lower min +0.0002),
PR **+0.122** (CI lower min +0.052) — CI-positive on every seed.

Reading: two findings. (1) **The closed-vocab incumbent advantage reverses on
realistic open-vocab data** — the embedding earns its place in the regime the
architecture was actually proposed for, with the PR gap (+0.12) being the
operationally meaningful one at 1:100. (2) **C3 is a binarization trap, not an
embedding property**: the smooth likelihood detects fleet contamination about
as well as raw cosine (a single training appearance is a low count, not an
allow-list pass), and wrapping *either* scorer in a binary known-device gate
zeroes it. The operational lesson is "never binarize on training-window
membership," independent of scoring family.

## A4 — No-subword cell (closed vocabulary)

| Scorer | Spoof ROC-AUC | Spoof PR-AUC |
|---|---|---|
| mp_nosub (max_n=0) | **0.887 ± 0.007** | **0.819 ± 0.011** |
| mp_trained (subwords) | 0.868 ± 0.012 | 0.787 ± 0.018 |
| mp_random | 0.836 ± 0.014 | 0.712 ± 0.025 |
| cat_nosub | 0.752 ± 0.006 | 0.502 ± 0.006 |
| trivial | 0.750 | 0.500 |

Verdicts: `a4_subwords_help_mp_roc` **0/5**; `a4_subwords_hurt_mp_roc` **4/5**
(the fifth CI spans zero); `a4_training_helps_nosub_roc` and `_pr` **5/5**
(mp_nosub − mp_random: +0.051 ROC, +0.111 PR, CI lower min +0.020).

Reading: **subwords are a net negative for in-vocabulary discrimination on
per-feature tokens** (their OOV benefit is not exercised by any configuration
here — a deliberate scope limit; an OOV stress test was considered and
declined 2026-07-11) —
shared feature-prefix n-grams (`os_`, `tz_`) pull same-feature tokens together
(within-feature cosine 0.424 with subwords vs 0.330 without) and blunt the
learned separation. With the confound removed, training beats frozen random on
every seed: the A1 near-tie was a subword artifact, not evidence that learning
fails. `cat_nosub` collapses to the trivial baseline (0.752 ≈ 0.750) because
92.7% ± 1.3% of spoof events are OOV as whole strings — concat *requires*
subwords to score unseen devices, which is precisely the dependency that
creates the C1 contamination pathology.

## OOV — Deliberate out-of-vocabulary stress test (open vocab, 1:100)

**Not pre-specified in plan 12** — added 2026-07-12 to run the OOV stress test A4
deferred ("an OOV stress test was considered and declined 2026-07-11"). Genuinely-OOV
tokens are injected into **both** benign and attack eval events at level `p`; the
training corpus and likelihood tables are kept OOV-free by construction, so the
incumbent floors every novel value to its smoothing constant while FastText composes
an OOV vector from character n-grams. 1:100 imbalance (~1% base rate, auto-scaled to
100 accounts). Four arms cross OOV construction (morphological version-suffix vs
arbitrary random string) × feature target (`os`/`browser`, `region`/`asn`, and
`region`-only for the disambiguation). Figures: `figures/oov_degradation_*_{roc,pr}_auc.png`,
`figures/oov_delta_attribution.png`.

### Symmetric OOV does not collapse either model on ROC

At every level both `mp_raw` and the incumbent hold spoof_k1 ROC ≥ 0.96; the trivial
set-membership baseline is the sole casualty (PR 0.108 → 0.010 ≈ chance; ROC → 0.500)
because any injected token breaks the `device_key`. The smoothing floor and the
mean-pool average both hit benign and attack events alike, so *symmetric* drift barely
moves ranking. ROC is the wrong lens here — the effect lives in precision under imbalance.

### PR-AUC under imbalance is where the models separate

| Arm (spoof_k1) | mp_raw PR  p=0 → p=1 | incumbent PR  p=0 → p=1 | dPR @ p=1 |
|---|---|---|---|
| arbitrary (region/asn, random) | 0.837 → **0.498 ± 0.083** | 0.764 → **0.258 ± 0.046** | **+0.241** |
| morphological (os/browser, version) | 0.837 → 0.895 ± 0.027 | 0.764 → 0.932 ± 0.008 | −0.037 |

On the realistic arm (arbitrary novelty on high-cardinality geo/network) the incumbent's
precision **craters** (0.76 → 0.26) while mean-pool holds (0.84 → 0.50); dPR peaks
**+0.310** at p=0.5. Over the same sweep the ROC gap peaked at only +0.017 — **ROC
understated the operational gap ~15×.** On the morphological arm the OOV lands on
low-cardinality, non-discriminative `os`/`browser`: the floored feature becomes a
constant for both classes (deletes a noisy dimension), both models *improve*, and the
incumbent edges ahead at p=1 (dPR −0.037).

### Same-feature disambiguation: dilution + subword recovery

Running both constructions on the *same* feature (`region`) removes the
feature-discriminativeness confound:

| region arm (spoof_k1) | mp_raw PR @ p=1 | incumbent PR @ p=1 | dPR @ p=1 |
|---|---|---|---|
| region_morph (version-suffix) | **0.746 ± 0.056** | 0.284 ± 0.049 | **+0.462** |
| region_arb (random string) | 0.473 ± 0.119 | 0.284 ± 0.049 | +0.189 |

The incumbent floors **identically** (PR 0.284) regardless of morphology — it cannot
tell `region_viken_v577` from `region_x9f3q`. The entire difference is FastText's: it
retains the value under morphological OOV (subword composition places the variant next
to the trained token) but loses it under arbitrary OOV. Attribution
`gap(region_morph) − gap(region_arb)` = **+0.273 ± 0.092 dPR at p=1**, strictly positive
and growing. So mean-pool's OOV advantage decomposes into two additive parts:
**dilution / floor-avoidance** (+0.19 dPR even on arbitrary OOV — one corrupted feature
of seven, and the shared `region_` prefix keeps it roughly placed) plus **subword
recovery** (a further +0.27 dPR when the novel value is a morphological variant). The
*cross-feature* attribution is negative (−0.278) and **must not be used** — it conflates
morphology with which feature carries signal.

Reading: **this complicates A4.** A4 found subwords a net negative for *in-vocabulary*
discrimination (max_n=0 improves spoof ROC). The OOV test shows the other side of the
same trade: subwords are net-*positive* under *morphological* OOV drift, recovering
signal the categorical incumbent floors identically. The subword machinery *looks*
like a regime-dependent trade against the incumbent — a liability on stable
vocabularies, an asset when novel values are variants of known ones. But that
comparison uses the wrong baseline: measured against the design's recommended
`max_n=0` + per-feature fallback encoder (next subsection), the subword benefit does
not survive.

### Subword composition vs the recommended fallback encoder

The +0.27 dPR "subword recovery" above is measured against the *incumbent*, which
floors OOV. The design does not recommend the incumbent — it recommends `max_n=0`
plus a per-feature fallback vector (`mp_nosub`: unseen value → mean of that feature's
trained token vectors). Measured against *that* baseline, subwords do not win:

| region arm (spoof_k1 PR) | subword (mp_raw) | fallback (mp_nosub) | dPR (mp−nosub) @ p=1 |
|---|---|---|---|
| in-vocab (p=0, both arms) | 0.837 | **0.914** | −0.076 (fallback; the A4 effect) |
| region_arb (arbitrary) | 0.473 | **0.662** | −0.189 (fallback, CI-robust) |
| region_morph (morphological) | **0.746** | 0.662 | +0.085 (tie, 5-seed CI crosses 0) |

The fallback is blind to surface form — every unseen value of a feature maps to the
feature mean, so it scores identically (0.662) on both region arms at p=1. Subwords
see surface form: they recover morphological variants (0.746) but fabricate misleading
vectors for arbitrary codes (0.473) that scatter the benign class into false positives
under 1:100 imbalance. Because device attribute codes carry no informative character
structure, the fallback matches or beats subwords at every level; subwords only tie
under heavy morphological drift. This **strengthens** the plain-token (`max_n=0`)
recommendation — it now holds under OOV, not just in-vocab, closing the trade-off A4
left open. FastText's subword OOV advantage requires morphologically-structured fields
(version strings, structured IDs); geo/network categorical codes are not that, and the
fallback is the better OOV mechanism there.

**Scope — benign-only arm not pursued (by design).** A benign-only (asymmetric) OOV
arm has no realistic generating process: vocabulary drift is a property of time and
the population, so novel values hit legitimate and attacker traffic alike. If any
asymmetry exists it runs the *other* way — attacker devices (emulators, fresh
infrastructure, anti-fingerprinting) skew toward less-represented values, i.e.
marginally *more* OOV-prone — and that direction is already partly captured by the
spoof/novel attack construction. "Both sides drift" is therefore the realistic model,
and the arm is intentionally not run rather than deferred.

## A3p — Correlated-marginals collapse precursor

Pooled within-feature cosine (wv, threshold 0.9, decision a2b73375):

| Cell | H6 correlated marginals | Closed-vocab independent (rerun factorial) |
|---|---|---|
| SG + per-event | 0.758 ± 0.011 (**0/5 collapse**) | 0.939 ± 0.007 (5/5 collapse) |
| SG + per-account | 0.459 ± 0.011 | 0.424 ± 0.013 |
| CBOW + per-event | 0.904 ± 0.003 (marginal, >0.9) | 0.982 ± 0.003 (5/5 collapse) |
| CBOW + per-account | 0.515 ± 0.011 | −0.113 ± 0.006 |

Per-feature gradient (SG + per-event, wv): rtt_bucket 0.961 > asn_bucket 0.801
> browser 0.776 > country 0.753 > region 0.706 > device_type 0.580 > os 0.551.
Verdicts (all **5/5**): `a3p_rtt_max_cosine`, `a3p_rtt_min_jsd`,
`a3p_conditioned_below_threshold`; `a3p_collapse_all` **0/5**.

The embedding-free JSD diagnostic confirms the mechanism: rtt_bucket (the one
independently sampled feature) has the lowest within-feature context JSD in the
per-event corpus on every seed (0.26 vs 0.35–0.53 for coupled features). The
prefix-controlled contrast is rtt_bucket vs asn_bucket (equal-length token
prefixes): 0.961 vs 0.801 per-event, and rtt stays elevated (0.786) even
per-account — because RTT is re-sampled per event, neither marginal coupling
nor account persistence differentiates its values.

Reading: **C2's severity is independence-driven.** On realistically coupled
features, SG per-event corpora degrade but do not collapse; only the
independent feature collapses. The C2 claim must be scoped to
weakly-coupled feature sets (or the full ρ-sweep run to characterize the
boundary), and the mechanism section rewritten around context-distribution
overlap, for which feature independence is a necessary condition.

### Collapse-monitor caveat (subword prefix inflation)

Subword-free input vectors (`vectors_vocab`) show within-feature cosine ≈ 0 in
all H6 cells — the wv-space similarity is carried almost entirely by shared
feature-prefix character n-grams on this open vocabulary. The monitor operates
in wv space (which is what the scorer uses), but the 0.9 threshold is not
prefix-safe: CBOW + per-account (a healthy cell) already puts rtt_bucket at
0.918. The proposed "deployable collapse monitor" needs per-feature baselines,
not a single global threshold.

## Implications for the record

- **C1 as written is dead; a revised recommendation survives.** The honest
  architecture statement: per-feature *plain token* embeddings (no subwords),
  per-account corpus, mean-pool + centroid cosine, no gate — validated against
  both the frozen-random control and the likelihood incumbent, with the
  incumbent winning on small closed vocabularies and the embedding winning on
  realistic open vocabularies at 1:100.
- **C2 survives with independence scoping** (A3p): mechanism = context-
  distribution overlap; severity requires weak feature coupling; synthetic
  benchmarks with independent generators overstate it.
- **C3 generalizes upward**: binarization on training-window membership
  suppresses contaminated-fleet detection for *any* smooth scorer — shown for
  both cosine and likelihood. This is a stronger, cleaner claim than the
  original.
- The collapse monitor needs per-feature baselines (prefix inflation; see A3p
  caveat) — or, given A4, the cleaner fix is dropping subwords, which removes
  the inflation at its source.
- **Subwords are a regime-dependent trade, not a pure negative** (A4 + OOV):
  net-negative for in-vocabulary discrimination (A4), net-positive under
  morphological OOV drift (OOV section, +0.27 dPR subword recovery, incumbent
  floors regardless). The plain-token recommendation is correct for stable
  vocabularies; deployments expecting version/ID drift should re-weigh the
  subword OOV benefit — and evaluate OOV robustness with PR under imbalance,
  never ROC (which understated the gap ~15×).

Remaining phase 2 (optional, not decision-changing): A3 full ρ-sweep
(dose-response for the coupling effect), A5 1:100 re-pool of the closed-vocab
evaluation.
