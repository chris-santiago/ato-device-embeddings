# H2 Rerun — Debate Transcript

## Contested points entering debate

| ID | Issue | Critique claim | Defense position |
|----|-------|---------------|-----------------|
| C1 | Single seed / no CIs | Results are sampling noise | Spoof + silhouette survive; fleet/novel may not |
| C2 | Window mismatch | window=6 vs window=1 confounds result | Window is principled; predict gap persists at all windows |
| C3 | Worst-case encoding | Underscore amplifies noise; prefixed-concat closes gap | Gap persists even with non-overlapping delimiter |
| C4 | Fleet contamination | Fleet injection suppresses AUC asymmetrically | Contamination is symmetric; delta preserved across strata |
| C5 | No trivial baseline | Set-membership baseline may match mean_pool | mean_pool > set-membership on spoof (graded similarity) |
| C6 | Mechanism: n-gram density vs position | Mechanism is density, not position | Prefixed-concat test resolves this (same test as C3) |
| C7 | Positional mechanism unverified | Tz at position 0 recovers spoof gap | No ordering recovers full gap |

---

## Round 1 — Critic

**C1:** The defense concedes the fleet and novel deltas may be noise. I sharpen: the _only_
reliable result entering the experiment is spoof AUC and silhouette. Fleet and novel AUC
support for H2 should be suspended pending CIs. Agreed test: bootstrap N=1000, percentile CIs.
**Resolution: empirical test agreed.** Test condition: if mean_pool spoof AUC 95% CI lower bound
strictly exceeds concat spoof AUC upper bound, defense wins. If CIs overlap, critique wins.

**C2:** The defense claims window=1 for concat is "principled" because each concatenated token
is one login event. I dispute: by this logic, mean_pool should also use window=1 for the same
reason — each 6-token sequence is one login event, and adjacent events provide the session
context. The defense chose window=6 for mean_pool specifically to give feature tokens access
to each other within a login event. This is a design choice that favors mean_pool. An equivalent
design for concat would be window=1 (one event of context) only if we accept that feature
tokens within the same event do not benefit from mutual context. I concede that concat window=1
vs. window=6 is a genuine empirical question. The window sweep remains necessary.
**Resolution: empirical test agreed.** Concat at window ∈ {1, 3, 6}. If window=6 concat closes
>50% of the spoof delta, critique wins C2.

**C3:** The defense predicts that even with a non-overlapping delimiter, the gap persists.
I accept this as a testable claim. The prefixed-concat test (key:val|key:val|...) will reveal
whether the mechanism is delimiter-sensitive.
**Resolution: empirical test agreed.** If prefixed-concat silhouette >= plain concat silhouette
AND prefixed-concat spoof AUC >= mean_pool spoof AUC − 0.01, critique wins C3.

**C4:** The defense claims fleet contamination is symmetric. I partially concede: if both signals
are embedded in the same vector space geometry, the centroid shift from fleet injection would
be symmetric in the abstract. However, mean_pool centroids may be more resistant to outlier
shift because they are means of sparse, high-dimensional feature token vectors, while concat
centroids may be more diffuse. I will not push for a separate stratification test — this
is an interaction effect that the bootstrap CIs will partially address. **Resolved: defense
wins C4 on the theoretical argument; no separate test required.**

**C5:** The defense concedes this point. A trivial baseline is required. I accept the defense's
prediction as testable: if set-membership baseline AUC >= mean_pool on spoof, critique wins.
**Resolution: empirical test agreed.** Exact-profile set-membership baseline on all three attack
types. Critique wins C5 if set-membership >= mean_pool on spoof AUC.

**C6:** The defense correctly notes this collapses into C3. **Resolved: merged into C3 test.**

**C7:** The defense predicts no permutation recovers the full gap. I sharpen: I only need one
ordering to recover >50% of the spoof delta (not 100%). If tz-first concat recovers >50% of
the gap, it demonstrates the mechanism is positional, and the fix is reordering, not mean-pooling.
**Resolution: empirical test agreed.** Test all 6 orderings placing tz at each position.
If any ordering recovers >50% of mean_pool vs. baseline-concat spoof delta, critique wins C7.

---

## Round 1 — Defender

**C1:** Agreed on the test condition. I add a sharpening: the _relevant_ CI test is whether
the mean_pool advantage on spoof is statistically distinguishable from zero, not whether the
two confidence intervals are disjoint (which is a more conservative criterion). I propose:
compute the bootstrap distribution of (mean_pool_AUC - concat_AUC) on spoof; if the 2.5th
percentile of the delta distribution is > 0, defense wins C1 on spoof. **Agreed on this
operationalisation.**

**C2:** I accept the window sweep. My specific prediction: concat at window=6 will improve
over concat at window=1, but the silhouette gap will persist because silhouette measures
embedding geometry (not training signal richness), and the n-gram boundary problem is
independent of window size. I narrow the defense-wins condition: if concat at window=6
does NOT close the silhouette gap (i.e., mean_pool silhouette remains higher at all window
values), defense wins the silhouette claim within C2. **Agreed.**

**C3:** I add a specific condition for defense-wins: if the silhouette gap between mean_pool
and prefixed-concat is > 0.05 (same direction as plain concat, both negative), defense wins
because the gap persists even with non-overlapping delimiters. **Agreed.**

**C5:** For the trivial baseline, I add: the set-membership baseline should be evaluated
using exact profile matching (all 6 features must match). This is the natural definition and
the one the defense's prediction is based on. A relaxed match (5/6 features) would be a
different claim. **Agreed: exact 6/6 match.**

**C7:** I accept the >50% recovery criterion as the critique's bar, but I add: the baseline
for the delta should be the mean_pool vs. the *best-performing* concat configuration from
the window sweep (not window=1 baseline-concat). If concat at window=1 already gives spoof
AUC = 0.763, and any tz-permutation gives spoof AUC = 0.782 (recovering 35% of the 0.054
delta), but the window=6 concat already gives 0.800, then the permutation test is measuring
improvement over the *wrong* baseline. I propose: the delta for C7 is measured vs.
window=1 concat spoof AUC. **Agreed: delta measured vs. window=1 concat.**

---

## Round 2 — Critic

All points are now either resolved or have agreed empirical tests. No new rounds needed.

**Status review:**
- C1: Empirical test agreed (bootstrap delta CIs, 2.5th percentile of delta > 0)
- C2: Empirical test agreed (window sweep on concat; does silhouette gap persist?)
- C3: Empirical test agreed (prefixed-concat; does silhouette gap > 0.05 persist?)
- C4: Defense wins (symmetric contamination argument accepted)
- C5: Empirical test agreed (set-membership baseline, exact 6/6 match, vs mean_pool on spoof)
- C6: Merged into C3
- C7: Empirical test agreed (tz-position permutation; >50% delta recovery vs window=1 concat)

---

## Resolution summary

| Point | Status | Conditions |
|-------|--------|------------|
| C1 | Empirical test | Bootstrap 2.5th pct of (mp-cc) delta on spoof > 0 → defense wins; else critique wins |
| C2 | Empirical test | If concat window=6 closes silhouette gap → critique wins; if gap persists → defense wins |
| C3 | Empirical test | If prefixed-concat silhouette gap > 0.05 vs mean_pool → defense wins; else critique wins |
| C4 | Defense wins | No separate test; symmetric contamination accepted |
| C5 | Empirical test | If set-membership AUC >= mean_pool AUC on spoof → critique wins; else defense wins |
| C6 | Merged into C3 | — |
| C7 | Empirical test | If any tz-ordering recovers >50% of window=1 spoof delta → critique wins; else defense wins |

---

## Empirical test list for experiment phase

### T1 — Bootstrap confidence intervals (addresses C1)
- Run N=1000 bootstrap on all AUC metrics and silhouette.
- Compute 95% CI for (mean_pool - concat) delta on each metric.
- **Defense wins:** 2.5th percentile of spoof delta > 0.
- **Critique wins:** 2.5th percentile of spoof delta <= 0 (CI includes zero).
- **Ambiguous:** CI crosses zero but mean_pool point estimate still higher.

### T2 — Concat window sweep (addresses C2)
- Train concat FastText at window ∈ {1, 3, 6}.
- Compare silhouette and spoof AUC vs mean_pool at each window.
- **Defense wins:** Silhouette gap (mean_pool > concat) persists at all window values.
- **Critique wins:** Concat at window=6 closes silhouette gap by >50%.
- **Ambiguous:** Silhouette gap shrinks but does not close; AUC ordering changes.

### T3 — Prefixed-concat format (addresses C3/C6)
- Train FastText on "os:ios|browser:safari|tz:utc-5|lang:en_us|net:wifi|screen:small" tokens.
- Compare silhouette and spoof AUC vs plain concat and mean_pool.
- **Defense wins:** Silhouette gap between mean_pool and prefixed-concat > 0.05.
- **Critique wins:** Prefixed-concat silhouette within 0.05 of mean_pool AND spoof AUC within 0.01 of mean_pool.
- **Ambiguous:** Prefixed-concat improves over plain concat but does not reach mean_pool level.

### T4 — Trivial baseline (addresses C5) [NON-NEGOTIABLE]
- Implement exact-profile (6/6 feature) set-membership baseline.
- Score eval events: 1 if profile not in account's training set, 0 if it is.
- Compute AUC on novel, fleet, spoof vs enrollment-inclusive negative class.
- **Defense wins:** mean_pool AUC > set-membership AUC on spoof.
- **Critique wins:** set-membership AUC >= mean_pool AUC on spoof.
- **Ambiguous:** Baseline beats mean_pool on one attack type but not spoof.

### T5 — Tz-position permutation (addresses C7)
- Test all 6 orderings that place tz at positions 0–5 in the concat string.
- Compare spoof AUC for each ordering vs mean_pool and window=1 concat.
- **Defense wins:** No ordering recovers >50% of the spoof delta vs. window=1 concat.
- **Critique wins:** Any ordering recovers >50% of the spoof delta.
- **Ambiguous:** Some orderings improve over window=1 but none recover >50% of the delta.
