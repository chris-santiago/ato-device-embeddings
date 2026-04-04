# H2 CRITIQUE — Mean-Pool vs. Concatenated-String FastText

**Adversarial review from a skeptical ML engineer.**
Each issue states: (1) the claim, (2) the failure mechanism, (3) what would settle it.

---

## Foundational choices that drive most issues

Two design decisions in the PoC are load-bearing for the comparison:

**F1. Window size is not equivalent across the two models.**
Mean-pool uses `window=6` on 6-token sequences. Concat uses `window=1` on single-token
sequences. In mean-pool, every token within a single login event is within context range of
every other token in that event (and tokens from adjacent events). In concat, the single-token
representation means the skip-gram context is the adjacent *events*, not tokens within an event.
This is not the same context. Mean-pool gets an explicit cross-feature co-occurrence signal baked
into training; concat does not.

**F2. The concat format strips the structured feature key prefixes.**
Mean-pool tokens are `os_ios`, `browser_safari`, `tz_utc-5` — each with a semantically
structured prefix (`os_`, `browser_`, `tz_`). The concat format is values-only:
`ios_safari_utc-5_en_us_wifi_small`. The prefix `os_` is the mechanism by which mean-pool
FastText achieves structured n-gram clustering across OS variants. The concat format discards
this structure at the token level. Any AUC gap may be measuring prefix-loss, not positional
weighting or cross-boundary n-gram noise.

These two foundational differences mean the PoC is testing a **bundle of three effects
simultaneously**: (a) cross-boundary n-gram noise, (b) positional front-loading, and
(c) structured prefix loss + unequal context. H2 attributes the result to (a) and (b), but
(c) is a confound that could produce the same result for a different reason.

---

## Issues

### 1. The window sizes are not equivalent — mean-pool has a training advantage unrelated to H2

**Claim:** Any AUC or silhouette advantage for mean-pool is caused by cross-boundary n-gram
noise or positional weighting in the concat string.

**Mechanism of failure:** Mean-pool trains with `window=6` over 6-token sequences. Each token
in event E is within context range of all other tokens in event E. Skip-gram with window=6
explicitly trains `os_ios` to predict `browser_safari`, `tz_utc-5`, etc. — direct cross-feature
co-occurrence signal. Concat trains with `window=1`. The model never sees cross-event context
beyond 1 adjacent login. This is a structural training advantage for mean-pool that has nothing
to do with n-gram noise. If mean-pool wins on AUC, it may be because of superior cross-feature
co-occurrence signal during training, not because of anything happening in the n-gram
decomposition at inference.

**What would settle it:** Run concat with `window=6` (6 adjacent events as context). If the
AUC gap narrows substantially relative to `window=1`, the window effect explains a significant
portion of the advantage and the n-gram noise story is overstated.

---

### 2. The concat string omits structured feature key prefixes — the mechanism is different from what H2 claims

**Claim:** The n-gram advantage of mean-pool comes from structured prefixes (`os_`, `browser_`)
that reinforce feature-dimension clustering and are absent in concat due to cross-boundary noise.

**Mechanism of failure:** The PoC's mean-pool tokens are `os_ios`, `browser_safari` etc. Their
n-grams include `os_` and `browser_` as explicit prefix n-grams. The concat string
`ios_safari_utc-5_en_us_wifi_small` has none of these prefix n-grams — not because they are
*drowned out by cross-boundary noise* but because they were *never included*. This is a format
difference, not a noise effect. The hypothesis in the report says "n-gram bleed across feature
boundaries" is the mechanism. The PoC also tests "prefix structure absent in concat format" —
a different mechanism that was not the stated cause. The two effects cannot be separated in
this PoC.

**What would settle it:** Test a third concat format that includes the key prefixes and uses a
delimiter that cannot appear within feature values (e.g., `|` or `-`):
`os:ios|browser:safari|tz:utc-5|lang:en_us|net:wifi|screen:small`. If this format closes the
AUC gap relative to the values-only concat, the prefix structure (not cross-boundary n-grams)
was the primary driver. If the gap persists with the richer format, n-gram noise is genuinely
the cause.

---

### 3. Feature values that contain underscores amplify cross-boundary noise artificially

**Claim:** The cross-boundary n-gram effect in the concat format is representative of real-world
token behavior.

**Mechanism of failure:** `lang` values (`en_us`, `en_gb`, `es_mx`) contain underscores. When
joined with `_`, the concat produces `...utc-5_en_us_wifi...`. The boundary between lang and net
is now `en_us_wifi` — two consecutive `_` separators, creating n-grams like `_en`, `en_`, `n_u`,
`_us`, `us_`, `s_w`, `_wi`, `wi_` that span lang and net features at multiple character positions.
A single-underscore separator between features of the form `a_b` creates a clean single crossing.
The compound `en_us_wifi` creates a three-part n-gram crossing that is not representative of the
generic claim about cross-boundary noise. This makes concat look artificially worse than it would
with cleaner feature values.

**What would settle it:** Test the concat format with the same values but using a delimiter that
cannot appear within feature values — e.g., `|`. Then re-run. If AUC gap narrows, the underscore
compounding was inflating the effect. (Note: the user confirmed raw values-as-is is the intended
design; this remains a potential confound regardless of intent.)

---

### 4. Both silhouette scores are negative — the cluster signal is already weak for a different reason

**Claim:** Mean-pool produces better cluster separation than concat (higher silhouette).

**Mechanism of failure:** Both approaches produce negative silhouette (−0.049 vs −0.165). This
means that under *both* approaches, the average device embedding is *closer to devices from other
accounts* than to devices from its own account. The bounded feature vocabulary (~30 tokens) is
shared across all 400 accounts. An account's centroid is dominated by the account's primary
profile, but any two accounts with similar primary profiles (both iOS/Safari users in UTC-5)
will have nearly identical centroids regardless of method. The negative silhouette is an
inherent property of the shared feature vocabulary — it does not trace to either cross-boundary
n-gram noise or positional weighting. Mean-pool's less-negative silhouette (−0.049 vs −0.165)
could reflect the window effect (issue 1) or prefix structure (issue 2) rather than reduced
cross-boundary noise.

**What would settle it:** Compute the *theoretical ceiling* for silhouette under this evaluation
design — what is the maximum achievable silhouette when two accounts share any feature tokens?
If the ceiling is already negative (because many accounts share all tokens), then the silhouette
metric is not diagnostic for this hypothesis and AUC is the sole operative measure.

---

### 5. The concat window choice penalizes the concat model's cross-event context

**Claim:** `window=1` for concat is the appropriate training window because one event is
one context unit.

**Mechanism of failure:** The mean-pool model with `window=6` sees, for each token in event E,
all tokens from events E−1 and E+1 as well as event E. With Zipf-weighted device draws, two
adjacent events often come from the same device — so the model learns very strong within-device
co-occurrence at cross-event boundaries. The concat model with `window=1` sees one adjacent
event token but with *different n-gram structure* (because the token is a concatenated string).
Increasing concat's window would not replicate what mean-pool gets, because mean-pool's context
is 6 fine-grained tokens per neighboring event, not 1 opaque token per neighboring event. The
debate about window equivalence is genuinely underdetermined — there is no obviously correct
mapping between the two training schemas.

**What would settle it:** Run concat at `window=3`, `window=6`, and compare AUC. If AUC
increases monotonically with window size, the model is context-starved and the window choice
is a material confound. If AUC plateaus, the window effect is exhausted and n-gram noise is
the dominant remaining difference.

---

### 6. The spoof attack gap (0.073 AUC) may reflect positional weighting OR it may reflect the shared n-gram structure between spoof and legitimate profiles

**Claim:** The largest AUC gap on spoof attacks (0.826 vs 0.752) is caused by positional
front-loading — the timezone mismatch at position 3 is partially cancelled by high OS+browser
n-gram overlap at positions 1–2.

**Mechanism of failure:** A spoof attack has OS, browser, and lang matching the victim but
a different timezone. In the concat string `ios_safari_utc-5_en_us_wifi_small` (victim) vs
`ios_safari_utc+5_en_us_wifi_small` (spoof), the OS and browser characters are identical up
to position 10 (`ios_safari_`). The n-gram overlap up to the tz boundary is very high. The
mean-pool approach embeds `tz_utc-5` and `tz_utc+5` as distinct tokens whose distance is
learned from training co-occurrence — the model explicitly learns that accounts always
appear with a particular `tz_*` token, making the timezone mismatch highly penalizing. But
this is *not purely* a positional effect — it also reflects that mean-pool gives the timezone
feature equal embedding weight regardless of position, while concat discounts it because it
appears deep in the string. These two effects (positional weighting AND distinct token learning)
cannot be separated in the current PoC.

**What would settle it:** Permute the concat order so that timezone appears first:
`utc-5_ios_safari_en_us_wifi_small`. If spoof AUC improves substantially with tz at position 1,
positional weighting is the mechanism. If it does not improve, the mechanism is the distinct
token embedding effect — mean-pool's ability to learn the exact tz token vs the concat string's
n-gram dilution.

---

### 7. The concat model is not evaluated on OOV token behavior — the one production advantage it could retain

**Claim:** The PoC fully characterizes the concat approach.

**Mechanism of failure:** The PoC only evaluates in-vocabulary profiles (all 30 feature token
combinations are seen during training, and concatenated strings cover all combinations seen
in eval). The concat format's theoretical OOV advantage — FastText embeds `ios_safari_utcnew-5_...`
via n-gram averaging — is not tested. Mean-pool's OOV advantage is also not directly tested
(it was established in Experiment 3). If the concat format degrades *less* than mean-pool on
genuinely OOV feature tokens (new browser family, new OS), the production ranking could be
different from the in-vocabulary ranking. The PoC tests H2 only in the in-vocabulary setting.

**What would settle it:** Inject two genuinely OOV feature tokens — one new OS variant not
seen during training (`os_harmonyos`) and one new timezone (`tz_utc+9`) — into eval events and
measure AUC degradation for both signals. If concat degrades less on OOV tokens, it retains
a production advantage that the silhouette and in-vocabulary AUC numbers conceal.
