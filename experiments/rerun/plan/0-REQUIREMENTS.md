# Research Requirements

Working from the technical report and novelty assessment from the literature review.
Four publishable contributions identified, ordered from most to least novel.

## Contribution 1 — Within-Feature Embedding Collapse in Structured Categorical Corpora

**The claim:** Per-event corpus construction causes within-feature cosine similarity → 0.9993 in
structured categorical token sequences, silently destroying discriminability on the hardest
attack type while leaving easy attack metrics intact. The mechanism is deterministic: rigid
positional structure enforces identical co-occurrence distributions for within-feature tokens
regardless of training objective — both CBOW and Skip-gram collapse on per-event corpus
(within=0.997 and 0.996 respectively); both recover on per-account corpus (within=0.188 and
0.487). The training objective is not the causal factor; corpus construction is. Per-account
corpus resolves the collapse by breaking positional rigidity across variable-length sequences.

**Why novel:** Not documented in NLP, recsys, or fraud literature in this form. Closest prior
work (recsys embedding collapse literature) describes dimensional collapse in feature-interaction
models, not training-objective-induced within-feature collapse in structured token corpora.

## Contribution 2 — Rank-Normalization Collapse Under Realistic Class Imbalance

**The claim:** Per-user CDF rank-normalization, a common practitioner technique for handling
variable centroid quality across accounts, destroys PR-AUC at realistic 1:100
attack-to-benign imbalance (0.892 → 0.215) while ROC-AUC appears only modestly affected
(0.995 → 0.972). The mechanism is CDF compression: the transform reduces score margin between
positives and negatives, and at realistic imbalance this margin is the only thing separating
the precision-recall curve from the baseline.

**Why novel:** The general warning that ROC-AUC is misleading under imbalance is well-established
(Davis & Goadrich 2006). The specific mechanism — that a per-user normalization transform causes
this collapse, and that ROC-AUC actively hides it — is not documented anywhere. Practitioners
apply rank-normalization routinely without understanding this failure mode.

## Contribution 3 — Known-Device Gate Blinds Fleet Detection

**The claim:** A per-account "known device" gate — intended to reduce false positives by
suppressing alerts on devices seen in the training window — scores zero true positives at
top-1% on fleet attacks. The mechanism is structural: fleet devices appear in the training
window by construction (via injection events), so the gate fires on every fleet device
regardless of when the attack occurs. Raw cosine distance without the gate detects the same
fleet events at 91.8% top-1% precision because the fleet device's embedding is anomalous
relative to the account centroid even when the device is technically "known."

> **Claim updated from original:** The finding holds on ALL fleet events (fleet aggregate),
> not only the pre-lag cold-start population. Empirically, `two_stage` ROC-AUC is identical
> to `trivial` to 6 decimal places on both `fleet_aggregate` (0.459079) and `fleet_residual`
> (0.457436). The "pre-lag cold-start" framing is the operational motivation — blocklist
> handles post-lag events anyway, so the residual population is where the gate failure
> matters most — but it is not the scope of the empirical finding.

**Why novel:** The finding that a standard false-positive-reduction mechanism produces zero
detections on its target population is counterintuitive and practically significant. The
population-conditional evaluation methodology — evaluating each detection layer on the
population it actually serves — is not documented in the RBA or ATO literature.

## Contribution 4 — Mean-Pool Independent Token Embeddings vs. Concatenated String for ATO Device Fingerprint Detection

**The claim:** Mean-pooling one embedding per feature token outperforms embedding the full
device string as a single concatenated token for cosine-distance anomaly detection, with the
largest advantage on spoof attacks where only one feature differs. The mechanism is
cross-boundary character n-gram contamination in FastText: the concat string's n-grams span
feature boundaries, injecting signal uncorrelated with any single feature dimension and
diluting the contribution of the differing feature.

**Why novel:** The comparison hasn't been made in the ATO or fraud detection domain. Closest
prior work (categorical embedding literature: CURE IJCAI 2017, unsupervised categorical
embeddings WCCI 2020) embeds features independently but doesn't apply FastText to structured
login telemetry or evaluate against a concat baseline in an anomaly detection context. The
application to ATO with the spoof attack taxonomy is the contribution.
