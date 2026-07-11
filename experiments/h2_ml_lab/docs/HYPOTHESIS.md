## Hypothesis — Cycle 1

**Claim:** Mean-pooling six feature-token embeddings into a device vector will outperform directly embedding a single concatenated device string with FastText, on both silhouette score and ROC-AUC for ATO (Account Takeover) detection.

**Mechanism:** Character n-grams spanning feature boundaries in a concatenated string (e.g., `ios_safari_utc-5_en_us_wifi_small`) inject spurious signal uncorrelated with any semantic dimension. Mean-pooling embeds each feature token independently (e.g., `os_ios`, `browser_safari`), eliminating cross-boundary contamination. The advantage should be largest on spoof attacks, where the attacker matches the victim's OS, browser, and language and differs only on timezone — concentrating the diagnostic signal in one feature dimension that must compete against five matching ones.

**Signal:** The observable signal is cosine distance from the per-account embedding centroid to a test login event's embedding. Mean-pooling should produce tighter, more compact per-device clusters (higher silhouette score) and sharper separation between legitimate and attack logins (higher AUC), especially on spoof attacks where only one feature differs.

**Expected observable:** 
- Mean-pool silhouette score > concat silhouette score (cosine metric, per-device embeddings)
- Mean-pool ROC-AUC > concat ROC-AUC on spoof attacks (the most diagnostic condition)
- Mean-pool ROC-AUC >= concat ROC-AUC on novel and fleet attacks (no degradation)
- The trivial baseline (exact 6/6 feature set-membership) will be beaten on spoof attacks by both embedding strategies, because spoof profiles appear novel by definition (different timezone)

## Evaluation Metrics

**Primary:** 
- ROC-AUC per attack type (novel, fleet, spoof) — measures ranking quality of cosine distance scores; attack-type stratification tests the mechanism claim directly
- Silhouette score (cosine metric, per-device embeddings) — measures cluster compactness; tests whether mean-pooling produces more coherent device representations

**Secondary:**
- Bootstrap 95% CI on all AUC and silhouette values (N=1,000, percentile method) — required to claim significance

**Domain:** ato_device_embedding

---

## Configuration Specification

### Canonical Configuration

The canonical (robust) configuration used in the H2_RERUN experiment and all supplemental tests is:

| Parameter | Value |
|-----------|-------|
| Architecture | skip-gram (`sg=1`) |
| Corpus unit | per-account (all events for one account form one document) |
| Epochs | 20 |
| Negative samples | 10 |
| Min n-gram length | 3 |
| Max n-gram length | 6 |
| Window | 6 |

### Why This Configuration Is Canonical

**Skip-gram (`sg=1`) + per-account corpus** is the load-bearing combination for mean-pool correctness.

Skip-gram (sg=1) predicts context tokens from the center token. For feature tokens like `tz_utc-8` and `tz_utc+8`, this objective forces the model to assign distinct embeddings: two tokens that predict different context distributions (i.e., different co-occurring feature values from different devices) will diverge in embedding space. Within-feature values are thereby differentiated — `tz_utc-8` and `tz_utc+8` land in different regions of the embedding space, making the tz dimension genuinely discriminative in a mean-pool device vector.

The per-account corpus provides the cross-event context diversity necessary for this differentiation to occur. When the document is the full account history (all events concatenated), tokens from different events appear in the same context window. A timezone token from a spoof event appears adjacent to OS and browser tokens from the primary device's events, giving the model signal that `tz_utc+8` is contextually different from `tz_utc-5` even when the surrounding device features are identical.

### Why the Degenerate Configuration Fails

The degenerate configuration — sg=0 (CBOW), per-event corpus, epochs=10 — was used in the ml-lab experiment and produces **within-feature embedding collapse**.

CBOW predicts the center token from its context tokens. When the corpus consists of individual six-token events (per-event), the six feature tokens always appear as each other's context, regardless of which account or device they belong to. Because feature values within a single dimension (e.g., all timezone values) share the same co-occurrence distribution — they all appear with the same OS/browser/language/network/screen tokens — CBOW assigns them nearly identical embeddings. The result is within-feature cosine similarity of **0.9993**: `tz_utc-8` and `tz_utc+8` are effectively the same vector.

The collapse is documented and verified in `h2_ml_lab/experiments/config_verification.py`:

- **ml-lab config (degenerate):** within-feature similarity = 0.9993
- **H2_RERUN config (robust):** within-feature similarity = 0.427

When within-feature collapse is present, mean-pool device vectors carry no discriminative information within any feature dimension. The tz dimension contributes a token embedding that is nearly identical regardless of which timezone value is present. Mean-pool's architectural advantage — independent feature embedding — is nullified, and concat's cross-boundary n-grams become the dominant (and superior) signal source.

**Practical deployment note:** Any production use of mean-pool FastText for device fingerprinting must verify that within-feature collapse has not occurred. The T8 diagnostic (within-feature cosine similarity analysis) provides this check. A within-feature similarity above 0.9 should be treated as a configuration failure requiring remediation before trusting spoof detection results.
