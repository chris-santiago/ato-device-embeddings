# ATO Detection via Device Embeddings

FastText trained on structured login feature tokens (OS, browser, country, region, ASN,
device type) detects account takeover in real time. At realistic 1:100 attack-to-benign
imbalance, single-stage mean-pool cosine distance (`mp_raw`) achieves PR-AUC 0.892 on
the hardest attack type — a VPN-style single-field country change — an 8.6x lift over
the trivial set-membership baseline (PR-AUC 0.104). The model scores each login in under
a millisecond and requires no per-account gate or score normalization.

**Final recommendation: system-level blocklist + single-stage `mp_raw`. No per-account
gate. No rank-normalization.** An earlier two-stage gate architecture and a
rank-normalization variant were both retired after H6 showed they degrade PR-AUC
catastrophically under realistic class imbalance.

---

## Recommended architecture

Two decoupled layers, each evaluated on the population it actually serves:

```
Layer 1 — System-level blocklist (upstream deny-list):
  Input:  confirmed fleet/abuse device keys (customer complaints,
          downstream triage, threat intel).
  Action: if device_key in blocklist → deny. Event never reaches Layer 2.
  Coverage: post-lag fleet accounts (~61% of fleet attacks in H6).
  Precision: 1.0 by construction.

Layer 2 — Single-stage mp_raw scorer (no gate, no rank-norm):
  tokens     = [f"os_{os}", f"br_{browser}", f"country_{country}",
                f"region_{region}", f"asn_{asn_bucket}", f"dev_{device_type}"]
  embedding  = mean([fasttext_model[t] for t in tokens])
  risk_score = cosine_distance(embedding, account_centroid)
  → step-up auth if risk_score > threshold
  Coverage: all events that clear the blocklist (spoof, novel device,
            cold-start fleet).
  H6 performance: spoof k=1 PR=0.892; novel PR=0.965; fleet residual PR=0.948.

Monthly retraining:
  corpus: one sentence per account (all login events concatenated)
  FastText: sg=1, epochs=20, negative=10, min_n=3, max_n=6, window=6, vector_size=64
  Health check: compute within-feature cosine similarity → halt deployment if > 0.5
                (indicates embedding collapse; see "Critical configuration warning" below)

Offline (daily/weekly):
  device_id_history → Word2Vec centroid distance → fleet/reuse review queue
  (feeds Layer 1 blocklist population)

Fallback (< 20 confirmed events per account, or embedding service down):
  per-account known-device hash set → step-up auth if unseen profile
```

**Do not use:** per-account set-membership gate (two-stage) or rank-normalization.
Both degrade PR-AUC catastrophically under 1:100 imbalance. The two-stage gate also
blinds the model during the cold-start fleet window — the exact period when it is
the only available defense. See the narrative below for the mechanism.

---

## Results

PR-AUC is the operative metric at 1:100 attack-to-benign imbalance. ROC-AUC
compresses into a narrow band at this ratio where all scorers appear competitive
(trivial ROC=0.943, mp_raw ROC=0.995); only PR-AUC surfaces the 8.6x operational
gap between them.

| Scorer | Spoof k=1 PR-AUC | Novel PR-AUC | Fleet residual PR-AUC |
|--------|:----------------:|:------------:|:---------------------:|
| **`mp_raw` (recommended)** | **0.892** | **0.965** | **0.948** |
| Two-stage (gate + mp_raw) | 0.890 | 0.940 | 0.010 — blind in cold-start |
| Rank-norm (`mp_rank_norm`) | 0.215 | 0.273 | 0.229 |
| Trivial set-membership | 0.104 | 0.103 | 0.010 |

All numbers from H6 (RBA-calibrated hybrid dataset, 1:100 imbalance, 229-token
vocabulary). Bootstrap 95% CIs for mp_raw: spoof k=1 [0.880, 0.903], novel
[0.959, 0.971].

**Column notes.** Spoof k=1 = attacker changes country only (VPN-realistic, the
hardest case). Novel = fully new device tuple unseen in training. Fleet residual =
pre-lag cold-start accounts only — the 39 accounts where the blocklist had not yet
activated and `mp_raw` is the only defense; two-stage scores these as `known → 0`,
producing 0 true positives.

**Real-world replication (RBA dataset).** Applying the same pipeline to the DAS Group
RBA dataset (~31M real SSO logins, 141 ATO events) produces mean-pool ROC-AUC 0.852
[0.689, 0.975] vs. trivial 0.661. The mean-pool > trivial ordering holds across all
three tested temporal splits. Result is exploratory (n=9 test positives) but
directionally consistent. See `h2_rba/docs/REPORT.md` and `TECHNICAL_REPORT.md` §6.

---

## Quickstart

Scripts are self-contained with inline dependencies ([PEP 723](https://peps.python.org/pep-0723/)). No virtualenv required.

```bash
# H6: final experiment — RBA-calibrated vocabulary, 1:100 imbalance, temporal blocklist
uv run h6_hybrid/experiments/data_prep.py         # one-time: extract RBA marginals (~5 min)
uv run h6_hybrid/experiments/hybrid_experiment.py

# H2: mean-pool vs. concat head-to-head (the core architectural comparison)
uv run pre_ml_lab/experiments/h2_rerun_experiment1.py

# Real-world replication on the DAS Group RBA dataset
uv run h2_rba/experiments/data_prep.py            # one-time: downloads ~1 GB, writes parquet
uv run h2_rba/experiments/rba_rerun.py --smoke    # fast pipeline check (~30 sec)
```

---

## How we got here

### The signal that works — and why it isn't obvious

Early experiments (Experiments 1–2) tested FastText trained on raw opaque device IDs
(e.g., `device_8472a`). Character n-grams on random hex strings destroy account cluster
structure — silhouette score −0.051, meaning no per-account separation at all. Raw device
IDs do not work.

Experiment 3 replaced device IDs with structured feature tokens: `os_ios`,
`browser_safari`, `country_us`, and so on. FastText's n-gram mechanism now operates on
meaningful prefixes (`os_`, `browser_`) rather than random characters. Per-account
clusters form cleanly, and AUC on novel attacks reaches 0.985 under a corrected
evaluation design (enrollment events — legitimate new devices — included in the negative
class, so any signal that fires on all unseen devices is penalized).

An apparent 0.989 AUC for a simple out-of-vocabulary binary baseline in Experiment 2 was
an evaluation artifact: every account had unique device IDs, making attack detection
equivalent to flagging any globally unseen device. Under the corrected evaluation the
OOV baseline collapses to 0.750 on novel and spoof attacks and 0.250 on fleet attacks.
Feature embeddings are immune to this collapse because they measure fit-to-cluster, not
seen/unseen membership.

### Mean-pool vs. concatenated string (H2)

Once structured feature tokens were established, the question became: embed features as
one concatenated string (`ios_safari_us_en-us_wifi_small`) or mean-pool six independent
token embeddings? **Mean-pool wins — 7/7 pre-specified tests.** The mechanism: character
n-grams in the concatenated string span feature boundaries, injecting spurious signal
that dilutes the discriminative information in any single differing feature (e.g., a
changed country). Mean-pooling embeds each feature independently, so a single mismatched
feature contributes its full signal to the distance score.

This was confirmed three ways across independent investigations, all under the same
robust training configuration.

### Critical configuration warning

A degenerate training configuration (CBOW objective + per-event corpus) causes
**within-feature embedding collapse**: all values of a single feature (e.g., every
country code) converge to nearly identical vectors (cosine similarity 0.9993). Under
collapse, mean-pool carries no country signal and the concat string appears to win — the
opposite conclusion. This failure is **silent at the novel and fleet AUC level** and
visible only via the T8 token similarity diagnostic.

| Configuration | Within-feature similarity | Spoof AUC | Conclusion |
|--------------|:-------------------------:|:---------:|:----------:|
| CBOW + per-event corpus (broken) | 0.9993 — collapse | 0.384 (below chance) | Mean-pool appears to fail |
| Skip-gram + per-account corpus (correct) | 0.392 — healthy | 0.869 | Mean-pool confirmed |

**The skip-gram + per-account corpus configuration is not optional.** Run the T8
within-feature similarity check after every retraining cycle. If any feature dimension
shows similarity > 0.5, halt deployment and inspect corpus construction.

### Intermediate experiments (H3–H5)

Three follow-on experiments tested further variants: H3 per-feature normalization (no
improvement over mp_raw), H4 GRU temporal model (not confirmed at this dataset scale),
and H5 k=1 spoof stress test on a 30-token synthetic vocabulary (mp_raw scored 0.530,
below the trivial baseline of 0.750 — a failure since resolved). H6 resolved the H5
finding by using a 229-token RBA-calibrated vocabulary: with richer co-occurrence
structure, raw mp_raw at k=1 reaches ROC-AUC 0.995 / PR-AUC 0.892. The k=1 failure was
vocabulary poverty, not an architectural limit.

### Rank-normalization — retired

Rank-normalization converts raw cosine distances to empirical percentile scores within
each account's calibration set. On a balanced (1:1) evaluation with the 30-token
vocabulary, it improved k=1 ROC-AUC from 0.522 to 0.714 — a real gain on the balanced
metric. Under the operationally realistic 1:100 imbalance studied in H6, the same
transform collapses PR-AUC from 0.892 to 0.215 — a 4x degradation. The CDF compression
that helps on balanced data destroys precision at realistic imbalance. The earlier
recommendation to apply rank-normalization is retired. Use raw cosine distance.

### Two-stage gate — retired

Earlier analysis recommended a per-account set-membership gate: if the device is in the
account's known-device set, score it 0 (pass through); otherwise score with mp_raw. H6's
fleet-residual analysis retired this architecture. During the cold-start fleet window —
the period between the first fleet attack and blocklist activation — the fleet device is
already in the account's training set as a legitimate prior login. The gate fires
(`known → 0`) and blinds the model precisely when it is the only available defense.
After the blocklist activates, events from confirmed fleet devices never reach the model
at all, making the gate irrelevant on both sides of the lag boundary. The two-layer
blocklist + mp_raw architecture replaces it: the blocklist catches post-lag fleet
accounts with certainty (precision=1.0), and mp_raw handles cold-start fleet via the
continuous cosine-distance signal (fleet residual PR-AUC 0.948).

---

## Investigation artifacts

| Directory | Contents |
|-----------|----------|
| `pre_ml_lab/` | Experiments 1–3 and original/rerun H2 investigation |
| `h2_ml_lab/` | Structured H2 investigation with adversarial critique and peer review |
| `h2_rba/` | Real-world replication on the DAS Group RBA dataset (~31M logins) |
| `h3_pfn/` | H3: per-feature normalized scoring — not confirmed |
| `h4_gru/` | H4: GRU temporal model vs. mean-pool — not confirmed at this dataset scale |
| `h5_stress/` | H5: k=1 stress test on 30-token vocabulary — failure resolved by H6 |
| `h6_hybrid/` | H6: RBA-calibrated vocabulary, 1:100 imbalance, temporal fleet blocklist — final recommendation |
| `TECHNICAL_REPORT.md` | Full synthesis: H2 through H6, configuration sensitivity, RBA replication |
| `archive/` | Earlier process documents |

<details>
<summary>Full file inventory</summary>

### Scripts

| File | Purpose |
|------|---------|
| `pre_ml_lab/experiments/ato_fasttext_poc.py` | Original PoC: FastText on device ID sequences |
| `pre_ml_lab/experiments/ato_experiment2.py` | Experiment 2: FastText vs. Word2Vec vs. OOV baseline |
| `pre_ml_lab/experiments/ato_experiment3.py` | Experiment 3: fleet corpus, feature token embeddings, corrected enrollment evaluation |
| `pre_ml_lab/experiments/plot_conclusions.py` | Generates Experiment 2 figures |
| `pre_ml_lab/experiments/ato_concat_poc.py` | H2 original PoC: mean-pool vs. concat |
| `pre_ml_lab/experiments/ato_concat_experiment.py` | H2 five-test experiment |
| `pre_ml_lab/experiments/h2_rerun_poc.py` | H2 rerun PoC (independent implementation) |
| `pre_ml_lab/experiments/h2_rerun_experiment1.py` | H2 rerun: bootstrap CIs, window sweep, trivial baseline, permutation tests |
| `h2_ml_lab/experiments/ato_device_embedding_poc.py` | H2 ml-lab PoC (reveals CBOW collapse) |
| `h2_ml_lab/experiments/ato_device_embedding_experiment2.py` | H2 ml-lab experiment iteration 1 |
| `h2_ml_lab/experiments/ato_device_embedding_experiment3.py` | H2 ml-lab experiment iteration 2 |
| `h2_ml_lab/experiments/robust_config_experiment.py` | Token similarity and compactness diagnostics under robust config |
| `h2_ml_lab/experiments/config_verification.py` | Side-by-side comparison: degenerate vs. robust training config |
| `h2_ml_lab/experiments/variable_spoof_experiment.py` | Variable-K spoof (k=1/2/3) × raw vs. rank-norm |
| `h2_rba/experiments/data_prep.py` | One-shot: download RBA dataset, write parquet |
| `h2_rba/experiments/rba_rerun.py` | RBA replication: tokenize, train FastText, score, metrics |
| `h6_hybrid/experiments/data_prep.py` | One-shot: extract RBA clean-login marginals for chain-sampling |
| `h6_hybrid/experiments/hybrid_experiment.py` | H6: chain-sampled accounts, variable-K spoof, novel, fleet with temporal blocklist |

### Research documents

| File | Purpose |
|------|---------|
| `TECHNICAL_REPORT.md` | Definitive synthesis: H2 through H6 |
| `pre_ml_lab/docs/REPORT.md` | Full report: Experiments 1–3, production constraints |
| `pre_ml_lab/docs/CONCLUSIONS.md` | Per-finding verdicts and signal hierarchy |
| `pre_ml_lab/docs/REPORT_ADDENDUM.md` | Production deployment analysis |
| `pre_ml_lab/docs/H2_REPORT.md` | H2 original investigation report |
| `pre_ml_lab/docs/H2_RERUN_REPORT.md` | H2 rerun report |
| `h2_ml_lab/docs/REPORT.md` | H2 ml-lab investigation report |
| `h2_ml_lab/docs/CONCLUSIONS.md` | H2 ml-lab per-test verdicts |
| `h2_ml_lab/docs/PEER_REVIEW_R1.md` | Round 1 peer review (3 major issues resolved) |
| `h2_ml_lab/docs/PEER_REVIEW_R2.md` | Round 2 peer review (2 minor issues, no major) |
| `h2_rba/docs/HYPOTHESIS.md` | Pre-run hypothesis for RBA replication |
| `h2_rba/docs/REPORT.md` | RBA replication report with design audit and sensitivity analysis |
| `h6_hybrid/docs/HYPOTHESIS.md` | Pre-registered H6 hypothesis |
| `h6_hybrid/docs/REPORT.md` | H6 report: final architecture recommendation and fleet-residual analysis |

</details>
