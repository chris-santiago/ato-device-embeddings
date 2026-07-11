# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim==4.4.0",
#   "numpy==2.2.6",
#   "scikit-learn==1.7.2",
#   "matplotlib==3.10.8",
# ]
# ///
#
# H6 Rerun — Phase 2 Reproducibility Script
# ==========================================
# Seed-parameterized rerun of the H6 hybrid experiment.
# Reads RBA marginals from the original h6_hybrid location (deterministic, seed-invariant).
# Writes results.json to experiments/rerun/seeds/seed_{seed}/h6/ matching
# the Section 4 metrics schema exactly.
#
# Addresses issues: da184294 (FastText seed — verified correct in source),
#   3b3a777b (T8 absent), 25f721cf (verdict schema mismatch), dd39e09c (neg-ratio
#   default), 27ef710e (dep version pins).
#
# Usage:
#   uv run experiments/rerun/scripts/h6/h6_rerun.py --seed 42 --smoke
#   uv run experiments/rerun/scripts/h6/h6_rerun.py --seed 42
#   uv run experiments/rerun/scripts/h6/h6_rerun.py --seed 123 --neg-ratio 100

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gensim.models import FastText
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parent.parent.parent.parent.parent
MARGINALS_PATH = Path(__file__).resolve().parent / "rba_marginals.json"
SEEDS_ROOT     = Path(__file__).resolve().parent.parent.parent / "seeds"

# ---------------------------------------------------------------------------
# Config — unchanged from h6_hybrid/experiments/hybrid_experiment.py
# ---------------------------------------------------------------------------
FEATURE_ORDER       = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]
DEVICE_KEY_FEATURES = ["os", "browser", "device_type", "country", "region", "asn_bucket"]

ROBUST_KWARGS = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    seed=42, workers=1,
)

N_ACCOUNTS_FULL  = 400
N_ACCOUNTS_SMOKE = 50
N_TRAIN          = 60
N_CENTROID       = 40
N_SPOOF_PER_ACCT = 5
N_ENROLL_NEG     = 5
FLEET_FRAC       = 0.25
N_BOOTSTRAP      = 1000
DEFAULT_SEED     = 42

BLOCKLIST_LAG_DAYS  = 10
ATTACK_WINDOW_DAYS  = 30
FLEET_EXTRA_SCORERS = ["trivial_blocklist", "two_stage_blocklist", "combined"]

SCORER_NAMES = ["mp_raw", "mp_rank_norm", "trivial", "two_stage", "two_stage_rank_norm"]
ATTACK_TYPES = ["spoof_k1", "spoof_k2", "spoof_k3", "novel", "fleet", "fleet_residual"]

# T8 collapse threshold — canonical across all phases (decision a2b73375)
T8_COLLAPSE_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_token(f: str, v: str) -> str:
    return f"{f}_{v}"

def device_key(d: dict) -> tuple:
    return tuple(str(d.get(f, "unknown")) for f in DEVICE_KEY_FEATURES)

def device_to_tokens(d: dict) -> list:
    return [make_token(f, d.get(f, "unknown")) for f in FEATURE_ORDER]

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))

def embed_mp(event: dict, model) -> np.ndarray:
    vecs = [model.wv[make_token(f, event.get(f, "unknown"))] for f in FEATURE_ORDER]
    return np.mean(vecs, axis=0)

def compute_centroid(events: list, model) -> np.ndarray:
    return np.mean([embed_mp(e, model) for e in events], axis=0)

def weighted_choice(counts: dict, rng: np.random.Generator) -> str:
    keys = list(counts.keys())
    weights = np.array([counts[k] for k in keys], dtype=float)
    weights /= weights.sum()
    return keys[rng.choice(len(keys), p=weights)]


# ---------------------------------------------------------------------------
# Marginals — unchanged from source
# ---------------------------------------------------------------------------
class Marginals:
    def __init__(self, path: Path):
        with open(path) as f:
            data = json.load(f)
        self._od_flat = data["os_device_joint"]
        self._od_nested: dict = defaultdict(dict)
        for key, cnt in self._od_flat.items():
            os_val, dev_val = key.split("::", 1)
            self._od_nested[os_val][dev_val] = cnt
        self._od_nested = dict(self._od_nested)
        self._browser_given_os     = data["browser_given_os"]
        self._country_marginal     = data["country_marginal"]
        self._region_given_country = data["region_given_country"]
        self._region_marginal      = data["region_marginal"]
        self._asn_given_country    = data["asn_given_country"]
        self._asn_marginal         = data["asn_marginal"]
        self._rtt_marginal         = data["rtt_marginal"]
        self._country_keys = list(self._country_marginal.keys())
        self._country_weights = np.array(
            [self._country_marginal[c] for c in self._country_keys], dtype=float
        )
        self._country_weights /= self._country_weights.sum()
        meta = data.get("meta", {})
        print(f"  Marginals loaded — {meta.get('total_clean_rows', '?'):,} clean RBA rows")

    def sample_device_tuple(self, rng: np.random.Generator) -> tuple[str, str, str]:
        od_keys = list(self._od_flat.keys())
        od_weights = np.array(list(self._od_flat.values()), dtype=float)
        od_weights /= od_weights.sum()
        chosen_key = od_keys[rng.choice(len(od_keys), p=od_weights)]
        os_val, dev_val = chosen_key.split("::", 1)
        browser_counts = self._browser_given_os.get(
            os_val, self._browser_given_os.get("unknown", {"chrome": 1})
        )
        browser_val = weighted_choice(browser_counts, rng)
        return os_val, dev_val, browser_val

    def sample_location_tuple(self, rng: np.random.Generator,
                               exclude_country: str | None = None) -> tuple[str, str, str]:
        if exclude_country is not None:
            eligible_keys = [c for c in self._country_keys if c != exclude_country]
            eligible_weights = np.array(
                [self._country_marginal[c] for c in eligible_keys], dtype=float
            )
            eligible_weights /= eligible_weights.sum()
            country = eligible_keys[rng.choice(len(eligible_keys), p=eligible_weights)]
        else:
            country = self._country_keys[rng.choice(len(self._country_keys), p=self._country_weights)]
        region = self._sample_region(country, rng)
        asn    = self._sample_asn(country, rng)
        return country, region, asn

    def _sample_region(self, country: str, rng: np.random.Generator) -> str:
        if country in self._region_given_country:
            return weighted_choice(self._region_given_country[country], rng)
        return weighted_choice(self._region_marginal, rng)

    def _sample_asn(self, country: str, rng: np.random.Generator) -> str:
        if country in self._asn_given_country:
            return weighted_choice(self._asn_given_country[country], rng)
        return weighted_choice(self._asn_marginal, rng)

    def sample_rtt(self, rng: np.random.Generator) -> str:
        return weighted_choice(self._rtt_marginal, rng)

    def sample_event(self, rng: np.random.Generator) -> dict:
        os_val, dev_val, browser_val = self.sample_device_tuple(rng)
        country, region, asn = self.sample_location_tuple(rng)
        rtt = self.sample_rtt(rng)
        return {
            "os": os_val, "device_type": dev_val, "browser": browser_val,
            "country": country, "region": region, "asn_bucket": asn, "rtt_bucket": rtt,
        }

    def sample_spoof_k1(self, benign_event: dict, rng: np.random.Generator) -> dict:
        home_country = benign_event.get("country", "unknown")
        new_country, new_region, new_asn = self.sample_location_tuple(
            rng, exclude_country=home_country
        )
        spoof = dict(benign_event)
        spoof["country"] = new_country
        spoof["region"]  = new_region
        spoof["asn_bucket"] = new_asn
        return spoof

    def sample_spoof_k2(self, benign_event: dict, rng: np.random.Generator) -> dict:
        spoof = self.sample_spoof_k1(benign_event, rng)
        os_val, dev_val, browser_val = self.sample_device_tuple(rng)
        spoof["os"] = os_val
        spoof["device_type"] = dev_val
        spoof["browser"] = browser_val
        return spoof

    def sample_spoof_k3(self, benign_event: dict, rng: np.random.Generator) -> dict:
        spoof = self.sample_spoof_k1(benign_event, rng)
        os_val, dev_val, browser_val = self.sample_device_tuple(rng)
        spoof["os"] = os_val
        spoof["device_type"] = dev_val
        spoof["browser"] = browser_val
        spoof["rtt_bucket"] = self.sample_rtt(rng)
        return spoof


# ---------------------------------------------------------------------------
# Dataset generation — unchanged from source
# ---------------------------------------------------------------------------
def generate_accounts(marginals: Marginals, n_accounts: int, rng: np.random.Generator,
                      n_enroll_neg: int = N_ENROLL_NEG,
                      blocklist_lag: int = BLOCKLIST_LAG_DAYS,
                      attack_window: int = ATTACK_WINDOW_DAYS) -> tuple[list[dict], dict]:
    accounts = []
    for _ in range(n_accounts):
        home_event   = marginals.sample_event(rng)
        home_country = home_event["country"]
        train = []
        for i in range(N_TRAIN):
            if rng.random() < 0.70:
                evt = dict(home_event)
                evt["rtt_bucket"] = marginals.sample_rtt(rng)
            else:
                evt = marginals.sample_event(rng)
                if rng.random() < 0.80:
                    _, home_region, home_asn = marginals.sample_location_tuple(rng)
                    evt["country"]    = home_country
                    evt["region"]     = home_event["region"]
                    evt["asn_bucket"] = home_event["asn_bucket"]
            train.append(evt)

        known_devices = {device_key(e) for e in train}

        enroll_neg = []
        for _ in range(n_enroll_neg):
            if rng.random() < 0.70:
                evt = dict(home_event)
                evt["rtt_bucket"] = marginals.sample_rtt(rng)
            else:
                evt = marginals.sample_event(rng)
                evt["country"]    = home_country
                evt["region"]     = home_event["region"]
                evt["asn_bucket"] = home_event["asn_bucket"]
            enroll_neg.append(evt)

        base_event = train[int(rng.integers(0, N_CENTROID))]
        spoof_k1 = [marginals.sample_spoof_k1(base_event, rng) for _ in range(N_SPOOF_PER_ACCT)]
        spoof_k2 = [marginals.sample_spoof_k2(base_event, rng) for _ in range(N_SPOOF_PER_ACCT)]
        spoof_k3 = [marginals.sample_spoof_k3(base_event, rng) for _ in range(N_SPOOF_PER_ACCT)]

        novel_atk = []
        for _ in range(N_SPOOF_PER_ACCT):
            e = marginals.sample_event(rng)
            while e["country"] == home_event["country"]:
                e = marginals.sample_event(rng)
            novel_atk.append(e)

        accounts.append({
            "train": train, "centroid_events": train[:N_CENTROID],
            "known_devices": known_devices, "home_event": home_event,
            "enroll_neg": enroll_neg,
            "spoof_k1": spoof_k1, "spoof_k2": spoof_k2, "spoof_k3": spoof_k3,
            "novel_atk": novel_atk, "has_fleet": False, "fleet_atk": [],
        })

    fleet_device = marginals.sample_event(rng)
    fleet_key    = device_key(fleet_device)
    n_fleet = 0
    for acc in accounts:
        if rng.random() < FLEET_FRAC:
            replace_idx = int(rng.integers(0, N_TRAIN))
            acc["train"][replace_idx] = dict(fleet_device)
            acc["known_devices"].add(fleet_key)
            acc["has_fleet"] = True
            acc["fleet_atk"] = [dict(fleet_device) for _ in range(N_SPOOF_PER_ACCT)]
            n_fleet += 1

    fleet_accounts = [acc for acc in accounts if acc["has_fleet"]]
    if fleet_accounts:
        for acc in fleet_accounts:
            acc["fleet_attack_time"] = float(rng.uniform(0, attack_window))
        t_fleet_first = min(acc["fleet_attack_time"] for acc in fleet_accounts)
        t_blocklist   = t_fleet_first + blocklist_lag
        for acc in fleet_accounts:
            acc["fleet_blocklist_active"] = acc["fleet_attack_time"] >= t_blocklist
        n_pre_lag  = sum(1 for acc in fleet_accounts if not acc["fleet_blocklist_active"])
        n_post_lag = sum(1 for acc in fleet_accounts if acc["fleet_blocklist_active"])
    else:
        t_fleet_first, t_blocklist = 0.0, float(blocklist_lag)
        n_pre_lag, n_post_lag = 0, 0

    return accounts, {
        "fleet_device": fleet_device, "fleet_key": fleet_key,
        "n_fleet_accounts": n_fleet,
        "t_fleet_first": t_fleet_first, "t_blocklist": t_blocklist,
        "blocklist_lag": blocklist_lag,
        "n_pre_lag": n_pre_lag, "n_post_lag": n_post_lag,
    }


# ---------------------------------------------------------------------------
# Corpus and training — unchanged from source
# ---------------------------------------------------------------------------
def build_corpus(accounts: list[dict]) -> list[list[str]]:
    sentences = []
    for acc in accounts:
        flat = []
        for e in acc["train"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences


def train_model(corpus: list[list[str]], seed: int):
    return FastText(sentences=corpus, **{**ROBUST_KWARGS, "seed": seed})


def compute_centroids(accounts: list[dict], model) -> list[np.ndarray]:
    return [compute_centroid(acc["centroid_events"], model) for acc in accounts]


# ---------------------------------------------------------------------------
# Scoring — unchanged from source
# ---------------------------------------------------------------------------
def score_single_stage(test_event: dict, centroid: np.ndarray, model) -> float:
    return cosine_dist(embed_mp(test_event, model), centroid)

def score_trivial(test_event: dict, known_devices: set) -> float:
    return 0.0 if device_key(test_event) in known_devices else 1.0

def score_two_stage(test_event: dict, centroid: np.ndarray, known_devices: set, model) -> float:
    if device_key(test_event) in known_devices:
        return 0.0
    return score_single_stage(test_event, centroid, model)

def compute_baseline(calib_events: list, centroid: np.ndarray, model) -> np.ndarray:
    return np.array([cosine_dist(embed_mp(e, model), centroid) for e in calib_events])

def score_rank_norm(test_event: dict, centroid: np.ndarray, baseline: np.ndarray, model) -> float:
    raw = cosine_dist(embed_mp(test_event, model), centroid)
    return float(np.mean(baseline < raw))

def score_rank_norm_two_stage(test_event: dict, centroid: np.ndarray, baseline: np.ndarray,
                               known_devices: set, model) -> float:
    if device_key(test_event) in known_devices:
        return 0.0
    return score_rank_norm(test_event, centroid, baseline, model)


def collect_scores_for_attack(accounts: list[dict], centroids: list, model,
                               attack_key: str) -> dict:
    scores: dict = {s: ([], []) for s in SCORER_NAMES}
    for acc, centroid in zip(accounts, centroids):
        kd       = acc["known_devices"]
        calib    = acc["train"][N_CENTROID:]
        baseline = compute_baseline(calib, centroid, model)
        if attack_key == "novel":
            atk_events = acc["novel_atk"]
        elif attack_key in ("spoof_k1", "spoof_k2", "spoof_k3"):
            atk_events = acc[attack_key]
        else:
            if not acc["has_fleet"]:
                continue
            atk_events = acc["fleet_atk"]
        pairs = [(e, 0) for e in acc["enroll_neg"]] + [(e, 1) for e in atk_events]
        for e, lbl in pairs:
            scores["mp_raw"][0].append(score_single_stage(e, centroid, model))
            scores["mp_rank_norm"][0].append(score_rank_norm(e, centroid, baseline, model))
            scores["trivial"][0].append(score_trivial(e, kd))
            scores["two_stage"][0].append(score_two_stage(e, centroid, kd, model))
            scores["two_stage_rank_norm"][0].append(
                score_rank_norm_two_stage(e, centroid, baseline, kd, model))
            for s in SCORER_NAMES:
                scores[s][1].append(lbl)
    return {s: (np.array(v[0]), np.array(v[1])) for s, v in scores.items()}


def collect_fleet_scores(accounts: list[dict], centroids: list, model,
                          fleet_info: dict) -> dict:
    all_scorers = SCORER_NAMES + FLEET_EXTRA_SCORERS
    scores: dict = {s: ([], []) for s in all_scorers}
    fleet_key = fleet_info["fleet_key"]
    for acc, centroid in zip(accounts, centroids):
        if not acc["has_fleet"]:
            continue
        kd       = acc["known_devices"]
        calib    = acc["train"][N_CENTROID:]
        baseline = compute_baseline(calib, centroid, model)
        blocklist_active = acc.get("fleet_blocklist_active", False)
        pairs = [(e, 0) for e in acc["enroll_neg"]] + [(e, 1) for e in acc["fleet_atk"]]
        for e, lbl in pairs:
            is_fleet_device = (device_key(e) == fleet_key)
            blocklist_fires = is_fleet_device and blocklist_active
            mp   = score_single_stage(e, centroid, model)
            rn   = score_rank_norm(e, centroid, baseline, model)
            triv = score_trivial(e, kd)
            ts   = score_two_stage(e, centroid, kd, model)
            ts_rn = score_rank_norm_two_stage(e, centroid, baseline, kd, model)
            scores["mp_raw"][0].append(mp)
            scores["mp_rank_norm"][0].append(rn)
            scores["trivial"][0].append(triv)
            scores["two_stage"][0].append(ts)
            scores["two_stage_rank_norm"][0].append(ts_rn)
            scores["trivial_blocklist"][0].append(1.0 if blocklist_fires else triv)
            scores["two_stage_blocklist"][0].append(1.0 if blocklist_fires else ts)
            scores["combined"][0].append(1.0 if blocklist_fires else mp)
            for s in all_scorers:
                scores[s][1].append(lbl)
    return {s: (np.array(v[0]), np.array(v[1])) for s, v in scores.items()}


def collect_fleet_residual_scores(accounts: list[dict], centroids: list, model) -> dict:
    """
    Pre-lag (cold-start) fleet accounts only. Blocklist inactive here by
    definition, so blocklist variants fall through to their base scorers.
    All 8 scorers included to satisfy the Section 4 schema.
    """
    all_scorers = SCORER_NAMES + FLEET_EXTRA_SCORERS
    scores: dict = {s: ([], []) for s in all_scorers}
    for acc, centroid in zip(accounts, centroids):
        if not acc["has_fleet"] or acc.get("fleet_blocklist_active", False):
            continue
        kd       = acc["known_devices"]
        calib    = acc["train"][N_CENTROID:]
        baseline = compute_baseline(calib, centroid, model)
        pairs = [(e, 0) for e in acc["enroll_neg"]] + [(e, 1) for e in acc["fleet_atk"]]
        for e, lbl in pairs:
            mp   = score_single_stage(e, centroid, model)
            rn   = score_rank_norm(e, centroid, baseline, model)
            triv = score_trivial(e, kd)
            ts   = score_two_stage(e, centroid, kd, model)
            ts_rn = score_rank_norm_two_stage(e, centroid, baseline, kd, model)
            # Pre-lag: blocklist never fires (fleet_blocklist_active=False)
            scores["mp_raw"][0].append(mp)
            scores["mp_rank_norm"][0].append(rn)
            scores["trivial"][0].append(triv)
            scores["two_stage"][0].append(ts)
            scores["two_stage_rank_norm"][0].append(ts_rn)
            scores["trivial_blocklist"][0].append(triv)
            scores["two_stage_blocklist"][0].append(ts)
            scores["combined"][0].append(mp)
            for s in all_scorers:
                scores[s][1].append(lbl)
    return {s: (np.array(v[0]), np.array(v[1])) for s, v in scores.items()}


def top_pct_metrics(scores: np.ndarray, labels: np.ndarray, pct: float = 0.01,
                    seed: int = DEFAULT_SEED) -> dict:
    rng = np.random.default_rng(seed)
    n_flag = max(1, int(len(scores) * pct))
    perturbed = scores + rng.uniform(0, 1e-9, len(scores))
    threshold = np.sort(perturbed)[-n_flag]
    flagged = perturbed >= threshold
    tp = int(np.sum(flagged & (labels == 1)))
    fp = int(np.sum(flagged & (labels == 0)))
    fn = int(np.sum(~flagged & (labels == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec,
            "n_flagged": int(np.sum(flagged))}


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_auc(scores: np.ndarray, labels: np.ndarray, n: int,
                   rng: np.random.Generator) -> tuple[float, float, float, float, float, float]:
    """Returns (roc_mean, roc_lo, roc_hi, pr_mean, pr_lo, pr_hi) at 95%."""
    point_roc = roc_auc_score(labels, scores)
    point_pr  = average_precision_score(labels, scores)
    boot_roc, boot_pr = [], []
    idx = np.arange(len(labels))
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        sl, ss = labels[sample], scores[sample]
        if sl.sum() == 0 or sl.sum() == len(sl):
            continue
        boot_roc.append(roc_auc_score(sl, ss))
        boot_pr.append(average_precision_score(sl, ss))
    roc_lo, roc_hi = np.percentile(boot_roc, [2.5, 97.5])
    pr_lo,  pr_hi  = np.percentile(boot_pr,  [2.5, 97.5])
    return point_roc, roc_lo, roc_hi, point_pr, pr_lo, pr_hi


# ---------------------------------------------------------------------------
# T8: Token similarity (adapted for open H6 vocabulary)
# ---------------------------------------------------------------------------
def _feature_for_token(token: str) -> str | None:
    """Identify which feature a vocabulary token belongs to by prefix match."""
    for f in FEATURE_ORDER:
        if token.startswith(f + "_"):
            return f
    return None


def compute_t8(model) -> dict:
    """
    Within-feature vs cross-feature cosine similarity on H6 vocabulary.
    Adapts the H2 T8 approach for an open vocabulary: feature membership is
    determined by the '{feature}_' prefix used by make_token().
    """
    tokens, feat_types = [], []
    for token in model.wv.key_to_index:
        feat = _feature_for_token(token)
        if feat is not None:
            tokens.append(token)
            feat_types.append(feat)

    if not tokens:
        return {"within_feature_mean": 0.0, "cross_feature_mean": 0.0,
                "within_cross_ratio": 0.0, "collapse_detected": False}

    vecs = np.array([model.wv[t] for t in tokens])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / (norms + 1e-10)
    sim_mat = vecs_n @ vecs_n.T

    feat_arr = np.array(feat_types)
    n = len(tokens)
    within, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if feat_arr[i] == feat_arr[j]:
                within.append(sim_mat[i, j])
            else:
                cross.append(sim_mat[i, j])

    within_mean = float(np.mean(within)) if within else 0.0
    cross_mean  = float(np.mean(cross))  if cross  else 0.0
    ratio = within_mean / cross_mean if cross_mean > 1e-10 else float("inf")
    return {
        "within_feature_mean": within_mean,
        "cross_feature_mean":  cross_mean,
        "within_cross_ratio":  ratio,
        "collapse_detected":   within_mean > T8_COLLAPSE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------
def compute_verdicts(raw_results: dict) -> dict:
    """Derive the three Section 4 verdict booleans from raw bootstrap tuples.

    raw_results maps attack_key -> {scorer: (roc, roc_lo, roc_hi, pr, pr_lo, pr_hi, top1)}.
    Primary criterion uses PR-AUC and PR CIs: ROC-AUC hides the rank_norm collapse
    at 1:100 imbalance, so PR-AUC is the honest pass/fail metric for this regime.
    """
    k1 = raw_results.get("spoof_k1", {})

    # primary_criterion_confirmed: mp_raw PR-AUC > trivial PR-AUC, non-overlapping PR CIs
    if "mp_raw" in k1 and "trivial" in k1:
        mp_pr, mp_pr_lo, _mp_pr_hi = k1["mp_raw"][3], k1["mp_raw"][4], k1["mp_raw"][5]
        triv_pr, _triv_pr_lo, triv_pr_hi = k1["trivial"][3], k1["trivial"][4], k1["trivial"][5]
        primary_confirmed = bool(mp_pr_lo > triv_pr_hi and mp_pr > triv_pr)
    else:
        primary_confirmed = False

    # rank_norm_collapse_confirmed: mp_rank_norm PR-AUC < 50% of mp_raw PR-AUC
    if "mp_raw" in k1 and "mp_rank_norm" in k1:
        mp_raw_pr  = k1["mp_raw"][3]
        mp_rank_pr = k1["mp_rank_norm"][3]
        rank_norm_collapse = bool(mp_raw_pr > 0.0 and mp_rank_pr < 0.5 * mp_raw_pr)
    else:
        rank_norm_collapse = False

    # gate_blinds_fleet_confirmed: two_stage top-1% TP=0 on fleet_residual
    fr = raw_results.get("fleet_residual", {})
    if "two_stage" in fr:
        ts_tp = fr["two_stage"][6]["tp"]
        gate_blinds = bool(ts_tp == 0)
    else:
        gate_blinds = False

    return {
        "primary_criterion_confirmed":  primary_confirmed,
        "rank_norm_collapse_confirmed": rank_norm_collapse,
        "gate_blinds_fleet_confirmed":  gate_blinds,
    }


# ---------------------------------------------------------------------------
# Schema-conformant results builder
# ---------------------------------------------------------------------------
def build_scorer_block(roc: float, pr: float, pr_lo: float, pr_hi: float) -> dict:
    return {"roc_auc": roc, "pr_auc": pr, "ci_lower": pr_lo, "ci_upper": pr_hi}


def build_fleet_residual_scorer_block(roc: float, pr: float, pr_lo: float, pr_hi: float,
                                       top1: dict) -> dict:
    return {
        "roc_auc": roc, "pr_auc": pr, "ci_lower": pr_lo, "ci_upper": pr_hi,
        "top1pct_tp":   top1["tp"],
        "top1pct_prec": top1["precision"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="H6 Phase 2 Rerun")
    parser.add_argument("--smoke",     action="store_true",
                        help="Fast smoke check: 50 accounts, 200 bootstrap resamples")
    parser.add_argument("--seed",      type=int, default=DEFAULT_SEED)
    parser.add_argument("--neg-ratio",     type=int, default=100,
                        help="Enrollment negatives per spoof (default 100 per decision dd39e09c)")
    parser.add_argument("--blocklist-lag", type=int, default=BLOCKLIST_LAG_DAYS)
    parser.add_argument("--attack-window", type=int, default=ATTACK_WINDOW_DAYS)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_accounts  = N_ACCOUNTS_SMOKE if args.smoke else N_ACCOUNTS_FULL
    n_bootstrap = 200 if args.smoke else N_BOOTSTRAP
    n_enroll_neg = N_ENROLL_NEG * args.neg_ratio

    # Output directory
    out_dir = SEEDS_ROOT / f"seed_{args.seed}" / "h6"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"H6 Rerun  (seed={args.seed}, n_accounts={n_accounts}, "
          f"neg_ratio={args.neg_ratio}:1, "
          f"blocklist_lag={args.blocklist_lag}d"
          + (", SMOKE" if args.smoke else "") + ")")
    print("=" * 60)

    # [1] Load marginals
    if not MARGINALS_PATH.exists():
        sys.exit(
            f"ERROR: {MARGINALS_PATH} not found.\n"
            "Run: uv run experiments/rerun/scripts/h6/data_prep.py\n"
            "Requires: data/rba/rba.parquet at repo root."
        )
    print("\n[1/5] Loading marginals ...")
    marginals = Marginals(MARGINALS_PATH)

    # [2] Generate accounts
    print(f"\n[2/5] Generating {n_accounts} accounts ...")
    t0 = time.time()
    accounts, fleet_info = generate_accounts(
        marginals, n_accounts, rng, n_enroll_neg=n_enroll_neg,
        blocklist_lag=args.blocklist_lag, attack_window=args.attack_window,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Fleet accounts: {fleet_info['n_fleet_accounts']} "
          f"(pre-lag={fleet_info['n_pre_lag']}, post-lag={fleet_info['n_post_lag']})")

    # [3] Build corpus and train
    print("\n[3/5] Building corpus and training FastText (sg=1, per-account) ...")
    t0 = time.time()
    corpus    = build_corpus(accounts)
    vocab_size = len(set(tok for sent in corpus for tok in sent))
    print(f"  Corpus: {len(corpus)} sentences, {vocab_size} unique tokens")
    model = train_model(corpus, seed=args.seed)
    print(f"  Training done in {time.time()-t0:.1f}s")

    # [4] T8 — run before downstream metrics; abort if collapse detected
    print("\n[4/5] T8 token similarity check ...")
    t8 = compute_t8(model)
    print(f"  within_feature_mean={t8['within_feature_mean']:.4f}  "
          f"cross_feature_mean={t8['cross_feature_mean']:.4f}  "
          f"ratio={t8['within_cross_ratio']:.4f}  "
          f"collapse={t8['collapse_detected']}")
    if t8["collapse_detected"]:
        sys.exit(
            f"ABORT: T8 within_feature_mean={t8['within_feature_mean']:.4f} "
            f"> {T8_COLLAPSE_THRESHOLD} threshold under robust config. "
            f"Debug corpus construction before continuing. (seed={args.seed})"
        )

    # [5] Score all attack types
    print(f"\n[5/5] Computing centroids and evaluating (bootstrap n={n_bootstrap}) ...")
    centroids = compute_centroids(accounts, model)

    raw_scores:  dict = {}
    raw_results: dict = {}
    for atk in ATTACK_TYPES:
        if atk == "fleet":
            score_dict = collect_fleet_scores(accounts, centroids, model, fleet_info)
        elif atk == "fleet_residual":
            score_dict = collect_fleet_residual_scores(accounts, centroids, model)
        else:
            score_dict = collect_scores_for_attack(accounts, centroids, model, atk)

        raw_scores[atk] = score_dict
        raw_results[atk] = {}
        print(f"\n  [{atk}]:")
        for scorer, (sc, lab) in score_dict.items():
            if len(np.unique(lab)) < 2:
                print(f"    {scorer:25s}: skipped (no positive labels)")
                continue
            roc, roc_lo, roc_hi, pr, pr_lo, pr_hi = bootstrap_auc(sc, lab, n_bootstrap, rng)
            t1 = top_pct_metrics(sc, lab, seed=args.seed)
            raw_results[atk][scorer] = (roc, roc_lo, roc_hi, pr, pr_lo, pr_hi, t1)
            print(f"    {scorer:25s}: ROC={roc:.4f} [{roc_lo:.4f},{roc_hi:.4f}]  "
                  f"PR={pr:.4f}  top1%: tp={t1['tp']} prec={t1['precision']:.3f}")

    # ---------------------------------------------------------------------------
    # Build Section 4-conformant results schema
    # ---------------------------------------------------------------------------
    def _scorer_block(atk_key: str, scorer: str) -> dict | None:
        if scorer not in raw_results.get(atk_key, {}):
            return None
        v = raw_results[atk_key][scorer]
        return build_scorer_block(v[0], v[3], v[4], v[5])

    def _fleet_res_block(scorer: str) -> dict | None:
        if scorer not in raw_results.get("fleet_residual", {}):
            return None
        v = raw_results["fleet_residual"][scorer]
        return build_fleet_residual_scorer_block(v[0], v[3], v[4], v[5], v[6])

    def _roc_delta(atk_key: str, scorer_a: str, scorer_b: str) -> float | None:
        a = raw_results.get(atk_key, {}).get(scorer_a)
        b = raw_results.get(atk_key, {}).get(scorer_b)
        if a is None or b is None:
            return None
        return round(abs(a[0] - b[0]), 8)

    spoof_k1_scorers = ["mp_raw", "two_stage", "mp_rank_norm", "two_stage_rank_norm", "trivial"]
    spoof_k23_scorers = ["mp_raw", "two_stage", "mp_rank_norm", "two_stage_rank_norm", "trivial"]
    novel_scorers     = ["mp_raw", "two_stage", "mp_rank_norm", "two_stage_rank_norm", "trivial"]
    fleet_agg_scorers = ["mp_raw", "two_stage", "trivial_blocklist", "trivial"]
    fleet_res_scorers = ["mp_raw", "two_stage", "mp_rank_norm", "two_stage_rank_norm",
                         "trivial_blocklist", "two_stage_blocklist", "combined", "trivial"]

    results_schema: dict = {
        "seed":      args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_accounts":          n_accounts,
            "vocab_size":          vocab_size,
            "imbalance_ratio":     args.neg_ratio,
            "fleet_frac":          FLEET_FRAC,
            "blocklist_lag_days":  args.blocklist_lag,
            "attack_window_days":  args.attack_window,
            "neg_ratio":           args.neg_ratio,
            "n_fleet_accounts":    fleet_info["n_fleet_accounts"],
            "n_pre_lag":           fleet_info["n_pre_lag"],
            "n_post_lag":          fleet_info["n_post_lag"],
        },
        "spoof_k1": {
            s: _scorer_block("spoof_k1", s)
            for s in spoof_k1_scorers if _scorer_block("spoof_k1", s) is not None
        },
        "spoof_k2": {
            s: _scorer_block("spoof_k2", s)
            for s in spoof_k23_scorers if _scorer_block("spoof_k2", s) is not None
        },
        "spoof_k3": {
            s: _scorer_block("spoof_k3", s)
            for s in spoof_k23_scorers if _scorer_block("spoof_k3", s) is not None
        },
        "novel": {
            s: _scorer_block("novel", s)
            for s in novel_scorers if _scorer_block("novel", s) is not None
        },
        "fleet_aggregate": {
            s: _scorer_block("fleet", s)
            for s in fleet_agg_scorers if _scorer_block("fleet", s) is not None
        } | ({"two_stage_vs_trivial_roc_delta": d}
             if (d := _roc_delta("fleet", "two_stage", "trivial")) is not None else {}),
        "fleet_residual": {
            s: _fleet_res_block(s)
            for s in fleet_res_scorers if _fleet_res_block(s) is not None
        } | ({"two_stage_vs_trivial_roc_delta": d}
             if (d := _roc_delta("fleet_residual", "two_stage", "trivial")) is not None else {}),
        "t8_token_similarity": t8,
        "verdicts": compute_verdicts(raw_results),
    }

    # Write results.json
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_schema, f, indent=2)
    print(f"\n  Results: {results_path}")

    # Save raw score arrays for figure generation (PR curves, histograms)
    npz_arrays: dict = {}
    for atk, sdict in raw_scores.items():
        for scorer, (sc, lab) in sdict.items():
            key = f"{atk}__{scorer}"
            npz_arrays[f"{key}__s"] = sc
            npz_arrays[f"{key}__l"] = lab
    scores_path = out_dir / "scores.npz"
    np.savez_compressed(scores_path, **npz_arrays)
    print(f"  Scores:  {scores_path}")

    # ---------------------------------------------------------------------------
    # Smoke assertions — exit non-zero if any sanity check fails
    # ---------------------------------------------------------------------------
    if args.smoke:
        errors = []
        k1 = results_schema.get("spoof_k1", {})
        if "mp_raw" in k1 and k1["mp_raw"]["roc_auc"] <= 0.5:
            errors.append(f"smoke: mp_raw spoof_k1 ROC-AUC={k1['mp_raw']['roc_auc']:.4f} <= 0.5")
        if t8["collapse_detected"]:
            errors.append(f"smoke: T8 collapse detected (within={t8['within_feature_mean']:.4f})")
        if errors:
            for e in errors:
                print(f"ASSERTION FAILED: {e}", file=sys.stderr)
            sys.exit(1)
        print("\nSMOKE PASS — all assertions satisfied.")

    print("\nDone.")


if __name__ == "__main__":
    main()
