# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
#   "matplotlib>=3.7",
# ]
# ///
#
# H6 Hybrid Experiment
# ====================
# Tests whether FastText skip-gram mean-pool centroid scoring (two-stage
# architecture) beats trivial set-membership on k=1 country-change spoofs,
# using a hybrid dataset generated from real RBA marginals.
#
# Requires: h6_hybrid/experiments/rba_marginals.json
# (run data_prep.py first)
#
# Usage:
#   uv run h6_hybrid/experiments/hybrid_experiment.py           # full run
#   uv run h6_hybrid/experiments/hybrid_experiment.py --smoke   # 50 accounts, fast check
#   uv run h6_hybrid/experiments/hybrid_experiment.py --seed 99 # reproducibility

import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from gensim.models import FastText
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
MARGINALS_PATH = Path(__file__).resolve().parent / "rba_marginals.json"
FIGURES_DIR   = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Feature ordering must match H2-RBA FEATURE_ORDER exactly
FEATURE_ORDER = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]
DEVICE_KEY_FEATURES = ["os", "browser", "device_type", "country", "region", "asn_bucket"]

# FastText — identical to H2/H2-RBA robust config
ROBUST_KWARGS = dict(
    vector_size=64, window=6, sg=1, negative=10,
    min_count=1, epochs=20, min_n=3, max_n=6,
    seed=42, workers=1,
)

# Dataset parameters
N_ACCOUNTS_FULL  = 400
N_ACCOUNTS_SMOKE = 50
N_TRAIN          = 60    # target events per account
N_CENTROID       = 40    # events used for centroid (first N_CENTROID of training)
N_SPOOF_PER_ACCT = 5     # spoof/novel/fleet attack events per account
N_ENROLL_NEG     = 5     # enrollment negatives per account (scaled by neg_ratio)
FLEET_FRAC       = 0.25  # fraction of accounts with fleet device in training
N_BOOTSTRAP      = 1000
SEED             = 42

# Fleet blocklist model: cross-account deny-list that activates after a lag
# from the FIRST ATTACK event (not training appearance — training logins don't
# trigger customer complaints). rng state changes vs. pre-blocklist runs; do
# not diff against h6_metrics.json; run from scratch.
BLOCKLIST_LAG_DAYS  = 10   # days from first attack reported to blocklist activation
ATTACK_WINDOW_DAYS  = 30   # days over which fleet attacks are spread (CLI-overridable)
FLEET_EXTRA_SCORERS = ["trivial_blocklist", "two_stage_blocklist", "combined"]

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


# ---------------------------------------------------------------------------
# Weighted sampler from a counts dict
# ---------------------------------------------------------------------------
def weighted_choice(counts: dict, rng: np.random.Generator) -> str:
    keys = list(counts.keys())
    weights = np.array([counts[k] for k in keys], dtype=float)
    weights /= weights.sum()
    return keys[rng.choice(len(keys), p=weights)]


# ---------------------------------------------------------------------------
# Marginals loader
# ---------------------------------------------------------------------------
class Marginals:
    def __init__(self, path: Path):
        with open(path) as f:
            data = json.load(f)

        # os_device_joint: flat dict "os::device_type" -> count
        self._od_flat = data["os_device_joint"]
        # Rebuild nested dict
        self._od_nested: dict = defaultdict(dict)
        for key, cnt in self._od_flat.items():
            os_val, dev_val = key.split("::", 1)
            self._od_nested[os_val][dev_val] = cnt
        self._od_nested = dict(self._od_nested)

        self._browser_given_os   = data["browser_given_os"]
        self._country_marginal   = data["country_marginal"]
        self._region_given_country = data["region_given_country"]
        self._region_marginal    = data["region_marginal"]
        self._asn_given_country  = data["asn_given_country"]
        self._asn_marginal       = data["asn_marginal"]
        self._rtt_marginal       = data["rtt_marginal"]

        # Cache sorted list of countries for spoof (excluding home)
        self._country_keys = list(self._country_marginal.keys())
        self._country_weights = np.array(
            [self._country_marginal[c] for c in self._country_keys], dtype=float
        )
        self._country_weights /= self._country_weights.sum()

        meta = data.get("meta", {})
        print(f"  Marginals loaded — {meta.get('total_clean_rows', '?'):,} clean RBA rows")

    def sample_device_tuple(self, rng: np.random.Generator) -> tuple[str, str, str]:
        """Returns (os, device_type, browser) using chain sampling."""
        # Sample (os, device_type) jointly
        od_keys = list(self._od_flat.keys())
        od_weights = np.array(list(self._od_flat.values()), dtype=float)
        od_weights /= od_weights.sum()
        chosen_key = od_keys[rng.choice(len(od_keys), p=od_weights)]
        os_val, dev_val = chosen_key.split("::", 1)
        # Sample browser conditioned on os
        browser_counts = self._browser_given_os.get(os_val, self._browser_given_os.get("unknown", {"chrome": 1}))
        browser_val = weighted_choice(browser_counts, rng)
        return os_val, dev_val, browser_val

    def sample_location_tuple(self, rng: np.random.Generator, exclude_country: str | None = None) -> tuple[str, str, str]:
        """Returns (country, region, asn_bucket) using chain sampling."""
        if exclude_country is not None:
            # Sample from countries excluding home country (for k=1 spoof)
            eligible_keys = [c for c in self._country_keys if c != exclude_country]
            eligible_weights = np.array(
                [self._country_marginal[c] for c in eligible_keys], dtype=float
            )
            eligible_weights /= eligible_weights.sum()
            country = eligible_keys[rng.choice(len(eligible_keys), p=eligible_weights)]
        else:
            country = self._country_keys[rng.choice(len(self._country_keys), p=self._country_weights)]

        region  = self._sample_region(country, rng)
        asn     = self._sample_asn(country, rng)
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

    def sample_event(self, rng: np.random.Generator, home_country: str | None = None) -> dict:
        """Sample a full login event; home_country=None means normal login."""
        os_val, dev_val, browser_val = self.sample_device_tuple(rng)
        country, region, asn = self.sample_location_tuple(rng)
        rtt = self.sample_rtt(rng)
        return {
            "os": os_val,
            "device_type": dev_val,
            "browser": browser_val,
            "country": country,
            "region": region,
            "asn_bucket": asn,
            "rtt_bucket": rtt,
        }

    def sample_spoof_k1(self, benign_event: dict, rng: np.random.Generator) -> dict:
        """k=1 spoof: country changed to different country; region+asn resampled from
        new country's conditional. Device features and rtt_bucket unchanged."""
        home_country = benign_event.get("country", "unknown")
        new_country, new_region, new_asn = self.sample_location_tuple(rng, exclude_country=home_country)
        spoof = dict(benign_event)
        spoof["country"] = new_country
        spoof["region"]  = new_region
        spoof["asn_bucket"] = new_asn
        return spoof

    def sample_spoof_k2(self, benign_event: dict, rng: np.random.Generator) -> dict:
        """k=2 spoof: k=1 plus os+device_type changed (re-sample device tuple)."""
        spoof = self.sample_spoof_k1(benign_event, rng)
        os_val, dev_val, browser_val = self.sample_device_tuple(rng)
        spoof["os"] = os_val
        spoof["device_type"] = dev_val
        spoof["browser"] = browser_val
        return spoof

    def sample_spoof_k3(self, benign_event: dict, rng: np.random.Generator) -> dict:
        """k=3 spoof: k=1 plus all device features changed."""
        spoof = self.sample_spoof_k1(benign_event, rng)
        os_val, dev_val, browser_val = self.sample_device_tuple(rng)
        spoof["os"] = os_val
        spoof["device_type"] = dev_val
        spoof["browser"] = browser_val
        spoof["rtt_bucket"] = self.sample_rtt(rng)
        return spoof


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_accounts(marginals: Marginals, n_accounts: int, rng: np.random.Generator,
                      n_enroll_neg: int = N_ENROLL_NEG,
                      blocklist_lag: int = BLOCKLIST_LAG_DAYS,
                      attack_window: int = ATTACK_WINDOW_DAYS) -> tuple[list[dict], dict]:
    """
    Generate n_accounts synthetic accounts with training events and test events.

    Each account:
      - Picks a home device (consistent os/device_type/browser) and home country
      - Generates N_TRAIN training events, most using the home profile (with variation)
      - Generates test events:
        * N_ENROLL_NEG enrollment negatives (known training devices, benign)
        * N_SPOOF_PER_ACCT each of k=1, k=2, k=3 spoofs
    """
    accounts = []
    for _ in range(n_accounts):
        # Establish home profile by sampling once
        home_event = marginals.sample_event(rng)
        home_country = home_event["country"]

        # Generate training history: mix of home device events with some variation
        train = []
        for i in range(N_TRAIN):
            if rng.random() < 0.70:
                # Re-use home device profile, vary only rtt
                evt = dict(home_event)
                evt["rtt_bucket"] = marginals.sample_rtt(rng)
            else:
                # Sample a fresh event (introduces device diversity)
                evt = marginals.sample_event(rng)
                # But stay in home country 80% of the time (realistic travel pattern)
                if rng.random() < 0.80:
                    country, region, asn = marginals.sample_location_tuple(rng)
                    evt["country"] = home_country  # keep country fixed
                    # region/asn from home country conditional
                    _, home_region, home_asn = marginals.sample_location_tuple(
                        rng, exclude_country=None
                    )
                    # Override with a region/asn still in home country space
                    evt["region"] = home_event["region"]
                    evt["asn_bucket"] = home_event["asn_bucket"]
            train.append(evt)

        known_devices = {device_key(e) for e in train}

        # Enrollment negatives: fresh benign events using same home-profile generator.
        # NOT copies of training events — that would make trivial AUC = 1.0 by
        # construction (all negatives known, all spoofs novel).
        # With the 70/30 home-vs-fresh split, ~70% will hit the home device_key
        # (known, trivial score=0) and ~30% will be novel (trivial score=1),
        # producing a realistic non-degenerate trivial AUC.
        enroll_neg = []
        for _ in range(n_enroll_neg):
            if rng.random() < 0.70:
                evt = dict(home_event)
                evt["rtt_bucket"] = marginals.sample_rtt(rng)
            else:
                evt = marginals.sample_event(rng)
                evt["country"] = home_country
                evt["region"] = home_event["region"]
                evt["asn_bucket"] = home_event["asn_bucket"]
            enroll_neg.append(evt)

        # Spoofs: generate from one benign event as the base
        base_event = train[int(rng.integers(0, N_CENTROID))]

        spoof_k1 = [marginals.sample_spoof_k1(base_event, rng) for _ in range(N_SPOOF_PER_ACCT)]
        spoof_k2 = [marginals.sample_spoof_k2(base_event, rng) for _ in range(N_SPOOF_PER_ACCT)]
        spoof_k3 = [marginals.sample_spoof_k3(base_event, rng) for _ in range(N_SPOOF_PER_ACCT)]

        # Novel attack: completely new device sampled freely, guaranteed different country
        novel_atk = []
        for _ in range(N_SPOOF_PER_ACCT):
            e = marginals.sample_event(rng)
            while e["country"] == home_event["country"]:
                e = marginals.sample_event(rng)
            novel_atk.append(e)

        accounts.append({
            "train": train,
            "centroid_events": train[:N_CENTROID],
            "known_devices": known_devices,
            "home_event": home_event,
            "enroll_neg": enroll_neg,
            "spoof_k1": spoof_k1,
            "spoof_k2": spoof_k2,
            "spoof_k3": spoof_k3,
            "novel_atk": novel_atk,
            "has_fleet": False,   # filled in below
            "fleet_atk": [],
        })

    # Fleet device: one global device injected into FLEET_FRAC of accounts' training
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

    # Assign temporal attack times uniformly over attack_window days.
    # The first fleet attack triggers the complaint; blocklist activates at
    # t_fleet_first + blocklist_lag. Training logins with the fleet device
    # are normal usage and never trigger complaints.
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
        "fleet_device":     fleet_device,
        "fleet_key":        fleet_key,
        "n_fleet_accounts": n_fleet,
        "t_fleet_first":    t_fleet_first,
        "t_blocklist":      t_blocklist,
        "blocklist_lag":    blocklist_lag,
        "n_pre_lag":        n_pre_lag,
        "n_post_lag":       n_post_lag,
    }


# ---------------------------------------------------------------------------
# Corpus and embedding
# ---------------------------------------------------------------------------
def build_corpus(accounts: list[dict]) -> list[list[str]]:
    """Per-account concatenated corpus — identical invariant to H2/H2-RBA."""
    sentences = []
    for acc in accounts:
        flat = []
        for e in acc["train"]:
            flat.extend(device_to_tokens(e))
        if flat:
            sentences.append(flat)
    return sentences


def train_model(corpus: list[list[str]], seed: int = SEED):
    model = FastText(sentences=corpus, **{**ROBUST_KWARGS, "seed": seed})
    return model


def compute_centroids(accounts: list[dict], model) -> list[np.ndarray]:
    return [compute_centroid(acc["centroid_events"], model) for acc in accounts]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_single_stage(test_event: dict, centroid: np.ndarray, model) -> float:
    """Cosine distance from event embedding to centroid."""
    emb = embed_mp(test_event, model)
    return cosine_dist(emb, centroid)


def score_trivial(test_event: dict, known_devices: set) -> float:
    """0 if known device, 1 if novel."""
    return 0.0 if device_key(test_event) in known_devices else 1.0


def score_two_stage(test_event: dict, centroid: np.ndarray, known_devices: set, model) -> float:
    """Stage 1: known device → 0. Stage 2: embedding distance for novel."""
    if device_key(test_event) in known_devices:
        return 0.0
    return score_single_stage(test_event, centroid, model)


def compute_baseline(calib_events: list, centroid: np.ndarray, model) -> np.ndarray:
    """Per-account calibration CDF: cosine distances of held-out calib events."""
    return np.array([cosine_dist(embed_mp(e, model), centroid) for e in calib_events])


def score_rank_norm(test_event: dict, centroid: np.ndarray, baseline: np.ndarray, model) -> float:
    """Rank-normalize raw cosine distance against per-user calibration distribution."""
    raw = cosine_dist(embed_mp(test_event, model), centroid)
    return float(np.mean(baseline < raw))


def score_rank_norm_two_stage(test_event: dict, centroid: np.ndarray, baseline: np.ndarray,
                              known_devices: set, model) -> float:
    if device_key(test_event) in known_devices:
        return 0.0
    return score_rank_norm(test_event, centroid, baseline, model)


ATTACK_TYPES = ["spoof_k1", "spoof_k2", "spoof_k3", "novel", "fleet", "fleet_residual"]
SCORER_NAMES = ["mp_raw", "mp_rank_norm", "trivial", "two_stage", "two_stage_rank_norm"]


def collect_scores_for_attack(accounts: list[dict], centroids: list, model,
                               attack_key: str) -> dict:
    """
    Collect (score, label) pairs for a given attack type.
    attack_key: one of spoof_k1/k2/k3 / novel / fleet
    Negatives: enrollment negatives (label=0). Fleet only uses fleet accounts.
    """
    scores: dict = {s: ([], []) for s in SCORER_NAMES}

    for acc, centroid in zip(accounts, centroids):
        kd       = acc["known_devices"]
        calib    = acc["train"][N_CENTROID:]
        baseline = compute_baseline(calib, centroid, model)

        # Select attack events; skip account if it has no fleet events
        if attack_key == "fleet":
            if not acc["has_fleet"]:
                continue
            atk_events = acc["fleet_atk"]
        elif attack_key == "novel":
            atk_events = acc["novel_atk"]
        else:
            atk_events = acc[attack_key]

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
    """
    Fleet scorer collection with 8 scorers: 5 standard + 3 blocklist variants.
    Blocklist activates per-account at fleet_info['t_blocklist']. Pre-lag attacks
    are in the cold-start window (only mp_raw can detect); post-lag attacks are
    caught by blocklist scorers as well.
    """
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

            mp    = score_single_stage(e, centroid, model)
            rn    = score_rank_norm(e, centroid, baseline, model)
            triv  = score_trivial(e, kd)
            ts    = score_two_stage(e, centroid, kd, model)
            ts_rn = score_rank_norm_two_stage(e, centroid, baseline, kd, model)

            scores["mp_raw"][0].append(mp)
            scores["mp_rank_norm"][0].append(rn)
            scores["trivial"][0].append(triv)
            scores["two_stage"][0].append(ts)
            scores["two_stage_rank_norm"][0].append(ts_rn)
            # Blocklist variants: 1.0 if blocklist fires; else fall through to base scorer
            scores["trivial_blocklist"][0].append(1.0 if blocklist_fires else triv)
            scores["two_stage_blocklist"][0].append(1.0 if blocklist_fires else ts)
            scores["combined"][0].append(1.0 if blocklist_fires else mp)

            for s in all_scorers:
                scores[s][1].append(lbl)

    return {s: (np.array(v[0]), np.array(v[1])) for s, v in scores.items()}


def collect_fleet_residual_scores(accounts: list[dict], centroids: list, model) -> dict:
    """
    Fleet evaluation restricted to cold-start (pre-lag) accounts only.
    Post-lag accounts are excluded — in production they're blocked upstream
    before reaching the model. This gives a clean view of residual risk:
    what fraction of cold-start fleet attacks does the embedding catch?
    """
    scores: dict = {s: ([], []) for s in SCORER_NAMES}

    for acc, centroid in zip(accounts, centroids):
        if not acc["has_fleet"]:
            continue
        if acc.get("fleet_blocklist_active", False):
            continue  # excluded: blocked upstream, never reaches the model
        kd       = acc["known_devices"]
        calib    = acc["train"][N_CENTROID:]
        baseline = compute_baseline(calib, centroid, model)

        pairs = [(e, 0) for e in acc["enroll_neg"]] + [(e, 1) for e in acc["fleet_atk"]]
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


def top_pct_metrics(scores: np.ndarray, labels: np.ndarray, pct: float = 0.01,
                    seed: int = SEED) -> dict:
    """Threshold at top-pct of scores; return precision/recall/F1/counts."""
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
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec,
            "recall": rec, "f1": f1, "n_flagged": int(np.sum(flagged))}


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_auc(scores: np.ndarray, labels: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float, float, float, float, float]:
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
# Plotting
# ---------------------------------------------------------------------------
def plot_auc_comparison(results: dict, n_accounts: int, seed: int, smoke: bool) -> None:
    """Bar chart of AUC ± CI for each scorer at each k level."""
    scorer_labels = {
        "mp_raw": "MP-Raw", "mp_rank_norm": "Rank-Norm",
        "trivial": "Trivial", "two_stage": "2-Stage", "two_stage_rank_norm": "2-Stage RN",
        "trivial_blocklist": "Trivial+BL", "two_stage_blocklist": "2-Stage+BL",
        "combined": "Combined",
    }
    colors = {
        "mp_raw": "#4878CF", "mp_rank_norm": "#9467BD",
        "trivial": "#D65F5F", "two_stage": "#6ACC65", "two_stage_rank_norm": "#2CA02C",
        "trivial_blocklist": "#FF7F0E", "two_stage_blocklist": "#BCBD22", "combined": "#17BECF",
    }

    plot_atks = [atk for atk in ATTACK_TYPES if atk in results and results[atk]]
    fig, axes = plt.subplots(1, len(plot_atks), figsize=(4 * len(plot_atks) + 2, 5), sharey=True)
    if len(plot_atks) == 1:
        axes = [axes]
    for ax, atk in zip(axes, plot_atks):
        atk_scorers = list(results[atk].keys())
        x = np.arange(len(atk_scorers))
        aucs, lo_errs, hi_errs = [], [], []
        for scorer in atk_scorers:
            auc, lo, hi = results[atk][scorer][0], results[atk][scorer][1], results[atk][scorer][2]
            aucs.append(auc); lo_errs.append(auc - lo); hi_errs.append(hi - auc)
        bars = ax.bar(x, aucs, yerr=[lo_errs, hi_errs],
                      color=[colors.get(s, "#888888") for s in atk_scorers],
                      capsize=5, width=0.6, alpha=0.85)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{atk} (n={n_accounts})")
        ax.set_xticks(x)
        ax.set_xticklabels([scorer_labels.get(s, s) for s in atk_scorers], fontsize=7, rotation=20)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("ROC-AUC" if atk == plot_atks[0] else "")
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{auc:.3f}", ha="center", va="bottom", fontsize=7)

    suffix = "_smoke" if smoke else ""
    fig.suptitle(f"H6 Hybrid: AUC by Spoof Level (seed={seed}{', SMOKE' if smoke else ''})", y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / f"h6_hybrid_auc{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_roc_curves(accounts: list, centroids: list, model, attack_key: str, smoke: bool,
                    score_dict: dict | None = None) -> None:
    """ROC curves for all scorers at a given attack type."""
    if score_dict is None:
        score_dict = collect_scores_for_attack(accounts, centroids, model, attack_key)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {
        "mp_raw": "#4878CF", "mp_rank_norm": "#9467BD",
        "trivial": "#D65F5F", "two_stage": "#6ACC65", "two_stage_rank_norm": "#2CA02C",
        "trivial_blocklist": "#FF7F0E", "two_stage_blocklist": "#BCBD22", "combined": "#17BECF",
    }
    labels = {
        "mp_raw": "MP-Raw", "mp_rank_norm": "Rank-Norm",
        "trivial": "Trivial", "two_stage": "2-Stage", "two_stage_rank_norm": "2-Stage RN",
        "trivial_blocklist": "Trivial+BL", "two_stage_blocklist": "2-Stage+BL",
        "combined": "Combined (BL+Embed)",
    }
    for scorer, (sc, lab) in score_dict.items():
        auc = roc_auc_score(lab, sc)
        fpr, tpr, _ = roc_curve(lab, sc)
        ax.plot(fpr, tpr, label=f"{labels.get(scorer, scorer)} (AUC={auc:.3f})",
                color=colors.get(scorer, "#888888"), lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"H6 ROC — {attack_key}{' (smoke)' if smoke else ''}")
    ax.legend(loc="lower right")
    suffix = "_smoke" if smoke else ""
    out = FIGURES_DIR / f"h6_roc_{attack_key}{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="H6 Hybrid Experiment")
    parser.add_argument("--smoke",     action="store_true", help="Fast check: 50 accounts, no bootstrap")
    parser.add_argument("--seed",      type=int, default=SEED, help="Random seed")
    parser.add_argument("--neg-ratio",     type=int, default=1,
                        help="Enrollment negatives per spoof (1=balanced, 100=realistic imbalance)")
    parser.add_argument("--blocklist-lag", type=int, default=BLOCKLIST_LAG_DAYS,
                        help="Days from first fleet attack to blocklist activation")
    parser.add_argument("--attack-window", type=int, default=ATTACK_WINDOW_DAYS,
                        help="Days over which fleet attacks are spread")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_accounts = N_ACCOUNTS_SMOKE if args.smoke else N_ACCOUNTS_FULL
    n_bootstrap = 200 if args.smoke else N_BOOTSTRAP
    n_enroll_neg = N_ENROLL_NEG * args.neg_ratio

    print("=" * 60)
    print(f"H6 Hybrid Experiment  (seed={args.seed}, n_accounts={n_accounts}, "
          f"neg_ratio={args.neg_ratio}:1, "
          f"blocklist_lag={args.blocklist_lag}d, attack_window={args.attack_window}d"
          + (", SMOKE" if args.smoke else "") + ")")
    print("=" * 60)

    # Load marginals
    if not MARGINALS_PATH.exists():
        import sys
        sys.exit(f"ERROR: {MARGINALS_PATH} not found. Run data_prep.py first.")
    print("\n[1/5] Loading marginals ...")
    marginals = Marginals(MARGINALS_PATH)

    # Generate accounts
    print(f"\n[2/5] Generating {n_accounts} accounts (neg_ratio={args.neg_ratio}) ...")
    t0 = time.time()
    accounts, fleet_info = generate_accounts(
        marginals, n_accounts, rng, n_enroll_neg=n_enroll_neg,
        blocklist_lag=args.blocklist_lag, attack_window=args.attack_window,
    )
    n_pos = sum(len(a["spoof_k1"]) for a in accounts)
    n_neg = sum(len(a["enroll_neg"]) for a in accounts)
    n_fleet_accts = fleet_info["n_fleet_accounts"]
    print(f"  Generated in {time.time()-t0:.1f}s")
    print(f"  Spoofs per k: {n_pos} | Enrollment negs: {n_neg} | Fleet accounts: {n_fleet_accts}")
    print(f"  Fleet blocklist: lag={fleet_info['blocklist_lag']}d, "
          f"first_attack=t{fleet_info['t_fleet_first']:.1f}d → "
          f"active=t{fleet_info['t_blocklist']:.1f}d")
    print(f"  Fleet cold-start (pre-lag): {fleet_info['n_pre_lag']} accts | "
          f"Post-lag (caught): {fleet_info['n_post_lag']} accts")

    # Build corpus and train
    print("\n[3/5] Building corpus and training FastText ...")
    t0 = time.time()
    corpus = build_corpus(accounts)
    vocab_size = len(set(tok for sent in corpus for tok in sent))
    print(f"  Corpus: {len(corpus)} sentences, {vocab_size} unique tokens")
    model = train_model(corpus, seed=args.seed)
    print(f"  Training done in {time.time()-t0:.1f}s")

    # Compute centroids
    print("\n[4/5] Computing centroids ...")
    centroids = compute_centroids(accounts, model)

    # Evaluate all attack types
    print(f"\n[5/5] Evaluating (bootstrap n={n_bootstrap}) ...")
    results: dict = {}
    fleet_score_dict: dict | None = None
    fleet_residual_score_dict: dict | None = None
    for atk in ATTACK_TYPES:
        if atk == "fleet":
            score_dict = collect_fleet_scores(accounts, centroids, model, fleet_info)
            fleet_score_dict = score_dict
        elif atk == "fleet_residual":
            score_dict = collect_fleet_residual_scores(accounts, centroids, model)
            fleet_residual_score_dict = score_dict
        else:
            score_dict = collect_scores_for_attack(accounts, centroids, model, atk)
        results[atk] = {}
        print(f"\n  [{atk}]:")
        for scorer, (sc, lab) in score_dict.items():
            if len(np.unique(lab)) < 2:
                print(f"    {scorer:22s}: skipped (no positive labels)")
                continue
            roc, roc_lo, roc_hi, pr, pr_lo, pr_hi = bootstrap_auc(sc, lab, n_bootstrap, rng)
            t1 = top_pct_metrics(sc, lab)
            results[atk][scorer] = (roc, roc_lo, roc_hi, pr, pr_lo, pr_hi, t1)
            print(f"    {scorer:22s}: ROC={roc:.4f}  PR={pr:.4f}  "
                  f"top1%: prec={t1['precision']:.3f} rec={t1['recall']:.3f} "
                  f"(tp={t1['tp']} fp={t1['fp']})")

    # Determine success
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    k1_two_stage = results["spoof_k1"]["two_stage"]
    k1_trivial   = results["spoof_k1"]["trivial"]

    two_stage_auc, two_stage_lo, two_stage_hi = k1_two_stage[0], k1_two_stage[1], k1_two_stage[2]
    trivial_auc,   trivial_lo,   trivial_hi   = k1_trivial[0],   k1_trivial[1],   k1_trivial[2]

    trivial_expected = abs(trivial_auc - 0.75) < 0.10
    if trivial_expected:
        print(f"  Trivial AUC={trivial_auc:.4f} (near expected ~0.75)")
    else:
        print(f"  NOTE: Trivial AUC={trivial_auc:.4f} deviates from expected ~0.75")
        print(f"  Pre-registered: restate as additive lift over trivial")

    lift = two_stage_auc - trivial_auc
    ci_non_overlap = two_stage_lo > trivial_hi
    print(f"\n  Two-Stage k=1 AUC: {two_stage_auc:.4f}  [{two_stage_lo:.4f}, {two_stage_hi:.4f}]")
    print(f"  Trivial k=1 AUC:   {trivial_auc:.4f}  [{trivial_lo:.4f}, {trivial_hi:.4f}]")
    print(f"  Lift over trivial: {lift:+.4f}")
    print(f"  Non-overlapping CIs: {ci_non_overlap}")

    if ci_non_overlap and lift > 0:
        print("\n  RESULT: CONFIRMED — two-stage embedding beats trivial on k=1 (non-overlapping CIs)")
    elif lift > 0:
        print("\n  RESULT: INCONCLUSIVE — two-stage embedding higher than trivial but CIs overlap")
    else:
        print("\n  RESULT: REFUTED — two-stage embedding does not beat trivial on k=1")

    # Save metrics JSON
    metrics_path = FIGURES_DIR / ("h6_metrics_smoke.json" if args.smoke else "h6_metrics.json")
    metrics_out = {}
    for atk in ATTACK_TYPES:
        metrics_out[atk] = {
            scorer: {"roc_auc": v[0], "roc_lo": v[1], "roc_hi": v[2],
                     "pr_auc":  v[3], "pr_lo":  v[4], "pr_hi":  v[5],
                     "top1pct": v[6]}
            for scorer, v in results[atk].items()
        }
    metrics_out["verdict"] = {
        "lift_k1": float(lift),
        "ci_non_overlap_k1": bool(ci_non_overlap),
        "trivial_k1_near_075": bool(trivial_expected),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  Metrics: {metrics_path}")

    # Plots
    print("\nGenerating plots ...")
    plot_auc_comparison(results, n_accounts, args.seed, args.smoke)
    for atk in ATTACK_TYPES:
        if results.get(atk):
            if atk == "fleet":
                sd = fleet_score_dict
            elif atk == "fleet_residual":
                sd = fleet_residual_score_dict
            else:
                sd = None
            plot_roc_curves(accounts, centroids, model, atk, args.smoke, score_dict=sd)

    print("\nDone.")


if __name__ == "__main__":
    main()
