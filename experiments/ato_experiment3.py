# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.8",
#   "scipy>=1.12",
# ]
# ///
"""
ATO Experiment 3 — Realistic Data Redesign & Feature-Based Embeddings
======================================================================
Builds on Experiment 2's finding that global OOV (AUC 0.989) was trivially
dominant because synthetic data gave every account unique device IDs.

This experiment:
  1. Redesigns data so fleet devices appear in global vocabulary (25% of
     training accounts receive injected fleet-device login events at training
     time, simulating unconfirmed prior attacks).
  2. Tests account-level OOV as a signal that should survive the fleet case.
  3. Replaces opaque device-ID embeddings with structured feature-token
     embeddings (os, browser, tz, lang, net, screen), providing a stable,
     bounded vocabulary for novel devices.
  4. Introduces a spoof attack type: same OS+browser+lang as victim, different
     timezone — tests whether embeddings or exact-profile novelty catches evasion.

Evaluation redesign (v2)
-----------------------
The initial run revealed a false-positive blind spot: legit eval events were
drawn exclusively from the account's existing known devices. Every OOV-based
signal (global_oov, account_oov, feature_novelty) achieved AUC 1.000 trivially
because the legit class always looked "known" and the attack class always looked
"new." The critical false-positive case — a legitimate user enrolling a new
device — was absent.

Fix: a fourth eval event type, `enroll`, represents legitimate new device
enrollment. It has a brand-new device ID (OOV globally and per-account) and a
feature profile consistent with the victim's primary profile (same OS+browser+
tz+lang, randomised net/screen). Label = 0 (legitimate). The negative class for
AUC computation is legit + enroll combined. This forces each signal to
distinguish "new device that belongs here" from "attack device."

Deliberate scope exclusions
----------------------------
- No temporal decay of account history (all history weighted equally).
- No Procrustes alignment across account embedding spaces.
- Fleet devices are included in centroid computation for targeted accounts
  (realistic: at training time we don't know those logins were attacks).
- Profile spoofing is deterministic: same OS+browser+lang, different TZ only.
- No account-level feature novelty with continuous scoring (feature_novelty
  is binary exact-match on profile tuple).
"""

import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy.linalg import norm
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec, FastText

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)

# ── constants ─────────────────────────────────────────────────────────────────
N_ACCOUNTS = 400
N_FLEET_DEVICES = 80
EVENTS_PER_ACCT = 60
ATTACK_EVENTS = 80  # per attack type
FLEET_INJECT = 8  # fleet events injected into targeted training accts
FLEET_TARGET_FRAC = 0.25  # fraction of accounts previously targeted
STAY_PROB = 0.70  # Markov stay probability
VEC_DIM = 64
ID_WINDOW = 5
FEAT_WINDOW = 6  # one full login event = 6 feature tokens
N_NEGATIVE = 10
EPOCHS = 20
N_BOOTSTRAP = 1000

FEATURE_KEYS = ["os", "browser", "tz", "lang", "net", "screen"]
FEATURE_VALUES = {
    "os": ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz": ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang": ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "net": ["wifi", "lte", "5g", "broadband"],
    "screen": ["small", "medium", "large", "xlarge"],
}

# Pre-built token strings: "os_ios", "browser_safari", etc.
ALL_FEATURE_TOKENS = [f"{k}_{v}" for k in FEATURE_KEYS for v in FEATURE_VALUES[k]]

Profile = Tuple[str, str, str, str, str, str]  # (os, browser, tz, lang, net, screen)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def make_device_id(prefix: str = "dev") -> str:
    suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=12))
    return f"{prefix}_{suffix}"


def sample_primary_profile() -> Profile:
    return tuple(rng.choice(FEATURE_VALUES[k]) for k in FEATURE_KEYS)  # type: ignore[return-value]


def profile_to_tokens(profile: Profile) -> List[str]:
    """Convert a profile tuple to its 6 feature tokens."""
    return [f"{k}_{v}" for k, v in zip(FEATURE_KEYS, profile)]


def vary_profile_net_screen(base: Profile) -> Profile:
    """Keep os/browser/tz/lang, randomise net/screen."""
    os_, browser, tz, lang = base[0], base[1], base[2], base[3]
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    denom = norm(a) * norm(b)
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def bootstrap_auc_ci(
    y_true: List[int],
    y_score: List[float],
    n: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    """Returns (auc, ci_lo, ci_hi) using percentile bootstrap."""
    yt = np.array(y_true)
    ys = np.array(y_score)
    if len(np.unique(yt)) < 2:
        return (float("nan"), float("nan"), float("nan"))
    auc = roc_auc_score(yt, ys)
    rs = np.random.RandomState(seed)
    boot_aucs = []
    for _ in range(n):
        idx = rs.choice(len(yt), size=len(yt), replace=True)
        if len(np.unique(yt[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt[idx], ys[idx]))
    lo = float(np.percentile(boot_aucs, 2.5))
    hi = float(np.percentile(boot_aucs, 97.5))
    return auc, lo, hi


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Device:
    device_id: str
    profile: Profile


@dataclass
class Account:
    account_id: str
    primary_profile: Profile
    known_devices: List[Device]  # 2–4 devices
    known_device_ids: Set[str] = field(default_factory=set)
    observed_profiles: Set[Profile] = field(default_factory=set)
    # corpus: list of device_id strings (login sequence)
    id_corpus: List[str] = field(default_factory=list)
    # feature corpus: list of token lists, one per login event
    feature_corpus: List[List[str]] = field(default_factory=list)


@dataclass
class EvalEvent:
    account: Account
    device_id: str
    profile: Profile
    label: int  # 1 = attack, 0 = legitimate


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def build_fleet(n: int) -> List[Device]:
    """
    80 shared fleet devices. Each has an 'attacker profile':
    - Different OS from victim (we randomise it; attack evaluation picks victims
      with a different OS family).
    - Timezone at least 2 slots away from reference (enforced at attack time).
    - Different language.
    We just give fleet devices a fixed 'attacker profile' here; attack
    construction will re-use whichever fleet device's stored profile.
    """
    devices = []
    for i in range(n):
        # Attacker-skewed profile: pick a distinctive combination.
        os_ = rng.choice(["windows", "linux"])  # attacker OS pool
        browser = rng.choice(["chrome", "firefox"])
        tz = rng.choice(["utc+5", "utc+8"])  # "foreign" timezones
        lang = rng.choice(["zh_cn", "de_de", "fr_fr"])  # non-en language
        net = rng.choice(FEATURE_VALUES["net"])
        screen = rng.choice(FEATURE_VALUES["screen"])
        profile: Profile = (os_, browser, tz, lang, net, screen)
        devices.append(Device(device_id=make_device_id("fleet"), profile=profile))
    return devices


def build_accounts(fleet: List[Device]) -> List[Account]:
    """
    Build 400 accounts. 25% are 'previously targeted': receive FLEET_INJECT
    fleet-device login events mixed into their training corpus.
    """
    accounts = []
    targeted_count = int(N_ACCOUNTS * FLEET_TARGET_FRAC)
    targeted_indices = set(rng.sample(range(N_ACCOUNTS), targeted_count))

    for i in range(N_ACCOUNTS):
        acct_id = f"acct_{i:04d}"
        primary = sample_primary_profile()

        # 2–4 known devices, each a slight variation of primary profile
        n_devs = rng.randint(2, 4)
        known_devices = []
        for _ in range(n_devs):
            dev_profile = vary_profile_net_screen(primary)
            known_devices.append(
                Device(device_id=make_device_id("legit"), profile=dev_profile)
            )

        acct = Account(
            account_id=acct_id,
            primary_profile=primary,
            known_devices=known_devices,
        )
        acct.known_device_ids = {d.device_id for d in known_devices}
        accounts.append(acct)

    # Populate corpora; targeted accounts also get fleet injections.
    for i, acct in enumerate(accounts):
        devices = acct.known_devices
        is_targeted = i in targeted_indices

        # Inject fleet events first (so fleet device IDs enter global vocab)
        if is_targeted:
            injected = rng.choices(fleet, k=FLEET_INJECT)
            for dev in injected:
                acct.id_corpus.append(dev.device_id)
                acct.feature_corpus.append(profile_to_tokens(dev.profile))
                acct.observed_profiles.add(dev.profile)
                # NOTE: we do NOT add fleet IDs to known_device_ids — that set
                # represents the account owner's own devices.

        # Normal login events (i.i.d. uniform over known devices)
        for _ in range(EVENTS_PER_ACCT):
            dev = rng.choice(devices)
            acct.id_corpus.append(dev.device_id)
            acct.feature_corpus.append(profile_to_tokens(dev.profile))
            acct.observed_profiles.add(dev.profile)

    return accounts


def build_accounts_markov(fleet: List[Device]) -> List[Account]:
    """Same as build_accounts but uses Markov device switching."""
    accounts = []
    targeted_count = int(N_ACCOUNTS * FLEET_TARGET_FRAC)
    targeted_indices = set(rng.sample(range(N_ACCOUNTS), targeted_count))

    for i in range(N_ACCOUNTS):
        acct_id = f"acct_m_{i:04d}"
        primary = sample_primary_profile()
        n_devs = rng.randint(2, 4)
        known_devices = []
        for _ in range(n_devs):
            dev_profile = vary_profile_net_screen(primary)
            known_devices.append(
                Device(device_id=make_device_id("mlegit"), profile=dev_profile)
            )

        acct = Account(
            account_id=acct_id,
            primary_profile=primary,
            known_devices=known_devices,
        )
        acct.known_device_ids = {d.device_id for d in known_devices}
        accounts.append(acct)

    for i, acct in enumerate(accounts):
        devices = acct.known_devices
        is_targeted = i in targeted_indices

        if is_targeted:
            injected = rng.choices(fleet, k=FLEET_INJECT)
            for dev in injected:
                acct.id_corpus.append(dev.device_id)
                acct.feature_corpus.append(profile_to_tokens(dev.profile))
                acct.observed_profiles.add(dev.profile)

        # Markov walk
        current = rng.choice(devices)
        for _ in range(EVENTS_PER_ACCT):
            acct.id_corpus.append(current.device_id)
            acct.feature_corpus.append(profile_to_tokens(current.profile))
            acct.observed_profiles.add(current.profile)
            if rng.random() < STAY_PROB or len(devices) == 1:
                pass  # stay
            else:
                others = [d for d in devices if d.device_id != current.device_id]
                current = rng.choice(others)

    return accounts


def make_novel_attack_profile(victim_primary: Profile) -> Profile:
    """
    Foreign feature profile: different OS family from victim.
    Pick an OS that differs from victim's, plus foreign TZ and language.
    """
    victim_os = victim_primary[0]
    other_os = [o for o in FEATURE_VALUES["os"] if o != victim_os]
    os_ = rng.choice(other_os)
    browser = rng.choice(FEATURE_VALUES["browser"])
    # TZ far from victim's
    victim_tz_idx = FEATURE_VALUES["tz"].index(victim_primary[2])
    far_tz_indices = [
        j for j, _ in enumerate(FEATURE_VALUES["tz"]) if abs(j - victim_tz_idx) >= 2
    ]
    tz = FEATURE_VALUES["tz"][rng.choice(far_tz_indices)]
    # Non-English language
    lang = rng.choice(
        [l for l in FEATURE_VALUES["lang"] if l not in ("en_us", "en_gb")]
    )
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def make_spoof_attack_profile(victim_primary: Profile) -> Profile:
    """
    Mimics victim: same OS+browser+lang, but different timezone.
    """
    os_ = victim_primary[0]
    browser = victim_primary[1]
    # Different TZ from victim's primary
    victim_tz = victim_primary[2]
    other_tz = [t for t in FEATURE_VALUES["tz"] if t != victim_tz]
    tz = rng.choice(other_tz)
    lang = victim_primary[3]
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def make_enroll_profile(victim_primary: Profile) -> Profile:
    """
    Legitimate new device enrollment: same OS+browser+tz+lang as victim's
    primary profile, randomised net+screen. This device is OOV (new ID) but
    its features are consistent with the account's region and OS family.
    OOV-based signals (global_oov, account_oov) cannot distinguish this from
    an attack; feature_w2v should score it as low-distance (legit).
    """
    os_, browser, tz, lang = victim_primary[:4]
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def generate_eval_events(
    accounts: List[Account],
    fleet: List[Device],
    n_per_type: int = ATTACK_EVENTS,
) -> Dict[str, List[EvalEvent]]:
    """
    Generate eval events:
      - legit:  random known device from random account
      - novel:  new device ID + foreign profile
      - fleet:  existing fleet device ID + attacker profile
      - spoof:  new device ID + victim-mimic profile
    Returns dict keyed by event type.
    """
    events: Dict[str, List[EvalEvent]] = {
        "legit": [],
        "enroll": [],  # legitimate new device — OOV but profile-consistent (label=0)
        "novel": [],
        "fleet": [],
        "spoof": [],
    }

    for _ in range(n_per_type):
        acct = rng.choice(accounts)

        # legit — returning known device
        dev = rng.choice(acct.known_devices)
        events["legit"].append(EvalEvent(acct, dev.device_id, dev.profile, label=0))

        # enroll — legitimate new device enrollment (new ID, profile matches primary)
        enroll_id = make_device_id("enroll")
        enroll_profile = make_enroll_profile(acct.primary_profile)
        events["enroll"].append(EvalEvent(acct, enroll_id, enroll_profile, label=0))

        # novel attack
        novel_id = make_device_id("novel")
        novel_profile = make_novel_attack_profile(acct.primary_profile)
        events["novel"].append(EvalEvent(acct, novel_id, novel_profile, label=1))

        # fleet attack — pick a fleet device, use its stored attacker profile
        fleet_dev = rng.choice(fleet)
        events["fleet"].append(
            EvalEvent(acct, fleet_dev.device_id, fleet_dev.profile, label=1)
        )

        # spoof attack
        spoof_id = make_device_id("spoof")
        spoof_profile = make_spoof_attack_profile(acct.primary_profile)
        events["spoof"].append(EvalEvent(acct, spoof_id, spoof_profile, label=1))

    return events


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_id_model(accounts: List[Account]) -> Word2Vec:
    """Word2Vec on device ID sequences (one sequence = one account's id_corpus)."""
    sentences = [acct.id_corpus for acct in accounts if acct.id_corpus]
    model = Word2Vec(
        sentences=sentences,
        vector_size=VEC_DIM,
        window=ID_WINDOW,
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
    )
    return model


def train_feature_model(accounts: List[Account]) -> Word2Vec:
    """
    Word2Vec on feature token sequences.
    Each login event emits 6 tokens; we concatenate across all events for each
    account to form one long sentence per account.
    """
    sentences = []
    for acct in accounts:
        flat_tokens: List[str] = []
        for token_list in acct.feature_corpus:
            flat_tokens.extend(token_list)
        if flat_tokens:
            sentences.append(flat_tokens)
    model = Word2Vec(
        sentences=sentences,
        vector_size=VEC_DIM,
        window=FEAT_WINDOW,
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
    )
    return model


def train_feature_fasttext(accounts: List[Account]) -> FastText:
    """
    FastText on feature token sequences (same corpus as Word2Vec).

    Motivation: structured feature tokens (os_ios, browser_safari, tz_utc-5)
    share meaningful prefixes across feature dimensions. FastText's character
    n-grams can embed a token like 'os_harmonyos' that never appeared in
    training by averaging the n-gram vectors for 'os_', 'har', 'arm', etc.
    This makes the feature vocabulary effectively unbounded at inference time —
    new OS versions, browsers, or timezone strings get reasonable embeddings
    without requiring a model retrain.

    This is explicitly NOT the case for device ID tokens (random alphanumeric
    strings where n-gram sharing is spurious noise, as shown in Experiment 2).
    FastText on feature tokens is a different hypothesis from FastText on IDs.

    min_n=3, max_n=6: standard gensim defaults; captures prefix patterns like
    'os_' (3 chars) through 'os_ios' (6 chars).
    """
    sentences = []
    for acct in accounts:
        flat_tokens: List[str] = []
        for token_list in acct.feature_corpus:
            flat_tokens.extend(token_list)
        if flat_tokens:
            sentences.append(flat_tokens)
    model = FastText(
        sentences=sentences,
        vector_size=VEC_DIM,
        window=FEAT_WINDOW,
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
        min_n=3,
        max_n=6,
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# CENTROID COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_id_centroids(
    accounts: List[Account], id_model: Word2Vec
) -> Dict[str, np.ndarray]:
    """Per-account centroid of device ID embeddings seen in corpus."""
    wv = id_model.wv
    # Global mean vector for OOV fallback
    global_mean = wv.vectors.mean(axis=0)

    centroids: Dict[str, np.ndarray] = {}
    for acct in accounts:
        vecs = []
        seen = set()
        for did in acct.id_corpus:
            if did in seen:
                continue
            seen.add(did)
            if did in wv.key_to_index:
                vecs.append(wv[did])
            else:
                vecs.append(global_mean)
        if vecs:
            centroids[acct.account_id] = np.mean(vecs, axis=0)
        else:
            centroids[acct.account_id] = global_mean
    return centroids


def get_device_feature_vec(profile: Profile, feat_model: Word2Vec) -> np.ndarray:
    """Mean of the 6 feature-token embeddings for a device's profile (Word2Vec)."""
    wv = feat_model.wv
    tokens = profile_to_tokens(profile)
    vecs = [wv[t] for t in tokens if t in wv.key_to_index]
    if not vecs:
        return wv.vectors.mean(axis=0)
    return np.mean(vecs, axis=0)


def get_device_feature_vec_ft(profile: Profile, ft_model: FastText) -> np.ndarray:
    """
    Mean of the 6 feature-token embeddings via FastText.
    FastText embeds any token — including OOV — via character n-gram averaging,
    so no key_to_index guard is needed. This is the OOV-safe path.
    """
    wv = ft_model.wv
    tokens = profile_to_tokens(profile)
    return np.mean([wv[t] for t in tokens], axis=0)


def compute_feature_centroids(
    accounts: List[Account], feat_model: Word2Vec
) -> Dict[str, np.ndarray]:
    """
    Per-account centroid: mean of per-device feature embeddings (Word2Vec).
    Each device's embedding = mean of its 6 feature token embeddings.
    Uses the unique profile tuples observed in the account's corpus.
    """
    centroids: Dict[str, np.ndarray] = {}
    global_mean = feat_model.wv.vectors.mean(axis=0)
    for acct in accounts:
        if not acct.observed_profiles:
            centroids[acct.account_id] = global_mean
            continue
        device_vecs = [
            get_device_feature_vec(p, feat_model) for p in acct.observed_profiles
        ]
        centroids[acct.account_id] = np.mean(device_vecs, axis=0)
    return centroids


def compute_feature_fasttext_centroids(
    accounts: List[Account], ft_model: FastText
) -> Dict[str, np.ndarray]:
    """Per-account centroid using FastText feature embeddings (OOV-safe)."""
    centroids: Dict[str, np.ndarray] = {}
    global_mean = ft_model.wv.vectors.mean(axis=0)
    for acct in accounts:
        if not acct.observed_profiles:
            centroids[acct.account_id] = global_mean
            continue
        device_vecs = [
            get_device_feature_vec_ft(p, ft_model) for p in acct.observed_profiles
        ]
        centroids[acct.account_id] = np.mean(device_vecs, axis=0)
    return centroids


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════


def score_events(
    eval_events: Dict[str, List[EvalEvent]],
    id_model: Word2Vec,
    feat_model: Word2Vec,
    ft_model: FastText,
    id_centroids: Dict[str, np.ndarray],
    feat_centroids: Dict[str, np.ndarray],
    ft_centroids: Dict[str, np.ndarray],
    id_global_vocab: Set[str],
) -> Dict[str, Dict[str, List]]:
    """
    For every eval event, compute 6 signals.
    Structure: scores[signal][event_type] = {"scores": [...], "labels": [...]}
    """
    _SIGNALS = [
        "global_oov",
        "account_oov",
        "id_w2v",
        "feature_w2v",
        "feature_fasttext",
        "feature_novelty",
    ]
    results: Dict[str, Dict[str, Dict[str, list]]] = {
        sig: {et: {"scores": [], "labels": []} for et in eval_events}
        for sig in _SIGNALS
    }

    wv_id = id_model.wv
    id_global_mean = wv_id.vectors.mean(axis=0)

    for event_type, events in eval_events.items():
        for ev in events:
            acct = ev.account
            did = ev.device_id
            prof = ev.profile
            lbl = ev.label
            acct_id = acct.account_id

            # 1. global_oov
            g_oov = 0.0 if did in id_global_vocab else 1.0
            results["global_oov"][event_type]["scores"].append(g_oov)
            results["global_oov"][event_type]["labels"].append(lbl)

            # 2. account_oov
            a_oov = 0.0 if did in acct.known_device_ids else 1.0
            results["account_oov"][event_type]["scores"].append(a_oov)
            results["account_oov"][event_type]["labels"].append(lbl)

            # 3. id_w2v — cosine distance to account ID centroid
            dev_id_vec = wv_id[did] if did in wv_id.key_to_index else id_global_mean
            results["id_w2v"][event_type]["scores"].append(
                cos_dist(dev_id_vec, id_centroids[acct_id])
            )
            results["id_w2v"][event_type]["labels"].append(lbl)

            # 4. feature_w2v — Word2Vec centroid distance (OOV falls back to global mean)
            dev_feat_vec = get_device_feature_vec(prof, feat_model)
            results["feature_w2v"][event_type]["scores"].append(
                cos_dist(dev_feat_vec, feat_centroids[acct_id])
            )
            results["feature_w2v"][event_type]["labels"].append(lbl)

            # 5. feature_fasttext — FastText centroid distance (OOV-safe via n-grams)
            dev_ft_vec = get_device_feature_vec_ft(prof, ft_model)
            results["feature_fasttext"][event_type]["scores"].append(
                cos_dist(dev_ft_vec, ft_centroids[acct_id])
            )
            results["feature_fasttext"][event_type]["labels"].append(lbl)

            # 6. feature_novelty — exact profile tuple not in account's observed set
            f_nov = 0.0 if prof in acct.observed_profiles else 1.0
            results["feature_novelty"][event_type]["scores"].append(f_nov)
            results["feature_novelty"][event_type]["labels"].append(lbl)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

ATTACK_TYPES = ["novel", "fleet", "spoof"]
SIGNALS = [
    "global_oov",
    "account_oov",
    "id_w2v",
    "feature_w2v",
    "feature_fasttext",
    "feature_novelty",
]

# Pre-specified verdicts (written as comments for reference)
# ─────────────────────────────────────────────────────────
# Negative class = legit (known device) + enroll (new legit device, OOV).
#
# global_oov   on fleet:  expected AUC ≈ 0.50  (fleet device in vocab, both legit
#                          and enroll negatives exist → signal partially degraded)
# global_oov   on novel:  expected AUC < 0.80  (novel attack = OOV; so is enroll
#                          → signal cannot distinguish them)
# account_oov  on any:    expected AUC < 0.80  (enroll is also not in known set,
#                          same signal value as any attack → high false-positive rate)
# feature_w2v  on novel:  expected AUC > 0.80  (enroll profile is close to centroid;
#                          attack profile is far → signal preserved)
# feature_w2v  on spoof:  expected AUC notably lower than on novel (spoof profile
#                          close to victim → separates less cleanly from enroll)
# feature_novelty on spoof: expected AUC reduced from v1 (enroll also has novel
#                          exact-match profile → signal degrades like account_oov)
# id_w2v       on fleet:  expected AUC > 0.75  (fleet device has trained embedding
#                          far from victim cluster; enroll gets OOV mean)


def evaluate_all(
    raw_results: Dict[str, Dict[str, Dict[str, list]]],
    mode_label: str,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    For each (signal, attack_type), compute AUC with CI.
    Uses legit events (label=0) pooled with attack events (label=1) for that type.
    Returns aucs[signal][attack_type] = (auc, ci_lo, ci_hi).
    """
    aucs: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    for sig in SIGNALS:
        aucs[sig] = {}
        for at in ATTACK_TYPES:
            # Negative class = legit (returning known device) + enroll (new legit device).
            # This forces signals to distinguish "new device that belongs here" from attack.
            neg_scores = (
                raw_results[sig]["legit"]["scores"]
                + raw_results[sig]["enroll"]["scores"]
            )
            neg_labels = (
                raw_results[sig]["legit"]["labels"]
                + raw_results[sig]["enroll"]["labels"]
            )
            atk_scores = raw_results[sig][at]["scores"]
            atk_labels = raw_results[sig][at]["labels"]

            y_true = neg_labels + atk_labels
            y_score = neg_scores + atk_scores

            auc, lo, hi = bootstrap_auc_ci(y_true, y_score)
            aucs[sig][at] = (auc, lo, hi)

    return aucs


def print_results_table(
    aucs: Dict[str, Dict[str, Tuple[float, float, float]]], mode: str
) -> None:
    header = f"\n{'=' * 70}\n  Results — {mode} corpus\n{'=' * 70}"
    print(header)
    col_w = 22
    print(f"{'Signal':<18}" + "".join(f"{at:>{col_w}}" for at in ATTACK_TYPES))
    print("-" * (18 + col_w * len(ATTACK_TYPES)))
    for sig in SIGNALS:
        row = f"{sig:<18}"
        for at in ATTACK_TYPES:
            auc, lo, hi = aucs[sig][at]
            cell = f"{auc:.3f} [{lo:.3f}–{hi:.3f}]"
            row += f"{cell:>{col_w}}"
        print(row)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════


def fig1_auc_heatmap(
    aucs_iid: Dict[str, Dict[str, Tuple[float, float, float]]],
    aucs_markov: Dict[str, Dict[str, Tuple[float, float, float]]],
    out_path: str = str(FIGURES_DIR / "exp3_fig1_auc_heatmap.png"),
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Experiment 3 — AUC Heatmap by Signal × Attack Type",
        fontsize=13,
        fontweight="bold",
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#d62728", "#ffff00", "#2ca02c"]
    )

    for ax, aucs, title in [
        (axes[0], aucs_iid, "i.i.d. corpus"),
        (axes[1], aucs_markov, "Markov corpus"),
    ]:
        matrix = np.array(
            [[aucs[sig][at][0] for at in ATTACK_TYPES] for sig in SIGNALS]
        )
        im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(ATTACK_TYPES)))
        ax.set_xticklabels(ATTACK_TYPES, fontsize=10)
        ax.set_yticks(range(len(SIGNALS)))
        ax.set_yticklabels(SIGNALS, fontsize=9)
        ax.set_title(title, fontsize=11)

        # Annotate cells
        for r, sig in enumerate(SIGNALS):
            for c, at in enumerate(ATTACK_TYPES):
                auc = aucs[sig][at][0]
                text_color = "white" if auc < 0.65 else "black"
                ax.text(
                    c,
                    r,
                    f"{auc:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUC")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig2_score_distributions(
    raw_results: Dict[str, Dict[str, Dict[str, list]]],
    out_path: str = str(FIGURES_DIR / "exp3_fig2_score_distributions.png"),
) -> None:
    """KDE/histogram of feature_w2v scores for legit, novel, fleet, spoof (i.i.d.)."""
    from scipy.stats import gaussian_kde

    sig = "feature_w2v"
    classes = [
        ("legit", "#1f77b4", "Legitimate (known device)"),
        ("enroll", "#17becf", "Legitimate (new enrollment)"),
        ("novel", "#d62728", "Novel attack"),
        ("fleet", "#ff7f0e", "Fleet attack"),
        ("spoof", "#9467bd", "Spoof attack"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        "feature_w2v Score Distributions — i.i.d. Corpus",
        fontsize=12,
        fontweight="bold",
    )

    for key, color, label in classes:
        scores = np.array(raw_results[sig][key]["scores"])
        if len(scores) < 3:
            continue
        xs = np.linspace(scores.min() - 0.05, scores.max() + 0.05, 300)
        try:
            kde = gaussian_kde(scores, bw_method="scott")
            ax.plot(xs, kde(xs), color=color, linewidth=2, label=label)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=color)
        except Exception:
            ax.hist(scores, bins=30, alpha=0.4, color=color, label=label, density=True)

    ax.set_xlabel("Cosine Distance (feature_w2v)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig3_signal_comparison(
    aucs_iid: Dict[str, Dict[str, Tuple[float, float, float]]],
    out_path: str = str(FIGURES_DIR / "exp3_fig3_signal_comparison.png"),
) -> None:
    """Grouped bar chart: x=attack types, bars=signals, y=AUC with CI error bars."""
    n_signals = len(SIGNALS)
    n_attacks = len(ATTACK_TYPES)
    x = np.arange(n_attacks)
    bar_width = 0.15
    offsets = (
        np.linspace(-(n_signals - 1) / 2, (n_signals - 1) / 2, n_signals) * bar_width
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(
        "Signal Comparison by Attack Type — i.i.d. Corpus",
        fontsize=12,
        fontweight="bold",
    )

    for i, (sig, color) in enumerate(zip(SIGNALS, colors)):
        aucs_vals = [aucs_iid[sig][at][0] for at in ATTACK_TYPES]
        ci_lo = [aucs_iid[sig][at][0] - aucs_iid[sig][at][1] for at in ATTACK_TYPES]
        ci_hi = [aucs_iid[sig][at][2] - aucs_iid[sig][at][0] for at in ATTACK_TYPES]
        yerr = np.array([ci_lo, ci_hi])

        ax.bar(
            x + offsets[i],
            aucs_vals,
            width=bar_width,
            color=color,
            label=sig,
            alpha=0.85,
            zorder=3,
        )
        ax.errorbar(
            x + offsets[i],
            aucs_vals,
            yerr=yerr,
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_TYPES, fontsize=11)
    ax.set_xlabel("Attack Type", fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(0.3, 1.08)
    ax.axhline(
        0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance (0.5)"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def run_pipeline(accounts: List[Account], fleet: List[Device], mode_label: str):
    print(f"\n{'─' * 60}")
    print(f"  Pipeline: {mode_label}")
    print(f"{'─' * 60}")

    # Train models
    print("  Training ID Word2Vec...")
    id_model = train_id_model(accounts)

    print("  Training feature Word2Vec...")
    feat_model = train_feature_model(accounts)

    print("  Training feature FastText...")
    ft_model = train_feature_fasttext(accounts)

    # Build global ID vocabulary (all device IDs seen in training corpora)
    id_global_vocab: Set[str] = set(id_model.wv.key_to_index.keys())

    # Compute centroids
    print("  Computing centroids...")
    id_centroids = compute_id_centroids(accounts, id_model)
    feat_centroids = compute_feature_centroids(accounts, feat_model)
    ft_centroids = compute_feature_fasttext_centroids(accounts, ft_model)

    # Generate eval events
    print("  Generating eval events...")
    eval_events = generate_eval_events(accounts, fleet, n_per_type=ATTACK_EVENTS)

    # Score
    print("  Scoring...")
    raw_results = score_events(
        eval_events,
        id_model,
        feat_model,
        ft_model,
        id_centroids,
        feat_centroids,
        ft_centroids,
        id_global_vocab,
    )

    # Evaluate
    aucs = evaluate_all(raw_results, mode_label)
    print_results_table(aucs, mode_label)

    return aucs, raw_results


def main() -> None:
    print("=" * 70)
    print("  ATO Experiment 3 — Realistic Data Redesign & Feature Embeddings")
    print("=" * 70)

    # Build shared fleet (same fleet for both corpus modes)
    print("\nBuilding fleet devices...")
    fleet = build_fleet(N_FLEET_DEVICES)

    # ── i.i.d. pipeline ───────────────────────────────────────────────────────
    print("\nBuilding i.i.d. accounts...")
    accounts_iid = build_accounts(fleet)
    aucs_iid, raw_iid = run_pipeline(accounts_iid, fleet, "i.i.d.")

    # ── Markov pipeline ───────────────────────────────────────────────────────
    print("\nBuilding Markov accounts...")
    accounts_markov = build_accounts_markov(fleet)
    aucs_markov, raw_markov = run_pipeline(accounts_markov, fleet, "Markov")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig1_auc_heatmap(aucs_iid, aucs_markov)
    fig2_score_distributions(raw_iid)
    fig3_signal_comparison(aucs_iid)

    # ── Pre-specified verdict check ───────────────────────────────────────────
    print("\nPre-specified verdict check (i.i.d.):")
    print("─" * 55)
    novel_w2v_auc = aucs_iid["feature_w2v"]["novel"][0]
    aucs_iid["feature_w2v"]["spoof"][0]
    novel_ft_auc = aucs_iid["feature_fasttext"]["novel"][0]
    aucs_iid["feature_fasttext"]["spoof"][0]

    checks = [
        (
            "global_oov",
            "fleet",
            "≈ 0.50 (blind to fleet, partially degraded by enroll)",
            lambda a: a < 0.80,
        ),
        (
            "global_oov",
            "novel",
            "< 0.80 (enroll also OOV → can't distinguish)",
            lambda a: a < 0.80,
        ),
        (
            "account_oov",
            "fleet",
            "< 0.80 (enroll also not in known set)",
            lambda a: a < 0.80,
        ),
        (
            "feature_w2v",
            "novel",
            "> 0.80 (enroll close to centroid, attack far)",
            lambda a: a > 0.80,
        ),
        (
            "feature_w2v",
            "spoof",
            "< feature_w2v/novel (spoof closer to centroid)",
            lambda a: a < novel_w2v_auc,
        ),
        (
            "feature_fasttext",
            "novel",
            "> 0.80 (structured token n-grams reinforce cluster)",
            lambda a: a > 0.80,
        ),
        (
            "feature_fasttext",
            "spoof",
            "< feature_fasttext/novel (spoof closer via shared n-grams)",
            lambda a: a < novel_ft_auc,
        ),
        (
            "id_w2v",
            "fleet",
            "> 0.75 (fleet embedding far from victim cluster)",
            lambda a: a > 0.75,
        ),
    ]

    for sig, at, expectation, check_fn in checks:
        auc = aucs_iid[sig][at][0]
        status = "PASS" if check_fn(auc) else "FAIL"
        print(f"  [{status}] {sig:22s} on {at:6s}: {auc:.3f}  (expected {expectation})")

    print("\nDone.")


if __name__ == "__main__":
    main()
