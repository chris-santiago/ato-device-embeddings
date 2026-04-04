# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3",
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.8",
# ]
# ///
"""
H2 Experiment — Mean-Pool vs. Concat FastText: Debate-Agreed Tests
===================================================================

Implements all five tests from H2_DEBATE.md:

  T1  Window sweep: concat at window ∈ {1, 3, 6} vs mean-pool (baseline)
  T2  Prefixed-concat format: key:val|key:val|... joined with |
  T3  Tz-position permutation: 6 orderings placing tz at each of 6 positions
  T4  OOV injection: 5% of eval events with unseen os_harmonyos / tz_utc+9
  T5  Bootstrap CIs (N=1000, percentile) on all AUC numbers

Pre-specified verdicts are stated in each results block.
All signals evaluated on identical data splits.

Deliberate exclusions from this experiment:
  - No Markov corpus (i.i.d. only; Markov would not change the mechanism conclusions).
  - id_w2v, global_oov, account_oov not re-evaluated (established in Experiment 3).
"""

import random
import string
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics import roc_auc_score, silhouette_score
from gensim.models import FastText

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)

# ── constants ─────────────────────────────────────────────────────────────────
N_ACCOUNTS = 400
N_FLEET_DEVICES = 80
EVENTS_PER_ACCT = 60
ATTACK_EVENTS = 80
FLEET_INJECT = 8
FLEET_TARGET_FRAC = 0.25
VEC_DIM = 64
FEAT_WINDOW = 6
N_NEGATIVE = 10
EPOCHS = 20
N_BOOTSTRAP = 1000
OOV_INJECT_FRAC = 0.50  # T4 — 50% gives ~40 OOV events for stable bootstrap CIs

FEATURE_KEYS = ["os", "browser", "tz", "lang", "net", "screen"]
FEATURE_VALUES = {
    "os": ["ios", "android", "windows", "macos", "linux"],
    "browser": ["safari", "chrome", "firefox", "edge", "samsung"],
    "tz": ["utc-8", "utc-5", "utc+0", "utc+1", "utc+5", "utc+8"],
    "lang": ["en_us", "en_gb", "es_mx", "fr_fr", "de_de", "zh_cn"],
    "net": ["wifi", "lte", "5g", "broadband"],
    "screen": ["small", "medium", "large", "xlarge"],
}

# OOV tokens injected in T4 — never seen during training
OOV_OS = "harmonyos"   # → concat value "harmonyos"; mean-pool token "os_harmonyos"
OOV_TZ = "utc+9"      # → concat value "utc+9"; mean-pool token "tz_utc+9"

Profile = Tuple[str, str, str, str, str, str]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def make_device_id(prefix: str = "dev") -> str:
    suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=12))
    return f"{prefix}_{suffix}"


def sample_primary_profile() -> Profile:
    return tuple(rng.choice(FEATURE_VALUES[k]) for k in FEATURE_KEYS)  # type: ignore[return-value]


def profile_to_tokens(profile: Profile) -> List[str]:
    """Mean-pool: 6 key-prefixed tokens."""
    return [f"{k}_{v}" for k, v in zip(FEATURE_KEYS, profile)]


def profile_to_concat(profile: Profile, order: Optional[List[int]] = None) -> str:
    """Values-only concat, optionally reordered (for T3 permutation test)."""
    vals = list(profile)
    if order is not None:
        vals = [vals[i] for i in order]
    return "_".join(vals)


def profile_to_prefixed_concat(profile: Profile) -> str:
    """T2: key:val pairs joined with | delimiter."""
    return "|".join(f"{k}:{v}" for k, v in zip(FEATURE_KEYS, profile))


def vary_profile_net_screen(base: Profile) -> Profile:
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
    yt = np.array(y_true)
    ys = np.array(y_score)
    if len(np.unique(yt)) < 2:
        return float("nan"), float("nan"), float("nan")
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
    known_devices: List[Device]
    known_device_ids: Set[str] = field(default_factory=set)
    observed_profiles: Set[Profile] = field(default_factory=set)
    # mean-pool corpus: flat list of feature tokens across all events
    feature_corpus: List[List[str]] = field(default_factory=list)
    # raw values corpus (reused for all concat variants at eval time)
    concat_corpus: List[str] = field(default_factory=list)
    prefixed_corpus: List[str] = field(default_factory=list)


@dataclass
class EvalEvent:
    account: Account
    device_id: str
    profile: Profile
    label: int
    is_oov: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def build_fleet(n: int) -> List[Device]:
    devices = []
    for _ in range(n):
        os_ = rng.choice(["windows", "linux"])
        browser = rng.choice(["chrome", "firefox"])
        tz = rng.choice(["utc+5", "utc+8"])
        lang = rng.choice(["zh_cn", "de_de", "fr_fr"])
        net = rng.choice(FEATURE_VALUES["net"])
        screen = rng.choice(FEATURE_VALUES["screen"])
        profile: Profile = (os_, browser, tz, lang, net, screen)
        devices.append(Device(device_id=make_device_id("fleet"), profile=profile))
    return devices


def build_accounts(fleet: List[Device]) -> List[Account]:
    accounts = []
    targeted_count = int(N_ACCOUNTS * FLEET_TARGET_FRAC)
    targeted_indices = set(rng.sample(range(N_ACCOUNTS), targeted_count))

    for i in range(N_ACCOUNTS):
        primary = sample_primary_profile()
        n_devs = rng.randint(2, 4)
        known_devices = [
            Device(
                device_id=make_device_id("legit"),
                profile=vary_profile_net_screen(primary),
            )
            for _ in range(n_devs)
        ]
        acct = Account(
            account_id=f"acct_{i:04d}",
            primary_profile=primary,
            known_devices=known_devices,
        )
        acct.known_device_ids = {d.device_id for d in known_devices}
        accounts.append(acct)

    for i, acct in enumerate(accounts):
        devices = acct.known_devices
        is_targeted = i in targeted_indices

        if is_targeted:
            for dev in rng.choices(fleet, k=FLEET_INJECT):
                _add_event(acct, dev.profile)

        for _ in range(EVENTS_PER_ACCT):
            dev = rng.choice(devices)
            _add_event(acct, dev.profile)

    return accounts


def _add_event(acct: Account, profile: Profile) -> None:
    acct.feature_corpus.append(profile_to_tokens(profile))
    acct.concat_corpus.append(profile_to_concat(profile))
    acct.prefixed_corpus.append(profile_to_prefixed_concat(profile))
    acct.observed_profiles.add(profile)


def make_novel_attack_profile(victim_primary: Profile) -> Profile:
    victim_os = victim_primary[0]
    other_os = [o for o in FEATURE_VALUES["os"] if o != victim_os]
    os_ = rng.choice(other_os)
    browser = rng.choice(FEATURE_VALUES["browser"])
    victim_tz_idx = FEATURE_VALUES["tz"].index(victim_primary[2])
    far_tz = [j for j, _ in enumerate(FEATURE_VALUES["tz"]) if abs(j - victim_tz_idx) >= 2]
    tz = FEATURE_VALUES["tz"][rng.choice(far_tz)]
    lang = rng.choice([l for l in FEATURE_VALUES["lang"] if l not in ("en_us", "en_gb")])
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def make_spoof_attack_profile(victim_primary: Profile) -> Profile:
    os_ = victim_primary[0]
    browser = victim_primary[1]
    victim_tz = victim_primary[2]
    other_tz = [t for t in FEATURE_VALUES["tz"] if t != victim_tz]
    tz = rng.choice(other_tz)
    lang = victim_primary[3]
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def make_enroll_profile(victim_primary: Profile) -> Profile:
    os_, browser, tz, lang = victim_primary[:4]
    net = rng.choice(FEATURE_VALUES["net"])
    screen = rng.choice(FEATURE_VALUES["screen"])
    return (os_, browser, tz, lang, net, screen)


def generate_eval_events(
    accounts: List[Account],
    fleet: List[Device],
    n_per_type: int = ATTACK_EVENTS,
    oov_frac: float = 0.0,
) -> Dict[str, List[EvalEvent]]:
    """
    oov_frac: fraction of novel attack events to replace with OOV-token profiles.
    OOV profiles substitute os with OOV_OS or tz with OOV_TZ randomly.
    """
    events: Dict[str, List[EvalEvent]] = {
        "legit": [], "enroll": [], "novel": [], "fleet": [], "spoof": [],
    }
    for idx in range(n_per_type):
        acct = rng.choice(accounts)
        dev = rng.choice(acct.known_devices)
        events["legit"].append(EvalEvent(acct, dev.device_id, dev.profile, label=0))

        enroll_id = make_device_id("enroll")
        events["enroll"].append(
            EvalEvent(acct, enroll_id, make_enroll_profile(acct.primary_profile), label=0)
        )

        novel_profile = make_novel_attack_profile(acct.primary_profile)
        is_oov = rng.random() < oov_frac
        if is_oov:
            # Replace OS with OOV value
            novel_profile = (OOV_OS,) + novel_profile[1:]  # type: ignore[assignment]
        events["novel"].append(
            EvalEvent(acct, make_device_id("novel"), novel_profile, label=1, is_oov=is_oov)
        )

        fleet_dev = rng.choice(fleet)
        events["fleet"].append(
            EvalEvent(acct, fleet_dev.device_id, fleet_dev.profile, label=1)
        )

        spoof_profile = make_spoof_attack_profile(acct.primary_profile)
        events["spoof"].append(
            EvalEvent(acct, make_device_id("spoof"), spoof_profile, label=1)
        )

    return events


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_mean_pool_ft(accounts: List[Account]) -> FastText:
    sentences = []
    for acct in accounts:
        flat: List[str] = []
        for toks in acct.feature_corpus:
            flat.extend(toks)
        if flat:
            sentences.append(flat)
    return FastText(
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


def train_concat_ft(accounts: List[Account], window: int) -> FastText:
    sentences = [acct.concat_corpus for acct in accounts if acct.concat_corpus]
    return FastText(
        sentences=sentences,
        vector_size=VEC_DIM,
        window=window,
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
        min_n=3,
        max_n=6,
    )


def train_prefixed_concat_ft(accounts: List[Account]) -> FastText:
    """T2: FastText on key:val|key:val|... tokens."""
    sentences = [acct.prefixed_corpus for acct in accounts if acct.prefixed_corpus]
    return FastText(
        sentences=sentences,
        vector_size=VEC_DIM,
        window=1,
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
        min_n=3,
        max_n=6,
    )


def train_perm_concat_ft(accounts: List[Account], order: List[int]) -> FastText:
    """T3: FastText on reordered concat tokens."""
    sentences = [
        [profile_to_concat(p, order) for p in acct.observed_profiles]
        for acct in accounts
        if acct.observed_profiles
    ]
    return FastText(
        sentences=sentences,
        vector_size=VEC_DIM,
        window=1,
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
        min_n=3,
        max_n=6,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def embed_mean_pool(profile: Profile, model: FastText) -> np.ndarray:
    tokens = profile_to_tokens(profile)
    return np.mean([model.wv[t] for t in tokens], axis=0)


def embed_concat(profile: Profile, model: FastText, order: Optional[List[int]] = None) -> np.ndarray:
    return model.wv[profile_to_concat(profile, order)]


def embed_prefixed_concat(profile: Profile, model: FastText) -> np.ndarray:
    return model.wv[profile_to_prefixed_concat(profile)]


def compute_centroids(
    accounts: List[Account],
    model: FastText,
    embed_fn: Callable[[Profile, FastText], np.ndarray],
) -> Dict[str, np.ndarray]:
    centroids: Dict[str, np.ndarray] = {}
    global_mean = model.wv.vectors.mean(axis=0)
    for acct in accounts:
        if not acct.observed_profiles:
            centroids[acct.account_id] = global_mean
            continue
        vecs = [embed_fn(p, model) for p in acct.observed_profiles]
        centroids[acct.account_id] = np.mean(vecs, axis=0)
    return centroids


def compute_perm_centroids(
    accounts: List[Account],
    model: FastText,
    order: List[int],
) -> Dict[str, np.ndarray]:
    fn = lambda p, m: embed_concat(p, m, order=order)
    return compute_centroids(accounts, model, fn)


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════


def score_events(
    eval_events: Dict[str, List[EvalEvent]],
    model: FastText,
    centroids: Dict[str, np.ndarray],
    embed_fn: Callable[[Profile, FastText], np.ndarray],
    oov_only: bool = False,
) -> Dict[str, Dict[str, list]]:
    results = {et: {"scores": [], "labels": []} for et in eval_events}
    for et, events in eval_events.items():
        for ev in events:
            if oov_only and not ev.is_oov:
                continue
            vec = embed_fn(ev.profile, model)
            results[et]["scores"].append(cos_dist(vec, centroids[ev.account.account_id]))
            results[et]["labels"].append(ev.label)
    return results


def auc_ci(results: Dict[str, Dict[str, list]], attack_type: str) -> Tuple[float, float, float]:
    neg_s = results["legit"]["scores"] + results["enroll"]["scores"]
    neg_l = results["legit"]["labels"] + results["enroll"]["labels"]
    pos_s = results[attack_type]["scores"]
    pos_l = results[attack_type]["labels"]
    return bootstrap_auc_ci(neg_l + pos_l, neg_s + pos_s)


def compute_silhouette(
    accounts: List[Account],
    model: FastText,
    embed_fn: Callable[[Profile, FastText], np.ndarray],
    max_accounts: int = 200,
) -> float:
    sample = accounts[:max_accounts]
    vecs, labels = [], []
    for i, acct in enumerate(sample):
        for p in acct.observed_profiles:
            vecs.append(embed_fn(p, model))
            labels.append(i)
    arr = np.array(vecs)
    lab = np.array(labels)
    unique, counts = np.unique(lab, return_counts=True)
    valid = unique[counts >= 2]
    if len(valid) < 2:
        return float("nan")
    mask = np.isin(lab, valid)
    return float(silhouette_score(arr[mask], lab[mask], metric="cosine"))


# ═══════════════════════════════════════════════════════════════════════════════
# T1 — WINDOW SWEEP
# ═══════════════════════════════════════════════════════════════════════════════


def run_t1_window_sweep(
    accounts: List[Account],
    eval_events: Dict[str, List[EvalEvent]],
    mp_model: FastText,
    mp_centroids: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    print("\n[T1] Window sweep: concat at window ∈ {1, 3, 6}")
    print("     Verdict threshold: gap < 0.005 → window is load-bearing; gap ≥ 0.010 → n-gram noise has independent power")

    mp_embed = lambda p, m: embed_mean_pool(p, m)
    mp_aucs = {at: auc_ci(score_events(eval_events, mp_model, mp_centroids, mp_embed), at) for at in ("novel", "fleet", "spoof")}
    print(f"     mean_pool: novel={mp_aucs['novel'][0]:.4f} [{mp_aucs['novel'][1]:.3f}–{mp_aucs['novel'][2]:.3f}]  fleet={mp_aucs['fleet'][0]:.4f}  spoof={mp_aucs['spoof'][0]:.4f}")

    results = {"mean_pool": mp_aucs}
    best_concat_novel = 0.0
    best_window = 1

    for w in (1, 3, 6):
        print(f"     Training concat FastText window={w}...", end=" ", flush=True)
        cc_model = train_concat_ft(accounts, window=w)
        cc_centroids = compute_centroids(accounts, cc_model, lambda p, m: embed_concat(p, m))
        cc_embed = lambda p, m: embed_concat(p, m)
        cc_aucs = {at: auc_ci(score_events(eval_events, cc_model, cc_centroids, cc_embed), at) for at in ("novel", "fleet", "spoof")}
        print(f"novel={cc_aucs['novel'][0]:.4f} [{cc_aucs['novel'][1]:.3f}–{cc_aucs['novel'][2]:.3f}]  fleet={cc_aucs['fleet'][0]:.4f}  spoof={cc_aucs['spoof'][0]:.4f}")
        results[f"concat_w{w}"] = cc_aucs
        if cc_aucs["novel"][0] > best_concat_novel:
            best_concat_novel = cc_aucs["novel"][0]
            best_window = w

    gap = mp_aucs["novel"][0] - best_concat_novel
    if gap < 0.005:
        verdict = "WINDOW LOAD-BEARING — n-gram noise story overstated"
    elif gap >= 0.010:
        verdict = "N-GRAM NOISE HAS INDEPENDENT EXPLANATORY POWER"
    else:
        verdict = "AMBIGUOUS (gap 0.005–0.010)"
    print(f"     Gap (mean-pool vs best-concat w={best_window}): {gap:+.4f} → {verdict}")
    results["_verdict"] = verdict
    results["_best_window"] = best_window
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# T2 — PREFIXED-CONCAT FORMAT
# ═══════════════════════════════════════════════════════════════════════════════


def run_t2_prefixed_concat(
    accounts: List[Account],
    eval_events: Dict[str, List[EvalEvent]],
    mp_model: FastText,
    mp_centroids: Dict[str, np.ndarray],
    mp_aucs: Dict[str, Tuple],
    best_window: int,
) -> Dict[str, Dict]:
    """
    T2 tests whether structured key prefixes (key:val|...) recover mean-pool performance.
    CRITICAL: must run at best_window from T1 to avoid re-introducing the window confound.
    """
    print(f"\n[T2] Prefixed-concat format: key:val|key:val|... at window={best_window}")
    print("     Verdict threshold: prefixed ≈ mean-pool (within 0.005 on novel) → prefix structure is the driver")
    print("     Note: window matched to T1 best-window to isolate format effect from window effect.")

    # Re-train prefixed concat using best_window from T1 (not hardcoded window=1)
    sentences = [acct.prefixed_corpus for acct in accounts if acct.prefixed_corpus]
    pc_model = FastText(
        sentences=sentences,
        vector_size=VEC_DIM,
        window=best_window,
        sg=1,
        negative=N_NEGATIVE,
        min_count=1,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
        min_n=3,
        max_n=6,
    )
    print(f"     Training prefixed-concat FastText (window={best_window})...", end=" ", flush=True)
    pc_centroids = compute_centroids(accounts, pc_model, lambda p, m: embed_prefixed_concat(p, m))
    pc_embed = lambda p, m: embed_prefixed_concat(p, m)
    pc_aucs = {at: auc_ci(score_events(eval_events, pc_model, pc_centroids, pc_embed), at) for at in ("novel", "fleet", "spoof")}
    print(f"novel={pc_aucs['novel'][0]:.4f} [{pc_aucs['novel'][1]:.3f}–{pc_aucs['novel'][2]:.3f}]  fleet={pc_aucs['fleet'][0]:.4f}  spoof={pc_aucs['spoof'][0]:.4f}")

    # Silhouette comparison
    mp_sil = compute_silhouette(accounts, mp_model, lambda p, m: embed_mean_pool(p, m))
    pc_sil = compute_silhouette(accounts, pc_model, lambda p, m: embed_prefixed_concat(p, m))
    print(f"     Silhouette — mean_pool: {mp_sil:.4f}  prefixed_concat (w={best_window}): {pc_sil:.4f}")

    # Compare against mean-pool on spoof (the residual gap after window equalization)
    gap_novel = mp_aucs["novel"][0] - pc_aucs["novel"][0]
    gap_spoof = mp_aucs["spoof"][0] - pc_aucs["spoof"][0]

    if abs(gap_novel) <= 0.005:
        verdict_novel = "PREFIXED CONCAT ≈ MEAN-POOL on novel — prefix structure accounts for remaining gap"
    elif gap_novel > 0.010:
        verdict_novel = "PREFIXED CONCAT STILL LAGS on novel — cross-boundary n-gram noise independent of prefix structure"
    else:
        verdict_novel = "AMBIGUOUS on novel (gap 0.005–0.010)"

    if abs(gap_spoof) <= 0.010:
        verdict_spoof = "PREFIXED CONCAT ≈ MEAN-POOL on spoof — format explains residual spoof gap"
    elif gap_spoof > 0.020:
        verdict_spoof = "PREFIXED CONCAT STILL LAGS on spoof — n-gram noise persists after prefix restoration"
    else:
        verdict_spoof = "AMBIGUOUS on spoof (gap 0.010–0.020)"

    print(f"     Novel gap (mean-pool vs prefixed, w={best_window}): {gap_novel:+.4f} → {verdict_novel}")
    print(f"     Spoof gap (mean-pool vs prefixed, w={best_window}): {gap_spoof:+.4f} → {verdict_spoof}")
    return {
        "prefixed_concat": pc_aucs,
        "prefixed_sil": pc_sil,
        "mp_sil": mp_sil,
        "_verdict_novel": verdict_novel,
        "_verdict_spoof": verdict_spoof,
        "_verdict": f"{verdict_novel} | {verdict_spoof}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# T3 — TZ POSITION PERMUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def run_t3_tz_permutation(
    accounts: List[Account],
    eval_events: Dict[str, List[EvalEvent]],
) -> Dict[str, object]:
    """
    Place tz (feature index 2) at each of 6 positions.
    Other features keep their original relative order.
    """
    print("\n[T3] Tz-position permutation: spoof AUC for tz at positions 0–5")
    print("     Verdict: monotonic decrease with tz position → positional weighting confirmed")

    tz_idx = FEATURE_KEYS.index("tz")  # = 2
    other_indices = [i for i in range(6) if i != tz_idx]  # [0, 1, 3, 4, 5]

    spoof_aucs = []
    for tz_pos in range(6):
        # Build ordering: tz at tz_pos, others in original relative order
        order = []
        others_iter = iter(other_indices)
        for pos in range(6):
            if pos == tz_pos:
                order.append(tz_idx)
            else:
                order.append(next(others_iter))

        print(f"     tz at position {tz_pos} (order={order})...", end=" ", flush=True)
        perm_model = train_perm_concat_ft(accounts, order)
        perm_centroids = compute_perm_centroids(accounts, perm_model, order)

        scored = score_events(
            eval_events,
            perm_model,
            perm_centroids,
            lambda p, m: embed_concat(p, m, order=order),
        )
        auc, lo, hi = auc_ci(scored, "spoof")
        spoof_aucs.append((tz_pos, order, auc, lo, hi))
        print(f"spoof AUC={auc:.4f} [{lo:.3f}–{hi:.3f}]")

    # Check monotonicity
    aucs_only = [x[2] for x in spoof_aucs]
    diffs = [aucs_only[i+1] - aucs_only[i] for i in range(5)]
    is_monotone_dec = all(d <= 0 for d in diffs)
    is_monotone_inc = all(d >= 0 for d in diffs)

    if is_monotone_dec:
        verdict = "MONOTONICALLY DECREASING — positional weighting confirmed"
    elif is_monotone_inc:
        verdict = "MONOTONICALLY INCREASING — opposite of prediction; positional weighting refuted"
    else:
        verdict = "NON-MONOTONIC — specific n-gram crossings at boundaries, not pure positional weighting"
    print(f"     {verdict}")
    return {"spoof_aucs_by_tz_pos": spoof_aucs, "_verdict": verdict}


# ═══════════════════════════════════════════════════════════════════════════════
# T4 — OOV INJECTION
# ═══════════════════════════════════════════════════════════════════════════════


def run_t4_oov(
    accounts: List[Account],
    fleet: List[Device],
    mp_model: FastText,
    mp_centroids: Dict[str, np.ndarray],
    cc_model: FastText,
    cc_centroids: Dict[str, np.ndarray],
) -> Dict[str, object]:
    print(f"\n[T4] OOV injection: {OOV_INJECT_FRAC*100:.0f}% of novel events with os={OOV_OS}")
    print("     Verdict: mean-pool degrades less → H2 holds in OOV setting too")

    eval_oov = generate_eval_events(accounts, fleet, oov_frac=OOV_INJECT_FRAC)

    mp_embed = lambda p, m: embed_mean_pool(p, m)
    cc_embed = lambda p, m: embed_concat(p, m)

    # In-vocab subset (is_oov=False)
    def score_subset(events, model, centroids, embed_fn, oov_flag):
        results = {et: {"scores": [], "labels": []} for et in events}
        for et, evs in events.items():
            for ev in evs:
                if ev.is_oov != oov_flag and et == "novel":
                    continue
                vec = embed_fn(ev.profile, model)
                results[et]["scores"].append(cos_dist(vec, centroids[ev.account.account_id]))
                results[et]["labels"].append(ev.label)
        return results

    mp_iv = score_subset(eval_oov, mp_model, mp_centroids, mp_embed, oov_flag=False)
    mp_ov = score_subset(eval_oov, mp_model, mp_centroids, mp_embed, oov_flag=True)
    cc_iv = score_subset(eval_oov, cc_model, cc_centroids, cc_embed, oov_flag=False)
    cc_ov = score_subset(eval_oov, cc_model, cc_centroids, cc_embed, oov_flag=True)

    def safe_auc_ci(r, at):
        neg_s = r["legit"]["scores"] + r["enroll"]["scores"]
        neg_l = r["legit"]["labels"] + r["enroll"]["labels"]
        pos_s = r[at]["scores"]
        pos_l = r[at]["labels"]
        combined_labels = neg_l + pos_l
        if not pos_s or len(set(combined_labels)) < 2:
            return float("nan"), float("nan"), float("nan")
        return bootstrap_auc_ci(combined_labels, neg_s + pos_s)

    mp_iv_auc = safe_auc_ci(mp_iv, "novel")
    mp_ov_auc = safe_auc_ci(mp_ov, "novel")
    cc_iv_auc = safe_auc_ci(cc_iv, "novel")
    cc_ov_auc = safe_auc_ci(cc_ov, "novel")

    print(f"     mean_pool  in-vocab novel AUC: {mp_iv_auc[0]:.4f} [{mp_iv_auc[1]:.3f}–{mp_iv_auc[2]:.3f}]")
    print(f"     mean_pool  OOV novel AUC:      {mp_ov_auc[0]:.4f} [{mp_ov_auc[1]:.3f}–{mp_ov_auc[2]:.3f}]")
    print(f"     concat     in-vocab novel AUC: {cc_iv_auc[0]:.4f} [{cc_iv_auc[1]:.3f}–{cc_iv_auc[2]:.3f}]")
    print(f"     concat     OOV novel AUC:      {cc_ov_auc[0]:.4f} [{cc_ov_auc[1]:.3f}–{cc_ov_auc[2]:.3f}]")

    mp_drop = mp_iv_auc[0] - mp_ov_auc[0]
    cc_drop = cc_iv_auc[0] - cc_ov_auc[0]
    print(f"     AUC drop — mean_pool: {mp_drop:+.4f}  concat: {cc_drop:+.4f}")

    if mp_drop <= cc_drop:
        verdict = "MEAN-POOL DEGRADES LESS — H2 holds in OOV setting"
    elif cc_drop < mp_drop - 0.01:
        verdict = "CONCAT DEGRADES LESS — concat retains OOV production advantage"
    else:
        verdict = "COMPARABLE DEGRADATION — OOV handling does not change ranking"
    print(f"     {verdict}")
    return {
        "mp_iv": mp_iv_auc, "mp_ov": mp_ov_auc,
        "cc_iv": cc_iv_auc, "cc_ov": cc_ov_auc,
        "_verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════


def plot_t1(t1_results: Dict) -> None:
    """T1: AUC by attack type for mean-pool and concat at each window size."""
    signals = ["mean_pool", "concat_w1", "concat_w3", "concat_w6"]
    labels = ["mean_pool\n(w=6)", "concat\n(w=1)", "concat\n(w=3)", "concat\n(w=6)"]
    colors = ["#2196F3", "#FF7043", "#FF5722", "#BF360C"]
    attack_types = ["novel", "fleet", "spoof"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, at in zip(axes, attack_types):
        x = np.arange(len(signals))
        vals, lo_err, hi_err = [], [], []
        for sig in signals:
            auc, lo, hi = t1_results[sig][at]
            vals.append(auc)
            lo_err.append(auc - lo)
            hi_err.append(hi - auc)
        bars = ax.bar(x, vals, color=colors, width=0.6)
        ax.errorbar(x, vals, yerr=[lo_err, hi_err], fmt="none", color="black", capsize=4, linewidth=1.2)
        ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8, label="OOV ceiling")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.5, 1.05)
        ax.set_title(f"{at} attacks", fontsize=10)
        ax.set_ylabel("ROC-AUC" if at == "novel" else "")
    axes[0].legend(fontsize=7)
    fig.suptitle("T1 — Window Sweep: AUC by attack type (enrollment in negative class)", fontsize=10)
    plt.tight_layout()
    out = FIGURES_DIR / "concat_exp_fig1_window_sweep.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_t2(t2_results: Dict, mp_auc: Tuple, mp_sil: float) -> None:
    """T2: Prefixed-concat vs mean-pool AUC and silhouette."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    attack_types = ["novel", "fleet", "spoof"]
    x = np.arange(len(attack_types))
    w = 0.35

    mp_vals = [mp_auc[at][0] for at in attack_types]
    pc_vals = [t2_results["prefixed_concat"][at][0] for at in attack_types]
    mp_lo = [mp_auc[at][0] - mp_auc[at][1] for at in attack_types]
    mp_hi = [mp_auc[at][2] - mp_auc[at][0] for at in attack_types]
    pc_lo = [t2_results["prefixed_concat"][at][0] - t2_results["prefixed_concat"][at][1] for at in attack_types]
    pc_hi = [t2_results["prefixed_concat"][at][2] - t2_results["prefixed_concat"][at][0] for at in attack_types]

    ax = axes[0]
    ax.bar(x - w/2, mp_vals, w, label="mean_pool_fasttext", color="#2196F3")
    ax.bar(x + w/2, pc_vals, w, label="prefixed_concat_fasttext", color="#9C27B0")
    ax.errorbar(x - w/2, mp_vals, yerr=[mp_lo, mp_hi], fmt="none", color="black", capsize=3)
    ax.errorbar(x + w/2, pc_vals, yerr=[pc_lo, pc_hi], fmt="none", color="black", capsize=3)
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("T2 — Prefixed-concat vs mean-pool AUC")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = ["mean_pool\nfasttext", "prefixed_concat\nfasttext"]
    sils = [mp_sil, t2_results["prefixed_sil"]]
    colors = ["#2196F3", "#9C27B0"]
    bars = ax2.bar(names, sils, color=colors)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    for bar, val in zip(bars, sils):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.3f}", ha="center", fontsize=9)
    ax2.set_ylabel("Silhouette score (cosine)")
    ax2.set_title("T2 — Silhouette comparison")

    plt.tight_layout()
    out = FIGURES_DIR / "concat_exp_fig2_prefixed_concat.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_t3(t3_results: Dict) -> None:
    """T3: Spoof AUC by tz position."""
    data = t3_results["spoof_aucs_by_tz_pos"]
    positions = [d[0] for d in data]
    aucs = [d[2] for d in data]
    lo_err = [d[2] - d[3] for d in data]
    hi_err = [d[4] - d[2] for d in data]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(positions, aucs, yerr=[lo_err, hi_err], marker="o", capsize=5,
                color="#E91E63", linewidth=1.8, markersize=6)
    ax.axhline(min(aucs), color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Timezone feature position in concat string (0 = first)")
    ax.set_ylabel("Spoof attack ROC-AUC")
    ax.set_title("T3 — Tz position permutation: spoof AUC\n(monotonic decrease → positional weighting confirmed)")
    ax.set_xticks(positions)
    ax.set_ylim(min(aucs) - 0.05, max(aucs) + 0.05)
    plt.tight_layout()
    out = FIGURES_DIR / "concat_exp_fig3_tz_permutation.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_t4(t4_results: Dict) -> None:
    """T4: In-vocab vs OOV AUC degradation for mean-pool and concat."""
    mp_iv, mp_ov = t4_results["mp_iv"], t4_results["mp_ov"]
    cc_iv, cc_ov = t4_results["cc_iv"], t4_results["cc_ov"]

    labels = ["mean_pool\nin-vocab", "mean_pool\nOOV", "concat\nin-vocab", "concat\nOOV"]
    vals = [mp_iv[0], mp_ov[0], cc_iv[0], cc_ov[0]]
    lo_errs = [v - ci[1] for v, ci in zip(vals, [mp_iv, mp_ov, cc_iv, cc_ov])]
    hi_errs = [ci[2] - v for v, ci in zip(vals, [mp_iv, mp_ov, cc_iv, cc_ov])]
    colors = ["#2196F3", "#90CAF9", "#FF5722", "#FFAB91"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    bars = ax.bar(x, vals, color=colors, width=0.6)
    ax.errorbar(x, vals, yerr=[lo_errs, hi_errs], fmt="none", color="black", capsize=4)
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8, label="OOV ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Novel attack ROC-AUC")
    ax.set_title("T4 — OOV injection: novel attack AUC\n(in-vocab vs OOV tokens for each signal)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIGURES_DIR / "concat_exp_fig4_oov_injection.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("H2 Experiment — Debate-Agreed Tests T1–T5")
    print("=" * 55)

    print("\n[Setup] Generating data and training base models...")
    fleet = build_fleet(N_FLEET_DEVICES)
    accounts = build_accounts(fleet)
    eval_events = generate_eval_events(accounts, fleet)

    # Base mean-pool model (used across T1, T2, T4)
    mp_model = train_mean_pool_ft(accounts)
    mp_centroids = compute_centroids(accounts, mp_model, lambda p, m: embed_mean_pool(p, m))
    mp_sil = compute_silhouette(accounts, mp_model, lambda p, m: embed_mean_pool(p, m))
    print(f"     mean_pool silhouette: {mp_sil:.4f}")

    # T1 — Window sweep
    t1 = run_t1_window_sweep(accounts, eval_events, mp_model, mp_centroids)
    best_w = t1["_best_window"]
    mp_aucs_t1 = t1["mean_pool"]

    # T2 — Prefixed concat
    t2 = run_t2_prefixed_concat(accounts, eval_events, mp_model, mp_centroids, mp_aucs_t1, best_window=best_w)

    # T3 — Tz permutation
    t3 = run_t3_tz_permutation(accounts, eval_events)

    # T4 — OOV injection (use best-window concat model)
    print(f"\n[T4 setup] Training best-window concat (w={best_w}) for OOV test...")
    cc_best_model = train_concat_ft(accounts, window=best_w)
    cc_best_centroids = compute_centroids(accounts, cc_best_model, lambda p, m: embed_concat(p, m))
    t4 = run_t4_oov(accounts, fleet, mp_model, mp_centroids, cc_best_model, cc_best_centroids)

    # Figures
    print("\n[Figures] Generating...")
    plot_t1(t1)
    plot_t2(t2, mp_aucs_t1, t2["mp_sil"])
    plot_t3(t3)
    plot_t4(t4)

    # Summary
    print("\n" + "=" * 55)
    print("VERDICT SUMMARY")
    print("=" * 55)
    print(f"T1 (window sweep):       {t1['_verdict']}")
    print(f"T2 (prefixed concat):    {t2['_verdict']}")
    print(f"T3 (tz permutation):     {t3['_verdict']}")
    print(f"T4 (OOV injection):      {t4['_verdict']}")
    print("T5 (bootstrap CIs):      Embedded in all AUC outputs above.")


if __name__ == "__main__":
    main()
