# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "polars==1.40.0",
# ]
# ///
#
# H6 — RBA Marginals Extraction (rerun copy)
# ===========================================
# Reads rba.parquet (clean logins only), computes co-occurrence marginals for
# chain-sampling, and writes rba_marginals.json to this script's directory
# (experiments/rerun/scripts/h6/rba_marginals.json).
#
# This script must be run once before h6_rerun.py. It is deterministic —
# output does not vary by seed.
#
# Prerequisite: data/rba/rba.parquet must exist at the repo root.
# Source: h6_hybrid/experiments/data_prep.py (copied; paths updated for rerun).
#
# Filtering: login_successful=True, is_attack_ip=False, is_ato=False
#
# Marginals written:
#   os_device_joint          : P(os, device_type)
#   browser_given_os         : P(browser | os)
#   country_marginal         : P(country)
#   region_given_country     : P(region | country)
#   asn_given_country        : P(asn_bucket | country)
#   rtt_marginal             : P(rtt_bucket)
#
# Usage (from repo root):
#   uv run experiments/rerun/scripts/h6/data_prep.py

import json
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent.parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "rba" / "rba.parquet"
OUT_PATH     = Path(__file__).resolve().parent / "rba_marginals.json"

N_MIN_COUNTRY = 5
N_MIN_DISTINCT = 3

FEATURE_COLS = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]


def counts_to_dict(df: pl.DataFrame, *key_cols: str, count_col: str = "count") -> dict:
    if len(key_cols) == 1:
        return {row[key_cols[0]]: row[count_col] for row in df.iter_rows(named=True)}
    elif len(key_cols) == 2:
        result: dict = defaultdict(dict)
        for row in df.iter_rows(named=True):
            result[row[key_cols[0]]][row[key_cols[1]]] = row[count_col]
        return dict(result)
    else:
        raise ValueError("Only 1 or 2 key columns supported")


def main() -> None:
    print("=" * 60)
    print("H6 — RBA Marginals Extraction (rerun)")
    print("=" * 60)

    if not PARQUET_PATH.exists():
        sys.exit(
            f"ERROR: {PARQUET_PATH} not found.\n"
            "Obtain rba.parquet and place it at data/rba/rba.parquet relative to the repo root.\n"
            "If you have the raw RBA dataset, run h2_rba/experiments/data_prep.py first."
        )

    print(f"\nLoading {PARQUET_PATH} ...")
    raw = pl.read_parquet(PARQUET_PATH)
    print(f"  Total rows: {len(raw):,}")

    missing = [c for c in ["login_successful", "is_attack_ip", "is_ato"] + FEATURE_COLS
               if c not in raw.columns]
    if missing:
        sys.exit(f"ERROR: missing columns in parquet: {missing}")

    clean = raw.filter(
        pl.col("login_successful") & ~pl.col("is_attack_ip") & ~pl.col("is_ato")
    )
    print(f"  Clean rows (successful, non-attack, non-ATO): {len(clean):,}")
    print(f"  Dropped: {len(raw) - len(clean):,}")

    clean = clean.with_columns(
        [pl.col(c).cast(pl.Utf8).fill_null("unknown") for c in FEATURE_COLS]
    )

    print("\nComputing marginals ...")

    os_device_df = (
        clean.group_by(["os", "device_type"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    os_device_joint = counts_to_dict(os_device_df, "os", "device_type")
    total_od = sum(v for d in os_device_joint.values() for v in d.values())
    print(f"  os_device_joint: {sum(len(v) for v in os_device_joint.values())} combos "
          f"over {total_od:,} events")

    browser_os_df = (
        clean.group_by(["os", "browser"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    browser_given_os = counts_to_dict(browser_os_df, "os", "browser")
    print(f"  browser_given_os: {len(browser_given_os)} os values")

    country_df = (
        clean.group_by("country")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    country_marginal = counts_to_dict(country_df, "country")
    print(f"  country_marginal: {len(country_marginal)} countries")

    region_country_df = (
        clean.group_by(["country", "region"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    region_given_country_raw = counts_to_dict(region_country_df, "country", "region")

    region_given_country = {}
    region_fallback_countries = []
    for country, regions in region_given_country_raw.items():
        total = sum(regions.values())
        if total < N_MIN_COUNTRY or len(regions) < N_MIN_DISTINCT:
            region_fallback_countries.append(country)
        else:
            region_given_country[country] = regions
    print(f"  region_given_country: {len(region_given_country)} eligible countries "
          f"({len(region_fallback_countries)} fall back to marginal)")

    region_marginal_df = (
        clean.group_by("region")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    region_marginal = counts_to_dict(region_marginal_df, "region")

    asn_country_df = (
        clean.group_by(["country", "asn_bucket"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    asn_given_country_raw = counts_to_dict(asn_country_df, "country", "asn_bucket")

    asn_given_country = {}
    asn_fallback_countries = []
    for country, asns in asn_given_country_raw.items():
        total = sum(asns.values())
        if total < N_MIN_COUNTRY or len(asns) < N_MIN_DISTINCT:
            asn_fallback_countries.append(country)
        else:
            asn_given_country[country] = asns
    print(f"  asn_given_country: {len(asn_given_country)} eligible countries "
          f"({len(asn_fallback_countries)} fall back to marginal)")

    asn_marginal_df = (
        clean.group_by("asn_bucket")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    asn_marginal = counts_to_dict(asn_marginal_df, "asn_bucket")

    rtt_df = (
        clean.group_by("rtt_bucket")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    rtt_marginal = counts_to_dict(rtt_df, "rtt_bucket")
    print(f"  rtt_marginal: {rtt_marginal}")

    os_device_joint_flat: dict = {}
    for os_val, device_dict in os_device_joint.items():
        for device_val, cnt in device_dict.items():
            os_device_joint_flat[f"{os_val}::{device_val}"] = cnt

    marginals = {
        "os_device_joint": os_device_joint_flat,
        "browser_given_os": browser_given_os,
        "country_marginal": country_marginal,
        "region_given_country": region_given_country,
        "region_marginal": region_marginal,
        "asn_given_country": asn_given_country,
        "asn_marginal": asn_marginal,
        "rtt_marginal": rtt_marginal,
        "region_fallback_countries": region_fallback_countries,
        "asn_fallback_countries": asn_fallback_countries,
        "meta": {
            "total_clean_rows": len(clean),
            "total_raw_rows": len(raw),
            "n_min_country": N_MIN_COUNTRY,
            "n_min_distinct": N_MIN_DISTINCT,
        }
    }

    with open(OUT_PATH, "w") as f:
        json.dump(marginals, f, indent=2)
    sz = OUT_PATH.stat().st_size / 1e6
    print(f"\nWrote {OUT_PATH}  ({sz:.2f} MB)")
    print("\nDone. Run: uv run experiments/rerun/scripts/h6/h6_rerun.py --seed 42")


if __name__ == "__main__":
    main()
