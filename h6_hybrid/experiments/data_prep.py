# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "polars>=0.20",
# ]
# ///
#
# H6 Hybrid — RBA Marginals Extraction
# =====================================
# Reads rba.parquet (clean logins only), computes co-occurrence marginals for
# chain-sampling, and writes h6_hybrid/experiments/rba_marginals.json.
#
# Filtering: login_successful=True, is_attack_ip=False, is_ato=False
# (clean user behavior only — no attack contamination in marginals)
#
# Marginals structure:
#   os_device_joint          : P(os, device_type)                     dict[(os,device)] -> count
#   browser_given_os         : P(browser | os)                        dict[os][browser] -> count
#   country_marginal         : P(country)                             dict[country] -> count
#   region_given_country     : P(region | country)                    dict[country][region] -> count
#   asn_given_country        : P(asn_bucket | country)                dict[country][asn] -> count
#   rtt_marginal             : P(rtt_bucket)                          dict[rtt_bucket] -> count
#
# Usage:
#   uv run h6_hybrid/experiments/data_prep.py

import json
import sys
from pathlib import Path
from collections import defaultdict

import polars as pl

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "rba" / "rba.parquet"
OUT_PATH     = Path(__file__).resolve().parent / "rba_marginals.json"

# Minimum observations required to keep a country in conditional lookups
N_MIN_COUNTRY = 5
# Minimum distinct values required in a conditional; else fall back to marginal
N_MIN_DISTINCT = 3

FEATURE_COLS = ["os", "browser", "device_type", "country", "region", "asn_bucket", "rtt_bucket"]


def counts_to_dict(df: pl.DataFrame, *key_cols: str, count_col: str = "count") -> dict:
    """Convert a polars groupby result to a nested Python dict."""
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
    print("H6 Hybrid — RBA Marginals Extraction")
    print("=" * 60)

    if not PARQUET_PATH.exists():
        sys.exit(f"ERROR: {PARQUET_PATH} not found. Run h2_rba/experiments/data_prep.py first.")

    print(f"\nLoading {PARQUET_PATH} ...")
    raw = pl.read_parquet(PARQUET_PATH)
    print(f"  Total rows: {len(raw):,}")

    # Verify required columns
    missing = [c for c in ["login_successful", "is_attack_ip", "is_ato"] + FEATURE_COLS
               if c not in raw.columns]
    if missing:
        sys.exit(f"ERROR: missing columns in parquet: {missing}")

    # Filter to clean logins only
    clean = raw.filter(
        pl.col("login_successful") & ~pl.col("is_attack_ip") & ~pl.col("is_ato")
    )
    print(f"  Clean rows (successful, non-attack, non-ATO): {len(clean):,}")
    print(f"  Dropped: {len(raw) - len(clean):,}")

    # Fill nulls with "unknown" for all feature columns
    for col in FEATURE_COLS:
        if col not in clean.columns:
            sys.exit(f"ERROR: feature column '{col}' missing after filter")
    clean = clean.with_columns(
        [pl.col(c).cast(pl.Utf8).fill_null("unknown") for c in FEATURE_COLS]
    )

    print("\nComputing marginals ...")

    # 1. os_device_joint: P(os, device_type)
    os_device_df = (
        clean.group_by(["os", "device_type"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    os_device_joint = counts_to_dict(os_device_df, "os", "device_type")
    total_od = sum(v for d in os_device_joint.values() for v in d.values())
    print(f"  os_device_joint: {sum(len(v) for v in os_device_joint.values())} combos "
          f"over {total_od:,} events")

    # 2. browser_given_os: P(browser | os)
    browser_os_df = (
        clean.group_by(["os", "browser"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    browser_given_os = counts_to_dict(browser_os_df, "os", "browser")
    print(f"  browser_given_os: {len(browser_given_os)} os values")

    # 3. country_marginal: P(country)
    country_df = (
        clean.group_by("country")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    country_marginal = counts_to_dict(country_df, "country")
    print(f"  country_marginal: {len(country_marginal)} countries")

    # 4. region_given_country: P(region | country)
    region_country_df = (
        clean.group_by(["country", "region"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    region_given_country_raw = counts_to_dict(region_country_df, "country", "region")

    # Apply N_MIN_COUNTRY and N_MIN_DISTINCT filters
    region_given_country = {}
    region_fallback_countries = []
    for country, regions in region_given_country_raw.items():
        total = sum(regions.values())
        n_distinct = len(regions)
        if total < N_MIN_COUNTRY or n_distinct < N_MIN_DISTINCT:
            region_fallback_countries.append(country)
        else:
            region_given_country[country] = regions
    print(f"  region_given_country: {len(region_given_country)} eligible countries "
          f"({len(region_fallback_countries)} fall back to marginal)")

    # Overall region marginal for fallback
    region_marginal_df = (
        clean.group_by("region")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    region_marginal = counts_to_dict(region_marginal_df, "region")

    # 5. asn_given_country: P(asn_bucket | country)
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
        n_distinct = len(asns)
        if total < N_MIN_COUNTRY or n_distinct < N_MIN_DISTINCT:
            asn_fallback_countries.append(country)
        else:
            asn_given_country[country] = asns
    print(f"  asn_given_country: {len(asn_given_country)} eligible countries "
          f"({len(asn_fallback_countries)} fall back to marginal)")

    # Overall asn_bucket marginal for fallback
    asn_marginal_df = (
        clean.group_by("asn_bucket")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    asn_marginal = counts_to_dict(asn_marginal_df, "asn_bucket")

    # 6. rtt_marginal: P(rtt_bucket)
    rtt_df = (
        clean.group_by("rtt_bucket")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    rtt_marginal = counts_to_dict(rtt_df, "rtt_bucket")
    print(f"  rtt_marginal: {rtt_marginal}")

    # Serialize to JSON
    # JSON keys must be strings; tuples become "os::device_type" keys
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
    print("\nDone. Run: uv run h6_hybrid/experiments/hybrid_experiment.py")


if __name__ == "__main__":
    main()
