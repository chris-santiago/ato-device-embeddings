# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "polars==1.40.0",
#   "requests==2.32.3",
#   "tqdm==4.67.1",
# ]
# ///
#
# RBA Dataset Preparation
# =======================
# Downloads rba-dataset.zip from GitHub releases, extracts the CSV,
# normalizes column names, derives feature buckets, and writes
# data/rba/rba.parquet for use by rba_rerun.py.
#
# Idempotent: re-running skips steps whose outputs already exist.
# Run: uv run experiments/rerun/scripts/rba/data_prep.py

import sys
import zipfile
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

DOWNLOAD_URL = (
    "https://github.com/das-group/rba-dataset/releases/download/v1.0.0/rba-dataset.zip"
)
REPO_ROOT    = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_DIR     = REPO_ROOT / "data" / "rba"
ZIP_PATH     = DATA_DIR / "rba-dataset.zip"
PARQUET_PATH = DATA_DIR / "rba.parquet"

RENAME = {
    "IP Address":                 "ip",
    "Country":                    "country",
    "Region":                     "region",
    "City":                       "city",
    "ASN":                        "asn",
    "User Agent String":          "user_agent",
    "OS Name and Version":        "os_raw",
    "Browser Name and Version":   "browser_raw",
    "Device Type":                "device_type",
    "User ID":                    "user_id",
    "Login Timestamp":            "login_timestamp",
    "Round-Trip Time [ms]":       "rtt_ms",
    "Login Successful":           "login_successful",
    "Is Attack IP":               "is_attack_ip",
    "Is Account Takeover":        "is_ato",
}


def _os_expr(col: str) -> pl.Expr:
    s = pl.col(col).cast(pl.Utf8).str.to_lowercase()
    return (
        pl.when(s.str.contains("windows")).then(pl.lit("windows"))
        .when(s.str.contains("ios|iphone|ipad")).then(pl.lit("ios"))
        .when(s.str.contains("android")).then(pl.lit("android"))
        .when(s.str.contains("mac|darwin")).then(pl.lit("macos"))
        .when(s.str.contains("linux|ubuntu|debian")).then(pl.lit("linux"))
        .otherwise(pl.col(col).cast(pl.Utf8).str.split(" ").list.first().str.to_lowercase())
        .fill_null("unknown")
        .alias("os")
    )


def _browser_expr(col: str) -> pl.Expr:
    s = pl.col(col).cast(pl.Utf8).str.to_lowercase()
    return (
        pl.when(s.str.contains("edge")).then(pl.lit("edge"))
        .when(s.str.contains("samsung")).then(pl.lit("samsung"))
        .when(s.str.contains("chrome")).then(pl.lit("chrome"))
        .when(s.str.contains("safari")).then(pl.lit("safari"))
        .when(s.str.contains("firefox")).then(pl.lit("firefox"))
        .when(s.str.contains("opera")).then(pl.lit("opera"))
        .otherwise(pl.col(col).cast(pl.Utf8).str.split(" ").list.first().str.to_lowercase())
        .fill_null("unknown")
        .alias("browser")
    )


def _rtt_bucket_expr(col: str) -> pl.Expr:
    r = pl.col(col).cast(pl.Float64, strict=False).clip(lower_bound=0)
    return (
        pl.when(r < 50).then(pl.lit("rtt_lt50"))
        .when(r < 150).then(pl.lit("rtt_50_150"))
        .when(r < 400).then(pl.lit("rtt_150_400"))
        .when(r.is_not_null()).then(pl.lit("rtt_gt400"))
        .otherwise(pl.lit("rtt_missing"))
        .alias("rtt_bucket")
    )


def download_zip() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print(f"  Already present: {ZIP_PATH}  ({ZIP_PATH.stat().st_size / 1e9:.2f} GB)")
        return
    print(f"  GET {DOWNLOAD_URL}")
    resp = requests.get(DOWNLOAD_URL, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(ZIP_PATH, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="  download") as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved {ZIP_PATH}  ({ZIP_PATH.stat().st_size / 1e9:.2f} GB)")


def find_csv_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    print(f"  Zip contents: {names}")
    csvs = [n for n in names if n.lower().endswith(".csv")]
    if not csvs:
        sys.exit(f"ERROR: no CSV found in zip. Contents: {names}")
    return csvs[0]


def extract_csv(zip_path: Path, csv_name: str) -> Path:
    out = DATA_DIR / Path(csv_name).name
    if out.exists():
        print(f"  Already extracted: {out}")
        return out
    print(f"  Extracting {csv_name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(csv_name, DATA_DIR)
    extracted = DATA_DIR / csv_name
    if extracted != out:
        extracted.rename(out)
    print(f"  Extracted to {out}  ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def build_parquet(csv_path: Path) -> None:
    if PARQUET_PATH.exists():
        print(f"  Already present: {PARQUET_PATH}  ({PARQUET_PATH.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"  Scanning {csv_path.name} ...")
    schema = pl.read_csv_batched(csv_path, n_rows=1).next_batches(1)[0].columns
    print(f"  CSV columns: {schema}")
    rename_map = {k: v for k, v in RENAME.items() if k in schema}
    missing = [k for k in RENAME if k not in schema]
    if missing:
        print(f"  WARNING: columns not found in CSV (skipped): {missing}")
    df = (
        pl.scan_csv(csv_path, infer_schema_length=10_000)
        .rename(rename_map)
        .with_columns([
            _os_expr("os_raw"),
            _browser_expr("browser_raw"),
            pl.col("device_type").cast(pl.Utf8).str.to_lowercase().str.strip_chars().fill_null("unknown"),
            pl.col("country").cast(pl.Utf8).str.to_lowercase().str.strip_chars().fill_null("unknown"),
            pl.col("region").cast(pl.Utf8).str.to_lowercase().str.strip_chars().fill_null("unknown"),
            (pl.col("asn").cast(pl.Int64, strict=False).fill_null(-1) // 1000)
                .cast(pl.Utf8).alias("asn_bucket"),
            _rtt_bucket_expr("rtt_ms"),
            pl.col("is_ato").cast(pl.Boolean, strict=False).fill_null(False),
            pl.col("user_id").cast(pl.Utf8),
            pl.col("login_timestamp").cast(pl.Utf8)
                .str.to_datetime(format=None, strict=False)
                .dt.epoch(time_unit="s")
                .alias("login_timestamp"),
        ])
        .collect()
    )

    df.write_parquet(PARQUET_PATH)
    mb = PARQUET_PATH.stat().st_size / 1e6
    print(f"  Wrote {PARQUET_PATH}  ({mb:.0f} MB)")
    print("\n  Dataset summary:")
    print(f"    Total logins:  {len(df):>14,}")
    print(f"    Unique users:  {df['user_id'].n_unique():>14,}")
    ato_count = df["is_ato"].sum()
    print(f"    ATO events:    {ato_count:>14,}  ({100 * ato_count / len(df):.4f}%)")
    print(f"    ATO users:     {df.filter(pl.col('is_ato'))['user_id'].n_unique():>14,}")
    preview_cols = [c for c in ["user_id", "os", "browser", "device_type", "country",
                                "region", "asn_bucket", "rtt_bucket",
                                "login_timestamp", "is_ato"] if c in df.columns]
    print("\n  Sample row:")
    print(df.select(preview_cols).head(1))


def main() -> None:
    print("=" * 60)
    print("RBA Dataset Preparation")
    print("=" * 60)
    print("\n[1/3] Download zip ...")
    download_zip()
    print("\n[2/3] Locate and extract CSV ...")
    csv_name = find_csv_in_zip(ZIP_PATH)
    csv_path = extract_csv(ZIP_PATH, csv_name)
    print("\n[3/3] Build parquet ...")
    build_parquet(csv_path)
    print("\nDone. Run: uv run experiments/rerun/scripts/rba/rba_rerun.py --seed 42")


if __name__ == "__main__":
    main()
