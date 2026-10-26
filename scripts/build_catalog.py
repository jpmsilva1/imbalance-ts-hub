#!/usr/bin/env python3
"""Turns a raw scan_results.csv (from scan.py) into the two catalog CSVs
defined by the design plan's "Catalog schema": catalog/series.csv (accepted
series only, full schema) and catalog/scanned.csv (every scanned series,
audit trail).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Fixed to match Moniz, Branco & Torgo 2017 -- same constants as scan.py.
REL_THRES = 0.9
REL_COEF = 1.5
REL_XTRM_TYPE = "both"
K = 9  # lags; paper's create.data(ts, 10) -- see scan.py
EMBED = True
DIFF = False

# Sources report the same frequency under different spellings (GluonTS vs.
# TSLib, or inconsistent pandas-alias casing) -- collapse those to one
# canonical string before anything downstream (seasonal_period lookup,
# published catalog) has to deal with duplicates.
GRANULARITY_ALIASES = {
    "1H": "H", "h": "H", "1h": "H",
    "1B": "B",
    "0.5h": "30min",
}

# ponytail: coarse granularity -> period lookup, not a fitted seasonal
# decomposition (matches the plan's explicit "24/hourly, 7/daily, 12/monthly"
# example). Covers every canonical granularity string observed in
# scan_results.csv (post GRANULARITY_ALIASES normalization); extend when a
# new one shows up rather than trying to parse freq strings.
SEASONAL_PERIOD = {
    "H": 24,
    "D": 7, "B": 5,
    "W": 52, "W-TUE": 52,
    "M": 12, "1M": 12,
    "Q": 4,
    "Y": 1,
    "min": 1440, "10min": 144, "15min": 96, "30min": 48,
}

SERIES_COLUMNS = [
    "id", "name", "source", "collection", "license",
    "granularity", "time_column", "length", "dtype", "size_bytes",
    "N", "n_normal", "n_rare", "IR", "%Rare", "imbalance_level",
    "rel_thres", "rel_coef", "rel_xtrm_type", "k", "embed", "diff",
    "missing_pct", "mean", "std", "cv", "skewness", "kurtosis",
    "autocorr_lag1", "seasonal_period",
    "content_hash", "blob_path", "hf_revision", "pipeline_version", "ingested_at",
]

SCANNED_COLUMNS = ["id", "content_hash", "N", "n_rare", "%Rare", "IR", "imbalance_level", "verdict"]


def build_catalog(scan_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a scan_results.csv-shaped DataFrame into (series_df, scanned_df).

    series_df: accepted series only (status=="ok" and n_rare>0), full catalog
    schema, sorted by id. blob_path/hf_revision/pipeline_version/ingested_at
    are left blank -- those are filled by the future HF blob-upload step, not
    known at scan time.
    scanned_df: every row, with a derived "accepted"/"rejected" verdict --
    the full audit trail, including skipped/error rows.
    """
    accepted_mask = (scan_df["status"] == "ok") & (scan_df["n_rare"] > 0)

    series_df = scan_df[accepted_mask].copy()
    series_df["granularity"] = series_df["granularity"].replace(GRANULARITY_ALIASES)
    # Per dataset_card.md's Licensing section: M4 (GluonTS's m4_* collections)
    # carries no declared license; TSLib's bundled CSV licenses are unstated;
    # every other GluonTS collection comes from the Monash/Zenodo archive
    # under CC BY 4.0.
    series_df["license"] = "cc-by-4.0"
    series_df.loc[series_df["collection"].astype(str).str.startswith("m4"), "license"] = "unlicensed"
    series_df.loc[series_df["source"] == "tslib", "license"] = "unknown"
    series_df["size_bytes"] = pd.NA
    series_df["rel_thres"] = REL_THRES
    series_df["rel_coef"] = REL_COEF
    series_df["rel_xtrm_type"] = REL_XTRM_TYPE
    series_df["k"] = K
    series_df["embed"] = EMBED
    series_df["diff"] = DIFF
    series_df["seasonal_period"] = series_df["granularity"].map(SEASONAL_PERIOD)
    for col in ("blob_path", "hf_revision", "pipeline_version", "ingested_at"):
        series_df[col] = pd.NA
    series_df = series_df[SERIES_COLUMNS].sort_values("id").reset_index(drop=True)

    scanned_df = scan_df.copy()
    scanned_df["verdict"] = "rejected"
    scanned_df.loc[accepted_mask, "verdict"] = "accepted"
    scanned_df = scanned_df[SCANNED_COLUMNS].sort_values("id").reset_index(drop=True)

    return series_df, scanned_df


BLOB_COLUMNS = ["size_bytes", "blob_path", "hf_revision", "pipeline_version", "ingested_at"]


def carry_over_blob_columns(series_df: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Keep already-uploaded blob metadata for ids whose content_hash is unchanged,
    so a rebuild doesn't orphan every blob (upload_blobs only stages rows with
    blob_path NA)."""
    prev = previous.set_index("id")
    same = series_df["id"].map(prev["content_hash"]) == series_df["content_hash"]
    out = series_df.astype({c: "object" for c in BLOB_COLUMNS})
    for c in BLOB_COLUMNS:
        out.loc[same, c] = out.loc[same, "id"].map(prev[c]).values
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", default="scan_results.csv")
    parser.add_argument("--out-dir", default="catalog")
    args = parser.parse_args()

    scan_df = pd.read_csv(args.scan, low_memory=False)
    deduped = scan_df.drop_duplicates(subset="id")
    n_dropped = len(scan_df) - len(deduped)
    if n_dropped:
        dupes = scan_df.loc[scan_df["id"].duplicated(keep=False), "id"].unique()
        print(f"warning: dropped {n_dropped} duplicate-id rows, ids: {list(dupes)}", file=sys.stderr)
    scan_df = deduped
    series_df, scanned_df = build_catalog(scan_df)
    prev_path = Path(args.out_dir) / "series.csv"
    if prev_path.exists():
        series_df = carry_over_blob_columns(series_df, pd.read_csv(prev_path, low_memory=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    series_df.to_csv(out_dir / "series.csv", index=False)
    scanned_df.to_csv(out_dir / "scanned.csv", index=False)
    print(f"series.csv: {len(series_df)} accepted series")
    print(f"scanned.csv: {len(scanned_df)} total scanned")


if __name__ == "__main__":
    main()
