#!/usr/bin/env python3
"""Phase 1: scan candidate time series datasets and score each with imbalance_eval.

Pulls univariate series from GluonTS and the TSLib HuggingFace repo, embeds
each with time_delay_embedding(k=10), scores it with compute_imbalance(), and
appends one row per series to a single results CSV. Designed to run
unattended for a long time (~150k series) -- see README.md for usage.
"""
import argparse
import csv
import hashlib
import os
import socket
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# Global socket timeout so a stalled TCP handshake (observed hanging
# indefinitely on a GluonTS download) fails into the existing error_row path
# instead of blocking the whole scan forever. Covers urllib/requests-based
# downloads (gluonts, huggingface_hub) since they fall back to this when no
# per-call timeout is given.
SOCKET_TIMEOUT_S = 60
socket.setdefaulttimeout(SOCKET_TIMEOUT_S)

try:
    import imbalance_eval as ie
except ImportError:
    # Not on PyPI yet, so not in requirements.txt. Point IMBALANCE_EVAL_PATH
    # at a local checkout (e.g. `export IMBALANCE_EVAL_PATH=~/imbalance_eval`)
    # rather than pip-installing it, until it's published.
    _fallback = os.environ.get("IMBALANCE_EVAL_PATH")
    if not _fallback:
        raise ImportError(
            "imbalance_eval not found. Install it, or set IMBALANCE_EVAL_PATH "
            "to a local checkout, e.g. export IMBALANCE_EVAL_PATH=~/imbalance_eval"
        ) from None
    sys.path.insert(0, os.path.expanduser(_fallback))
    import imbalance_eval as ie

CSV_COLUMNS = [
    "id", "source", "collection", "name", "granularity", "time_column", "length", "dtype", "content_hash",
    "N", "n_normal", "n_rare", "IR", "%Rare", "imbalance_level",
    "missing_pct", "mean", "std", "cv", "skewness", "kurtosis", "autocorr_lag1",
    "status", "note",
]

# Params fixed to match Moniz, Branco & Torgo 2017.
REL_THRES = 0.9
REL_COEF = 1.5
REL_XTRM_TYPE = "both"
K = 10

# Excluded because they're huge (would dominate runtime) or need manual setup.
GLUONTS_EXCLUDE = {
    "kaggle_web_traffic_with_missing": "145,063 series",
    "kaggle_web_traffic_without_missing": "145,063 series",
    "kaggle_web_traffic_weekly": "145,063 series",
    "dominick": "115,704 series",
    "temperature_rain_without_missing": "32,072 series",
    "m5": "requires manual Kaggle download",
    "constant": "synthetic",
}

TSLIB_REPO = "thuml/Time-Series-Library"
TSLIB_FOLDERS = {"ETT-small", "electricity", "exchange_rate", "illness", "traffic", "weather"}


def make_id(source, collection, key):
    raw = f"{source}:{collection}:{key}".lower()
    return raw.replace(" ", "-")


def content_hash(values):
    arr = np.asarray(values, dtype=float)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def imbalance_level(n_rare, pct_rare):
    if n_rare == 0:
        return "none"
    if pct_rare < 1:
        return "extreme"
    if pct_rare < 5:
        return "severe"
    if pct_rare < 15:
        return "moderate"
    return "mild"


def score_series(series: pd.Series) -> dict:
    """Embed + score a single numeric series. Returns a dict of CSV fields
    (everything except id/source/collection/name/granularity/time_column)."""
    missing_pct = round(100 * series.isna().mean(), 4)
    clean = series.dropna()

    row = {
        "length": len(series), "dtype": str(series.dtype), "content_hash": content_hash(clean.values),
        "N": "", "n_normal": "", "n_rare": "", "IR": "", "%Rare": "", "imbalance_level": "",
        "missing_pct": missing_pct, "mean": "", "std": "", "cv": "", "skewness": "", "kurtosis": "",
        "autocorr_lag1": "", "status": "error", "note": "",
    }

    if len(clean) < K + 2:
        row["status"] = "skipped_short"
        row["note"] = f"only {len(clean)} non-null values, need >= {K + 2} to embed"
        return row

    mean = clean.mean()
    std = clean.std()
    row["mean"] = mean
    row["std"] = std
    row["cv"] = (std / abs(mean)) if abs(mean) > 1e-12 else np.nan
    row["skewness"] = clean.skew()
    row["kurtosis"] = clean.kurtosis()
    row["autocorr_lag1"] = clean.autocorr(1)

    try:
        embedded = ie.time_delay_embedding(clean, k=K)
        with warnings.catch_warnings():
            # compute_imbalance warns when a series has no boxplot outliers
            # (n_rare=0, IR=0) -- expected, not an error; see imbalance_eval.py.
            warnings.simplefilter("ignore")
            stats = ie.compute_imbalance(
                embedded["target"], rel_thres=REL_THRES, rel_xtrm_type=REL_XTRM_TYPE, rel_coef=REL_COEF
            )
        row.update(stats)
        row["imbalance_level"] = imbalance_level(stats["n_rare"], stats["%Rare"])
        row["status"] = "ok"
    except ValueError as e:
        row["status"] = "skipped_short"
        row["note"] = str(e)[:200]
    except Exception as e:
        row["status"] = "error"
        row["note"] = str(e)[:200]

    return row


def build_row(id_, source, collection, name, granularity, time_column, series):
    base = {"id": id_, "source": source, "collection": collection, "name": name,
            "granularity": granularity, "time_column": time_column}
    try:
        base.update(score_series(series))
    except Exception as e:
        # Last-resort catch so one pathological series never aborts a collection.
        base.update({c: "" for c in CSV_COLUMNS if c not in base})
        base["status"] = "error"
        base["note"] = str(e)[:200]
    return base


def _build_row_star(args):
    return build_row(*args)


def _score_tasks(tasks, workers):
    """Run build_row over a list of build_row-argument tuples, spread across
    worker processes -- each series is scored independently, so this is
    embarrassingly parallel and worth it once there are enough series to
    amortize process startup. Falls back to sequential for workers<=1."""
    if not tasks:
        return []
    if workers <= 1:
        return [build_row(*t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_build_row_star, tasks))


def write_rows(out_path, rows):
    if not rows:
        return
    new_file = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def existing_ids(out_path):
    if not out_path.exists():
        return set()
    return set(pd.read_csv(out_path, usecols=["id"])["id"])


def error_row(source, collection, note):
    """One blank CSV row recording a whole collection that failed to download."""
    return {c: "" for c in CSV_COLUMNS} | {
        "id": make_id(source, collection, "dataset"), "source": source, "collection": collection,
        "status": "error", "note": note[:200],
    }


def iter_gluonts_series(collection, entries, freq):
    """Yield (id_, key, granularity, time_column, series) for each entry of
    ds.train. One malformed entry is skipped (printed), not fatal -- shared by
    scan.py and upload_blobs.py so both normalize identically."""
    for n_seen, entry in enumerate(entries):
        try:
            # `is None` matters, not `or`: item_id can legitimately be 0, which
            # `or` would treat as falsy and overwrite with the counter. Some
            # datasets also set item_id to None explicitly, hence the check.
            key = entry.get("item_id")
            if key is None:
                key = n_seen
            series = pd.Series(np.asarray(entry["target"], dtype=float))
        except Exception as e:
            print(f"gluonts:{collection} SKIPPING bad entry #{n_seen}: {e}")
            continue
        id_ = make_id("gluonts", collection, key)
        yield id_, key, freq, "", series


def iter_tslib_series(collection, df):
    """Yield (id_, key, granularity, time_column, series) for each numeric
    column of a raw TSLib CSV DataFrame. Does the date parse -> sort -> drop
    in here -- shared by scan.py and upload_blobs.py so both normalize
    identically."""
    date_col = "date" if "date" in df.columns else None
    granularity = ""
    index = None
    if date_col:
        # Parse to datetime BEFORE sorting. These files use non-zero-padded
        # dates ("1990-1-2"), so sorting the raw strings puts 1990-1-10
        # before 1990-1-2 and scrambles the whole series. Same order of
        # operations as imbalance_eval.load_csv.
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)
        try:
            granularity = pd.infer_freq(df[date_col]) or ""
        except Exception:
            granularity = ""
        index = pd.DatetimeIndex(df[date_col])
        df = df.drop(columns=[date_col])

    for col in df.select_dtypes(include="number").columns:
        id_ = make_id("tslib", collection, col)
        series = pd.Series(df[col].to_numpy(), index=index) if index is not None else df[col]
        yield id_, col, granularity, (date_col or ""), series


def scan_gluonts(out_path, seen_ids, collections_filter, limit, workers):
    try:
        from gluonts.dataset.repository.datasets import get_dataset, dataset_names
    except ImportError:
        from gluonts.dataset.repository import get_dataset, dataset_names

    names = [n for n in dataset_names if n not in GLUONTS_EXCLUDE]
    if collections_filter:
        names = [n for n in names if n in collections_filter]

    for name in names:
        err_id = make_id("gluonts", name, "dataset")
        if err_id in seen_ids:
            continue
        t0 = time.time()
        try:
            ds = get_dataset(name)
        except Exception as e:
            write_rows(out_path, [error_row("gluonts", name, f"download failed: {e}")])
            seen_ids.add(err_id)
            print(f"gluonts:{name} FAILED to download ({time.time() - t0:.1f}s): {e}")
            continue

        freq = getattr(ds.metadata, "freq", "")
        tasks = []
        for id_, key, granularity, time_column, series in iter_gluonts_series(name, ds.train, freq):
            if id_ in seen_ids:
                continue
            # Bound on NEW rows added this run, not on entries examined -- so
            # --resume with the same --limit keeps making progress instead of
            # immediately exhausting the budget on already-scanned entries.
            if limit and len(tasks) >= limit:
                break
            tasks.append((id_, "gluonts", name, str(key), granularity, time_column, series))
            seen_ids.add(id_)

        rows = _score_tasks(tasks, workers)
        write_rows(out_path, rows)
        print(f"gluonts:{name} {len(rows)} series in {time.time() - t0:.1f}s")


def read_tslib_csv(repo_path):
    """Read one TSLib CSV, preferring the cached hf_hub_download path.

    Falls back to the plain resolve URL because some brotlicffi builds raise
    "decoder process called with data when 'can_accept_more_data() is False'"
    on the larger files (ETT/electricity/traffic). urllib doesn't request
    brotli at all, so the direct URL sidesteps the broken decoder.
    """
    from huggingface_hub import hf_hub_download
    try:
        return pd.read_csv(hf_hub_download(TSLIB_REPO, repo_path, repo_type="dataset"))
    except Exception:
        return pd.read_csv(f"https://huggingface.co/datasets/{TSLIB_REPO}/resolve/main/{repo_path}")


def list_tslib_collections(collections_filter=None):
    """{collection: repo_path} for every TSLib CSV under the in-scope folders,
    shared by scan.py and upload_blobs.py so both resolve the same file."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(TSLIB_REPO, repo_type="dataset")

    by_collection = {}
    for f in files:
        if not f.endswith(".csv"):
            continue
        parts = f.split("/")
        if len(parts) < 2 or parts[0] not in TSLIB_FOLDERS:
            continue
        collection = Path(f).stem
        if collections_filter and collection not in collections_filter:
            continue
        by_collection.setdefault(collection, f)  # one csv per collection expected
    return by_collection


def scan_tslib(out_path, seen_ids, collections_filter, limit, workers):
    by_collection = list_tslib_collections(collections_filter)

    for collection, repo_path in by_collection.items():
        err_id = make_id("tslib", collection, "dataset")
        if err_id in seen_ids:
            continue
        t0 = time.time()
        try:
            df = read_tslib_csv(repo_path)
        except Exception as e:
            write_rows(out_path, [error_row("tslib", collection, f"download/read failed: {e}")])
            seen_ids.add(err_id)
            print(f"tslib:{collection} FAILED to download ({time.time() - t0:.1f}s): {e}")
            continue

        tasks = []
        for id_, key, granularity, time_column, series in iter_tslib_series(collection, df):
            if id_ in seen_ids:
                continue
            # Bound on NEW rows added this run, not on column position -- so
            # --resume with the same --limit keeps making progress instead of
            # re-slicing the same already-scanned leading columns.
            if limit and len(tasks) >= limit:
                break
            tasks.append((id_, "tslib", collection, key, granularity, time_column, series))
            seen_ids.add(id_)

        rows = _score_tasks(tasks, workers)
        write_rows(out_path, rows)
        print(f"tslib:{collection} {len(rows)} series in {time.time() - t0:.1f}s")


def print_summary(out_path):
    df = pd.read_csv(out_path)
    print("\n=== summary ===")
    print(f"total rows: {len(df)}")
    print("\nstatus:")
    print(df["status"].value_counts().to_string())
    print("\nimbalance_level:")
    print(df["imbalance_level"].value_counts(dropna=False).to_string())
    passed = (df["n_rare"].fillna(0).astype(float) > 0).sum()
    print(f"\npassed n_rare > 0 gate: {passed} / {len(df)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="scan_results.csv")
    parser.add_argument("--source", choices=["gluonts", "tslib", "both"], default="both")
    parser.add_argument("--collections", default=None, help="comma-separated subset of collections")
    parser.add_argument("--limit", type=int, default=0, help="max series per collection, 0=all")
    parser.add_argument("--resume", action="store_true", help="skip ids already present in --out")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                         help="parallel worker processes for scoring (default: all cores)")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.resume:
        print(f"error: {out_path} already exists. Pass --resume to continue it, or remove it first.", file=sys.stderr)
        sys.exit(1)

    seen_ids = existing_ids(out_path) if args.resume else set()
    if seen_ids:
        print(f"resuming: {len(seen_ids)} ids already in {out_path}")

    collections_filter = set(args.collections.split(",")) if args.collections else None

    if args.source in ("gluonts", "both"):
        scan_gluonts(out_path, seen_ids, collections_filter, args.limit, args.workers)
    if args.source in ("tslib", "both"):
        scan_tslib(out_path, seen_ids, collections_filter, args.limit, args.workers)

    if out_path.exists():
        print_summary(out_path)


if __name__ == "__main__":
    main()
