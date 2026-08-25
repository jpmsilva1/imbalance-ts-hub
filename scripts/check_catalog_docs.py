#!/usr/bin/env python3
"""Fails if catalog/filter_reference.json (the numbers quoted in README.md's
"Filterable values" table and dataset_card.md's "Filterable columns" table)
has drifted from what's actually in catalog/series.csv.

Not a public API -- internal safety net for the hand-written docs tables.
Regenerate the reference after intentionally updating the catalog:

    python scripts/check_catalog_docs.py --write

Then update the matching tables in README.md and dataset_card.md by hand.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

NUMERIC_COLUMNS = ["N", "IR", "%Rare", "length", "missing_pct", "seasonal_period"]
ROUND_DP = {"N": 0, "length": 0, "IR": 2, "%Rare": 2, "missing_pct": 4, "seasonal_period": 0}

# Must match the "examples" rows actually quoted in the docs tables.
COLLECTION_EXAMPLES = [
    "m4_monthly", "wiki-rolling_nips", "m4_quarterly",
    "london_smart_meters_without_missing", "m4_yearly",
    "car_parts_without_missing", "weather", "wiki2000_nips",
]


def _round(value, col):
    dp = ROUND_DP[col]
    v = round(float(value), dp)
    return int(v) if dp == 0 else v


def compute_reference(df: pd.DataFrame) -> dict:
    ref = {"row_count": len(df)}
    ref["source"] = df["source"].value_counts().to_dict()
    ref["license"] = df["license"].value_counts().to_dict()
    ref["imbalance_level"] = df["imbalance_level"].value_counts().to_dict()
    ref["dtype"] = df["dtype"].value_counts().to_dict()

    ref["time_column_by_source"] = {
        src: (vals.dropna().iloc[0] if not vals.dropna().empty else None)
        for src, vals in df.groupby("source")["time_column"]
    }

    ref["granularity"] = df["granularity"].fillna("__null__").value_counts().to_dict()

    ref["collection_count_distinct"] = int(df["collection"].nunique())
    counts = df["collection"].value_counts()
    ref["collection_examples"] = {c: int(counts.get(c, 0)) for c in COLLECTION_EXAMPLES}

    numeric = {}
    for col in NUMERIC_COLUMNS:
        s = df[col].dropna()
        q = s.quantile([0.25, 0.5, 0.75])
        entry = {label: _round(q[frac], col) for label, frac in
                 [("25%", 0.25), ("50%", 0.5), ("75%", 0.75)]}
        # min/max via .min()/.max() rather than quantile(0)/quantile(1.0):
        # pandas' quantile() does linear interpolation even at the endpoints,
        # which yields NaN (inf - inf) when the column contains inf (e.g. IR
        # for a series that's all rare points, n_normal == 0).
        entry["min"] = _round(s.min(), col)
        entry["max"] = _round(s.max(), col)
        if df[col].isna().any():
            entry["null_count"] = int(df[col].isna().sum())
        numeric[col] = entry
    ref["numeric"] = numeric
    return ref


def diff_reference(expected: dict, actual: dict, path="") -> list[str]:
    """Flat, human-readable drift lines; empty list = no drift."""
    diffs = []
    for key in sorted(set(expected) | set(actual), key=str):
        p = f"{path}.{key}" if path else str(key)
        if key not in expected:
            diffs.append(f"{p}: new in catalog (value={actual[key]!r}), not in reference")
        elif key not in actual:
            diffs.append(f"{p}: in reference (value={expected[key]!r}) but missing from catalog")
        elif isinstance(expected[key], dict) and isinstance(actual[key], dict):
            diffs.extend(diff_reference(expected[key], actual[key], p))
        elif expected[key] != actual[key]:
            diffs.append(f"{p}: reference={expected[key]!r} actual={actual[key]!r}")
    return diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="catalog/series.csv")
    parser.add_argument("--reference", default="catalog/filter_reference.json")
    parser.add_argument("--write", action="store_true", help="regenerate the reference instead of checking it")
    args = parser.parse_args()

    df = pd.read_csv(args.catalog, low_memory=False)
    actual = compute_reference(df)
    ref_path = Path(args.reference)

    if args.write:
        ref_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        print(f"wrote {ref_path}")
        return

    expected = json.loads(ref_path.read_text())
    diffs = diff_reference(expected, actual)
    if diffs:
        print(f"catalog docs reference ({ref_path}) is out of date with {args.catalog}:\n")
        for d in diffs:
            print(f"  - {d}")
        print("\nIf this drift is expected, regenerate with "
              "`python scripts/check_catalog_docs.py --write`, then manually "
              "update the matching tables in README.md and dataset_card.md.")
        sys.exit(1)
    print("catalog docs reference is up to date.")


if __name__ == "__main__":
    main()
