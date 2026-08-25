import pandas as pd

from scripts.build_catalog import build_catalog


def _scan_row(**overrides):
    """One scan_results.csv row with realistic ok/accepted defaults."""
    row = {
        "id": "gluonts:m4_hourly:h1", "source": "gluonts", "collection": "m4_hourly",
        "name": "h1", "granularity": "H", "time_column": "", "length": 500,
        "dtype": "float64", "content_hash": "abc123",
        "N": 490, "n_normal": 470, "n_rare": 20, "IR": 0.04, "%Rare": 4.08,
        "imbalance_level": "severe",
        "missing_pct": 0.0, "mean": 100.0, "std": 10.0, "cv": 0.1,
        "skewness": 0.5, "kurtosis": 1.2, "autocorr_lag1": 0.8,
        "status": "ok", "note": "",
    }
    row.update(overrides)
    return row


def test_accepted_series_lands_in_series_csv_with_full_catalog_schema():
    scan_df = pd.DataFrame([_scan_row()])

    series_df, scanned_df = build_catalog(scan_df)

    assert len(series_df) == 1
    row = series_df.iloc[0]
    assert row["id"] == "gluonts:m4_hourly:h1"
    assert row["source"] == "gluonts"
    assert row["collection"] == "m4_hourly"
    assert row["N"] == 490
    assert row["n_rare"] == 20
    assert row["imbalance_level"] == "severe"
    # Fixed scoring params, recorded per row per the catalog schema.
    assert row["rel_thres"] == 0.9
    assert row["rel_coef"] == 1.5
    assert row["rel_xtrm_type"] == "both"
    assert row["k"] == 10
    assert row["embed"] == True
    assert row["diff"] == False
    # Granularity-keyed lookup, not a fitted decomposition.
    assert row["seasonal_period"] == 24
    assert row["content_hash"] == "abc123"

    assert len(scanned_df) == 1
    assert scanned_df.iloc[0]["verdict"] == "accepted"


def test_series_with_no_rare_regime_is_excluded_from_series_csv_but_audited():
    scan_df = pd.DataFrame([_scan_row(
        id="gluonts:m4_hourly:h2", n_rare=0, n_normal=490, IR=0.0, **{"%Rare": 0.0},
        imbalance_level="none",
    )])

    series_df, scanned_df = build_catalog(scan_df)

    assert len(series_df) == 0
    assert len(scanned_df) == 1
    assert scanned_df.iloc[0]["verdict"] == "rejected"
    assert scanned_df.iloc[0]["id"] == "gluonts:m4_hourly:h2"


def test_skipped_and_error_rows_are_audited_but_excluded_from_series_csv():
    # None/NaN, not "" -- matches what pd.read_csv actually produces for a
    # blank cell in a numeric column (a real scan_results.csv row for a
    # skipped/error series has empty N/n_rare/etc fields).
    blank = dict(N=None, n_normal=None, n_rare=None, IR=None, imbalance_level=None)
    scan_df = pd.DataFrame([
        _scan_row(id="gluonts:m1_yearly:9", status="skipped_short", **blank, **{"%Rare": None}),
        _scan_row(id="gluonts:m3_monthly:dataset", status="error", **blank, **{"%Rare": None}),
    ])

    series_df, scanned_df = build_catalog(scan_df)

    assert len(series_df) == 0
    assert len(scanned_df) == 2
    assert set(scanned_df["verdict"]) == {"rejected"}


def test_granularity_aliases_are_normalized_to_one_canonical_spelling():
    scan_df = pd.DataFrame([
        _scan_row(id="gluonts:m4_hourly:h1", granularity="1H"),
        _scan_row(id="gluonts:m4_hourly:h2", granularity="h"),
        _scan_row(id="gluonts:m4_hourly:h3", granularity="H"),
        _scan_row(id="gluonts:m1_business:b1", granularity="1B"),
        _scan_row(id="tslib:weather:w1", granularity="0.5h"),
    ])

    series_df, _ = build_catalog(scan_df)

    by_id = series_df.set_index("id")["granularity"]
    assert by_id["gluonts:m4_hourly:h1"] == "H"
    assert by_id["gluonts:m4_hourly:h2"] == "H"
    assert by_id["gluonts:m4_hourly:h3"] == "H"
    assert by_id["gluonts:m1_business:b1"] == "B"
    assert by_id["tslib:weather:w1"] == "30min"
    # seasonal_period lookup still resolves correctly post-normalization.
    assert series_df.set_index("id")["seasonal_period"]["tslib:weather:w1"] == 48


def test_license_is_assigned_by_source_and_collection():
    scan_df = pd.DataFrame([
        _scan_row(id="gluonts:m4_monthly:1", source="gluonts", collection="m4_monthly"),
        _scan_row(id="gluonts:nn5_daily:1", source="gluonts", collection="nn5_daily"),
        _scan_row(id="tslib:traffic:1", source="tslib", collection="traffic"),
    ])

    series_df, _ = build_catalog(scan_df)

    by_id = series_df.set_index("id")["license"]
    assert by_id["gluonts:m4_monthly:1"] == "unlicensed"
    assert by_id["gluonts:nn5_daily:1"] == "cc-by-4.0"
    assert by_id["tslib:traffic:1"] == "unknown"


def test_series_csv_is_sorted_by_id():
    scan_df = pd.DataFrame([
        _scan_row(id="gluonts:z_collection:9"),
        _scan_row(id="gluonts:a_collection:1"),
        _scan_row(id="gluonts:m_collection:5"),
    ])

    series_df, _ = build_catalog(scan_df)

    assert series_df["id"].tolist() == [
        "gluonts:a_collection:1", "gluonts:m_collection:5", "gluonts:z_collection:9",
    ]
