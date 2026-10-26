import pandas as pd

from scripts.scan import _score_tasks, build_row, iter_gluonts_series, iter_tslib_series


def _task(i):
    series = pd.Series(range(i, i + 50), dtype=float)
    return (f"gluonts:test:{i}", "gluonts", "test", str(i), "D", "", series)


def test_score_tasks_parallel_matches_serial():
    tasks = [_task(i) for i in range(4)]

    serial = [build_row(*t) for t in tasks]
    parallel = _score_tasks(tasks, workers=2)

    assert parallel == serial


def test_score_tasks_empty_returns_empty():
    assert _score_tasks([], workers=4) == []


def test_iter_gluonts_series_numbers_entries_with_null_item_id():
    # `entry.get("item_id")` can legitimately be 0 (falsy but valid) or
    # explicitly None (needs the positional fallback) -- both must be
    # distinguished from "missing key" via `is None`, not `or`.
    entries = [{"item_id": 0, "target": [1.0, 2.0]}, {"item_id": None, "target": [3.0, 4.0]}]

    rows = list(iter_gluonts_series("test", entries, "D"))

    assert [id_ for id_, *_ in rows] == ["gluonts:test:0", "gluonts:test:1"]
    assert rows[0][4].tolist() == [1.0, 2.0]


def test_iter_gluonts_series_skips_malformed_entry_without_aborting():
    entries = [{"item_id": 0, "target": [1.0]}, {"item_id": 1}, {"item_id": 2, "target": [2.0]}]

    rows = list(iter_gluonts_series("test", entries, "D"))

    assert [id_ for id_, *_ in rows] == ["gluonts:test:0", "gluonts:test:2"]


def test_iter_tslib_series_sorts_by_parsed_date_before_dropping_it():
    # Unpadded dates ("1990-1-10" vs "1990-1-2") sort wrong as raw strings --
    # must parse to datetime first, matching imbalance_eval.load_csv's order.
    df = pd.DataFrame({"date": ["1990-1-10", "1990-1-2"], "value": [10.0, 2.0]})

    rows = list(iter_tslib_series("test", df))

    assert len(rows) == 1
    id_, key, granularity, time_column, series = rows[0]
    assert id_ == "tslib:test:value"
    assert time_column == "date"
    assert series.tolist() == [2.0, 10.0]
    assert isinstance(series.index, pd.DatetimeIndex)


def test_iter_tslib_series_without_date_column_yields_range_index():
    df = pd.DataFrame({"value": [1.0, 2.0]})

    rows = list(iter_tslib_series("test", df))

    id_, key, granularity, time_column, series = rows[0]
    assert time_column == ""
    assert not isinstance(series.index, pd.DatetimeIndex)


def test_embedding_matches_paper_create_data_10():
    # create.data(ts, 10) = 9 lags + target, so a 50-value series leaves 41 rows, not 40.
    assert build_row("x", "gluonts", "t", "k", "D", "", pd.Series(range(50), dtype=float) ** 2)["N"] == 41


def test_all_rare_series_is_marked_degenerate_not_ok():
    # Zero-IQR series: R uba/UBL score every value rare. That is a collapsed phi, not a rare regime.
    y = pd.Series([0.0] * 60 + [1.0, 2.0, 0.0, 0.0, 5.0])
    row = build_row("x", "gluonts", "t", "k", "D", "", y)
    assert row["status"] == "degenerate_phi"
    assert row["n_normal"] == 0
