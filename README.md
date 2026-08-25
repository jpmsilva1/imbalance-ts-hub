# Imbalanced Time-Series Hub

[![Code: MIT](https://img.shields.io/github/license/jpmsilva1/imbalance-ts-hub?label=code)](LICENSE) 
[![Ingest](https://github.com/jpmsilva1/imbalance-ts-hub/actions/workflows/ingest.yml/badge.svg)](https://github.com/jpmsilva1/imbalance-ts-hub/actions/workflows/ingest.yml)

A curated, versioned catalog of **imbalanced time series**. Every series in it has been scored with [`imbalance_eval`](https://github.com/jpmsilva1/imbalance_eval) (the relevance-function methodology from Moniz, Branco & Torgo 2017) and kept only if it actually has a rare regime worth studying.

Instead of hunting down datasets and hand-checking whether they're actually imbalanced, browse the catalog's metadata, filter with pandas, and pull only the series you want:

```python
from imbalance_hub import load_catalog, pull

catalog = load_catalog()
severe_hourly = catalog[(catalog.imbalance_level == "severe") & (catalog.granularity == "H")]

series = pull(severe_hourly.id.iloc[0])   # -> pd.Series, values ready to use
```

## Contents

- [License](#license)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Catalog schema](#catalog-schema)
- [Client API](#client-api)
- [How the catalog is built](#how-the-catalog-is-built)
- [Development](#development)

## License

The code in this repo (`imbalance_hub/`, `scripts/`) is [MIT](LICENSE). That does **not**
cover the catalog data — the series values redistributed via the Hugging Face mirror come
from third-party sources with their own, differing terms. See
[`DATA_LICENSES.md`](DATA_LICENSES.md) for the scope split and `dataset_card.md`'s
"Licensing" section for the full per-collection breakdown.

## Quick start

```bash
pip install git+https://github.com/jpmsilva1/imbalance-ts-hub.git
```

*(Not yet published to PyPI — installing from the git repo works today; `pip install imbalance-hub` will once it is.)*

```python
from imbalance_hub import load_catalog, pull, pull_many

# The whole catalog as a DataFrame -- one row per accepted series.
catalog = load_catalog()

# Filter with plain pandas. No bespoke query language.
candidates = catalog[
    (catalog.imbalance_level.isin(["severe", "extreme"]))
    & (catalog.source == "gluonts")
    & (catalog.length > 1000)
].sort_values("%Rare")

print(candidates[["id", "collection", "N", "n_rare", "%Rare", "imbalance_level"]].head())

# Pull just the series you actually want.
series = pull(candidates.id.iloc[0])          # -> pd.Series
batch = pull_many(candidates.id.head(20))     # -> dict[id, pd.Series]
# `gluonts` series come back with an integer step index; only `tslib` series
# carry real timestamps (`pull` returns a `DatetimeIndex` when the source had one).

# Long severe/extreme series from a specific collection.
m4_candidates = catalog[
    (catalog.collection == "m4_monthly")
    & (catalog.imbalance_level.isin(["severe", "extreme"]))
    & (catalog.length >= 200)
]

# Hourly-granularity series with no missing values and a strong rare regime.
hourly_clean = catalog[
    (catalog.granularity == "H")
    & (catalog.missing_pct == 0)
    & (catalog["%Rare"] < 5)
]
```

See [Filterable values](#filterable-values) below for the full domain of every column above.

`load_catalog()` caches `catalog/series.csv` locally (`~/.cache/imbalance_hub/`). Pinned versions (`load_catalog(version="<tag-or-sha>")`) are served straight from that cache once fetched — they can't change. The default `version="latest"` always re-checks with the server, but via a conditional request: if the catalog hasn't changed since your last call, the ~24MB body isn't re-downloaded, only a small "not modified" response is. `pull()`/`pull_many()` only download the Parquet blob for the specific series you ask for — the catalog is metadata-only, so browsing 59k+ series doesn't mean downloading 59k+ series.

## How it works

```
GluonTS + TSLib (raw sources)
        │  scored with imbalance_eval.compute_imbalance()
        ▼
catalog/scanned.csv    -- every series ever scanned, incl. rejects (audit trail)
catalog/series.csv     -- accepted series only (n_rare > 0), full metadata
        │
        ▼
Hugging Face dataset repo -- one Parquet blob per accepted series
        │
        ▼
imbalance_hub client  -- load_catalog() / pull() / pull_many()
```

**The inclusion gate is `n_rare > 0`.** A series with no boxplot outliers at the paper's `coef=1.5` has no rare regime by construction — it's definitionally not imbalanced, so it's excluded. Every band of severity above that line is kept; there's no further curation beyond the gate.

**Severity is computed, not hand-labeled**, from `%Rare` (note the inversion — *lower* `%Rare` means *more* imbalanced, since `IR = n_rare / n_normal`, so *lower* `IR` also means *more* imbalanced):

| `imbalance_level` | `%Rare` | approx. `IR` |
|---|---|---|
| `extreme` | `< 1` | `< 0.01` |
| `severe` | `1 – 5` | `0.01 – 0.05` |
| `moderate` | `5 – 15` | `0.05 – 0.18` |
| `mild` | `≥ 15` | `≥ 0.18` |

Scoring params are fixed to match the paper across the whole catalog: `rel_thres=0.9`, `rel_coef=1.5`, `rel_xtrm_type="both"`, time-delay embedding with `k=10`, scored on **raw series** (not differenced — see `diff` column). They're also recorded per row, so a future change to these defaults is traceable without re-deriving anything.

## Catalog schema

`catalog/series.csv` — one row per accepted series, sorted by `id`.

| Group | Columns |
|---|---|
| **Identity** | `id` (`source:collection:key`, e.g. `gluonts:m4_hourly:h1`), `name`, `source` (`gluonts`/`tslib`), `collection`, `license` (`cc-by-4.0`/`unlicensed`/`unknown` — see [Licensing](#license)) |
| **Structure** | `granularity`, `time_column`, `length` (raw, pre-embed), `dtype`, `size_bytes` |
| **Imbalance** | `N`, `n_normal`, `n_rare`, `IR`, `%Rare`, `imbalance_level`, plus the exact scoring params used: `rel_thres`, `rel_coef`, `rel_xtrm_type`, `k`, `embed`, `diff` |
| **Characteristics** | `missing_pct`, `mean`, `std`, `cv`, `skewness`, `kurtosis`, `autocorr_lag1`, `seasonal_period` (granularity-keyed lookup, e.g. `24` for hourly — not a fitted decomposition) |
| **Provenance** | `content_hash` (`sha256(values)[:16]`, for integrity checks), `blob_path`, `hf_revision`, `pipeline_version`, `ingested_at` |

### Filterable values

What you can actually pass to `.isin([...])` / comparisons in the Quick start example above, straight from `catalog/series.csv` as of the current catalog (59,489 rows). Kept in sync with the real data by `scripts/check_catalog_docs.py`, enforced in CI.

**`source`**

| Value | Count |
|---|---|
| `gluonts` | 58,354 |
| `tslib` | 1,135 |

**`license`** — see [License](#license) for what each value permits.

| Value | Count |
|---|---|
| `cc-by-4.0` | 28,868 |
| `unlicensed` | 29,486 |
| `unknown` | 1,135 |

**`imbalance_level`** — see the severity table in [How it works](#how-it-works) for how this is derived from `%Rare`.

| Value | Count |
|---|---|
| `moderate` | 32,867 |
| `severe` | 13,256 |
| `mild` | 12,194 |
| `extreme` | 1,172 |

**`dtype`**: `float64` (59,484), `int64` (5)

**`time_column`**: `NaN` for every `gluonts` row, `"date"` for every `tslib` row — effectively a `source`-keyed flag, not an independent axis to filter on.

**`granularity`** — normalized to one canonical spelling per frequency (raw source data used inconsistent notation for the same interval — e.g. `H`/`h`/`1H` all meant hourly, `1B` and `B` both meant business-day, `30min`/`0.5h` both meant a 30-minute interval; `scripts/build_catalog.py`'s `GRANULARITY_ALIASES` collapses these before publishing). Sorted below by actual duration, shortest first:

| Value | Duration | Count |
|---|---|---|
| `min` | 1 minute | 201 |
| `10min` | 10 minutes | 125 |
| `15min` | 15 minutes | 24 |
| `30min` | 30 minutes | 6,742 |
| `H` | 1 hour | 5,051 |
| `D` | 1 day | 15,606 |
| `B` | 1 business day | 10 |
| `W` | 1 week | 528 |
| `W-TUE` | 1 week, anchored Tuesday | 7 |
| `M` | 1 month | 20,129 |
| `Q` | 1 quarter | 6,529 |
| `Y` | 1 year | 4,517 |
| *(missing)* | — | 20 |

**`collection`** — 55 distinct values. The largest:

| Value | Count |
|---|---|
| `m4_monthly` | 16,998 |
| `wiki-rolling_nips` | 9,359 |
| `m4_quarterly` | 6,294 |
| `london_smart_meters_without_missing` | 5,556 |
| `m4_yearly` | 4,406 |
| `car_parts_without_missing` | 2,195 |
| `weather` | 2,142 |
| `wiki2000_nips` | 1,972 |

...down to single-series collections like `sunspot_without_missing`. Run `catalog.collection.value_counts()` for the full list of 55.

**Numeric columns** (min / 25% / median / 75% / max):

| Column | Min | 25% | Median | 75% | Max |
|---|---|---|---|---|---|
| `N` | 2 | 72 | 320 | 812 | 526,970 |
| `length` | 12 | 82 | 330 | 822 | 526,980 |
| `IR` | 0 | 0.05 | 0.10 | 0.16 | ∞ (all-rare series, `n_normal=0`) |
| `%Rare` | 0.01 | 5.08 | 8.70 | 13.54 | 100.00 |
| `missing_pct` | 0 | 0 | 0 | 0 | 3.95 |
| `seasonal_period` | 1 | 7 | 12 | 12 | 1,440 (20 null) |

`missing_pct` is a 0–100 percentage (percent of raw values that are `NaN`), not a 0–1 fraction — most series have none, hence the flat 0s through the 75th percentile.

`mean`, `std`, `cv`, `skewness`, `kurtosis`, `autocorr_lag1` are also present but heavy-tailed and not usefully summarized as a single range here — inspect them directly, e.g. `catalog["kurtosis"].describe()`.

`catalog/scanned.csv` holds every series ever scanned — accepted or not — with a `verdict` column (`accepted`/`rejected`). It's the audit trail proving *why* a series is absent from the catalog, and it doubles as the incremental-scan cache for future runs.

## Client API

```python
from imbalance_hub import load_catalog, pull, pull_many
```

- **`load_catalog(version="latest", refresh=False)`** → `pd.DataFrame`. One HTTP GET, cached at `~/.cache/imbalance_hub/{version}/series.csv`. Pass `refresh=True` to bypass the cache and re-fetch.
- **`pull(id)`** → `pd.Series`. Downloads the series' Parquet blob (cached via `huggingface_hub`'s own cache), verifies it against the catalog's `content_hash`, and returns it — `DatetimeIndex` if the source series had timestamps, integer index otherwise. Raises `KeyError` for an unknown id, `ValueError` if the blob doesn't exist yet or fails the hash check.
- **`pull_many(ids, errors="raise")`** → `dict[str, pd.Series]`. Same as `pull`, batched (concurrent, deduped, insertion-order preserved). `errors="raise"` (default) aborts on the first bad id, like `pull`; `errors="skip"` drops failed ids and returns only the series that succeeded.

There's deliberately no `search()`/query-DSL wrapper — the catalog is a plain DataFrame, and pandas is the query layer. You already know how to filter a DataFrame.

## How the catalog is built

The pipeline behind `catalog/series.csv` — useful if you want to reproduce it, extend the source list, or just understand where the numbers come from.

```bash
pip install -r requirements.txt
```

[`imbalance_eval`](https://github.com/jpmsilva1/imbalance_eval) isn't on PyPI yet, so it's
not in `requirements.txt`. Either `pip install` it from a local checkout, or point
`scripts/scan.py` at one directly:

```bash
export IMBALANCE_EVAL_PATH=~/imbalance_eval
```

**1. Scan** — pull every candidate series from GluonTS and TSLib, score each one, write a full audit row (including rejects/failures) to a flat CSV:

```bash
python scripts/scan.py --out scan_results.csv
```

Useful flags:

| Flag | What it does |
|---|---|
| `--source {gluonts,tslib,both}` | restrict to one source (default `both`) |
| `--collections a,b,c` | restrict to specific collections |
| `--limit N` | cap new rows added per collection (`0` = no limit) |
| `--resume` | skip ids already present in `--out`, so an interrupted run picks up where it left off |
| `--workers N` | parallel worker processes for scoring (default: all CPU cores) |

The script flushes results after each collection completes, and a stalled network call times out (60s) into a logged `error` row rather than hanging the whole run — so a crash or a bad connection never loses more than the collection in progress.

**What gets scanned:** all GluonTS collections in `dataset_names` except a short exclusion list of near-duplicate giants (`kaggle_web_traffic` ×3, `dominick`, `temperature_rain` — each tens to hundreds of thousands of similar short series that would swamp the catalog without adding diversity) plus `m5` (needs a manual Kaggle download) and `constant` (synthetic). From TSLib, only the forecasting-relevant folders (`ETT-small`, `electricity`, `exchange_rate`, `illness`, `traffic`, `weather`) — its anomaly-detection and UEA classification folders are out of scope. Around 150k series total, dominated by M4 (~100k). Expect anywhere from tens of minutes to several hours depending on flags, mostly download/compute time, plus 1-2 GB cached under `~/.gluonts` and `~/.cache/huggingface`.

**2. Build the catalog** — turn the raw scan into the two published CSVs:

```bash
python scripts/build_catalog.py --scan scan_results.csv --out-dir catalog
```

**3. Mirror accepted series to Hugging Face** as per-series Parquet blobs (one file per accepted `id`), so the catalog CSVs stay small and git-diffable while the actual data lives somewhere built for it. Re-fetches each collection from source (values aren't stored in `scan_results.csv`), stages a Parquet per series under `--stage-dir`, then pushes the whole staged tree in one commit and fills `blob_path`/`size_bytes`/`hf_revision`/`pipeline_version`/`ingested_at` back into the catalog. Run as a module, not a script directly, since it imports across the package boundary:

```bash
python -m scripts.upload_blobs --catalog catalog/series.csv --stage-dir /tmp/blobs
```

`--stage-only` stages without uploading (useful to sanity-check a collection first); `--collections a,b,c` restricts to a subset; already-staged files and already-committed blobs are both skipped on re-run, so a failed run (including through the documented flaky CloudFront route to `huggingface.co`) is fixed by running the same command again.

**CI**: `.github/workflows/ingest.yml` runs steps 1–2 on `workflow_dispatch` and opens a PR with the catalog diff — nothing runs on a schedule, since these sources don't change often enough to justify polling. Step 3 is not wired into CI; it's a local, manual step (no HF credentials are stored in this repo).

## Citation

```bibtex
@software{silva2026imbalancehub,
  author  = {Silva, João P. M.},
  title   = {imbalance-hub},
  year    = {2026},
  url     = {https://github.com/jpmsilva1/imbalance-ts-hub}
}
```

Machine-readable version in [`CITATION.cff`](CITATION.cff). No DOI yet — a Zenodo deposit
is planned but not done.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

House style: plain `pytest` functions and `assert`, no fixtures/classes beyond what a test actually needs. Tests exercise pure functions at their public seam — network/filesystem boundaries (`hf_hub_download`, HTTP fetches) are injectable rather than mocked internally, so nothing needs real credentials to run.
