---
pretty_name: imbalance-hub
license: other
license_name: mixed-per-collection
license_link: https://github.com/jpmsilva1/imbalance-ts-hub/blob/main/dataset_card.md#licensing
task_categories:
  - time-series-forecasting
tags:
  - time-series
  - imbalanced-regression
  - imbalanced-learning
  - forecasting
size_categories:
  - 10K<n<100K
---

# imbalance-hub

A curated, versioned catalog of **imbalanced time series** — "OpenML for imbalanced time
series." Every series here has been scored with
[`imbalance_eval`](https://github.com/jpmsilva1/imbalance_eval) (the relevance-function
methodology from Moniz, Branco & Torgo, 2017) and kept only if it actually has a rare
regime worth studying.

The catalog metadata and client library live in the companion GitHub repo,
**[jpmsilva1/imbalance-ts-hub](https://github.com/jpmsilva1/imbalance-ts-hub)** — start there for
the schema, the scoring methodology, and how the catalog is built. This dataset repo holds
the actual data: one Parquet file per accepted series.

## Quick start

```bash
pip install git+https://github.com/jpmsilva1/imbalance-ts-hub.git
```

```python
from imbalance_hub import load_catalog, pull

catalog = load_catalog()
severe_hourly = catalog[(catalog.imbalance_level == "severe") & (catalog.granularity == "H")]

series = pull(severe_hourly.id.iloc[0])   # -> pd.Series, values ready to use

# Filter by collection + severity + length before pulling anything.
candidates = catalog[
    (catalog.collection == "m4_monthly")
    & (catalog.imbalance_level.isin(["severe", "extreme"]))
    & (catalog.length >= 200)
]
```

`load_catalog()` fetches the metadata CSV from GitHub (not from this repo) and caches it
locally, re-checking with a conditional request on the default `version="latest"` so an
unchanged catalog isn't re-downloaded; `pull()` downloads only the one Parquet blob you
asked for from here, verified against a `content_hash` recorded in the catalog. Browsing
the catalog's 59k+ rows never downloads more than the metadata.

## Dataset structure

One Parquet file per accepted series, at `{source}/{collection}/{shard}/{key}.parquet`
(`shard` is a 2-hex-char directory keeping every folder under Hugging Face's per-directory
file limit — an implementation detail, not part of the catalog id). Each file has:

| Column | Type | Notes |
|---|---|---|
| `value` | float64 | Raw values, in original order. NaNs preserved where the source series has missing points. |
| `timestamp` | datetime64 | Present only when the source series carries real timestamps (TSLib); absent for GluonTS series, which are index-only. |

The catalog id (`source:collection:key`, e.g. `gluonts:m4_hourly:h1`) is what maps an id to
its blob path — see `blob_path_for()` in
[`scripts/upload_blobs.py`](https://github.com/jpmsilva1/imbalance-ts-hub/blob/main/scripts/upload_blobs.py)
in the GitHub repo, or just use the catalog CSV's `blob_path` column directly.

## Filterable columns

The columns you'll actually filter `load_catalog()`'s DataFrame on. Full column list/types
and the complete numeric-distribution table live in the
[GitHub README's Catalog schema section](https://github.com/jpmsilva1/imbalance-ts-hub#catalog-schema).

| Column | Values |
|---|---|
| `source` | `gluonts` (58,354), `tslib` (1,135) |
| `license` | `cc-by-4.0` (28,868), `unlicensed` (29,486), `unknown` (1,135) — see [Licensing](#licensing) |
| `imbalance_level` | `moderate` (32,867), `severe` (13,256), `mild` (12,194), `extreme` (1,172) |
| `granularity` | 12 canonical values (normalized from source-inconsistent notation), from `min` up to `Y` — see the GitHub README's [Filterable values](https://github.com/jpmsilva1/imbalance-ts-hub#catalog-schema) for the full duration-ordered table |
| `collection` | 55 distinct, from `m4_monthly` (16,998) down to single-series collections |

Numeric ranges (min / median / max) for the columns most people filter on:

| Column | Min | Median | Max |
|---|---|---|---|
| `length` | 12 | 330 | 526,980 |
| `%Rare` | 0.01 | 8.70 | 100.00 |
| `IR` | 0 | 0.10 | ∞ (all-rare series, `n_normal=0`) |

## Dataset summary

- **59,489 accepted series** out of 136,673 scanned (~43.5%) — the rest were excluded by
  the inclusion gate (no rare regime) or failed to embed (too short).
- **55 source collections**, from [GluonTS](https://github.com/awslabs/gluonts)
  (58,354 series — M4, M3, electricity, traffic, tourism, and more) and
  [TSLib](https://huggingface.co/datasets/thuml/Time-Series-Library) (1,135 series — ETT,
  weather, exchange rate, national illness).
- **Severity distribution**: 1,172 `extreme`, 13,256 `severe`, 32,867 `moderate`,
  12,194 `mild` (see the GitHub README for how `imbalance_level` is computed from `%Rare`).

## Licensing

**No single license applies — this catalog spans sources with different terms, checked
against the original loaders rather than assumed.** The curation layer (catalog metadata,
imbalance-scoring methodology, pipeline code) is separate from the underlying values: the
code is [MIT-licensed](https://github.com/jpmsilva1/imbalance-ts-hub/blob/main/LICENSE), which
does **not** extend to the data — see
[`DATA_LICENSES.md`](https://github.com/jpmsilva1/imbalance-ts-hub/blob/main/DATA_LICENSES.md)
for that scope split. For the raw values themselves:

- **Most GluonTS collections** (`electricity`, `traffic`, `tourism_*`, `nn5_*`, `weather`,
  `wind_farms_*`, `m1_*`, and most others fetched via GluonTS's
  [`_tsf_datasets.py`](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/dataset/repository/_tsf_datasets.py))
  come from the [Monash Time Series Forecasting Archive](https://forecastingdata.org/) on
  Zenodo, which licenses its datasets under
  **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** — verified directly against
  the [NN5 Daily record](https://zenodo.org/record/4656110).
- **M4** (`m4_*` — GluonTS's largest single contributor by series count) is fetched from
  [`M4Competition/M4-methods`](https://github.com/M4Competition/M4-methods) on GitHub.
  **That repository publishes no license** (confirmed via the GitHub API): no LICENSE
  file, no license declared anywhere. Under default copyright, that means no redistribution
  rights have been granted for this data, by imbalance-hub or by anyone else who mirrors
  it. It's rehosted here in line with common practice in the forecasting field, not because
  the rights question is resolved. If the M4 organizers ever ask, these series will be
  removed from the Hugging Face mirror on request.
- **TSLib** (`ETT-small`, `electricity`, `exchange_rate`, `illness`, `traffic`, `weather`)
  — the [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) repo
  itself is **MIT-licensed**, but that covers the code; the bundled benchmark CSVs'
  original licenses aren't stated in that repo.

If you plan to use a specific series commercially or redistribute it standalone, verify
that series' actual source collection rather than relying on this summary.

## Citation

To cite the catalog itself:

```bibtex
@software{silva2026imbalancehub,
  author  = {Silva, João P. M.},
  title   = {imbalance-hub},
  year    = {2026},
  url     = {https://github.com/jpmsilva1/imbalance-ts-hub}
}
```

If you use this catalog's imbalance scoring, also cite the methodology it implements:

```bibtex
@article{moniz2017resampling,
  title={Resampling strategies for imbalanced time series forecasting},
  author={Moniz, Nuno and Branco, Paula and Torgo, Lu{\'\i}s},
  journal={International Journal of Data Science and Analytics},
  volume={3},
  pages={161--181},
  year={2017},
  publisher={Springer}
}
```
