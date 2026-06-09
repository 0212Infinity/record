# Guangzhou Weather AI

[English](./README.md) | [简体中文](./README-zh.md)

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-f7931e)
![Status](https://img.shields.io/badge/status-local%20demo-2f7d32)

A local AI weather forecasting demo for **Guangzhou**, built from historical `QWeather` 7-day forecast snapshots.

This project trains a lightweight machine learning pipeline to **correct raw forecasts** and exposes the results through a small Flask web app. It predicts:

- daily maximum temperature
- daily minimum temperature
- daily precipitation
- daytime weather type

The current version is designed as a **forecast correction model**, not a replacement for professional numerical weather prediction systems.

## Features

- Uses only the local historical snapshot dataset in `data/`
- Cleans invalid or empty weather files automatically
- Aligns each forecast row with the corresponding `fxDate -> daily[0]` target
- Trains regression models for temperature and precipitation
- Trains a classification model for daytime weather text
- Serves a local dashboard with:
  - 7-day AI predictions
  - raw forecast vs AI forecast comparison
  - historical backtest metrics
  - data quality summary
- Exposes JSON APIs for integration or inspection

## Project Structure

```text
.
├── app.py                  # Flask entrypoint
├── requirements.txt
├── data/                   # Historical QWeather forecast snapshots
├── models/                 # Trained model artifact (generated locally)
├── src/
│   ├── data_pipeline.py    # Data loading, cleaning, feature building
│   ├── train.py            # Model training and artifact export
│   └── predict.py          # Dashboard payload generation
├── static/
│   ├── app.js              # Frontend logic
│   └── styles.css          # Dashboard styles
└── templates/
    └── index.html          # Flask HTML page
```

## Dataset

The dataset contains Guangzhou weather forecast snapshots saved as JSON files:

- source format: `data/YYYY/MM/YYYYMMDD.json`
- city: Guangzhou
- valid forecast horizon: 7 days per snapshot
- effective range in current dataset: `2023-06-22` to `2026-05-04`
- valid snapshots: `532`
- invalid or empty snapshots skipped during training: `25`
- aligned training rows produced: `3115`

Each valid file contains daily forecast fields such as:

- `tempMax`
- `tempMin`
- `precip`
- `humidity`
- `pressure`
- `cloud`
- `uvIndex`
- `textDay`
- `iconDay`

The original field notes are kept in [data.md](/D:/ai/record/data.md).

## Modeling Approach

Each snapshot contains a 7-day forecast. For every row:

- input features come from `daily[lead]`
- the label comes from the snapshot whose filename matches that row's `fxDate`
- the label value is taken from that matched file's `daily[0]`

This makes the project a practical **historical forecast correction pipeline**:

- regressors predict `tempMax`, `tempMin`, and `precip`
- a classifier predicts `textDay`
- categorical forecast fields are one-hot encoded
- train/test split is time-based to avoid future leakage

## Backtest Metrics

Current locally verified metrics:

| Metric | Value |
| --- | ---: |
| `tempMax_mae` | `1.472` |
| `tempMax_rmse` | `2.067` |
| `tempMin_mae` | `1.112` |
| `tempMin_rmse` | `1.508` |
| `precip_mae` | `0.889` |
| `precip_rmse` | `2.360` |
| `textDay_accuracy` | `0.553` |

These numbers are from the current time-based split in the local dataset and should be treated as a baseline, not a production benchmark.

## Requirements

- Windows or any environment that can run Python 3.14+
- Python environment with:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `flask`
  - `joblib`
  - `matplotlib`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you are using the local conda environment already configured on this machine:

```bash
python.exe -m pip install -r requirements.txt
```

## Quick Start

Train the model:

```bash
python -m src.train
```

Or with the configured interpreter:

```bash
python.exe -m src.train
```

Start the web app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:3000
```

`app.py` will automatically pick the next available port if `3000` is already occupied.

## API

### `GET /`

Returns the local dashboard page.

### `GET /api/summary`

Returns dataset and model summary, including:

- date range
- latest snapshot date
- valid and invalid snapshot counts
- aligned training row count

### `GET /api/predictions`

Returns the latest 7-day forecast, including:

- raw forecast values
- AI-corrected values
- predicted daytime weather type

### `GET /api/backtest`

Returns:

- backtest metrics
- per-row evaluation series for the held-out split

### `GET /api/quality`

Returns:

- total file count
- valid and invalid snapshot counts
- examples of invalid files
- unresolved target count

### `GET /api/refresh`

Reloads the cached model payload in the Flask app.

## Development Notes

- The project currently supports **Guangzhou only**.
- Empty files and malformed snapshots are skipped by design.
- File decoding falls back across multiple encodings because some files are not clean UTF-8.
- The target `daily[0]` is used as an approximate observed outcome, which introduces some label noise.
- No external weather API is called in this version.

## Limitations

- This is not an operational forecasting system.
- Labels are derived from forecast snapshots, not official ground-truth observation data.
- The dataset is limited to one city and a moderate number of historical snapshots.
- Weather category accuracy is meaningfully lower than temperature regression quality.

## Roadmap

- Add support for observed weather labels if station data becomes available
- Add multi-city training support
- Improve weather text classification by merging long-tail categories
- Add richer charts and confidence intervals
- Export prediction reports automatically

## License

This repository currently does not define a project license.

Please also review the weather data source and attribution terms referenced in the dataset itself.
