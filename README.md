# GridCast

Load and solar forecasting for commercial buildings and microgrids, with a battery dispatch optimizer that uses the forecasts. The repository covers the whole path: data loaders, validation, feature building, five forecasting models, conformal prediction intervals, rolling-origin backtests, MLflow tracking and model promotion, a scheduler for daily retrains and hourly forecasts, a FastAPI service and a static dashboard.

Everything here was run on the UCI ElectricityLoadDiagrams20112014 data (30 meters summed into one microgrid, hourly) with Open-Meteo weather for Lisbon. The numbers in [Results](#results) come from the scripts in this repository and are stored in `results/`.

## Architecture

```
  UCI load data        Open-Meteo archive        holiday calendar
        |                     |                         |
        v                     v                         v
  +---------------------------------------------------------------+
  |  data/  loaders, validation, gap filling, modelling frame     |
  +-------------------------------+-------------------------------+
                                  |
                                  v
  +---------------------------------------------------------------+
  |  features.py  lags, rolling stats, seasonal lags, calendar,   |
  |               weather at the target hour (direct strategy)    |
  +-------------------------------+-------------------------------+
                                  |
        +-------------+-----------+-----------+--------------+
        |             |                       |              |
   seasonal naive    ETS                  LightGBM       LSTM encoder-
   (24h, 168h)    (Holt-Winters)        24 direct       decoder, P10/
                                         models         P50/P90 loss
        |             |                       |              |
        +-------------+-----------+-----------+--------------+
                                  |
                                  v
  +---------------------------------------------------------------+
  |  models/calibrated.py  split conformal / conformalized        |
  |                        quantile regression per horizon        |
  +----------------+----------------------------+-----------------+
                   |                            |
                   v                            v
        backtest.py + benchmark.py        training.py + registry.py
        rolling-origin folds              MLflow runs, model versions,
        results/*.csv                     "production" alias promotion
                   |                            |
                   |                            v
                   |                     flows.py + scheduler.py
                   |                     daily retrain, hourly forecast
                   |                            |
                   v                            v
        dispatch/  LP battery dispatch  <-- forecasts
        TOU tariff + peak demand charge, savings vs baselines
                   |
                   v
        api/ (FastAPI)  <---------  artifacts/ and results/
                   |
                   v
        dashboard/ (HTML, CSS, JS, ApexCharts)
```

## Setup

Python 3.11 or newer.

```
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[models,deep,serve,dev]"
```

Fetch the load data (see `data/README.md`) and build the modelling frame:

```
curl -L -o eld.zip https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip eld.zip -d data/raw
gridcast build-data --raw data/raw/LD2011_2014.txt
```

The first run downloads three years of hourly weather from Open-Meteo and caches it in `data/cache`.

## Configuration

Settings are read from environment variables, see `.env.example`.

| variable | default | meaning |
| --- | --- | --- |
| `GRIDCAST_DATA_DIR` | `data` | raw, cache and processed data |
| `GRIDCAST_ARTIFACT_DIR` | `artifacts` | forecast and dispatch JSON written by the flows |
| `GRIDCAST_RESULTS_DIR` | `results` | benchmark output read by the API |
| `GRIDCAST_LATITUDE`, `GRIDCAST_LONGITUDE` | Lisbon | weather location |
| `GRIDCAST_COUNTRY` | `PT` | holiday calendar |
| `GRIDCAST_HORIZON` | `24` | forecast horizon in hours |
| `GRIDCAST_THREADS` | `2` | threads for LightGBM and PyTorch |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | tracking store, a local file store by default |
| `OPEN_METEO_URL` | archive endpoint | override for tests or a mirror |

## Usage

```
gridcast benchmark --folds 3 --fold-days 14 --stride 6
gridcast dispatch-eval --model lightgbm --model lstm_quantile --reference naive_168
gridcast retrain --model lightgbm
gridcast forecast --model lightgbm
gridcast serve
gridcast schedule
```

- `benchmark` runs the rolling-origin backtest for the chosen models and writes metric tables, predictions and metadata to `results/`. `--target solar_kw` benchmarks the simulated PV series, `--tag` suffixes the files. `gridcast combine` merges tagged runs.
- `dispatch-eval` replays day-ahead battery dispatch for the backtest test period and writes `results/dispatch_savings.csv`.
- `retrain` trains a candidate on everything except the last 14 days, logs it to MLflow, registers a new version and promotes it to the `production` alias when its holdout MAE is within 2 percent of the current production version (or better).
- `forecast` loads the production version and writes the next 24 hours with intervals, plus a dispatch schedule, to `artifacts/`.
- `schedule` runs both flows from a plain interval loop: retrain every day, forecast every hour.
- `serve` starts the API and mounts the dashboard at `/`.

`make` targets wrap the same commands.

### Models

| name | description |
| --- | --- |
| `naive_24`, `naive_168` | value from the same hour one day or one week earlier |
| `ets` | additive Holt-Winters with damped trend, daily season, fitted on the last 14 days at every origin |
| `lightgbm` | one gradient boosted model per horizon step, trained on lags, rolling statistics, seasonal lags, calendar and weather features |
| `lstm_quantile` | LSTM encoder-decoder (hidden size 32, 48 hour lookback) with known-future covariates and a pinball loss for P10, P50 and P90 |

Every model is wrapped in `ConformalForecaster`. It holds out the last 42 days of the training window, fits the model on the rest, and calibrates the interval half-width per horizon from the held-out errors (split conformal). For the LSTM it calibrates its own P10 and P90 outputs instead (conformalized quantile regression). The nominal interval is 80 percent.

Features never use values after the forecast origin. The one exception is weather and calendar at the target hour, which stand in for a weather forecast. `tests/test_features.py` corrupts everything after an origin and checks the features do not move.

### Dispatch

`gridcast.dispatch.optimizer.plan_dispatch` solves a linear program with SciPy HiGHS for each 24 hour window: minimise time-of-use energy cost plus a demand charge on the peak grid draw, subject to power and state of charge limits, round-trip efficiency, no export, and ending the day at the starting state of charge. The simulator plans each day from a forecast, executes the plan against realised load (clipping when the battery cannot follow) and tracks the running monthly peak so the demand charge is only paid on new peaks.

Tariff used in the evaluation: 0.10 / 0.18 / 0.32 per kWh (off-peak, shoulder, weekday 17:00 to 21:00 peak) and 14 per kW-month demand charge, prorated to the days covered. The battery is sized at 8 percent of peak load in power and 4 hours of energy. These are assumptions, not market data.

## API

Base URL `http://localhost:8000`. Interactive docs at `/docs`.

| route | returns |
| --- | --- |
| `GET /health` | status and whether forecast and backtest data exist |
| `GET /api/forecast?target=load_kw` | latest 24 hour forecast with interval bounds and recent history |
| `GET /api/intervals?target=load_kw` | interval bounds, widths and nominal coverage |
| `GET /api/history?target=&hours=` | recent observations |
| `GET /api/backtest/summary` | overall error, coverage and timing per model plus run metadata |
| `GET /api/backtest/horizon?model=` | MAE, RMSE, MAPE, sMAPE and coverage by horizon |
| `GET /api/backtest/origins?model=` | forecast origins available for replay |
| `GET /api/backtest/series?model=&origin=` | actual, forecast and interval for one origin |
| `GET /api/dispatch` | latest battery schedule |
| `GET /api/dispatch/savings` | dispatch cost comparison table |
| `GET /api/models` | last retrain report |
