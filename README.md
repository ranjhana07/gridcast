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

Download the load data (about 250 MB zipped, see `data/README.md`) and build the modelling frame:

```
gridcast fetch-data
gridcast build-data
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
gridcast importance
gridcast dispatch-eval --model lightgbm --model lstm_quantile --reference naive_168
gridcast retrain --model lightgbm
gridcast forecast --model lightgbm
gridcast serve
gridcast schedule
```

- `benchmark` runs the rolling-origin backtest for the chosen models and writes metric tables, predictions and metadata to `results/`. `--target solar_kw` benchmarks the simulated PV series, `--tag` suffixes the files, `--online` recalibrates the intervals online, `--track` logs the summary to MLflow. `gridcast combine` merges tagged runs.
- `importance` fits LightGBM on the whole frame and writes gain importance by feature and by feature family.
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

## Results

Hardware: 13th Gen Intel(R) Core(TM) i5-13420H (12 logical cores), Windows-11-10.0.26200-SP0, Python 3.12.10, torch 2.15.0+cpu (CPU), LightGBM 4.7.0. The machine was shared with other jobs while the benchmark ran, so the timing columns are indicative only. Training used one thread per model (`GRIDCAST_THREADS=1`).

Data: 26,305 hourly rows from 2012-01-01 to 2015-01-01. Rolling-origin backtest with 3 expanding-window folds of 14 days each (42 test days), a forecast origin every 6 hours, horizon 24 hours, models refit at the start of each fold. Intervals target 80 percent coverage (P10 to P90). Weather at the target hour is the observed value, so the models get a perfect weather forecast.

Commands:

```
gridcast benchmark --folds 3 --fold-days 14 --stride 6
gridcast benchmark -m naive_24 -m lightgbm --target solar_kw --folds 3 --fold-days 14 --stride 6 --tag solar
gridcast dispatch-eval --model lightgbm --model lstm_quantile --model ets --reference naive_168
```

### Point forecast error, all horizons

Source: `results/metrics_overall.csv`. Units are kW for MAE and RMSE, percent for MAPE and sMAPE.

| model | mae | rmse | mape | smape |
| --- | --- | --- | --- | --- |
| naive_24 | 191.7 | 282.7 | 7.92 | 7.84 |
| naive_168 | 142.3 | 204.1 | 5.87 | 5.83 |
| ets | 220.3 | 337.6 | 9.67 | 9.37 |
| lightgbm | 90.4 | 130.3 | 3.74 | 3.69 |
| lstm_quantile | 109.5 | 152.4 | 4.45 | 4.44 |

### MAE by horizon (kW)

Source: `results/metrics_by_horizon.csv`, which also has RMSE, MAPE and sMAPE for every horizon from 1 to 24.

| model | h=1 | h=6 | h=12 | h=18 | h=24 |
| --- | --- | --- | --- | --- | --- |
| naive_24 | 199.1 | 157.4 | 159.7 | 159.3 | 158.7 |
| naive_168 | 144.9 | 125.0 | 124.3 | 124.6 | 125.5 |
| ets | 85.3 | 192.0 | 207.0 | 235.7 | 198.8 |
| lightgbm | 63.1 | 73.9 | 81.9 | 85.6 | 88.4 |
| lstm_quantile | 111.2 | 100.3 | 101.1 | 102.4 | 100.4 |

### Interval coverage and width

Nominal coverage is 80 percent. Source: `results/coverage_overall.csv`, per-horizon values in `results/coverage_by_horizon.csv`.

| model | coverage % | mean width kW |
| --- | --- | --- |
| naive_24 | 74.8 | 513 |
| naive_168 | 75.9 | 382 |
| ets | 71.8 | 536 |
| lightgbm | 72.2 | 223 |
| lstm_quantile | 74.4 | 293 |

### Training and inference time

`fit` includes the conformal calibration pass. For ETS the model is fitted at prediction time, which is why its inference cost is large. Source: `results/timing.csv`.

| model | fit s (3 folds) | predict s | origins | predict ms/origin |
| --- | --- | --- | --- | --- |
| naive_24 | 0.0 | 0.00 | 159 | 0.02 |
| naive_168 | 0.0 | 0.00 | 159 | 0.02 |
| ets | 292.3 | 207.24 | 159 | 1303.41 |
| lightgbm | 127.9 | 1.35 | 159 | 8.48 |
| lstm_quantile | 66.4 | 0.10 | 159 | 0.65 |

### Battery dispatch

Day-ahead dispatch replayed over the 42 backtest days. Battery: 353 kW, 1412 kWh, round-trip efficiency 0.9. `perfect_foresight` plans with the actual load and bounds what any forecast can achieve. `naive_168` is the naive-forecast dispatch. Costs are in the tariff currency units described above. Source: `results/dispatch_savings.csv`.

| strategy | total cost | peak kW | saving vs no battery | saving % | saving vs naive_168 |
| --- | --- | --- | --- | --- | --- |
| no_battery | 484,256 | 4,268 | 0 | 0.00 | -13,088 |
| perfect_foresight | 469,965 | 3,915 | 14,291 | 2.95 | 1,202 |
| naive_168 | 471,168 | 3,989 | 13,088 | 2.70 | 0 |
| lightgbm | 471,274 | 3,989 | 12,982 | 2.68 | -106 |
| lstm_quantile | 471,010 | 3,989 | 13,245 | 2.74 | 157 |
| ets | 471,052 | 3,989 | 13,203 | 2.73 | 115 |

### Online interval recalibration

`gridcast benchmark -m naive_24 -m naive_168 -m lightgbm -m lstm_quantile --online --tag online --folds 3 --fold-days 14 --stride 6` recalibrates the interval at every origin from the 400 most recent errors that have already resolved (calibration window plus earlier test origins). Point forecasts are identical to the static run. ETS was not rerun. Source: `results/*_online.csv`.

| model | static coverage % | online coverage % | static width kW | online width kW |
| --- | --- | --- | --- | --- |
| naive_24 | 74.8 | 75.2 | 513 | 520 |
| naive_168 | 75.9 | 76.2 | 382 | 387 |
| lightgbm | 72.2 | 72.8 | 223 | 227 |
| lstm_quantile | 74.4 | 75.0 | 293 | 296 |

### LightGBM feature importance

`gridcast importance`, gain averaged over the 24 horizon models fitted on the whole frame. Per-feature values are in `results/feature_importance.csv`.

| feature family | share of gain % |
| --- | --- |
| seasonal lags | 86.7 |
| recent lags | 8.2 |
| calendar at target hour | 3.3 |
| rolling statistics | 0.8 |
| weather at target hour | 0.6 |
| weather at origin | 0.2 |

### Solar (simulated PV)

The dataset has no PV measurements, so the target is a 500 kW array simulated from Open-Meteo irradiance. The forecasters see no target-hour weather for this series, only lags, calendar and sun elevation, because the target is a deterministic function of irradiance. Treat it as a smoke test of the solar code path, not as a PV forecasting result. Source: `results/*_solar.csv`.

| model | MAE kW | RMSE kW | coverage % | mean width kW |
| --- | --- | --- | --- | --- |
| lightgbm | 7.57 | 17.30 | 85.3 | 37.6 |
| naive_24 | 9.58 | 24.01 | 87.8 | 53.7 |

### Reading the results

- LightGBM has the lowest MAE at each of the 24 horizons. Its overall MAE is 36.5 percent below the weekly seasonal naive, and the LSTM sits between the two (23.0 percent below).
- ETS is worse than both naive baselines. It is fitted on 14 days of history with a single daily season, which probably explains why it misses the weekday and weekend pattern the naive_168 baseline copies. It is also the slowest model at prediction time.
- The 80 percent intervals under-cover on this data: observed coverage is between 71.7 and 75.9 percent. The calibration window is the 42 days before each fold. The likely cause is drift between that window and the test days, which breaks the exchangeability that split conformal assumes; this was not isolated by a separate experiment. The conformal step still narrows LightGBM's interval to about 223 kW wide, the tightest of all models. The online recalibration option moves coverage by less than one percentage point for every model (table above), so it does not fix the gap with the 400-error window used here.
- Seasonal lags carry most of LightGBM's gain (86.7 percent), recent lags add 8.2 percent and observed target-hour weather only 0.6 percent. The calendar and weather inputs matter little on this series.
- Dispatch savings are 2.7 percent against no battery for every forecast-driven strategy and 2.95 percent for perfect foresight. The cost differences between forecast models are small (a few hundred currency units against roughly 13,000 saved over 42 days), and they do not follow the MAE ranking: with this tariff and battery size, the dispatch plan is not sensitive to the forecast differences seen here. The peak after dispatch is 3,989 kW for all forecast-driven runs.
- Limits: one dataset, 42 test days, observed weather used as the weather forecast, an assumed tariff and an assumed battery. Treat the ranking of models as indicative for this series only.

## Project layout

```
gridcast/
  config.py            settings from environment
  data/                loaders, weather client, calendar, validation, PV simulation, frame builder
  features.py          direct multi-horizon feature builder
  models/              naive, ets, gbm, deep, calibrated wrapper
  conformal.py         split conformal and CQR calibrators
  backtest.py          rolling-origin runner
  evaluation.py        metric tables
  benchmark.py         benchmark runner and result files
  zoo.py               model factories
  training.py          candidate training and MLflow logging
  registry.py          registration and promotion
  flows.py             retrain and forecast flows
  scheduler.py         interval scheduler
  dispatch/            tariff, battery, LP optimizer, simulator, evaluation
  api/                 FastAPI app and schemas
  cli.py               Typer entry point
dashboard/             static dashboard
deploy/                nginx config for the compose stack
results/               benchmark output used in this README
tests/                 pytest suite
```

## Testing

```
pytest
```

The suite needs no network, API keys or services. Weather requests use a fake fetcher, MLflow runs against a temporary file store and the API tests use a fake artifact directory. It covers feature leakage, conformal coverage on synthetic data, the optimizer on small hand-checked cases, the backtest folds, the registry and the API routes.

Docker was not available while building this project, so `Dockerfile` and `docker-compose.yml` are untested. `docker compose up --build` starts Postgres (TimescaleDB image), an MLflow server, the API, the scheduler and an nginx container for the dashboard on port 8080.
