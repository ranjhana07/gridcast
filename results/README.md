# Results

Raw output of the benchmark scripts. The headline numbers in the top-level README are read from these files.

| file | content |
| --- | --- |
| `meta.json` | backtest configuration, data range and hardware |
| `metrics_overall.csv` | MAE, RMSE, MAPE and sMAPE per model over all horizons |
| `metrics_by_horizon.csv` | the same metrics per model and horizon step |
| `metrics_by_fold.csv` | the same metrics per model and fold |
| `coverage_overall.csv`, `coverage_by_horizon.csv` | empirical coverage and mean width of the P10 to P90 interval |
| `timing.csv`, `timing_by_fold.csv` | fit and prediction time |
| `predictions.csv.gz` | every forecast: model, fold, origin, horizon, actual, point, q10, q90 |
| `dispatch_savings.csv`, `dispatch_savings.json` | battery dispatch cost comparison and its setup |
| `feature_importance*.csv` | LightGBM gain importance |

Files with the suffix `_solar` come from the simulated PV target and files with `_online` from the run with online interval recalibration.

Reproduce with the commands listed in the README results section. Timing depends on the machine; the other columns are deterministic for a fixed data build.
