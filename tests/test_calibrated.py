import numpy as np
import pandas as pd

from gridcast.backtest import BacktestConfig, run_backtest
from gridcast.evaluation import coverage_overall
from gridcast.models.calibrated import ConformalForecaster
from gridcast.models.naive import SeasonalNaive


def noisy_weekly(days: int = 160) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=24 * days, freq="1h", name="timestamp")
    rng = np.random.default_rng(9)
    base = 100 + 30 * np.sin(2 * np.pi * index.hour.to_numpy() / 24)
    return pd.DataFrame({"load_kw": base + rng.normal(0, 8, len(index))}, index=index)


def test_wrapper_produces_calibrated_intervals():
    frame = noisy_weekly()
    config = BacktestConfig(n_folds=2, fold_hours=24 * 21, stride=4)
    factory = lambda: ConformalForecaster(lambda: SeasonalNaive(period=24), alpha=0.2, stride=2)
    result = run_backtest(frame, {"naive": factory}, config)
    table = coverage_overall(result.predictions)
    assert abs(table["coverage"].iloc[0] - 0.8) < 0.07
    assert {"q10", "q90"} <= set(result.predictions.columns)


def test_calibration_never_touches_test_period():
    frame = noisy_weekly(120)
    model = ConformalForecaster(lambda: SeasonalNaive(period=24), calibration_hours=24 * 20, stride=3)
    train = frame.iloc[: 24 * 90]
    model.fit(train)
    positions = model.calibration_positions(len(train), len(train) - model.calibration_hours)
    assert positions.max() + model.horizon < len(train)


def test_wrapper_calibrates_on_the_models_own_target(synthetic_frame):
    from gridcast.features import solar_spec
    from gridcast.models.gbm import DirectLightGBM

    frame = synthetic_frame.assign(sun_elevation=0.5)
    spec = solar_spec(horizon=4)
    model = ConformalForecaster(
        lambda: DirectLightGBM(spec, {"n_estimators": 20, "n_jobs": 1}),
        calibration_hours=24 * 14,
        stride=6,
    )
    assert model.target == "solar_kw"
    model.fit(frame.iloc[: 24 * 80])
    assert model.split.radius.max() < 0.5 * frame["load_kw"].mean()
