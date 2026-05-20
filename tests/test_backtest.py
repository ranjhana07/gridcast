import numpy as np
import pandas as pd
import pytest

from gridcast.backtest import BacktestConfig, make_folds, run_backtest
from gridcast.evaluation import metrics_by_horizon, timing_summary
from gridcast.models.naive import SeasonalNaive


def periodic_frame(days: int = 150) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=24 * days, freq="1h", name="timestamp")
    hour = index.hour.to_numpy()
    dow = index.day_of_week.to_numpy()
    return pd.DataFrame({"load_kw": 100 + hour + 10 * dow}, index=index)


def test_folds_do_not_overlap_and_end_at_the_data_end():
    config = BacktestConfig(n_folds=3, fold_hours=24 * 14)
    folds = list(make_folds(24 * 150, config))
    assert folds[-1].test_end == 24 * 150
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end == later.train_end


def test_not_enough_history_is_rejected():
    with pytest.raises(ValueError):
        list(make_folds(24 * 30, BacktestConfig(n_folds=4, fold_hours=24 * 14)))


def test_origins_never_reach_past_the_fold():
    config = BacktestConfig(n_folds=2, fold_hours=24 * 14, stride=5)
    for fold in make_folds(24 * 150, config):
        positions = fold.origin_positions(config.horizon, config.stride)
        assert positions.max() + config.horizon < fold.test_end
        assert positions.min() == fold.train_end - 1


def test_seasonal_naive_is_exact_on_periodic_series():
    frame = periodic_frame()
    config = BacktestConfig(n_folds=2, fold_hours=24 * 14, stride=6)
    result = run_backtest(frame, {"naive": lambda: SeasonalNaive(period=168)}, config)
    table = metrics_by_horizon(result.predictions)
    assert np.allclose(table["mae"], 0.0)
    assert set(result.predictions["horizon"]) == set(range(1, 25))
    assert len(timing_summary(result.timings)) == 1
