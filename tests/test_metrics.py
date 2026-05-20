import numpy as np

from gridcast import metrics


def test_point_metrics_known_values():
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 180.0, 330.0])
    assert metrics.mae(actual, predicted) == 20.0
    assert np.isclose(metrics.rmse(actual, predicted), np.sqrt((100 + 400 + 900) / 3))
    assert np.isclose(metrics.mape(actual, predicted), 100 * (0.1 + 0.1 + 0.1) / 3)


def test_smape_is_symmetric():
    a = np.array([50.0, 80.0])
    b = np.array([60.0, 70.0])
    assert np.isclose(metrics.smape(a, b), metrics.smape(b, a))


def test_pinball_loss_penalises_under_prediction_more_for_high_quantile():
    actual = np.array([10.0])
    low = metrics.pinball(actual, np.array([5.0]), 0.9)
    high = metrics.pinball(actual, np.array([15.0]), 0.9)
    assert low > high


def test_interval_helpers():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 2.5, 2.0, 3.0])
    upper = np.array([2.0, 3.0, 4.0, 5.0])
    assert metrics.interval_coverage(actual, lower, upper) == 0.75
    assert metrics.interval_width(lower, upper) == 1.625
