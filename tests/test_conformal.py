import numpy as np
import pytest

from gridcast.conformal import ConformalizedQuantile, SplitConformal, conformal_quantile


def noisy_forecasts(n: int, horizon: int, seed: int, scale: float = 10.0):
    rng = np.random.default_rng(seed)
    point = rng.uniform(100, 200, size=(n, horizon))
    spread = scale * (1 + np.arange(horizon) / horizon)
    actual = point + rng.normal(0, 1, size=(n, horizon)) * spread
    return actual, point


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3])
def test_split_conformal_hits_nominal_coverage(alpha):
    cal_actual, cal_point = noisy_forecasts(4000, 6, seed=1)
    test_actual, test_point = noisy_forecasts(20000, 6, seed=2)
    calibrator = SplitConformal(alpha).fit(cal_actual, cal_point)
    lower, upper = calibrator.interval(test_point)
    coverage = np.mean((test_actual >= lower) & (test_actual <= upper), axis=0)
    assert np.all(np.abs(coverage - (1 - alpha)) < 0.02)


def test_intervals_widen_with_horizon_noise():
    actual, point = noisy_forecasts(5000, 8, seed=4)
    radius = SplitConformal(0.2).fit(actual, point).radius
    assert radius[-1] > radius[0]


def test_cqr_repairs_overconfident_quantiles():
    cal_actual, cal_point = noisy_forecasts(4000, 4, seed=5)
    test_actual, test_point = noisy_forecasts(20000, 4, seed=6)
    narrow = 3.0
    calibrator = ConformalizedQuantile(0.2).fit(cal_actual, cal_point - narrow, cal_point + narrow)
    raw = np.mean(np.abs(test_actual - test_point) <= narrow)
    lower, upper = calibrator.interval(test_point - narrow, test_point + narrow)
    fixed = np.mean((test_actual >= lower) & (test_actual <= upper))
    assert raw < 0.4
    assert abs(fixed - 0.8) < 0.02


def test_cqr_can_shrink_conservative_quantiles():
    cal_actual, cal_point = noisy_forecasts(4000, 4, seed=7)
    wide = 100.0
    calibrator = ConformalizedQuantile(0.2).fit(cal_actual, cal_point - wide, cal_point + wide)
    assert (calibrator.correction < 0).all()


def test_finite_sample_rank_is_capped():
    scores = np.arange(1, 6, dtype=float)[:, None]
    assert conformal_quantile(scores, 0.01)[0] == 5.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        SplitConformal(1.2)
    with pytest.raises(ValueError):
        conformal_quantile(np.empty((0, 2)), 0.1)
    with pytest.raises(RuntimeError):
        SplitConformal().interval(np.zeros((1, 2)))
