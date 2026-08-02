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


def test_online_corrections_follow_a_change_in_noise_level():
    from gridcast.conformal import online_corrections

    rng = np.random.default_rng(11)
    horizon = 4
    calibration = np.abs(rng.normal(0, 1, size=(500, horizon)))
    positions = np.arange(0, 3000, 3)
    test_scores = np.abs(rng.normal(0, 3, size=(len(positions), horizon)))
    corrections = online_corrections(calibration, test_scores, positions, 0.2, window=300)
    static = np.quantile(calibration, 0.8, axis=0)
    late = slice(len(positions) // 2, None)
    online_cover = np.mean(test_scores[late] <= corrections[late])
    static_cover = np.mean(test_scores[late] <= static)
    assert static_cover < 0.45
    assert abs(online_cover - 0.8) < 0.05


def test_online_corrections_only_use_scores_that_have_resolved():
    from gridcast.conformal import online_corrections

    positions = np.array([0, 1, 2, 3])
    calibration = np.zeros((5, 2))
    test_scores = np.array([[100.0, 100.0]] * 4)
    corrections = online_corrections(calibration, test_scores, positions, 0.2, window=50)
    assert corrections[0, 0] == 0.0
    assert corrections[0, 1] == 0.0
    assert corrections[1, 1] == 0.0
    assert corrections[1, 0] == 100.0
    assert corrections[2, 1] == 100.0
    big = online_corrections(calibration[:1], test_scores, positions, 0.2, window=50)
    assert big[3, 0] == 100.0


def test_online_requires_time_ordered_origins():
    from gridcast.conformal import online_corrections

    with pytest.raises(ValueError):
        online_corrections(np.zeros((3, 1)), np.zeros((2, 1)), np.array([5, 1]), 0.2)
