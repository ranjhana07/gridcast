from gridcast.explain import family_table, feature_family, fit_importance
from gridcast.features import FeatureSpec


def test_feature_families():
    assert feature_family("lag_24") == "recent lags"
    assert feature_family("roll_mean_168") == "rolling statistics"
    assert feature_family("seasonal_168") == "seasonal lags"
    assert feature_family("future_temperature_2m") == "weather at target hour"
    assert feature_family("future_hour_sin") == "calendar at target hour"
    assert feature_family("origin_temperature_2m") == "weather at origin"


def test_importance_shares_sum_to_one(synthetic_frame):
    table = fit_importance(synthetic_frame.iloc[: 24 * 60], FeatureSpec(horizon=2))
    assert abs(table["share"].sum() - 1.0) < 1e-9
    families = family_table(table)
    assert abs(families["share"].sum() - 1.0) < 1e-9
