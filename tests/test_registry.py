import numpy as np
import pandas as pd
import pytest

from gridcast.registry import (
    PRODUCTION_ALIAS,
    load_production_model,
    promote_if_better,
    register_run,
    registered_name,
)
from gridcast.tracking import configure
from gridcast.training import holdout_metrics, register_result, train_candidate


@pytest.fixture
def tracking_uri(tmp_path):
    return f"file:{(tmp_path / 'mlruns').as_posix()}"


@pytest.fixture
def weekly_frame():
    index = pd.date_range("2024-01-01", periods=24 * 140, freq="1h", name="timestamp")
    rng = np.random.default_rng(2)
    hour = index.hour.to_numpy()
    load = 300 + 60 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 4, len(index))
    return pd.DataFrame({"load_kw": load}, index=index)


def test_registered_name_is_slugged():
    assert registered_name("load_kw", "lstm_quantile") == "gridcast-load-kw-lstm-quantile"


def test_train_register_and_promote_roundtrip(tracking_uri, weekly_frame):
    first = train_candidate(weekly_frame.iloc[:-24 * 7], "naive_24", "load_kw", 24, tracking_uri, holdout_hours=24 * 14)
    assert first.metrics["holdout_mae"] > 0
    version = register_result(tracking_uri, first)
    client = configure(tracking_uri)
    decision = promote_if_better(client, first.registered_name, version)
    assert decision.promoted and decision.production_metric is None
    assert str(client.get_model_version_by_alias(first.registered_name, PRODUCTION_ALIAS).version) == version
    loaded = load_production_model(client, first.registered_name)
    origins = weekly_frame.index[[24 * 100]]
    assert np.allclose(loaded.predict(weekly_frame, origins).point, first.model.predict(weekly_frame, origins).point)


def test_worse_candidate_is_not_promoted(tracking_uri, weekly_frame):
    good = train_candidate(weekly_frame, "naive_24", "load_kw", 24, tracking_uri)
    noisy = weekly_frame.copy()
    noisy["load_kw"] = noisy["load_kw"] + np.random.default_rng(5).normal(0, 60, len(noisy))
    bad = train_candidate(noisy, "naive_24", "load_kw", 24, tracking_uri)
    name = good.registered_name
    client = configure(tracking_uri)
    first = register_run(client, name, good.run_id)
    assert promote_if_better(client, name, first).promoted
    second = register_run(client, name, bad.run_id)
    decision = promote_if_better(client, name, second)
    assert not decision.promoted
    assert decision.candidate_metric > decision.production_metric


def test_holdout_metrics_report_coverage(weekly_frame):
    from gridcast.zoo import build_factories

    model = build_factories(["naive_24"])["naive_24"]()
    train_end = len(weekly_frame) - 24 * 14
    model.fit(weekly_frame.iloc[:train_end])
    scores = holdout_metrics(model, weekly_frame, train_end)
    assert 0.6 < scores["holdout_coverage"] <= 1.0
    assert scores["holdout_width"] > 0


def test_benchmark_summary_is_logged(tracking_uri):
    import mlflow

    from gridcast.tracking import log_benchmark

    tables = {
        "metrics_overall": pd.DataFrame({"model": ["m"], "mae": [1.5], "rmse": [2.0], "mape": [1.0], "smape": [1.0]}),
        "coverage_overall": pd.DataFrame({"model": ["m"], "coverage": [0.8], "width": [10.0]}),
        "timing": pd.DataFrame({"model": ["m"], "fit_seconds": [3.0], "predict_seconds": [0.1]}),
    }
    meta = {"config": {"target": "load_kw", "n_folds": 2, "fold_hours": 336, "stride": 6}, "alpha": 0.2, "rows": 100}
    configure(tracking_uri)
    run_id = log_benchmark(tables, meta)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["m.mae"] == 1.5
    assert run.data.metrics["m.coverage"] == 0.8
    assert run.data.params["folds"] == "2"
