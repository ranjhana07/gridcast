from typer.testing import CliRunner

from gridcast.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("build-data", "benchmark", "dispatch-eval", "retrain", "forecast", "schedule", "serve"):
        assert command in result.output


def test_benchmark_fails_cleanly_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDCAST_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["benchmark", "-m", "naive_24"])
    assert result.exit_code != 0
