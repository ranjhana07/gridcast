from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from gridcast.backtest import BacktestConfig, run_backtest
from gridcast.evaluation import (
    coverage_overall,
    coverage_table,
    metrics_by_fold,
    metrics_by_horizon,
    metrics_overall,
    timing_summary,
)
from gridcast.zoo import build_factories


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor()
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def hardware_info() -> dict[str, str | int]:
    import lightgbm
    import torch

    return {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count() or 0,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "lightgbm": lightgbm.__version__,
    }


def run_benchmark(
    frame: pd.DataFrame,
    models: list[str],
    config: BacktestConfig,
    output_dir: Path,
    alpha: float = 0.2,
    tag: str = "",
) -> dict[str, pd.DataFrame]:
    factories = build_factories(models, config.target, config.horizon, alpha)
    result = run_backtest(frame, factories, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    tables = {
        "metrics_by_horizon": metrics_by_horizon(result.predictions),
        "metrics_overall": metrics_overall(result.predictions),
        "metrics_by_fold": metrics_by_fold(result.predictions),
        "coverage_by_horizon": coverage_table(result.predictions),
        "coverage_overall": coverage_overall(result.predictions),
        "timing": timing_summary(result.timings),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}{suffix}.csv", index=False, float_format="%.4f")
    result.predictions.to_csv(
        output_dir / f"predictions{suffix}.csv.gz", index=False, float_format="%.3f"
    )
    meta = {
        "config": asdict(config),
        "alpha": alpha,
        "models": models,
        "rows": len(frame),
        "start": str(frame.index[0]),
        "end": str(frame.index[-1]),
        "hardware": hardware_info(),
    }
    (output_dir / f"meta{suffix}.json").write_text(json.dumps(meta, indent=2))
    return tables
