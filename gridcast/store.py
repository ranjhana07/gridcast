from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / name

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(target)
        return target

    def read_json(self, name: str) -> dict[str, Any] | None:
        target = self.path(name)
        if not target.exists():
            return None
        return json.loads(target.read_text())

    def forecast_name(self, target: str) -> str:
        return f"forecast_{target}.json"

    def dispatch_name(self) -> str:
        return "dispatch_latest.json"


class ResultsStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def table(self, name: str, tag: str = "") -> pd.DataFrame | None:
        suffix = f"_{tag}" if tag else ""
        path = self.root / f"{name}{suffix}.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    def predictions(self, tag: str = "") -> pd.DataFrame | None:
        suffix = f"_{tag}" if tag else ""
        path = self.root / f"predictions{suffix}.csv.gz"
        if not path.exists():
            return None
        return pd.read_csv(path, parse_dates=["origin"])

    def meta(self, tag: str = "") -> dict[str, Any] | None:
        suffix = f"_{tag}" if tag else ""
        path = self.root / f"meta{suffix}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())
