from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _path("GRIDCAST_DATA_DIR", "data"))
    artifact_dir: Path = field(default_factory=lambda: _path("GRIDCAST_ARTIFACT_DIR", "artifacts"))
    results_dir: Path = field(default_factory=lambda: _path("GRIDCAST_RESULTS_DIR", "results"))
    mlflow_uri: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    latitude: float = field(default_factory=lambda: float(os.environ.get("GRIDCAST_LATITUDE", "38.72")))
    longitude: float = field(default_factory=lambda: float(os.environ.get("GRIDCAST_LONGITUDE", "-9.14")))
    country: str = field(default_factory=lambda: os.environ.get("GRIDCAST_COUNTRY", "PT"))
    horizon: int = field(default_factory=lambda: int(os.environ.get("GRIDCAST_HORIZON", "24")))
    open_meteo_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPEN_METEO_URL", "https://archive-api.open-meteo.com/v1/archive"
        )
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def worker_threads() -> int:
    return max(1, int(os.environ.get("GRIDCAST_THREADS", "2")))


def get_settings() -> Settings:
    return Settings()
