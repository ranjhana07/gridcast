from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Job:
    name: str
    every: timedelta
    action: Callable[[], object]
    next_run: datetime | None = None
    failures: int = field(default=0)

    def due(self, now: datetime) -> bool:
        return self.next_run is None or now >= self.next_run

    def run(self, now: datetime) -> None:
        try:
            self.action()
            self.failures = 0
        except Exception:
            self.failures += 1
        finally:
            self.next_run = now + self.every


def run_pending(jobs: list[Job], now: datetime) -> list[str]:
    ran = []
    for job in jobs:
        if job.due(now):
            job.run(now)
            ran.append(job.name)
    return ran


def serve(jobs: list[Job], poll_seconds: float = 30.0, clock: Callable[[], datetime] = datetime.now) -> None:
    while True:
        run_pending(jobs, clock())
        time.sleep(poll_seconds)
