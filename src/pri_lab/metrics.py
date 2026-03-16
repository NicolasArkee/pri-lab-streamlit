from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import platform
import resource
import time


def timestamp_utc() -> str:
  return datetime.now(UTC).isoformat(timespec="seconds")


def peak_memory_mb() -> float:
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if platform.system() == "Darwin":
    return rss / (1024 * 1024)
  return rss / 1024


@dataclass
class StageMetric:
  name: str
  started_at: str
  finished_at: str
  duration_seconds: float
  details: dict[str, object] = field(default_factory=dict)


@dataclass
class MetricsRecorder:
  command: str
  started_at: str = field(default_factory=timestamp_utc)
  stages: list[StageMetric] = field(default_factory=list)
  _start_perf: float = field(default_factory=time.perf_counter)

  @contextmanager
  def stage(self, name: str) -> Iterator[dict[str, object]]:
    started_at = timestamp_utc()
    started_perf = time.perf_counter()
    details: dict[str, object] = {}
    yield details
    finished_at = timestamp_utc()
    self.stages.append(
      StageMetric(
        name=name,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(time.perf_counter() - started_perf, 6),
        details=details,
      ),
    )

  def finish(self, extra: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
      "command": self.command,
      "started_at": self.started_at,
      "finished_at": timestamp_utc(),
      "duration_seconds": round(time.perf_counter() - self._start_perf, 6),
      "peak_memory_mb": round(peak_memory_mb(), 3),
      "stages": [
        {
          "name": stage.name,
          "started_at": stage.started_at,
          "finished_at": stage.finished_at,
          "duration_seconds": stage.duration_seconds,
          "details": stage.details,
        }
        for stage in self.stages
      ],
    }
    if extra:
      payload.update(extra)
    return payload
