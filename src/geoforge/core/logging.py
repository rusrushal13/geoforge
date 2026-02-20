"""Structured pipeline logging for debugging generation failures."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path


StageType = Literal["spec_generation", "code_generation", "validation", "export"]
StatusType = Literal["start", "success", "error", "retry"]


@dataclass
class PipelineEvent:
    """A single event in the pipeline execution."""

    timestamp: str
    stage: StageType
    status: StatusType
    provider: str | None = None
    attempt: int | None = None
    error: str | None = None
    duration_seconds: float | None = None
    details: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dict, omitting None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class PipelineLogger:
    """Tracks events through the generation pipeline."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider
        self.events: list[PipelineEvent] = []
        self._timers: dict[str, float] = {}

    def start(self, stage: StageType) -> None:
        """Record the start of a pipeline stage."""
        self._timers[stage] = time.monotonic()
        self.events.append(
            PipelineEvent(
                timestamp=datetime.now(tz=UTC).isoformat(),
                stage=stage,
                status="start",
                provider=self.provider,
            )
        )

    def success(self, stage: StageType, **details) -> None:
        """Record successful completion of a stage."""
        duration = self._elapsed(stage)
        self.events.append(
            PipelineEvent(
                timestamp=datetime.now(tz=UTC).isoformat(),
                stage=stage,
                status="success",
                provider=self.provider,
                duration_seconds=duration,
                details=details or None,
            )
        )

    def log_error(
        self, stage: StageType, error: str, attempt: int | None = None, **details
    ) -> None:
        """Record an error during a stage."""
        duration = self._elapsed(stage)
        self.events.append(
            PipelineEvent(
                timestamp=datetime.now(tz=UTC).isoformat(),
                stage=stage,
                status="error",
                provider=self.provider,
                attempt=attempt,
                error=error,
                duration_seconds=duration,
                details=details or None,
            )
        )

    def retry(self, stage: StageType, attempt: int, error: str) -> None:
        """Record a retry attempt."""
        self.events.append(
            PipelineEvent(
                timestamp=datetime.now(tz=UTC).isoformat(),
                stage=stage,
                status="retry",
                provider=self.provider,
                attempt=attempt,
                error=error,
            )
        )

    def _elapsed(self, stage: str) -> float | None:
        start = self._timers.pop(stage, None)
        if start is not None:
            return round(time.monotonic() - start, 3)
        return None

    def to_dict(self) -> dict:
        """Export full log as a dict."""
        return {
            "provider": self.provider,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

    def save(self, path: Path) -> None:
        """Save log to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
