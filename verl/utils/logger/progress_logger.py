"""Small, line-buffered progress log for long-running training jobs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional


class TrainingProgressLogger:
    """Write concise, human-readable training milestones that are safe to tail."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # A run directory is required to be new by the UI-S1 launcher, so a
        # fresh progress file cannot discard another run's information.
        with open(self.path, "w", encoding="utf-8"):
            pass

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value)

    def log(self, phase: str, status: str, step: Optional[int] = None, **fields: Any) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        parts = [timestamp]
        if step is not None:
            parts.append(f"STEP {step}")
        parts.extend([phase, status])
        parts.extend(
            f"{key}={self._format_value(value)}"
            for key, value in fields.items()
            if value is not None
        )
        # Open per event so an abrupt Ray worker exit does not leave a buffered
        # progress record invisible to `tail -f`.
        with open(self.path, "a", encoding="utf-8", buffering=1) as handle:
            handle.write(" | ".join(parts) + "\n")
            handle.flush()

