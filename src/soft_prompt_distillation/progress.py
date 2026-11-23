"""Small, unbuffered progress messages for long-running stages."""

from __future__ import annotations

from datetime import datetime, timezone


def log_progress(message: str) -> None:
    """Write one timestamped progress line immediately."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] {message}", flush=True)


__all__ = ["log_progress"]
