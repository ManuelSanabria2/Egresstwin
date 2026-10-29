"""Run trace: runs/<run_id>/trace.jsonl, one JSON event per line."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


class TraceWriter:
    """Append-only trace of turns, tool calls, tokens, cost, duration."""

    def __init__(self, run_id: str, runs_dir: str | Path = "runs") -> None:
        self.run_id = run_id
        self.run_dir = Path(runs_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "trace.jsonl"
        self.events: list[dict[str, Any]] = []

    def write(self, event: dict[str, Any]) -> dict[str, Any]:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
        self.events.append(entry)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry
