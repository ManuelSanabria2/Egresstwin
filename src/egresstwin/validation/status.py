"""Load the simulator verification status written by run_verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATUS_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "verification" / "status.json"
)


def load_verification_status(path: str | Path | None = None) -> dict[str, Any]:
    """Return the verification status dict (unavailable flag if absent)."""
    file = Path(path) if path is not None else STATUS_PATH
    if not file.is_file():
        return {"available": False, "passed": 0, "total": 0, "cases": []}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "passed": 0, "total": 0, "cases": []}
    data.setdefault("available", True)
    return data
