"""Shared fixtures: mock settings isolated in tmp dirs."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from egresstwin.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Mock-mode settings with cache/runs under tmp_path."""
    options: dict[str, object] = {
        "meta_mock": True,
        "meta_budget_usd": 60.0,
        "cache_dir": str(tmp_path / "cache"),
        "runs_dir": str(tmp_path / "runs"),
    }
    return Settings(**options)  # type: ignore[arg-type]


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """A tiny synthetic image for SAM calls."""
    path = tmp_path / "door_sample.png"
    Image.new("RGB", (32, 24), "white").save(path)
    return path
