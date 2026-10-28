"""Client tests: mock mode, disk cache (2nd call free), budget cap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from egresstwin.config import Settings
from egresstwin.meta.client import MetaClient
from egresstwin.meta.costs import BudgetExceeded


def test_mock_chat(settings: Settings) -> None:
    client = MetaClient(settings)
    result = client.chat_text("What exit do you see?")
    assert result.text and "Mock response" in result.text
    assert result.cached is False
    assert result.cost_usd > 0
    assert client.tracker.spent_usd == pytest.approx(result.cost_usd)


def test_mock_sam_door(settings: Settings, sample_image: Path) -> None:
    client = MetaClient(settings)
    result = client.segment_image(sample_image, "door")
    assert result.data["concept"] == "door"
    assert len(result.data["segments"]) == 1
    assert result.data["segments"][0]["label"] == "door"
    assert result.cost_usd == pytest.approx(0.0025)


def test_cache_second_call_is_free(settings: Settings, tmp_path: Path) -> None:
    client = MetaClient(settings)
    first = client.chat_text("What exit do you see?")
    spent_after_first = client.tracker.spent_usd
    second = client.chat_text("What exit do you see?")
    assert second.cached is True
    assert second.cost_usd == 0.0
    assert second.text == first.text
    assert client.tracker.spent_usd == spent_after_first
    log = Path(settings.runs_dir) / "costs.jsonl"
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1  # cache hit is not logged as a charged call
    assert json.loads(lines[0])["cost_usd"] == pytest.approx(first.cost_usd)


def test_budget_cap_blocks_call(tmp_path: Path) -> None:
    options: dict[str, object] = {
        "meta_mock": True,
        "meta_budget_usd": 0.000001,
        "cache_dir": str(tmp_path / "cache"),
        "runs_dir": str(tmp_path / "runs"),
    }
    client = MetaClient(Settings(**options))  # type: ignore[arg-type]
    with pytest.raises(BudgetExceeded):
        client.chat_text("What exit do you see?")
