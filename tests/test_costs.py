"""Cost table and budget guard tests."""

from __future__ import annotations

import pytest

from egresstwin.meta.costs import (
    BudgetExceeded,
    CostTracker,
    estimate_sam_image_cost,
    estimate_sam_video_cost,
    estimate_text_cost,
    estimate_voice_cost,
)


def test_text_cost_spark() -> None:
    assert estimate_text_cost("muse-spark-1.3", 1_000_000, 1_000_000) == pytest.approx(5.50)


def test_text_cost_cached_input() -> None:
    cost = estimate_text_cost("muse-spark-1.3", 1_000_000, 0, cached_input_tokens=500_000)
    assert cost == pytest.approx(0.5 * 1.25 + 0.5 * 0.15)


def test_text_cost_contributor() -> None:
    cost = estimate_text_cost("muse-spark-1.3-contributor", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.10 + 0.20)


def test_text_cost_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown text model"):
        estimate_text_cost("nope", 1, 1)


def test_sam_costs() -> None:
    assert estimate_sam_image_cost(1000) == pytest.approx(2.50)
    assert estimate_sam_image_cost(1) == pytest.approx(0.0025)
    assert estimate_sam_video_cost(10) == pytest.approx(50 * 0.20 / 1000)  # 50-frame minimum
    assert estimate_sam_video_cost(1000) == pytest.approx(0.20)


def test_voice_cost() -> None:
    assert estimate_voice_cost(1000) == pytest.approx(3.00)
    assert estimate_voice_cost(60) == pytest.approx(0.18)


def test_budget_exceeded() -> None:
    tracker = CostTracker(budget_usd=1.0)
    tracker.charge(0.60)
    with pytest.raises(BudgetExceeded):
        tracker.charge(0.50)
    assert tracker.spent_usd == pytest.approx(0.60)  # failed charge does not accumulate
