"""Config loading tests."""

from __future__ import annotations

import pytest

from egresstwin.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.meta_spark_model == "muse-spark-1.3"
    assert s.meta_sam_model == "sam-3.1"
    assert s.meta_voice_model == "muse-voice-transcribe-1.0"
    assert s.meta_budget_usd == 60.0
    assert s.numba_num_threads >= 1


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_MOCK", "1")
    monkeypatch.setenv("META_BUDGET_USD", "12.5")
    monkeypatch.setenv("META_SPARK_MODEL", "muse-spark-1.3-contributor")
    monkeypatch.setenv("NUMBA_NUM_THREADS", "4")
    s = Settings()
    assert s.meta_mock is True
    assert s.mock is True
    assert s.meta_budget_usd == 12.5
    assert s.meta_spark_model == "muse-spark-1.3-contributor"
    assert s.numba_num_threads == 4
