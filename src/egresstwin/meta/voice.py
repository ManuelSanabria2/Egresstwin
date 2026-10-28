"""Muse Voice Transcribe helpers (optional; cut first if time is short)."""

from __future__ import annotations

from egresstwin.meta.costs import estimate_voice_cost


def transcription_cost_usd(minutes: float) -> float:
    """Estimate transcription cost before any voice call is wired up."""
    return estimate_voice_cost(minutes)
