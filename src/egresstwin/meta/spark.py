"""Muse Spark helpers (vision and structured output live here later)."""

from __future__ import annotations

from egresstwin.meta.client import MetaClient, MetaResult


def ask_text(client: MetaClient, prompt: str, *, system: str | None = None) -> MetaResult:
    """Ask Muse Spark a plain-text question through the shared client."""
    return client.chat_text(prompt, system=system)
