"""Muse Spark helpers: text, vision and structured output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from egresstwin.meta.client import MetaClient, MetaResult


def ask_text(client: MetaClient, prompt: str, *, system: str | None = None) -> MetaResult:
    """Ask Muse Spark a plain-text question through the shared client."""
    return client.chat_text(prompt, system=system)


def read_image_structured(
    client: MetaClient,
    image_path: str | Path,
    prompt: str,
    schema: dict[str, Any],
    *,
    schema_name: str = "StructuredRead",
) -> MetaResult:
    """Read an image with Muse Spark and a strict JSON schema."""
    return client.vision_structured(image_path, prompt, schema, schema_name=schema_name)
