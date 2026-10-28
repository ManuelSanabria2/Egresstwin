"""SAM 3.1 helpers: one short noun-phrase concept per request."""

from __future__ import annotations

from pathlib import Path

from egresstwin.meta.client import MetaClient, MetaResult


def segment_concept(
    client: MetaClient, image_path: str | Path, concept: str
) -> MetaResult:
    """Segment a single concept (e.g. 'door'). Zero matches is valid."""
    return client.segment_image(image_path, concept)
