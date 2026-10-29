"""SAM 3.1 helpers: one short noun-phrase concept per request, box prompts."""

from __future__ import annotations

from pathlib import Path

from egresstwin.meta.client import MetaClient, MetaResult


def segment_concept(
    client: MetaClient, image_path: str | Path, concept: str
) -> MetaResult:
    """Segment a single concept (e.g. 'door'). Zero matches is valid."""
    return client.segment_image(image_path, concept)


def segment_boxes(
    client: MetaClient, image_path: str | Path, boxes: list[list[int]]
) -> MetaResult:
    """Segment with box prompts seeded by pixel boxes [x, y, w, h]."""
    return client.segment_image_boxes(image_path, boxes)


def build_box_prompt(boxes: list[list[int]], width_px: int, height_px: int) -> str:
    """Box-prompt input_text per the SAM docs grammar (x2/y2 inclusive)."""
    parts = []
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        parts.append(
            f"{i}<|box;x1={int(x)};y1={int(y)};x2={int(x + bw - 1)};"
            f"y2={int(y + bh - 1)};w={width_px};h={height_px}|>"
        )
    return "<0f>" + ",".join(parts)
