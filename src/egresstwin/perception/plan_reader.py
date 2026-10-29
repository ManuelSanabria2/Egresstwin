"""Muse Spark 1.3 plan reader: image -> structured PlanRead (pixel boxes).

The prompt lives in prompts/plan_reader.md and the wire schema asks for
coordinates on the normalized 0-1000 grounding grid; this module
converts them to pixels. Every API call goes through MetaClient, so
results are disk-cached by image hash and charged to the cost ledger.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

PROMPT_PATH = Path(__file__).parent / "prompts" / "plan_reader.md"
_FALLBACK_PROMPT = (
    "Analyse this image for an evacuation digital twin. Classify "
    "image_type, list every exit or door with a box and confidence, "
    "list dimension annotations, and add notes. Coordinates are "
    "integers on a normalized 0-1000 grid."
)

PLAN_READER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "image_type": {"type": "string", "enum": ["floor_plan", "aerial_photo", "other"]},
        "exits": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "x_min": {"type": "integer"},
                    "y_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "y_max": {"type": "integer"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "x_min", "y_min", "x_max", "y_max", "confidence"],
            },
        },
        "dimension_annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "x_min": {"type": "integer"},
                    "y_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "y_max": {"type": "integer"},
                },
                "required": ["text", "x_min", "y_min", "x_max", "y_max"],
            },
        },
        "scale_annotation": {"type": ["string", "null"]},
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "image_type",
        "exits",
        "dimension_annotations",
        "scale_annotation",
        "notes",
        "confidence",
    ],
}


class PerceptionError(ValueError):
    """Raised when an image cannot be turned into a layout."""


class BoxPx(BaseModel):
    """Axis-aligned box in pixels: x, y of the top-left corner plus size."""

    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def as_list(self) -> list[int]:
        return [round(self.x), round(self.y), round(self.w), round(self.h)]


class ExitDetection(BaseModel):
    """One exit/door detection: pixel box, label and confidence."""

    box: BoxPx
    label: str = "door"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DimensionAnnotation(BaseModel):
    """A visible dimension/scale text and where it sits (pixels)."""

    text: str
    box: BoxPx | None = None


class PlanRead(BaseModel):
    """Structured Spark read of one image, all boxes in pixels."""

    image_type: Literal["floor_plan", "aerial_photo", "other"] = "other"
    exits: list[ExitDetection] = Field(default_factory=list)
    dimension_annotations: list[DimensionAnnotation] = Field(default_factory=list)
    scale_annotation: str | None = None
    notes: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    image_width_px: int = 0
    image_height_px: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    model: str = ""

    @field_validator("exits", mode="before")
    @classmethod
    def _accept_spike_key(cls, value: Any) -> Any:
        return value


def load_prompt() -> str:
    """Prompt text from prompts/plan_reader.md (fallback: embedded copy)."""
    try:
        text = PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return _FALLBACK_PROMPT
    return text or _FALLBACK_PROMPT


def image_sha256(path: str | Path) -> str:
    """SHA-256 of the image file bytes (cache key for every stage)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_image(path: str | Path) -> tuple[int, int]:
    """Validate that an image can serve as a plan. Returns (width, height)."""
    p = Path(path)
    if not p.is_file():
        raise PerceptionError(f"image not found: {p}")
    try:
        with Image.open(p) as im:
            im.load()
            width, height = im.size
            gray = im.convert("L")
            extrema = gray.getextrema()
    except (UnidentifiedImageError, OSError) as exc:
        raise PerceptionError(f"image is not readable as an image: {p} ({exc})") from exc
    if width < 32 or height < 32:
        raise PerceptionError(f"image is too small to be a plan: {width}x{height} px")
    if extrema[0] == extrema[1]:
        raise PerceptionError("image is blank (a single flat colour): no plan content found")
    return width, height


def _box_from(item: dict[str, Any], width: int, height: int) -> BoxPx | None:
    """Accept box_px [x,y,w,h], box {x,y,w,h} / [x,y,w,h], or 0-1000 corners."""
    raw_px = item.get("box_px") or item.get("box_pixels")
    if isinstance(raw_px, (list, tuple)) and len(raw_px) == 4:
        x, y, w, h = (float(v) for v in raw_px)
        return BoxPx(x=x, y=y, w=max(w, 1.0), h=max(h, 1.0))
    box = item.get("box")
    if isinstance(box, dict) and {"x", "y", "w", "h"} <= set(box):
        return BoxPx(
            x=float(box["x"]), y=float(box["y"]),
            w=max(float(box["w"]), 1.0), h=max(float(box["h"]), 1.0),
        )
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x, y, w, h = (float(v) for v in box)
        return BoxPx(x=x, y=y, w=max(w, 1.0), h=max(h, 1.0))
    if {"x_min", "y_min", "x_max", "y_max"} <= set(item):
        x = float(item["x_min"]) * width / 1000.0
        y = float(item["y_min"]) * height / 1000.0
        w = max((float(item["x_max"]) - float(item["x_min"])) * width / 1000.0, 1.0)
        h = max((float(item["y_max"]) - float(item["y_min"])) * height / 1000.0, 1.0)
        return BoxPx(x=x, y=y, w=w, h=h)
    return None


def parse_plan_data(data: dict[str, Any], width: int, height: int) -> PlanRead:
    """Parse Spark JSON (0-1000 grid or pixels) into a pixel PlanRead."""
    raw_exits = data.get("exits") or data.get("exits_or_doors") or []
    exits: list[ExitDetection] = []
    for item in raw_exits:
        if not isinstance(item, dict):
            continue
        box = _box_from(item, width, height)
        if box is None:
            continue
        exits.append(
            ExitDetection(
                box=box,
                label=str(item.get("label", "door")),
                confidence=float(item.get("confidence", 0.5)),
            )
        )
    annotations: list[DimensionAnnotation] = []
    for item in data.get("dimension_annotations") or []:
        if not isinstance(item, dict):
            continue
        annotations.append(
            DimensionAnnotation(text=str(item.get("text", "")), box=_box_from(item, width, height))
        )
    image_type = str(data.get("image_type", "other"))
    if image_type not in ("floor_plan", "aerial_photo", "other"):
        image_type = "other"
    return PlanRead(
        image_type=image_type,  # type: ignore[arg-type]
        exits=exits,
        dimension_annotations=annotations,
        scale_annotation=data.get("scale_annotation"),
        notes=str(data.get("notes", "")),
        confidence=float(data.get("confidence", 0.5)),
        image_width_px=width,
        image_height_px=height,
    )


def read_plan(image_path: str | Path, client: Any = None) -> PlanRead:
    """Read one image with Muse Spark 1.3 via MetaClient (cached, charged)."""
    width, height = check_image(image_path)
    if client is None:
        from egresstwin.meta.client import MetaClient

        client = MetaClient()
    prompt = load_prompt()
    if hasattr(client, "vision_structured"):
        result = client.vision_structured(
            image_path, prompt, PLAN_READER_SCHEMA, schema_name="PlanRead"
        )
        data = dict(result.data or {})
        if not data and result.text:
            import json

            data = json.loads(result.text)
        read = parse_plan_data(data, width, height)
        read.cost_usd = float(getattr(result, "cost_usd", 0.0) or 0.0)
        read.cached = bool(getattr(result, "cached", False))
        read.model = str(getattr(result, "model", "") or "")
        return read
    if hasattr(client, "read_plan"):
        return client.read_plan(image_path)  # test doubles
    raise PerceptionError("client does not support vision_structured reads")
