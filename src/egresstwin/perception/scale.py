"""Scale resolution: pixels to meters, with assumed values flagged.

Priority: user two-point reference, then a known door width, then a
dimension annotation read by Spark (measured against the outer
outline), then an assumed standard door width. Anything not measured
from the image or given by the user is marked assumed so the report
never presents a guess as a measurement.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from egresstwin.perception.plan_reader import PerceptionError

ASSUMED_DOOR_WIDTH_M = 0.90
_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm|m)\b", re.IGNORECASE)


@dataclass
class ScaleEstimate:
    """Resolved scale plus how it was obtained."""

    meters_per_pixel: float
    source: Literal["user_two_points", "known_door", "annotation", "assumed"]
    assumed: bool
    detail: str

    def as_dict(self) -> dict:
        return {
            "meters_per_pixel": self.meters_per_pixel,
            "source": self.source,
            "assumed": self.assumed,
            "detail": self.detail,
        }


def from_two_points(
    point_a_px: tuple[float, float], point_b_px: tuple[float, float], distance_m: float
) -> ScaleEstimate:
    """Scale from two pixel points and the real distance between them."""
    if distance_m <= 0:
        raise PerceptionError("scale distance must be > 0 m")
    dist_px = math.dist(point_a_px, point_b_px)
    if dist_px < 1.0:
        raise PerceptionError("scale reference points are less than 1 px apart")
    return ScaleEstimate(
        meters_per_pixel=distance_m / dist_px,
        source="user_two_points",
        assumed=False,
        detail=f"user reference: {dist_px:.1f} px = {distance_m:g} m",
    )


def from_known_door(door_width_px: float, door_width_m: float) -> ScaleEstimate:
    """Scale from the pixel width of a door whose real width is given."""
    if door_width_px < 1.0:
        raise PerceptionError("known-door pixel width must be >= 1 px")
    if door_width_m <= 0:
        raise PerceptionError("known-door real width must be > 0 m")
    return ScaleEstimate(
        meters_per_pixel=door_width_m / door_width_px,
        source="known_door",
        assumed=False,
        detail=f"known door: {door_width_px:.1f} px = {door_width_m:g} m",
    )


def parse_annotation_meters(text: str | None) -> float | None:
    """First length in a dimension annotation, in meters (or None)."""
    if not text:
        return None
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower()
    return value / 1000.0 if unit == "mm" else value / 100.0 if unit == "cm" else value


def from_annotation(annotation_m: float, reference_px: float, detail: str = "") -> ScaleEstimate:
    """Scale from an annotation value measured over reference_px pixels."""
    if reference_px < 1.0:
        raise PerceptionError("annotation reference length must be >= 1 px")
    if annotation_m <= 0:
        raise PerceptionError("annotation length must be > 0 m")
    return ScaleEstimate(
        meters_per_pixel=annotation_m / reference_px,
        source="annotation",
        assumed=False,
        detail=detail or f"annotation: {reference_px:.1f} px = {annotation_m:g} m",
    )


def assumed_from_door(door_width_px: float) -> ScaleEstimate:
    """Fallback scale assuming a standard door width. Always flagged."""
    if door_width_px < 1.0:
        raise PerceptionError(
            "cannot resolve scale: no user reference, no annotation and no door to assume from"
        )
    return ScaleEstimate(
        meters_per_pixel=ASSUMED_DOOR_WIDTH_M / door_width_px,
        source="assumed",
        assumed=True,
        detail=(
            f"assumed standard door width {ASSUMED_DOOR_WIDTH_M:.2f} m "
            f"over {door_width_px:.1f} px; verify before trusting distances"
        ),
    )
