"""OpenCV wall extraction for line plans (decided in spike P01).

Walls come from OpenCV, not SAM: Otsu ink threshold plus a
morphological close gave walkable IoU 0.91 on the clean synthetic
plan and 0.89 on the hand-drawn style variant, while SAM 'wall'
contributed zero pixels. Also derives exit-gap candidates along
the outer wall loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from egresstwin.perception.plan_reader import PerceptionError


@dataclass
class CVResult:
    """Pixel masks from the OpenCV path, all at image resolution."""

    walls: np.ndarray  # bool
    walkable: np.ndarray  # bool
    interior: np.ndarray  # bool
    gaps: list[list[int]] = field(default_factory=list)  # [x, y, w, h] exit candidates
    image_shape: tuple[int, int] = (0, 0)  # (rows, cols)


def extract_walls(image_path: str | Path) -> CVResult:
    """Return wall/walkable masks and exit-gap candidates for one plan."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise PerceptionError(f"image is not readable as an image: {image_path}")
    if gray.size == 0 or int(gray.max()) == int(gray.min()):
        raise PerceptionError("image is blank (a single flat colour): no plan content found")
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    walls_u8 = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(walls_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise PerceptionError("no closed outline found: the image does not look like a plan")
    interior_u8 = np.zeros_like(walls_u8)
    cv2.drawContours(interior_u8, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    if float((interior_u8 > 0).mean()) < 0.02:
        raise PerceptionError("no closed outline found: the image does not look like a plan")
    wall_dil = cv2.dilate(
        walls_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2
    )
    walkable_u8 = cv2.bitwise_and(interior_u8, cv2.bitwise_not(wall_dil))
    gaps = _border_gaps(interior_u8, walls_u8)
    return CVResult(
        walls=walls_u8 > 0,
        walkable=walkable_u8 > 0,
        interior=interior_u8 > 0,
        gaps=gaps,
        image_shape=tuple(gray.shape),  # type: ignore[arg-type]
    )


def _border_gaps(interior: np.ndarray, walls: np.ndarray) -> list[list[int]]:
    """Find gaps along the outer wall loop as [x, y, w, h] boxes."""
    ys, xs = np.where(interior > 0)
    if len(xs) == 0:
        return []
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    band = 10
    strips: dict[str, Any] = {
        "top": walls[y1 : y1 + band, x1:x2],
        "bottom": walls[y2 - band : y2, x1:x2],
        "left": walls[y1:y2, x1 : x1 + band],
        "right": walls[y1:y2, x2 - band : x2],
    }
    gaps: list[list[int]] = []
    for side, strip in strips.items():
        if strip.size == 0:
            continue
        profile = (strip > 0).mean(axis=0 if side in ("top", "bottom") else 1)
        for a, b in _runs(profile < 0.5, min_len=25):
            if side == "top":
                gaps.append([x1 + a, y1, b - a, band])
            elif side == "bottom":
                gaps.append([x1 + a, y2 - band, b - a, band])
            elif side == "left":
                gaps.append([x1, y1 + a, band, b - a])
            else:
                gaps.append([x2 - band, y1 + a, band, b - a])
    return gaps


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks (any nonzero/True = positive)."""
    aa, bb = np.asarray(a) > 0, np.asarray(b) > 0
    union = int(np.logical_or(aa, bb).sum())
    return float(np.logical_and(aa, bb).sum() / union) if union else 1.0
