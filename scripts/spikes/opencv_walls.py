"""Task 4: OpenCV wall extraction for line plans (spike-only).

Method: grayscale -> Otsu threshold (dark ink) -> morphological close
to join stroke gaps -> walls. Walkable = pixels inside the outer wall
contour that are not walls (walls dilated by 2 px for stroke width).
Also derives door-gap candidates: white runs crossing the wall band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def extract_walls(image_path: Path) -> dict[str, Any]:
    """Return wall/walkable masks and exit-gap candidates for one plan."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(str(image_path))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    walls = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    # Interior = filled largest contour (the outer wall loop).
    contours, _ = cv2.findContours(walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    interior = np.zeros_like(walls)
    if contours:
        cv2.drawContours(interior, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    wall_dil = cv2.dilate(walls, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    walkable = cv2.bitwise_and(interior, cv2.bitwise_not(wall_dil))
    # Exit candidates: openings = wall-band pixels where ink is absent on
    # the outer contour. Approximate with white blobs touching the border
    # of the interior mask along the outer rectangle.
    gaps = _border_gaps(interior, walls)
    return {"walls": walls, "walkable": walkable, "interior": interior, "gaps": gaps}


def _border_gaps(interior: np.ndarray, walls: np.ndarray) -> list[list[int]]:
    """Find gaps along the outer wall loop as [x, y, w, h] boxes."""
    ys, xs = np.where(interior > 0)
    if len(xs) == 0:
        return []
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    band = 10
    strips = {
        "top": walls[y1 : y1 + band, x1:x2],
        "bottom": walls[y2 - band : y2, x1:x2],
        "left": walls[y1:y2, x1 : x1 + band],
        "right": walls[y2 - band : y2, x1:x2][:, :0],
    }
    strips["right"] = walls[y1:y2, x2 - band : x2]
    gaps: list[list[int]] = []
    for side, strip in strips.items():
        profile = (strip > 0).mean(axis=0 if side in ("top", "bottom") else 1)
        open_runs = _runs(profile < 0.5, min_len=25)
        for a, b in open_runs:
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
    out, start = [], None
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
    """IoU of two binary masks (any nonzero = positive)."""
    aa, bb = a > 0, b > 0
    union = int(np.logical_or(aa, bb).sum())
    return float(np.logical_and(aa, bb).sum() / union) if union else 1.0
