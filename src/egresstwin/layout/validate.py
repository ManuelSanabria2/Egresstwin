"""Layout validation: reachability, exit placement, narrow passages.

Errors raise LayoutValidationError with a clear message. Narrow
passages (local width < 0.6 m by default) are warnings, returned as a
list of strings.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage

from egresstwin.layout.schema import Layout

_SQRT2 = float(np.sqrt(2.0))
_NEIGHBORS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (1, 1, _SQRT2),
]


class LayoutValidationError(ValueError):
    """Raised when a layout cannot be simulated safely."""


def exit_source_cells(walkable: np.ndarray, exit_segment: tuple, cell_size_m: float) -> np.ndarray:
    """Walkable cells whose centre lies within 1.5 cells of the segment."""
    (x1, y1), (x2, y2) = exit_segment
    cells = np.argwhere(walkable)
    if len(cells) == 0:
        return cells.reshape(-1, 2)
    px = (cells[:, 1] + 0.5) * cell_size_m - x1
    py = (cells[:, 0] + 0.5) * cell_size_m - y1
    dx, dy = x2 - x1, y2 - y1
    length2 = dx * dx + dy * dy
    t = np.clip((px * dx + py * dy) / length2, 0.0, 1.0) if length2 > 0 else np.zeros(len(cells))
    dist = np.hypot(px - t * dx, py - t * dy)
    return cells[dist <= 1.5 * cell_size_m + 1e-9]


def reachable_mask(walkable: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Cells that can reach any source cell (Dijkstra over walkable)."""
    rows, cols = walkable.shape
    dist = np.full((rows, cols), np.inf)
    heap: list[tuple[float, int, int]] = []
    for r, c in sources:
        dist[r, c] = 0.0
        heapq.heappush(heap, (0.0, int(r), int(c)))
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc, w in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and walkable[nr, nc]:
                nd = d + w
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return np.isfinite(dist)


def validate_layout(layout: Layout, *, narrow_threshold_m: float = 0.6) -> list[str]:
    """Validate a layout. Returns warnings; raises on hard errors."""
    if not layout.exits:
        raise LayoutValidationError(
            "layout has no exits: add at least one exit so people can leave"
        )
    walkable = layout.effective_walkable()
    if not walkable.any():
        raise LayoutValidationError("layout has no walkable cells")
    warnings: list[str] = []
    all_sources = []
    for ex in layout.exits:
        sources = exit_source_cells(walkable, ex.segment, layout.cell_size_m)
        if len(sources) == 0:
            raise LayoutValidationError(
                f"exit {ex.id!r} does not touch any walkable cell: "
                "place its segment on the boundary of the walkable area"
            )
        # On the boundary: at least one source cell must neighbour a
        # non-walkable cell or the grid edge (otherwise it is interior).
        rows, cols = walkable.shape
        on_edge = False
        for r, c in sources:
            for dr, dc, _ in _NEIGHBORS:
                nr, nc = int(r) + dr, int(c) + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or not walkable[nr, nc]:
                    on_edge = True
        if not on_edge:
            raise LayoutValidationError(
                f"exit {ex.id!r} is not on the boundary of the walkable area: "
                "every neighbouring cell is walkable"
            )
        if ex.width_m < narrow_threshold_m:
            warnings.append(
                f"exit {ex.id!r} is {ex.width_m:.2f} m wide "
                f"(< {narrow_threshold_m:.2f} m recommended minimum)"
            )
        all_sources.append(sources)
    sources = np.concatenate(all_sources)
    reached = reachable_mask(walkable, sources)
    unreachable = int((walkable & ~reached).sum())
    if unreachable:
        total = int(walkable.sum())
        raise LayoutValidationError(
            f"{unreachable} of {total} walkable cells cannot reach any exit: "
            "connect them or mark them as obstacles"
        )
    # Narrow passages: ridge cells of the distance transform whose local
    # width (2 x distance to wall) is below the threshold.
    dist_m = ndimage.distance_transform_edt(walkable) * layout.cell_size_m
    ridge = (dist_m == ndimage.maximum_filter(dist_m, size=3)) & walkable & (dist_m > 0)
    narrow = ridge & (2.0 * dist_m < narrow_threshold_m)
    if narrow.any():
        width = float(2.0 * dist_m[narrow].min())
        warnings.append(
            f"passage narrower than {narrow_threshold_m:.2f} m detected "
            f"(minimum local width about {width:.2f} m)"
        )
    return warnings
