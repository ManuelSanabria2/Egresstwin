"""Geodesic floor field: distance to each exit over the walkable grid.

One Dijkstra run per exit (8-neighbourhood, diagonal cost sqrt(2),
distances in meters), so an array per exit is kept and blocking an
exit is just dropping its slice. The desired direction is the
normalised negative gradient, precomputed per exit.

Choice: Dijkstra on the grid graph instead of scikit-fmm. Dijkstra is
exact for the graph metric the simulator actually walks on, needs no
extra dependency (scikit-fmm is not in pyproject), is deterministic,
and per-exit arrays fall out naturally. The fast-marching eikonal
solution would be smoother but measures a continuum metric the
grid-stepping agents do not follow. See PROGRESS.md (P02).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from egresstwin.layout.schema import Exit
from egresstwin.layout.validate import exit_source_cells

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


@dataclass
class FloorField:
    """Per-exit geodesic distance (m) and desired direction grids."""

    distances: np.ndarray  # (n_exits, rows, cols), inf = unreachable
    dir_x: np.ndarray  # same shape, unit desired direction (x component)
    dir_y: np.ndarray
    exit_ids: list[str]
    cell_size_m: float

    def direction_at(self, exit_index: int, positions: np.ndarray) -> np.ndarray:
        """Desired direction (N, 2) at meter positions (nearest cell)."""
        rows = np.clip(
            (positions[:, 1] / self.cell_size_m).astype(int), 0, self.distances.shape[1] - 1
        )
        cols = np.clip(
            (positions[:, 0] / self.cell_size_m).astype(int), 0, self.distances.shape[2] - 1
        )
        out = np.stack(
            [self.dir_x[exit_index, rows, cols], self.dir_y[exit_index, rows, cols]], axis=1
        )
        return out

    def distance_at(self, exit_index: int, positions: np.ndarray) -> np.ndarray:
        """Geodesic distance (N,) at meter positions (nearest cell)."""
        rows = np.clip(
            (positions[:, 1] / self.cell_size_m).astype(int), 0, self.distances.shape[1] - 1
        )
        cols = np.clip(
            (positions[:, 0] / self.cell_size_m).astype(int), 0, self.distances.shape[2] - 1
        )
        return self.distances[exit_index, rows, cols]


def _dijkstra(walkable: np.ndarray, sources: np.ndarray, cell_size_m: float) -> np.ndarray:
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
                nd = d + w * cell_size_m
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return dist


def _directions(
    dist: np.ndarray, exit_mid: tuple[float, float], cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised negative gradient; source cells aim at the exit."""
    rows, cols = dist.shape
    filled = np.where(
        np.isfinite(dist),
        dist,
        np.nanmax(dist[np.isfinite(dist)]) + 10.0 * cell_size_m if np.isfinite(dist).any() else 0.0,
    )
    gy, gx = np.gradient(filled, cell_size_m)
    dx, dy = -gx, -gy
    norm = np.hypot(dx, dy)
    ok = norm > 1e-12
    dx = np.where(ok, dx / np.maximum(norm, 1e-12), 0.0)
    dy = np.where(ok, dy / np.maximum(norm, 1e-12), 0.0)
    # Cells at distance 0 (or with flat gradient) aim at the exit segment.
    ys, xs = np.mgrid[0:rows, 0:cols]
    vx = exit_mid[0] - (xs + 0.5) * cell_size_m
    vy = exit_mid[1] - (ys + 0.5) * cell_size_m
    vn = np.hypot(vx, vy)
    flat = ~ok
    dx = np.where(flat & (vn > 1e-12), vx / np.maximum(vn, 1e-12), dx)
    dy = np.where(flat & (vn > 1e-12), vy / np.maximum(vn, 1e-12), dy)
    return dx.astype(np.float64), dy.astype(np.float64)


def single_exit_field(
    walkable: np.ndarray, cell_size_m: float, ex: Exit
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distance, dir_x, dir_y) grids for one exit over ``walkable``."""
    sources = exit_source_cells(walkable, ex.segment, cell_size_m)
    if len(sources) == 0:
        raise ValueError(f"exit {ex.id!r} does not touch any walkable cell")
    dist = _dijkstra(walkable, sources, cell_size_m)
    dx, dy = _directions(dist, ex.midpoint, cell_size_m)
    return dist, dx, dy


def compute_floor_field(walkable: np.ndarray, cell_size_m: float, exits: list[Exit]) -> FloorField:
    """Distance and direction grids for every exit in the list."""
    distances, dir_xs, dir_ys = [], [], []
    for ex in exits:
        dist, dx, dy = single_exit_field(walkable, cell_size_m, ex)
        distances.append(dist), dir_xs.append(dx), dir_ys.append(dy)
    return FloorField(
        distances=np.stack(distances),
        dir_x=np.stack(dir_xs),
        dir_y=np.stack(dir_ys),
        exit_ids=[e.id for e in exits],
        cell_size_m=cell_size_m,
    )
