"""Wall field: distance to the nearest wall and its gradient.

Both are precomputed once with scipy's Euclidean distance transform,
so the wall force during simulation is an O(1) nearest-cell lookup per
agent instead of a geometry query.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class WallField:
    """Distance to wall (m) and unit gradient pointing away from walls."""

    distance: np.ndarray  # (rows, cols) in meters, 0 inside walls
    grad_x: np.ndarray  # unit gradient of distance (into walkable space)
    grad_y: np.ndarray
    cell_size_m: float

    def sample(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(distance, grad_x, grad_y) at meter positions, O(1) per agent."""
        rows = np.clip(
            (positions[:, 1] / self.cell_size_m).astype(int), 0, self.distance.shape[0] - 1
        )
        cols = np.clip(
            (positions[:, 0] / self.cell_size_m).astype(int), 0, self.distance.shape[1] - 1
        )
        return self.distance[rows, cols], self.grad_x[rows, cols], self.grad_y[rows, cols]


def compute_wall_field(walkable: np.ndarray, cell_size_m: float) -> WallField:
    """Precompute distance-to-wall and its normalised gradient."""
    distance = ndimage.distance_transform_edt(walkable) * cell_size_m
    gy, gx = np.gradient(distance, cell_size_m)
    norm = np.hypot(gx, gy)
    ok = norm > 1e-12
    grad_x = np.where(ok, gx / np.maximum(norm, 1e-12), 0.0)
    grad_y = np.where(ok, gy / np.maximum(norm, 1e-12), 0.0)
    return WallField(distance=distance, grad_x=grad_x, grad_y=grad_y, cell_size_m=cell_size_m)
