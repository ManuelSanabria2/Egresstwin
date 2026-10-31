"""Timed live events for a simulation run (P07).

An event fires once, at ``at_s`` seconds into the run:

- ``block_exit``: an exit stops capturing people. Its floor-field
  slice is dropped (no recompute); people heading there re-choose by
  geodesic distance over the exits still open.
- ``reopen_exit``: a blocked exit opens again. Only that exit's
  floor-field slice is recomputed.
- ``hazard_zone``: a polygon of cells turns non-walkable (smoke, a
  cordoned area). The floor fields of the exits still open are
  recomputed; blocked exits stay dropped.

Relocation rule for ``hazard_zone``: anyone whose current cell turns
non-walkable is moved, deterministically, to the centre of the
nearest still-walkable cell (Euclidean nearest, ties broken by the
distance transform's row-major scan order) and their velocity is
zeroed. If the hazard disconnects walkable cells from every open
exit, the engine records a warning instead of hiding it.

Events are pure simulation: no API calls, no cost.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter, model_validator
from scipy import ndimage

from egresstwin.layout.schema import Point, rasterize_polygon

__all__ = [
    "BlockExitEvent",
    "HazardZoneEvent",
    "ReopenExitEvent",
    "SimEvent",
    "coerce_events",
    "hazard_mask",
    "relocate_out_of_hazard",
]


class BlockExitEvent(BaseModel):
    """Block ``exit_id`` from ``at_s`` seconds into the run."""

    type: Literal["block_exit"] = "block_exit"
    exit_id: str = Field(min_length=1)
    at_s: float = Field(ge=0)


class ReopenExitEvent(BaseModel):
    """Reopen a previously blocked ``exit_id`` at ``at_s`` seconds."""

    type: Literal["reopen_exit"] = "reopen_exit"
    exit_id: str = Field(min_length=1)
    at_s: float = Field(ge=0)


class HazardZoneEvent(BaseModel):
    """Turn ``polygon`` (meter-space points) non-walkable at ``at_s``."""

    type: Literal["hazard_zone"] = "hazard_zone"
    polygon: list[Point]
    at_s: float = Field(ge=0)
    label: str = ""

    @model_validator(mode="after")
    def _check_polygon(self) -> HazardZoneEvent:
        if len(self.polygon) < 3:
            raise ValueError("hazard_zone polygon needs >= 3 points")
        return self


SimEvent = Annotated[
    BlockExitEvent | ReopenExitEvent | HazardZoneEvent,
    Field(discriminator="type"),
]

_ADAPTER = TypeAdapter(list[SimEvent])


def coerce_events(events) -> tuple[SimEvent, ...]:  # noqa: ANN001, ANN202
    """Validate dicts/models into a time-ordered tuple of events.

    The sort is stable, so events sharing ``at_s`` keep spec order.
    """
    validated = _ADAPTER.validate_python(list(events or []))
    return tuple(sorted(validated, key=lambda e: e.at_s))


def hazard_mask(
    polygon: list[Point], shape: tuple[int, int], cell_size_m: float
) -> np.ndarray:
    """Bool grid of the cells a hazard polygon makes non-walkable."""
    return rasterize_polygon(polygon, shape, cell_size_m)


def relocate_out_of_hazard(
    pos: np.ndarray,
    vel: np.ndarray,
    active: np.ndarray,
    walkable: np.ndarray,
    cell_size_m: float,
) -> int:
    """Move active agents standing on non-walkable cells. Returns count.

    Each such agent goes to the centre of its nearest walkable cell
    (see the module docstring for the exact rule); velocity is zeroed
    so the move never injects momentum.
    """
    rows = np.clip((pos[:, 1] / cell_size_m).astype(int), 0, walkable.shape[0] - 1)
    cols = np.clip((pos[:, 0] / cell_size_m).astype(int), 0, walkable.shape[1] - 1)
    stranded = active & ~walkable[rows, cols]
    n = int(stranded.sum())
    if n == 0 or not walkable.any():
        return n
    # distance_transform_edt on the non-walkable mask gives, per cell,
    # the index of the nearest zero (walkable) cell.
    _, indices = ndimage.distance_transform_edt(~walkable, return_indices=True)
    idx = np.flatnonzero(stranded)
    target_rows = indices[0][rows[idx], cols[idx]]
    target_cols = indices[1][rows[idx], cols[idx]]
    pos[idx] = np.stack(
        [(target_cols + 0.5) * cell_size_m, (target_rows + 0.5) * cell_size_m], axis=1
    )
    vel[idx] = 0.0
    return n
