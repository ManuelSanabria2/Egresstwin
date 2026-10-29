"""Apply scenario modifications to a Layout (pure, feasibility checked)."""

from __future__ import annotations

import numpy as np
from pydantic import TypeAdapter

from egresstwin.layout import edit
from egresstwin.layout.schema import Layout
from egresstwin.layout.validate import LayoutValidationError, validate_layout
from egresstwin.scenarios.spec import (
    AddExitModification,
    Modification,
    RemoveObstacleModification,
    WidenExitModification,
)

_ADAPTER = TypeAdapter(list[Modification])


def _carve_exit_throat(layout: Layout, segment, depth_m: float = 0.6) -> Layout:
    """Open a walkable throat outward from a new exit segment.

    Without it the wall field traps agents at the wall face and the
    engine's exit capture never triggers (same fix as perception).
    Outward is the side away from the walkable centroid.
    """
    (x1, y1), (x2, y2) = segment
    out = layout.model_copy(deep=True)
    walk = np.array(layout.walkable_mask, dtype=bool, copy=True)
    rows, cols = walk.shape
    cell = layout.cell_size_m
    ys, xs = np.where(layout.effective_walkable())
    cx, cy = (xs.mean() + 0.5) * cell, (ys.mean() + 0.5) * cell
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length <= 0:
        return layout
    nx, ny = -dy / length, dx / length
    if (cx - mx) * nx + (cy - my) * ny > 0:
        nx, ny = -nx, -ny  # normal pointing outward, away from the room
    cols_g, rows_g = np.meshgrid(np.arange(cols), np.arange(rows))
    px = (cols_g + 0.5) * cell - x1
    py = (rows_g + 0.5) * cell - y1
    t = (px * dx + py * dy) / (length * length)
    along = px * nx + py * ny
    throat = (t >= 0.0) & (t <= 1.0) & (along >= -cell) & (along <= depth_m)
    walk[throat] = True
    out.walkable_mask = walk
    return out


def apply_modifications(layout: Layout, modifications: list[Modification]) -> Layout:
    """Return a modified copy of layout. Raises ValueError if infeasible."""
    modifications = _ADAPTER.validate_python(list(modifications))
    out = layout
    for mod in modifications:
        if isinstance(mod, RemoveObstacleModification):
            target = next((o for o in out.obstacles if o.id == mod.obstacle_id), None)
            if target is None:
                raise ValueError(f"cannot remove unknown obstacle {mod.obstacle_id!r}")
            if not target.removable:
                raise ValueError(f"obstacle {mod.obstacle_id!r} is not removable")
            out = edit.mark_obstacle_walkable(out, mod.obstacle_id)
        elif isinstance(mod, AddExitModification):
            (x1, y1), (x2, y2) = mod.segment
            if not (
                0 <= x1 <= out.width_m and 0 <= x2 <= out.width_m
                and 0 <= y1 <= out.height_m and 0 <= y2 <= out.height_m
            ):
                raise ValueError("new exit segment lies outside the layout")
            out = edit.add_exit(out, mod.segment, exit_id=mod.exit_id)
            out = _carve_exit_throat(out, mod.segment)
        elif isinstance(mod, WidenExitModification):
            target = next((e for e in out.exits if e.id == mod.exit_id), None)
            if target is None:
                raise ValueError(f"cannot widen unknown exit {mod.exit_id!r}")
            if mod.width_m <= target.width_m:
                raise ValueError(
                    f"widen_exit width {mod.width_m:g} m does not exceed "
                    f"current width {target.width_m:.2f} m"
                )
            (x1, y1), (x2, y2) = target.segment
            mx, my = target.midpoint
            length = target.width_m
            ux, uy = (x2 - x1) / length, (y2 - y1) / length
            half = mod.width_m / 2.0
            segment = ((mx - ux * half, my - uy * half), (mx + ux * half, my + uy * half))
            if not (
                0 <= segment[0][0] <= out.width_m and 0 <= segment[1][0] <= out.width_m
                and 0 <= segment[0][1] <= out.height_m and 0 <= segment[1][1] <= out.height_m
            ):
                raise ValueError("widened exit segment lies outside the layout")
            widened = out.model_copy(deep=True)
            widened.walkable_mask = np.array(out.walkable_mask, copy=True)
            widened.exits = [
                type(e)(id=e.id, segment=segment) if e.id == mod.exit_id else e
                for e in out.exits
            ]
            out = widened
        else:  # pragma: no cover - discriminated union is exhaustive
            raise ValueError(f"unknown modification: {mod!r}")
    try:
        validate_layout(out)
    except LayoutValidationError as exc:
        raise ValueError(f"modified layout is not simulable: {exc}") from exc
    return out
