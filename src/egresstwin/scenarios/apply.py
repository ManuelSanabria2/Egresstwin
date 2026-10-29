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
