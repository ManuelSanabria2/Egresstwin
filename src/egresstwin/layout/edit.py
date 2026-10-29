"""Manual layout corrections used by the UI (P06).

Every function is pure: it returns a new Layout and never mutates
its input, so a bad AI read can always be fixed by hand and the
pre-edit layout stays available for undo.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from egresstwin.layout.schema import Exit, Layout, Obstacle, Point


def _copy(layout: Layout) -> Layout:
    clone = layout.model_copy(deep=True)
    clone.walkable_mask = np.array(layout.walkable_mask, dtype=bool, copy=True)
    clone.obstacles = [
        Obstacle(
            id=o.id,
            label=o.label,
            removable=o.removable,
            polygon=list(o.polygon) if o.polygon is not None else None,
            mask=np.array(o.mask, dtype=bool, copy=True) if o.mask is not None else None,
        )
        for o in layout.obstacles
    ]
    return clone


def _next_id(layout: Layout, prefix: str) -> str:
    existing = {e.id for e in layout.exits} | {o.id for o in layout.obstacles}
    n = 1
    while f"{prefix}-{n}" in existing:
        n += 1
    return f"{prefix}-{n}"


def add_exit(layout: Layout, segment: tuple[Point, Point], *, exit_id: str | None = None) -> Layout:
    """Return a copy with one exit added."""
    out = _copy(layout)
    new_id = exit_id or _next_id(out, "exit")
    if any(e.id == new_id for e in out.exits):
        raise ValueError(f"exit id {new_id!r} already exists")
    out.exits = [*out.exits, Exit(id=new_id, segment=segment)]
    return out


def remove_exit(layout: Layout, exit_id: str) -> Layout:
    """Return a copy without the named exit."""
    if not any(e.id == exit_id for e in layout.exits):
        raise ValueError(f"no exit with id {exit_id!r}")
    out = _copy(layout)
    out.exits = [e for e in out.exits if e.id != exit_id]
    return out


def add_obstacle(
    layout: Layout,
    polygon: list[Point],
    *,
    label: str = "obstacle",
    obstacle_id: str | None = None,
    removable: bool = False,
) -> Layout:
    """Return a copy with one polygon obstacle added (and blocked off)."""
    out = _copy(layout)
    obs = Obstacle(
        id=obstacle_id or _next_id(out, "obstacle"),
        label=label,
        removable=removable,
        polygon=polygon,
    )
    out.obstacles = [*out.obstacles, obs]
    out.walkable_mask = out.walkable_mask & ~obs.rasterized(out.shape, out.cell_size_m)
    return out


def remove_obstacle(layout: Layout, obstacle_id: str) -> Layout:
    """Return a copy without the named obstacle (its cells stay as-is)."""
    if not any(o.id == obstacle_id for o in layout.obstacles):
        raise ValueError(f"no obstacle with id {obstacle_id!r}")
    out = _copy(layout)
    out.obstacles = [o for o in out.obstacles if o.id != obstacle_id]
    return out


def mark_obstacle_walkable(layout: Layout, obstacle_id: str) -> Layout:
    """Return a copy where the named obstacle no longer blocks movement."""
    target = next((o for o in layout.obstacles if o.id == obstacle_id), None)
    if target is None:
        raise ValueError(f"no obstacle with id {obstacle_id!r}")
    out = _copy(layout)
    freed = target.rasterized(out.shape, out.cell_size_m)
    out.obstacles = [o for o in out.obstacles if o.id != obstacle_id]
    out.walkable_mask = out.walkable_mask | freed
    return out


def paint_wall(
    layout: Layout, points: list[Point], *, width_m: float = 0.2, wall_id: str | None = None
) -> Layout:
    """Return a copy with a wall stroked along a meter-space polyline."""
    if len(points) < 2:
        raise ValueError("paint_wall needs at least 2 points")
    if width_m <= 0:
        raise ValueError("wall width must be > 0 m")
    out = _copy(layout)
    rows, cols = out.shape
    img = Image.new("1", (cols, rows), 0)
    draw = ImageDraw.Draw(img)
    pts = [(x / out.cell_size_m, y / out.cell_size_m) for x, y in points]
    width_cells = max(1, round(width_m / out.cell_size_m))
    draw.line(pts, fill=1, width=width_cells, joint="curve")
    for x, y in pts:  # round caps at the vertices
        r = width_cells / 2.0
        draw.ellipse([x - r, y - r, x + r, y + r], fill=1)
    stroke = np.array(img, dtype=bool)
    new_id = wall_id or _next_id(out, "wall")
    if any(o.id == new_id for o in out.obstacles):
        raise ValueError(f"obstacle id {new_id!r} already exists")
    out.obstacles = [
        *out.obstacles,
        Obstacle(id=new_id, label="wall", removable=False, mask=stroke),
    ]
    out.walkable_mask = out.walkable_mask & ~stroke
    return out
