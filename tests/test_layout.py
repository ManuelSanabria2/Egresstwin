"""Layout schema and validation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from egresstwin.layout.schema import Exit, Layout, Obstacle, SpawnZone
from egresstwin.layout.validate import LayoutValidationError, validate_layout


def make_room(
    width_m: float = 10.0,
    height_m: float = 6.0,
    cell: float = 0.25,
    exits: list[Exit] | None = None,
    obstacles: list[Obstacle] | None = None,
) -> Layout:
    """Rectangular room: one-cell wall border, interior walkable."""
    rows, cols = int(height_m / cell), int(width_m / cell)
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    for obs in obstacles or []:
        walkable &= ~obs.rasterized((rows, cols), cell)
    return Layout(
        cell_size_m=cell,
        walkable_mask=walkable,
        obstacles=obstacles or [],
        exits=exits
        if exits is not None
        else [Exit(id="E", segment=((width_m, 2.0), (width_m, 4.0)))],
        spawn_zones=[
            SpawnZone(id="z", polygon=[(0.5, 0.5), (3.0, 0.5), (3.0, 3.0), (0.5, 3.0)], count=5)
        ],
    )


def test_npz_roundtrip(tmp_path: Path) -> None:
    layout = make_room(
        obstacles=[
            Obstacle(
                id="o1", label="block", polygon=[(4.0, 2.0), (5.0, 2.0), (5.0, 3.0), (4.0, 3.0)]
            )
        ]
    )
    path = tmp_path / "layout.npz"
    layout.save_npz(path)
    loaded = Layout.load_npz(path)
    assert np.array_equal(loaded.walkable_mask, layout.walkable_mask)
    assert loaded.exits[0].width_m == pytest.approx(2.0)
    assert loaded.obstacles[0].polygon is not None and len(loaded.obstacles[0].polygon) == 4
    assert loaded.cell_size_m == layout.cell_size_m


def test_no_exit_fails_clearly() -> None:
    layout = make_room(exits=[])
    with pytest.raises(LayoutValidationError, match="no exits"):
        validate_layout(layout)


def test_unreachable_area_fails() -> None:
    wall = Obstacle(id="w", polygon=[(4.9, 0.0), (5.1, 0.0), (5.1, 6.0), (4.9, 6.0)])
    layout = make_room(obstacles=[wall])
    with pytest.raises(LayoutValidationError, match="cannot reach any exit"):
        validate_layout(layout)


def test_interior_exit_fails() -> None:
    layout = make_room(exits=[Exit(id="X", segment=((5.0, 2.0), (5.0, 4.0)))])
    with pytest.raises(LayoutValidationError, match="not on the boundary"):
        validate_layout(layout)


def test_narrow_exit_warns() -> None:
    layout = make_room(exits=[Exit(id="E", segment=((10.0, 2.8), (10.0, 3.2)))])
    warnings = validate_layout(layout)
    assert any("0.40 m wide" in w for w in warnings)


def test_valid_room_passes() -> None:
    assert validate_layout(make_room()) == []
