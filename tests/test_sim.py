"""Simulator tests: correctness first (Helbing social-force engine)."""

from __future__ import annotations

import numpy as np
import pytest

from egresstwin.layout.schema import Exit, Layout, Obstacle, SpawnZone
from egresstwin.sim.engine import Scenario, run
from egresstwin.sim.floor_field import compute_floor_field
from egresstwin.sim.reference import compute_forces_reference
from egresstwin.sim.social_force import SFParams, compute_forces, validate_dt
from egresstwin.sim.wall_field import compute_wall_field


def room_layout(
    width_m: float,
    height_m: float,
    cell: float,
    exits: list[Exit],
    spawn_polygon: list,
    count: int,
    obstacles: list[Obstacle] | None = None,
) -> Layout:
    rows, cols = int(height_m / cell), int(width_m / cell)
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    return Layout(
        cell_size_m=cell,
        walkable_mask=walkable,
        obstacles=obstacles or [],
        exits=exits,
        spawn_zones=[SpawnZone(id="z", polygon=spawn_polygon, count=count)],
    )


def test_single_person_corridor_time() -> None:
    layout = room_layout(
        20.0,
        3.0,
        0.25,
        [Exit(id="E", segment=((20.0, 1.0), (20.0, 2.0)))],
        [(0.5, 1.0), (2.0, 1.0), (2.0, 2.0), (0.5, 2.0)],
        1,
    )
    params = SFParams(desired_speed_sd=0.0)
    result = run(layout, Scenario(params=params, max_time_s=60.0), seed=1)
    t = float(result.exit_times[0])
    expected = 18.0 / params.desired_speed_mean  # ~13.4 s of free walking
    assert np.isfinite(t)
    assert expected * 0.7 < t < expected * 1.45


def test_nobody_crosses_walls() -> None:
    wall_a = Obstacle(id="a", polygon=[(7.9, 0.0), (8.1, 0.0), (8.1, 4.0), (7.9, 4.0)])
    wall_b = Obstacle(id="b", polygon=[(7.9, 6.0), (8.1, 6.0), (8.1, 10.0), (7.9, 10.0)])
    layout = room_layout(
        16.0,
        10.0,
        0.25,
        [Exit(id="E", segment=((16.0, 4.0), (16.0, 6.0)))],
        [(0.5, 0.5), (6.0, 0.5), (6.0, 9.5), (0.5, 9.5)],
        30,
        obstacles=[wall_a, wall_b],
    )
    result = run(layout, Scenario(max_time_s=120.0), seed=2)
    assert np.isfinite(result.exit_times).all()
    walkable = layout.effective_walkable()
    for frame in result.frames[::7]:  # sampled frames
        pos = frame.astype(np.float64)
        inside = pos[(pos[:, 0] >= 0) & (pos[:, 1] >= 0)]
        rows = np.clip((inside[:, 1] / 0.25).astype(int), 0, walkable.shape[0] - 1)
        cols = np.clip((inside[:, 0] / 0.25).astype(int), 0, walkable.shape[1] - 1)
        assert walkable[rows, cols].all()


def test_same_seed_same_result() -> None:
    layout = room_layout(
        12.0,
        8.0,
        0.25,
        [Exit(id="E", segment=((12.0, 3.0), (12.0, 5.0)))],
        [(0.5, 0.5), (5.0, 0.5), (5.0, 7.5), (0.5, 7.5)],
        20,
    )
    first = run(layout, Scenario(max_time_s=120.0), seed=7)
    second = run(layout, Scenario(max_time_s=120.0), seed=7)
    assert np.array_equal(first.exit_times, second.exit_times)
    assert np.array_equal(first.frames, second.frames)


def test_symmetric_room_balanced_exits() -> None:
    layout = room_layout(
        20.0,
        12.0,
        0.25,
        [
            Exit(id="W", segment=((0.0, 5.0), (0.0, 7.0))),
            Exit(id="E", segment=((20.0, 5.0), (20.0, 7.0))),
        ],
        [(0.5, 0.5), (19.5, 0.5), (19.5, 11.5), (0.5, 11.5)],
        120,
    )
    result = run(layout, Scenario(max_time_s=180.0), seed=3)
    counts = np.bincount(result.exit_of_agent[result.exit_of_agent >= 0], minlength=2)
    assert counts.sum() == 120
    assert counts.min() >= 0.30 * 120


def test_numba_matches_reference() -> None:
    rng = np.random.default_rng(0)
    n = 10
    pos = rng.uniform(2.0, 8.0, size=(n, 2))
    vel = rng.uniform(-0.5, 0.5, size=(n, 2))
    radius = rng.uniform(0.25, 0.35, size=n)
    desired_dir = rng.normal(size=(n, 2))
    desired_dir /= np.hypot(desired_dir[:, 0], desired_dir[:, 1])[:, None]
    walkable = np.ones((20, 20), dtype=bool)
    wall = compute_wall_field(walkable, 0.5)
    wall_d, wall_gx, wall_gy = wall.sample(pos)
    params = SFParams()
    expected = compute_forces_reference(
        pos, vel, radius, np.full(n, 1.34), desired_dir, wall_d, wall_gx, wall_gy, params
    )
    got = compute_forces(
        pos,
        vel,
        radius,
        np.full(n, 1.34),
        desired_dir,
        wall_d,
        wall_gx,
        wall_gy,
        np.ones(n, dtype=bool),
        (20, 20),
        0.5,
        params,
    )
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-7)


def test_floor_field_detours_around_obstacle() -> None:
    layout = room_layout(
        10.0,
        6.0,
        0.25,
        [Exit(id="E", segment=((10.0, 2.0), (10.0, 4.0)))],
        [(0.5, 0.5), (2.0, 0.5), (2.0, 2.0), (0.5, 2.0)],
        1,
        obstacles=[Obstacle(id="o", polygon=[(4.9, 1.0), (5.1, 1.0), (5.1, 5.0), (4.9, 5.0)])],
    )
    field = compute_floor_field(layout.effective_walkable(), 0.25, layout.exits)
    behind = field.distance_at(0, np.array([[3.0, 3.0]]))[0]  # west of the wall
    assert np.isfinite(behind)
    assert behind > 7.0  # straight line to the exit would be ~7.0 via the gap detour


def test_wall_field_distance_grows_inward() -> None:
    walkable = np.zeros((21, 21), dtype=bool)
    walkable[1:-1, 1:-1] = True
    wall = compute_wall_field(walkable, 0.5)
    centre = wall.sample(np.array([[5.25, 5.25]]))[0][0]
    near = wall.sample(np.array([[5.25, 1.25]]))[0][0]
    assert centre > near > 0.0


def test_dt_stability_guard() -> None:
    validate_dt(SFParams(), 0.01)  # default dt must pass
    with pytest.raises(ValueError, match="unstable"):
        validate_dt(SFParams(), 0.05)
