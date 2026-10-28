"""Simulation engine: run(layout, scenario, seed) -> SimResult.

Deterministic: one NumPy Generator seeded per run, serial Numba
kernel, stable cell-list ordering. Same seed gives the same result.
Exit choice is the geodesically nearest exit at spawn; an optional
reconsideration interval (off by default) re-scores exits by
distance plus a congestion penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from egresstwin.layout.schema import Layout, rasterize_polygon
from egresstwin.layout.validate import validate_layout
from egresstwin.sim.floor_field import compute_floor_field
from egresstwin.sim.social_force import SFParams, compute_forces, validate_dt
from egresstwin.sim.wall_field import compute_wall_field


@dataclass(frozen=True)
class Scenario:
    """Run-time options (the scenario DSL arrives in a later prompt)."""

    dt: float = 0.01
    substeps: int = 5  # force-integration substeps per dt (stiff contacts)
    max_time_s: float = 900.0
    blocked_exit_ids: frozenset[str] = frozenset()
    removed_obstacle_ids: frozenset[str] = frozenset()
    reconsider_interval_s: float | None = None  # None = keep spawn choice
    params: SFParams = field(default_factory=SFParams)


@dataclass
class SimResult:
    """Everything metrics need, nothing they do not."""

    exit_times: np.ndarray  # (N,) seconds, inf if never exited
    exit_of_agent: np.ndarray  # (N,) index into exit_ids, -1 if none
    exit_ids: list[str]
    exits_per_step: np.ndarray  # (steps, n_exits) people leaving per step
    frames: np.ndarray  # (n_frames, N, 2) float16 positions at frame_dt
    frame_dt: float
    density_accum: np.ndarray  # (rows, cols) person-steps per cell
    n_steps: int
    dt: float
    total_time_s: float
    cell_size_m: float
    grid_shape: tuple[int, int]
    seed: int


def _segment_distance(pos: np.ndarray, segment: tuple) -> np.ndarray:
    """Distance from points (N, 2) to a segment ((x1,y1),(x2,y2))."""
    (x1, y1), (x2, y2) = segment
    px, py = pos[:, 0] - x1, pos[:, 1] - y1
    dx, dy = x2 - x1, y2 - y1
    length2 = dx * dx + dy * dy
    t = np.clip((px * dx + py * dy) / length2, 0.0, 1.0) if length2 > 0 else np.zeros(len(pos))
    return np.hypot(px - t * dx, py - t * dy)


def _spawn(layout: Layout, walkable: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Spawn positions (N, 2) sampled per zone with the run RNG."""
    positions = []
    for zone in layout.spawn_zones:
        zone_mask = rasterize_polygon(zone.polygon, layout.shape, layout.cell_size_m) & walkable
        cells = np.argwhere(zone_mask)  # (n, 2) as (row, col)
        if len(cells) == 0:
            raise ValueError(f"spawn zone {zone.id!r} has no walkable cells")
        n = zone.planned_count()
        pick = cells[rng.integers(0, len(cells), size=n)]
        jitter = rng.uniform(-0.5, 0.5, size=(n, 2)) * layout.cell_size_m
        pos = (pick[:, ::-1] + 0.5) * layout.cell_size_m + jitter  # (col,row)->(x,y)
        positions.append(pos)
    if not positions:
        raise ValueError("layout has no spawn zones: nobody to simulate")
    return np.concatenate(positions)


def run(layout: Layout, scenario: Scenario, seed: int) -> SimResult:
    """Simulate evacuation of a layout under a scenario."""
    params = scenario.params
    validate_dt(params, scenario.dt)
    validate_layout(layout)
    walkable = layout.effective_walkable(scenario.removed_obstacle_ids)
    exits = [e for e in layout.exits if e.id not in scenario.blocked_exit_ids]
    if not exits:
        raise ValueError("scenario blocks every exit: nobody can leave")
    # Exits are openings: cells the segment passes through become
    # walkable, otherwise the wall force seals its own exit.
    rows_g, cols_g = np.mgrid[0 : layout.shape[0], 0 : layout.shape[1]]
    centres = np.stack(
        [(cols_g + 0.5) * layout.cell_size_m, (rows_g + 0.5) * layout.cell_size_m], axis=-1
    ).reshape(-1, 2)
    for ex in exits:
        opening = _segment_distance(centres, ex.segment) <= 0.5 * layout.cell_size_m + 1e-9
        walkable.reshape(-1)[opening] = True
    floor = compute_floor_field(walkable, layout.cell_size_m, exits)
    wall = compute_wall_field(walkable, layout.cell_size_m)
    rng = np.random.default_rng(seed)
    pos = _spawn(layout, walkable, rng)
    n = len(pos)
    radius = rng.uniform(params.radius_min, params.radius_max, size=n)
    desired_speed = np.clip(
        rng.normal(params.desired_speed_mean, params.desired_speed_sd, size=n), 0.4, 2.5
    )
    vel = np.zeros((n, 2))
    active = np.ones(n, dtype=bool)
    # Nearest exit (geodesic) at spawn.
    dists = np.stack([floor.distance_at(k, pos) for k in range(len(exits))])
    assigned = np.argmin(dists, axis=0).astype(np.int64)
    if not np.isfinite(dists[assigned, np.arange(n)]).all():
        raise ValueError("some spawn positions cannot reach any unblocked exit")
    exit_times = np.full(n, np.inf)
    exit_of = np.full(n, -1, dtype=np.int64)
    capture = max(1.5 * layout.cell_size_m, 0.2)
    steps_per_frame = max(1, round(1.0 / (5.0 * scenario.dt)))
    max_steps = int(scenario.max_time_s / scenario.dt)
    exits_per_step = np.zeros((max_steps, len(exits)), dtype=np.int32)
    density_accum = np.zeros(layout.shape, dtype=np.float64)
    frames: list[np.ndarray] = []
    reconsider_every = (
        None
        if scenario.reconsider_interval_s is None
        else max(1, round(scenario.reconsider_interval_s / scenario.dt))
    )
    step = 0
    while step < max_steps and active.any():
        if step % steps_per_frame == 0:
            frame = pos.astype(np.float16)
            frame[~active] = np.float16(-10.0)  # sentinel: outside the grid
            frames.append(frame)
        rows = np.clip((pos[:, 1] / layout.cell_size_m).astype(int), 0, layout.shape[0] - 1)
        cols = np.clip((pos[:, 0] / layout.cell_size_m).astype(int), 0, layout.shape[1] - 1)
        np.add.at(density_accum, (rows[active], cols[active]), 1.0)
        if reconsider_every and step > 0 and step % reconsider_every == 0:
            d_now = np.stack([floor.distance_at(k, pos) for k in range(len(exits))])
            assigned_count = np.bincount(assigned[active], minlength=len(exits))
            penalty = 0.02 * assigned_count  # seconds per queued person, per exit
            cost = d_now / desired_speed[None, :] + penalty[:, None]
            assigned = np.where(active, np.argmin(cost, axis=0), assigned)
        desired_dir = np.zeros((n, 2))
        for k in range(len(exits)):
            sel = active & (assigned == k)
            if sel.any():
                desired_dir[sel] = floor.direction_at(k, pos[sel])
        wall_d, wall_gx, wall_gy = wall.sample(pos)
        # Semi-implicit Euler in substeps: the k = 1.2e5 contact spring
        # is stiff at dt = 0.01 in dense overlap (velocity flips each
        # step and jams never resolve), so forces are integrated at
        # dt/substeps while outputs stay at dt. Deterministic: the
        # substep count is fixed per scenario.
        sub_dt = scenario.dt / scenario.substeps
        for _ in range(scenario.substeps):
            forces = compute_forces(
                pos,
                vel,
                radius,
                desired_speed,
                desired_dir,
                wall_d,
                wall_gx,
                wall_gy,
                active,
                layout.shape,
                layout.cell_size_m,
                params,
            )
            vel = vel + forces / params.mass * sub_dt
            speed = np.hypot(vel[:, 0], vel[:, 1])
            cap = params.max_speed_factor * desired_speed
            over = (speed > cap) & active
            vel[over] *= (cap[over] / speed[over])[:, None]
            new_pos = pos + vel * sub_dt
            # Wall guard: a step into a non-walkable cell is cancelled.
            nr = np.clip((new_pos[:, 1] / layout.cell_size_m).astype(int), 0, layout.shape[0] - 1)
            nc = np.clip((new_pos[:, 0] / layout.cell_size_m).astype(int), 0, layout.shape[1] - 1)
            blocked = active & ~walkable[nr, nc]
            new_pos[blocked] = pos[blocked]
            vel[blocked] = 0.0
            pos = new_pos
            wall_d, wall_gx, wall_gy = wall.sample(pos)
        # Exit capture per assigned exit.
        for k, ex in enumerate(exits):
            sel = np.flatnonzero(active & (assigned == k))
            if len(sel) == 0:
                continue
            near = sel[_segment_distance(pos[sel], ex.segment) < capture]
            if len(near):
                active[near] = False
                exit_times[near] = (step + 1) * scenario.dt
                exit_of[near] = k
                exits_per_step[step, k] += len(near)
        step += 1
    total = (
        float(exit_times[np.isfinite(exit_times)].max())
        if np.isfinite(exit_times).any()
        else float(step * scenario.dt)
    )
    return SimResult(
        exit_times=exit_times,
        exit_of_agent=exit_of,
        exit_ids=[e.id for e in exits],
        exits_per_step=exits_per_step[:step],
        frames=np.array(frames, dtype=np.float16),
        frame_dt=steps_per_frame * scenario.dt,
        density_accum=density_accum,
        n_steps=step,
        dt=scenario.dt,
        total_time_s=total,
        cell_size_m=layout.cell_size_m,
        grid_shape=layout.shape,
        seed=seed,
    )
