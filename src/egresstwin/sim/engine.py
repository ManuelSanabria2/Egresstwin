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
from egresstwin.sim.events import (
    BlockExitEvent,
    HazardZoneEvent,
    ReopenExitEvent,
    SimEvent,
    coerce_events,
    hazard_mask,
    relocate_out_of_hazard,
)
from egresstwin.sim.floor_field import FloorField, single_exit_field
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
    events: tuple[SimEvent, ...] = ()  # timed live events (P07)
    # Pre-movement (reaction) times: uniform per agent in
    # [pre_movement_min_s, pre_movement_max_s]; both zero (default)
    # means everyone starts at t = 0, exactly as before.
    pre_movement_min_s: float = 0.0
    pre_movement_max_s: float = 0.0


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
    applied_events: list = field(default_factory=list)  # P07 event log
    warnings: list = field(default_factory=list)  # P07 hazard warnings
    reaction_times_s: np.ndarray = field(  # per-agent pre-movement times
        default_factory=lambda: np.zeros(0)
    )


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
    events = coerce_events(scenario.events)
    known_exit_ids = {e.id for e in layout.exits}
    for ev in events:
        if isinstance(ev, (BlockExitEvent, ReopenExitEvent)) and ev.exit_id not in known_exit_ids:
            raise ValueError(f"event references unknown exit {ev.exit_id!r}")
    base_walkable = layout.effective_walkable(scenario.removed_obstacle_ids)
    walkable = base_walkable.copy()
    # The exit universe never changes: a timed block only closes an
    # exit for a while, so exits_per_step keeps one column per exit.
    exits = [e for e in layout.exits if e.id not in scenario.blocked_exit_ids]
    if not exits:
        raise ValueError("scenario blocks every exit: nobody can leave")
    # Exits are openings: cells the segment passes through become
    # walkable, otherwise the wall force seals its own exit.
    rows_g, cols_g = np.mgrid[0 : layout.shape[0], 0 : layout.shape[1]]
    centres = np.stack(
        [(cols_g + 0.5) * layout.cell_size_m, (rows_g + 0.5) * layout.cell_size_m], axis=-1
    ).reshape(-1, 2)
    openings: dict[str, np.ndarray] = {}
    for ex in exits:
        opening = _segment_distance(centres, ex.segment) <= 0.5 * layout.cell_size_m + 1e-9
        openings[ex.id] = opening.reshape(layout.shape)
        walkable.reshape(-1)[opening] = True
    cell = layout.cell_size_m
    open_ids: set[str] = {e.id for e in exits}
    hazard_cells = np.zeros(layout.shape, dtype=bool)
    # Floor field over the universe; a blocked exit's slice is dropped
    # to inf (no Dijkstra), a reopened one is recomputed alone (P07).
    distances = np.full((len(exits), *layout.shape), np.inf)
    dir_x = np.zeros((len(exits), *layout.shape))
    dir_y = np.zeros((len(exits), *layout.shape))
    for k, ex in enumerate(exits):
        distances[k], dir_x[k], dir_y[k] = single_exit_field(walkable, cell, ex)
    floor = FloorField(
        distances=distances, dir_x=dir_x, dir_y=dir_y,
        exit_ids=[e.id for e in exits], cell_size_m=cell,
    )
    exit_index = {e.id: k for k, e in enumerate(exits)}
    wall = compute_wall_field(walkable, cell)
    rng = np.random.default_rng(seed)
    pos = _spawn(layout, walkable, rng)
    n = len(pos)
    radius = rng.uniform(params.radius_min, params.radius_max, size=n)
    desired_speed = np.clip(
        rng.normal(params.desired_speed_mean, params.desired_speed_sd, size=n), 0.4, 2.5
    )
    vel = np.zeros((n, 2))
    active = np.ones(n, dtype=bool)
    if scenario.pre_movement_max_s > 0:
        if scenario.pre_movement_min_s < 0 or (
            scenario.pre_movement_min_s > scenario.pre_movement_max_s
        ):
            raise ValueError("pre-movement range must satisfy 0 <= min <= max")
        reaction = rng.uniform(
            scenario.pre_movement_min_s, scenario.pre_movement_max_s, size=n
        )
    else:
        reaction = np.zeros(n)

    def reassign() -> np.ndarray:
        """Geodesically nearest open exit per agent (-1: none reachable)."""
        if not open_ids:
            return np.full(n, -1, dtype=np.int64)
        open_k = sorted(exit_index[eid] for eid in open_ids)
        d = np.stack([floor.distance_at(k, pos) for k in open_k])
        choice = np.argmin(d, axis=0)
        best = d[choice, np.arange(n)]
        return np.where(np.isfinite(best), np.array(open_k)[choice], -1).astype(np.int64)

    # Nearest exit (geodesic) at spawn.
    assigned = reassign()
    if (assigned < 0).any():
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
    applied_events: list[dict] = []
    warnings: list[str] = []
    event_idx = 0

    def apply_event(ev: SimEvent, at_step: int) -> None:
        """Apply one timed event; recompute only affected floor slices."""
        nonlocal wall, assigned, hazard_cells, walkable
        t_now = round(at_step * scenario.dt, 6)
        record: dict = {
            "type": ev.type, "at_s": float(ev.at_s), "applied_at_s": t_now,
            "recomputed_exit_ids": [], "relocated_agents": 0,
            "disconnected_cells": 0,
        }
        if isinstance(ev, (BlockExitEvent, ReopenExitEvent)):
            record["exit_id"] = ev.exit_id
            if ev.exit_id not in exit_index:
                record["detail"] = "exit is statically blocked; event ignored"
                applied_events.append(record)
                return
            k = exit_index[ev.exit_id]
            ex = exits[k]
            if isinstance(ev, BlockExitEvent):
                if ev.exit_id in open_ids:
                    open_ids.discard(ev.exit_id)
                    floor.distances[k] = np.inf
                    floor.dir_x[k] = 0.0
                    floor.dir_y[k] = 0.0
                    shared = np.zeros(layout.shape, dtype=bool)
                    for other in open_ids:
                        shared |= openings[other]
                    seal = openings[ev.exit_id] & ~shared
                    walkable[seal] = (base_walkable & ~hazard_cells)[seal]
                    wall = compute_wall_field(walkable, cell)
            else:
                if ev.exit_id not in open_ids:
                    open_ids.add(ev.exit_id)
                    walkable |= openings[ev.exit_id] & ~hazard_cells
                    try:
                        floor.distances[k], floor.dir_x[k], floor.dir_y[k] = (
                            single_exit_field(walkable, cell, ex)
                        )
                    except ValueError:
                        floor.distances[k] = np.inf
                        floor.dir_x[k] = 0.0
                        floor.dir_y[k] = 0.0
                    record["recomputed_exit_ids"] = [ev.exit_id]
                    wall = compute_wall_field(walkable, cell)
        elif isinstance(ev, HazardZoneEvent):
            record["polygon"] = [[float(x), float(y)] for x, y in ev.polygon]
            if ev.label:
                record["label"] = ev.label
            mask = hazard_mask(ev.polygon, layout.shape, cell)
            hazard_cells |= mask
            walkable &= ~mask
            record["relocated_agents"] = relocate_out_of_hazard(
                pos, vel, active, walkable, cell
            )
            recomputed: list[str] = []
            for eid in sorted(open_ids, key=lambda e: exit_index[e]):
                k = exit_index[eid]
                try:
                    floor.distances[k], floor.dir_x[k], floor.dir_y[k] = (
                        single_exit_field(walkable, cell, exits[k])
                    )
                except ValueError:
                    floor.distances[k] = np.inf
                    floor.dir_x[k] = 0.0
                    floor.dir_y[k] = 0.0
                recomputed.append(eid)
            record["recomputed_exit_ids"] = recomputed
            wall = compute_wall_field(walkable, cell)
            if open_ids:
                open_k = [exit_index[e] for e in open_ids]
                reachable = np.isfinite(floor.distances[open_k]).any(axis=0)
            else:
                reachable = np.zeros(layout.shape, dtype=bool)
            disconnected = int((walkable & ~reachable).sum())
            record["disconnected_cells"] = disconnected
            if disconnected:
                msg = (
                    f"hazard_zone at t={ev.at_s:g} s disconnects {disconnected} "
                    "walkable cell(s) from every open exit; people there cannot leave"
                )
                warnings.append(msg)
                record["warning"] = msg
        assigned = np.where(active, reassign(), assigned)
        applied_events.append(record)

    step = 0
    while step < max_steps and active.any():
        while event_idx < len(events) and events[event_idx].at_s <= step * scenario.dt + 1e-9:
            apply_event(events[event_idx], step)
            event_idx += 1
        if step % steps_per_frame == 0:
            frame = pos.astype(np.float16)
            frame[~active] = np.float16(-10.0)  # sentinel: outside the grid
            frames.append(frame)
        rows = np.clip((pos[:, 1] / layout.cell_size_m).astype(int), 0, layout.shape[0] - 1)
        cols = np.clip((pos[:, 0] / layout.cell_size_m).astype(int), 0, layout.shape[1] - 1)
        np.add.at(density_accum, (rows[active], cols[active]), 1.0)
        # People whose reaction time has not elapsed wait at spawn:
        # they do not move, cannot be captured, and (documented
        # simplification) do not yet exert or receive social forces.
        started = step * scenario.dt >= reaction
        if reconsider_every and step > 0 and step % reconsider_every == 0 and open_ids:
            open_k = sorted(exit_index[e] for e in open_ids)
            d_now = np.stack([floor.distance_at(k, pos) for k in open_k])
            valid = active & started & (assigned >= 0)
            assigned_count = np.bincount(
                assigned[valid], minlength=len(exits)
            )[open_k]
            penalty = 0.02 * assigned_count  # seconds per queued person, per exit
            cost = d_now / desired_speed[None, :] + penalty[:, None]
            choice = np.array(open_k)[np.argmin(cost, axis=0)]
            assigned = np.where(
                active, np.where(np.isfinite(d_now).any(axis=0), choice, -1), assigned
            )
        desired_dir = np.zeros((n, 2))
        for k in range(len(exits)):
            sel = active & started & (assigned == k)
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
                active & started,
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
        # Exit capture per assigned exit (open exits only).
        for k, ex in enumerate(exits):
            if ex.id not in open_ids:
                continue
            sel = np.flatnonzero(active & started & (assigned == k))
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
        applied_events=applied_events,
        warnings=warnings,
        reaction_times_s=reaction,
    )
