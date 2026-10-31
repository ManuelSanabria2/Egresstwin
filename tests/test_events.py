"""P07 tests: timed live events (blocked exits, hazard zones)."""

from __future__ import annotations

import numpy as np

from egresstwin.agent.tools import AgentTools
from egresstwin.layout.schema import Exit, Layout, SpawnZone
from egresstwin.parallel.store import ScenarioStore
from egresstwin.parallel.sweep import run_sweep
from egresstwin.scenarios.spec import ScenarioSpec
from egresstwin.sim.engine import Scenario, run
from egresstwin.sim.events import BlockExitEvent, HazardZoneEvent, ReopenExitEvent
from egresstwin.sim.social_force import SFParams


def corridor_layout(count: int = 6, height_m: float = 4.0) -> Layout:
    """60 m corridor: west exit near the spawn, east exit far away."""
    cell = 0.25
    rows, cols = int(height_m / cell), int(60.0 / cell)
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    mid = height_m / 2.0
    return Layout(
        cell_size_m=cell,
        walkable_mask=walkable,
        exits=[
            Exit(id="W", segment=((0.0, mid - 0.5), (0.0, mid + 0.5))),
            Exit(id="E", segment=((60.0, mid - 0.5), (60.0, mid + 0.5))),
        ],
        spawn_zones=[
            SpawnZone(
                id="z",
                polygon=[
                    (24.0, mid - 1.0), (26.0, mid - 1.0),
                    (26.0, mid + 1.0), (24.0, mid + 1.0),
                ],
                count=count,
            )
        ],
    )


SLOW = SFParams(desired_speed_mean=0.6, desired_speed_sd=0.0)


def test_block_nearest_exit_at_30_reroutes_everyone_remaining() -> None:
    layout = corridor_layout()
    events = (BlockExitEvent(exit_id="W", at_s=30.0),)
    result = run(
        layout,
        Scenario(params=SLOW, max_time_s=240.0, events=events),
        seed=11,
    )
    # Slow walkers (0.6 m/s) are still inside at t = 30 s; after the
    # block every remaining person must leave through the east exit.
    assert np.isfinite(result.exit_times).all()
    east = result.exit_ids.index("E")
    west = result.exit_ids.index("W")
    assert (result.exit_of_agent == east).all()
    assert result.exits_per_step[:, west].sum() == 0
    assert (result.exit_times > 30.0).all()
    applied = [e for e in result.applied_events if e["type"] == "block_exit"]
    assert len(applied) == 1
    assert applied[0]["exit_id"] == "W"
    # Blocking only drops the exit slice: no Dijkstra recompute at all.
    assert applied[0]["recomputed_exit_ids"] == []


def test_reopen_recomputes_only_that_exit() -> None:
    layout = corridor_layout(count=3)
    events = (
        BlockExitEvent(exit_id="W", at_s=1.0),
        ReopenExitEvent(exit_id="W", at_s=3.0),
    )
    result = run(layout, Scenario(max_time_s=240.0, events=events), seed=5)
    assert np.isfinite(result.exit_times).all()
    by_type = {e["type"]: e for e in result.applied_events}
    assert by_type["block_exit"]["recomputed_exit_ids"] == []
    assert by_type["reopen_exit"]["recomputed_exit_ids"] == ["W"]


def test_hazard_zone_disconnection_warns() -> None:
    # Both exits sit on the east wall; a hazard band across the whole
    # corridor at t = 1 s strands the west side with no route out.
    # That must warn, never pass silently.
    cell = 0.25
    rows, cols = int(4.0 / cell), int(40.0 / cell)
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    layout = Layout(
        cell_size_m=cell,
        walkable_mask=walkable,
        exits=[
            Exit(id="E1", segment=((40.0, 0.5), (40.0, 1.5))),
            Exit(id="E2", segment=((40.0, 2.5), (40.0, 3.5))),
        ],
        spawn_zones=[
            SpawnZone(
                id="z",
                polygon=[(4.0, 1.0), (6.0, 1.0), (6.0, 3.0), (4.0, 3.0)],
                count=4,
            )
        ],
    )
    hazard = HazardZoneEvent(
        polygon=[(19.0, 0.0), (21.0, 0.0), (21.0, 4.0), (19.0, 4.0)], at_s=1.0
    )
    result = run(layout, Scenario(max_time_s=20.0, events=(hazard,)), seed=3)
    assert result.warnings, "hazard disconnection must never be silent"
    assert any("hazard" in w for w in result.warnings)
    applied = [e for e in result.applied_events if e["type"] == "hazard_zone"]
    assert len(applied) == 1
    assert applied[0]["disconnected_cells"] > 0
    # Only the still-open exits got their floor field recomputed.
    assert sorted(applied[0]["recomputed_exit_ids"]) == ["E1", "E2"]


def test_hazard_zone_relocates_people_inside() -> None:
    layout = corridor_layout(count=5, height_m=6.0)
    # Hazard covers the spawn zone itself shortly after the start,
    # leaving wide passages on both sides.
    hazard = HazardZoneEvent(
        polygon=[(23.0, 1.5), (27.0, 1.5), (27.0, 4.5), (23.0, 4.5)], at_s=0.1
    )
    result = run(layout, Scenario(max_time_s=240.0, events=(hazard,)), seed=4)
    applied = [e for e in result.applied_events if e["type"] == "hazard_zone"]
    assert applied[0]["relocated_agents"] == 5
    # No recorded frame at or after the event keeps anyone in the zone.
    first_after = int(np.ceil(0.1 / result.frame_dt))
    for frame in result.frames[first_after:]:
        pos = frame.astype(np.float64)
        inside = pos[(pos[:, 0] >= 23.0) & (pos[:, 0] <= 27.0)
                     & (pos[:, 1] >= 1.5) & (pos[:, 1] <= 4.5)]
        assert len(inside) == 0
    assert np.isfinite(result.exit_times).all()


def test_same_spec_and_seed_same_result_with_events() -> None:
    layout = corridor_layout(count=5)
    spec = ScenarioSpec(
        name="block-west-late",
        max_time_s=240.0,
        events=[BlockExitEvent(exit_id="W", at_s=30.0)],
    )
    store_a, store_b = ScenarioStore(), ScenarioStore()
    rec_a = run_sweep(layout, [spec], replicas=1, seed=42, store=store_a)[0]
    rec_b = run_sweep(layout, [spec], replicas=1, seed=42, store=store_b)[0]
    assert rec_a.events == rec_b.events == [
        {"type": "block_exit", "exit_id": "W", "at_s": 30.0}
    ]
    assert rec_a.replicas[0].t95_s == rec_b.replicas[0].t95_s
    # Direct engine determinism under the same spec and seed.
    scenario = Scenario(max_time_s=240.0, events=tuple(spec.events))
    first = run(layout, scenario, seed=42)
    second = run(layout, scenario, seed=42)
    assert np.array_equal(first.exit_times, second.exit_times)
    assert np.array_equal(first.frames, second.frames)


def test_simulate_once_animation_carries_event() -> None:
    from egresstwin.app.api import SimulateRequest, simulate_once

    req = SimulateRequest(events=[{"type": "block_exit", "exit_id": "W", "at_s": 1.0}])
    assert req.events[0]["exit_id"] == "W"
    anim = simulate_once(
        corridor_layout(count=3), seed=7, max_time_s=120.0,
        events=tuple(req.events),
    )
    assert anim["events"][0]["type"] == "block_exit"
    assert anim["events"][0]["exit_id"] == "W"
    assert anim["warnings"] == []


def test_timed_stress_test_blocks_from_given_time() -> None:
    layout = corridor_layout(count=4)
    tools = AgentTools(layout, store=ScenarioStore(), seed=42, default_max_time_s=240.0)
    tools.get_baseline(replicas=1)
    out = tools.stress_test("baseline", replicas=1, block_at_s=30.0)
    entries = {e["blocked_exit_id"]: e for e in out["n_minus_1"]}
    assert entries["W"]["block_at_s"] == 30.0
    assert entries["W"]["t95_s"]["mean"] > 0
    # The default (from t = 0) behaviour is unchanged and still stored
    # under the key the verifier looks up.
    out0 = tools.stress_test("baseline", replicas=1)
    assert all("block_at_s" not in e or e["block_at_s"] is None for e in out0["n_minus_1"])
    assert ("baseline", "W") in tools.store.stress
