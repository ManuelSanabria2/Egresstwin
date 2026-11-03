"""Agent tests: scripted mock model, tools, verifier, trace.

The fake sweep runner returns deterministic replica metrics (better
for widened exits), so the whole loop runs without the simulator.
One integration test at the bottom uses the real engine.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from egresstwin.agent.loop import AgentLoop
from egresstwin.agent.report import AgentReport
from egresstwin.agent.tools import AgentTools, RunScenariosArgs, tool_schemas
from egresstwin.agent.verifier import verify_report
from egresstwin.layout.schema import Exit, Layout, Obstacle, SpawnZone
from egresstwin.parallel.store import ReplicaMetrics, ScenarioRecord, ScenarioStore
from egresstwin.scenarios.apply import apply_modifications


def room() -> Layout:
    cell, rows, cols = 0.25, 24, 48
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    obstacle = Obstacle(
        id="pile", label="boxes", removable=True,
        polygon=[(5.0, 2.0), (6.0, 2.0), (6.0, 3.0), (5.0, 3.0)],
    )
    walkable &= ~obstacle.rasterized((rows, cols), cell)
    return Layout(
        cell_size_m=cell,
        walkable_mask=walkable,
        obstacles=[obstacle],
        exits=[
            Exit(id="E", segment=((12.0, 2.75), (12.0, 3.25))),
            Exit(id="W", segment=((0.0, 2.75), (0.0, 3.25))),
        ],
        spawn_zones=[
            SpawnZone(id="z", polygon=[(7.0, 1.0), (11.0, 1.0), (11.0, 5.0), (7.0, 5.0)], count=12)
        ],
    )


def fake_runner(layout, specs, *, replicas, seed, store, scenario_ids):  # noqa: ANN001, ANN202
    records = []
    for spec, sid in zip(specs, scenario_ids, strict=True):
        mods = [m.model_dump(mode="json") for m in spec.modifications]
        t95, peak = 20.0, 5.0
        if any(m["type"] == "widen_exit" for m in mods):
            t95, peak = 12.0, 3.4
        elif any(m["type"] == "add_exit" for m in mods):
            t95, peak = 13.0, 3.8
        if spec.blocked_exit_ids:
            t95 *= 1.1
        rng = np.random.default_rng(zlib.crc32(sid.encode()))  # hash() varies per process
        modified = apply_modifications(layout, list(spec.modifications))
        blocked = set(spec.blocked_exit_ids)
        record = ScenarioRecord(
            scenario_id=sid, name=spec.name, modifications=mods,
            blocked_exit_ids=list(spec.blocked_exit_ids),
            exit_ids=[e.id for e in modified.exits if e.id not in blocked],
            replicas=[
                ReplicaMetrics(
                    seed=seed + i,
                    t95_s=float(t95 + rng.normal(0, 0.25)),
                    total_time_s=float(t95 + 2 + rng.normal(0, 0.25)),
                    mean_egress_s=float(t95 / 2 + rng.normal(0, 0.2)),
                    peak_density_per_m2=float(peak + rng.normal(0, 0.1)),
                    n_people=12, n_exited=12,
                    hotspots=[{"n_cells": 4, "area_m2": 1.0,
                               "peak_density_per_m2": float(peak), "centroid_m": [11.0, 3.0]}],
                    flow_per_exit={"E": {"total_people": 9.0, "peak_people_per_s": 3.0},
                                   "W": {"total_people": 3.0, "peak_people_per_s": 1.0}},
                )
                for i in range(replicas)
            ],
        )
        if store is not None:
            store.add(record)
        records.append(record)
    return records


WIDEN = {"type": "widen_exit", "exit_id": "E", "width_m": 2.4}


class ScriptedModel:
    """Model double: each turn is a list of (tool name, args) calls."""

    def __init__(self, turns, tools: AgentTools) -> None:  # noqa: ANN001
        self.turns = list(turns)
        self.tools = tools
        self.n_calls = 0
        self.seen_inputs: list = []

    def responses_turn(self, *, input_items, tools, instructions, previous_response_id=None):  # noqa: ANN001, ANN202
        self.seen_inputs.append(input_items)
        turn = self.turns.pop(0) if self.turns else []
        if callable(turn):
            turn = turn(self.tools)
        self.n_calls += 1
        calls = []
        for i, (name, args) in enumerate(turn):
            if callable(args):
                args = args(self.tools)
            calls.append(
                {"name": name, "arguments": json.dumps(args), "call_id": f"c{self.n_calls}-{i}"}
            )
        return SimpleNamespace(
            data={"response_id": f"resp-{self.n_calls}", "function_calls": calls, "text": ""},
            cost_usd=0.001, usage={"input_tokens": 100, "output_tokens": 20},
            model="muse-spark-1.3",
        )


def good_report(tools: AgentTools, *, t95_factor: float = 1.0) -> dict:
    store = tools.store
    widened = next(r for r in store.records.values() if r.name == "widen")
    robustness = []
    for exit_id in widened.exit_ids:
        stressed = store.stress[(widened.scenario_id, exit_id)]
        robustness.append(
            {"scenario_id": widened.scenario_id, "blocked_exit_id": exit_id,
             "claimed_t95_s": stressed.aggregate("t95_s")["mean"]}
        )
    return {
        "report": {
            "baseline_id": "baseline",
            "objective": "minimize_t95",
            "recommendations": [
                {
                    "modifications": [WIDEN],
                    "scenario_id": widened.scenario_id,
                    "claimed_metrics": {
                        "t95_s": widened.aggregate("t95_s")["mean"] * t95_factor,
                        "peak_density_per_m2": widened.aggregate("peak_density_per_m2")["mean"],
                    },
                    "rationale": "widen the saturated east exit",
                }
            ],
            "robustness": robustness,
            "limitations": ["decision support, not a safety certification"],
            "summary": "widen E",
        }
    }


def make_loop(tmp_path: Path, turns) -> tuple[AgentLoop, AgentTools]:  # noqa: ANN001
    store = ScenarioStore(path=tmp_path / "runs" / "test-run" / "scenarios.json")
    tools = AgentTools(room(), store=store, sweep_runner=fake_runner, default_max_time_s=60.0)
    model = ScriptedModel(turns, tools)
    loop = AgentLoop(
        room(), objective="minimize_t95", model_client=model, tools=tools,
        runs_dir=tmp_path / "runs", run_id="test-run",
    )
    return loop, tools


def test_tool_schemas_are_strict() -> None:
    schemas = tool_schemas()
    assert {s["name"] for s in schemas} == {
        "get_layout_summary", "get_baseline", "run_scenarios", "get_hotspots",
        "stress_test", "compare", "submit_report",
    }
    assert all(s["strict"] is True for s in schemas)
    assert all(s["parameters"]["additionalProperties"] is False for s in schemas)


def test_happy_flow_verified_with_multiple_calls_per_turn(tmp_path: Path) -> None:
    loop, tools = make_loop(
        tmp_path,
        [
            [("get_layout_summary", {}), ("get_baseline", {"replicas": 5})],
            [("run_scenarios", {"specs": [{"name": "widen", "modifications": [WIDEN]}],
                                "replicas": 5})],
            [("stress_test", lambda t: {"scenario_id": _widen_id(t), "replicas": 3})],
            [("submit_report", good_report)],
        ],
    )
    result = loop.run()
    assert result.status == "VERIFIED", result.verification
    assert result.turns == 4
    assert tools.store.baseline_id == "baseline"
    assert result.total_cost_usd == pytest.approx(0.004)
    assert result.total_cost_usd < 1.0
    trace = Path(result.trace_path)
    assert trace.is_file()
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_done"
    tool_calls = [e for e in events if e["event"] == "tool_call"]
    assert [c["tool"] for c in tool_calls[:2]] == ["get_layout_summary", "get_baseline"]
    assert any(e.get("verification_status") == "VERIFIED" for e in events)
    assert (tmp_path / "runs" / "test-run" / "report.json").is_file()
    assert (tmp_path / "runs" / "test-run" / "scenarios.json").is_file()


def _widen_id(tools: AgentTools) -> str:
    return next(r.scenario_id for r in tools.store.records.values() if r.name == "widen")


def test_invented_number_is_caught_then_corrected(tmp_path: Path) -> None:
    loop, _tools = make_loop(
        tmp_path,
        [
            [("get_baseline", {"replicas": 5})],
            [("run_scenarios", {"specs": [{"name": "widen", "modifications": [WIDEN]}],
                                "replicas": 5})],
            [("stress_test", lambda t: {"scenario_id": _widen_id(t), "replicas": 3})],
            [("submit_report", lambda t: good_report(t, t95_factor=0.5))],
            [("submit_report", good_report)],
        ],
    )
    result = loop.run()
    assert result.status == "VERIFIED"
    statuses = [
        e.get("verification_status") for e in result.events if e["event"] == "tool_result"
    ]
    assert statuses[0] != "VERIFIED"  # invented number rejected first
    assert statuses[-1] == "VERIFIED"


def test_verifier_catches_invented_number_directly() -> None:
    layout = room()
    store = ScenarioStore()
    tools = AgentTools(layout, store=store, sweep_runner=fake_runner, default_max_time_s=60.0)
    tools.get_baseline(replicas=5)
    tools.run_scenarios([{"name": "widen", "modifications": [WIDEN]}], replicas=5)
    widen_id = _widen_id(tools)
    tools.stress_test(widen_id, replicas=3)
    bad = AgentReport.model_validate(good_report(tools, t95_factor=0.5)["report"])
    verification = verify_report(bad, layout, store)
    assert verification.status != "VERIFIED"
    failed = [c for c in verification.checks if not c.passed]
    assert any("claim t95_s" in c.claim for c in failed)
    good = AgentReport.model_validate(good_report(tools)["report"])
    assert verify_report(good, layout, store).status == "VERIFIED"


def test_invalid_arguments_do_not_crash_the_loop(tmp_path: Path) -> None:
    loop, _tools = make_loop(
        tmp_path,
        [
            [("run_scenarios", {"specs": [], "replicas": 99})],
            [("get_baseline", {"replicas": 3})],
            [("submit_report", {"report": {
                "baseline_id": "baseline", "objective": "minimize_t95",
                "recommendations": [], "robustness": [],
                "limitations": ["no significant improvement found"],
                "summary": "no improvement demonstrated",
            }})],
        ],
    )
    result = loop.run()
    assert result.status == "VERIFIED"  # honest no-improvement report
    errors = [e for e in result.events if e["event"] == "tool_result"
              and "error" in e["summary"]]
    assert errors


def test_real_layout_agent_finds_verified_improvement(tmp_path: Path) -> None:
    """Real engine, scripted model: widen a clogged exit, verify it."""
    store = ScenarioStore(path=tmp_path / "runs" / "real" / "scenarios.json")
    tools = AgentTools(room(), store=store, default_max_time_s=60.0)
    turns = [
        [("get_layout_summary", {}), ("get_baseline", {"replicas": 3})],
        [("run_scenarios", {"specs": [{"name": "widen", "modifications": [WIDEN]}],
                            "replicas": 3})],
        [("stress_test", lambda t: {"scenario_id": _widen_id(t), "replicas": 2})],
        [("submit_report", good_report)],
    ]
    loop = AgentLoop(
        room(), model_client=ScriptedModel(turns, tools), tools=tools,
        runs_dir=tmp_path / "runs", run_id="real",
    )
    result = loop.run()
    assert result.status == "VERIFIED", result.verification
    widened = tools.store.get(_widen_id(tools))
    assert widened is not None
    assert all(r.n_exited == r.n_people for r in widened.replicas)


def test_apply_modifications_feasibility() -> None:
    layout = room()
    widened = apply_modifications(layout, [WIDEN])
    assert widened.exits[0].width_m == pytest.approx(2.4)
    with pytest.raises(ValueError, match="unknown obstacle"):
        apply_modifications(
            layout, [{"type": "remove_obstacle", "obstacle_id": "missing"}]
        )
    freed = apply_modifications(
        layout, [{"type": "remove_obstacle", "obstacle_id": "pile"}]
    )
    assert freed.obstacles == []
    with pytest.raises(ValueError, match="does not exceed"):
        apply_modifications(
            layout, [{"type": "widen_exit", "exit_id": "E", "width_m": 0.4}]
        )


# fake_runner leaves T95 unchanged for remove_obstacle: a change that does nothing.
CLEAR = {"type": "remove_obstacle", "obstacle_id": "pile"}


def report_for(tools: AgentTools, recs: list[tuple[str, dict]], *,
               baseline_id: str = "baseline") -> AgentReport:
    """Report that copies the store exactly for each (scenario_id, modification)."""
    store = tools.store
    recommendations, robustness = [], []
    for scenario_id, mod in recs:
        record = store.get(scenario_id)
        assert record is not None
        recommendations.append(
            {"modifications": [mod], "scenario_id": scenario_id,
             "claimed_metrics": {"t95_s": record.aggregate("t95_s")["mean"]}}
        )
        robustness += [
            {"scenario_id": scenario_id, "blocked_exit_id": e,
             "claimed_t95_s": store.stress[(scenario_id, e)].aggregate("t95_s")["mean"]}
            for e in record.exit_ids
        ]
    return AgentReport.model_validate(
        {"baseline_id": baseline_id, "objective": "minimize_t95",
         "recommendations": recommendations, "robustness": robustness}
    )


def test_report_cannot_pick_a_slower_baseline() -> None:
    """Naming a stress record (an exit blocked) as baseline must not prove a no-op."""
    layout = room()
    tools = AgentTools(layout, store=ScenarioStore(), sweep_runner=fake_runner,
                       default_max_time_s=60.0)
    tools.get_baseline(replicas=5)
    tools.stress_test("baseline", replicas=5)  # stores "baseline--no-W", T95 x 1.1
    tools.run_scenarios([{"name": "clear", "modifications": [CLEAR]}], replicas=5)
    tools.stress_test("clear", replicas=3)
    report = report_for(tools, [("clear", CLEAR)], baseline_id="baseline--no-W")
    verification = verify_report(report, layout, tools.store)
    assert verification.status != "VERIFIED"
    failed = {c.claim for c in verification.checks if not c.passed}
    assert "baseline_matches" in failed
    assert "[clear] significant_improvement_t95_s" in failed  # judged vs the real baseline


def test_no_change_report_must_name_the_real_baseline() -> None:
    """'Change nothing' measured against a slower record is not verified."""
    layout = room()
    tools = AgentTools(layout, store=ScenarioStore(), sweep_runner=fake_runner)
    tools.get_baseline(replicas=3)
    tools.stress_test("baseline", replicas=3)  # stores "baseline--no-W"
    report = AgentReport(baseline_id="baseline--no-W", objective="minimize_t95")
    verification = verify_report(report, layout, tools.store)
    assert verification.status == "FAILED"
    assert "baseline_matches" in {c.claim for c in verification.checks if not c.passed}


def test_agent_scenarios_use_baseline_run_settings() -> None:
    seen: list = []

    def capture(layout, specs, **kwargs):  # noqa: ANN001, ANN003, ANN202
        seen.extend(specs)
        return fake_runner(layout, specs, **kwargs)

    tools = AgentTools(room(), store=ScenarioStore(), sweep_runner=capture,
                       default_max_time_s=60.0)
    tools.get_baseline(replicas=3)
    tools.run_scenarios([{"name": "widen", "modifications": [WIDEN]}], replicas=3)
    assert len(seen) == 2
    for spec in seen:
        assert spec.max_time_s == 60.0
        assert spec.desired_speed_mps is None
        assert spec.blocked_exit_ids == [] and spec.events == []
        assert spec.pre_movement_min_s == spec.pre_movement_max_s == 0.0
    for extra in ({"desired_speed_mps": 2.5}, {"max_time_s": 5.0},
                  {"blocked_exit_ids": ["W"]}, {"pre_movement_max_s": 0.0}):
        with pytest.raises(ValidationError):
            RunScenariosArgs.model_validate(
                {"specs": [{"name": "x", "modifications": [WIDEN], **extra}]}
            )


def test_timed_stress_run_cannot_stand_in_for_n_minus_1() -> None:
    """An added exit named "W@999" must not inherit the timed run for W."""
    layout = room()
    tools = AgentTools(layout, store=ScenarioStore(), sweep_runner=fake_runner,
                       default_max_time_s=60.0)
    tools.get_baseline(replicas=5)
    add = {"type": "add_exit", "segment": [[8.0, 0.25], [9.0, 0.25]], "exit_id": "W@999"}
    tools.run_scenarios([{"name": "add", "modifications": [add]}], replicas=5)
    # Only the timed test: W "closes" at 999 s, after the run is over.
    tools.stress_test("add", replicas=3, block_at_s=999.0)
    assert ("add", "W@999") not in tools.store.stress
    report = AgentReport.model_validate({
        "baseline_id": "baseline", "objective": "minimize_t95",
        "recommendations": [{"modifications": [add], "scenario_id": "add",
                             "claimed_metrics": {"t95_s": tools.store.get("add")
                                                 .aggregate("t95_s")["mean"]}}],
        "robustness": [
            {"scenario_id": "add", "blocked_exit_id": e,
             "claimed_t95_s": tools.store.stress[("add", e.split("@")[0], 999.0)]
             .aggregate("t95_s")["mean"]}
            for e in ("E", "W", "W@999")
        ],
    })
    verification = verify_report(report, layout, tools.store)
    assert verification.status != "VERIFIED"
    failed = {c.claim for c in verification.checks if not c.passed}
    assert {"[add] n_minus_1[E]", "[add] n_minus_1[W]", "[add] n_minus_1[W@999]"} <= failed
    # The real N-1 test still runs W@999 blocked from t = 0.
    tools.stress_test("add", replicas=3)
    assert tools.store.stress[("add", "W@999")].blocked_exit_ids == ["W@999"]


def test_partial_status_needs_a_fully_verified_recommendation() -> None:
    layout = room()
    tools = AgentTools(layout, store=ScenarioStore(), sweep_runner=fake_runner,
                       default_max_time_s=60.0)
    tools.get_baseline(replicas=5)
    tools.run_scenarios(
        [{"name": "clear", "modifications": [CLEAR]},
         {"name": "widen", "modifications": [WIDEN]}],
        replicas=5,
    )
    tools.stress_test("clear", replicas=3)
    tools.stress_test("widen", replicas=3)
    # A feasible change with no improvement is FAILED, not partial credit.
    alone = verify_report(report_for(tools, [("clear", CLEAR)]), layout, tools.store)
    assert alone.status == "FAILED"
    # Next to a recommendation that passes every check, it is partial.
    mixed = verify_report(
        report_for(tools, [("clear", CLEAR), ("widen", WIDEN)]), layout, tools.store
    )
    assert mixed.status == "PARTIALLY_VERIFIED"
