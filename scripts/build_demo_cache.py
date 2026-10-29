"""Build the offline demo cache in data/demo (no network, no API key).

The perception read uses a ground-truth-backed stub (the synthetic
plan's doors are known exactly), but every simulation, the agent's
tool calls, the stress logic and the verifier run for real. On plan
B no modification beats the baseline with significance, so the
cached agent submits the honest no-improvement report and the
verifier stamps it VERIFIED.

Usage: .venv/Scripts/python.exe scripts/build_demo_cache.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from egresstwin.agent.loop import AgentLoop  # noqa: E402
from egresstwin.agent.tools import AgentTools  # noqa: E402
from egresstwin.app.api import default_bench_runner, simulate_once  # noqa: E402
from egresstwin.layout.schema import SpawnZone  # noqa: E402
from egresstwin.parallel.sweep import run_sweep  # noqa: E402
from egresstwin.perception.pipeline import image_to_layout  # noqa: E402
from egresstwin.scenarios.spec import ScenarioSpec  # noqa: E402

OUT = ROOT / "data" / "demo"
PLAN = ROOT / "data" / "samples" / "spike_plan_synth.png"
GT = ROOT / "data" / "samples" / "spike_plan_synth_gt.json"
DEMO_PEOPLE = 12


@dataclass
class FakeResult:
    data: dict = field(default_factory=dict)
    text: str = ""
    cost_usd: float = 0.0
    cached: bool = False
    model: str = "demo-stub"


class GtClient:
    """Spark/SAM stub seeded from the synthetic plan's ground truth."""

    def __init__(self) -> None:
        self.gt = json.loads(GT.read_text(encoding="utf-8"))

    def vision_structured(self, image_path, prompt, schema, schema_name="PlanRead"):  # noqa: ANN001, ANN202
        exits = [
            {"label": d["kind"], "confidence": 0.95, "box_px": d["box"]}
            for d in self.gt["doors"]
        ]
        return FakeResult(
            data={"image_type": "floor_plan", "exits": exits,
                  "dimension_annotations": [{"text": "13.6 m", "box_px": [372, 578, 60, 14]}],
                  "scale_annotation": "13.6 m", "notes": "demo cache read",
                  "confidence": 0.95},
            model="muse-spark-1.3",
        )

    def segment_image_boxes(self, image_path, boxes):  # noqa: ANN001, ANN202
        return FakeResult(
            data={"segments": [
                {"label": "door", "confidence": 0.97,
                 "box": {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}} for b in boxes
            ]},
            model="sam-3.1",
        )

    def segment_image(self, image_path, concept):  # noqa: ANN001, ANN202
        return FakeResult(data={"segments": [], "concept": concept}, model="sam-3.1")


class ScriptedModel:
    """Deterministic demo driver: explore, then report honestly."""

    def __init__(self, tools: AgentTools) -> None:
        self.tools = tools
        self.turns = [
            [("get_layout_summary", {}), ("get_baseline", {"replicas": 3})],
            [("run_scenarios", {"specs": [{
                "name": "widen-both",
                "modifications": [
                    {"type": "widen_exit", "exit_id": "E1", "width_m": 2.6},
                    {"type": "widen_exit", "exit_id": "E2", "width_m": 2.4},
                ],
                "max_time_s": 90.0}], "replicas": 3})],
            [("submit_report", {"report": {
                "baseline_id": "baseline", "objective": "minimize_t95",
                "recommendations": [], "robustness": [],
                "limitations": [
                    "Decision support, not a safety certification.",
                    "No simulated modification beat the baseline with "
                    "statistical significance on this plan, so no change "
                    "is recommended.",
                ],
                "summary": "Explored widening both exits; the simulations "
                           "do not demonstrate an improvement, so the "
                           "honest recommendation is to change nothing.",
            }})],
        ]
        self.n = 0

    def responses_turn(self, *, input_items, tools, instructions, previous_response_id=None):  # noqa: ANN001, ANN202
        turn = self.turns.pop(0) if self.turns else []
        self.n += 1
        calls = [
            {"name": name, "arguments": json.dumps(args), "call_id": f"demo-{self.n}-{i}"}
            for i, (name, args) in enumerate(turn)
        ]
        return SimpleNamespace(
            data={"response_id": f"demo-resp-{self.n}", "function_calls": calls, "text": ""},
            cost_usd=0.001, usage={"input_tokens": 100, "output_tokens": 20},
            model="muse-spark-1.3",
        )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result = image_to_layout(
        PLAN, client=GtClient(),
        scale_two_points=((60, 572), (740, 572)), scale_distance_m=13.6,
    )
    layout = result.layout.model_copy(deep=True)
    zone = layout.spawn_zones[0]
    layout.spawn_zones = [SpawnZone(id=zone.id, polygon=zone.polygon, count=DEMO_PEOPLE)]
    layout.save_npz(OUT / "layout.npz")
    print("layout:", layout.shape, "exits", [e.id for e in layout.exits])

    # Cached simulation tab content: baseline replicas + animation.
    records = run_sweep(layout, [ScenarioSpec(name="baseline", max_time_s=90.0)],
                        replicas=3, seed=42)
    rec = records[0]
    from egresstwin.parallel.store import METRIC_KEYS

    simulation = {
        "name": "baseline",
        "aggregates": {k: rec.aggregate(k) for k in METRIC_KEYS},
        "n_replicas": len(rec.replicas),
        "animation": simulate_once(layout, max_time_s=90.0),
    }
    (OUT / "simulation.json").write_text(json.dumps(simulation), encoding="utf-8")
    print("baseline t95:", rec.aggregate("t95_s"))

    # Cached agent run (real tools, verifier and trace).
    tools = AgentTools(layout, default_max_time_s=90.0)
    loop = AgentLoop(
        layout, objective="minimize_t95",
        model_client=ScriptedModel(tools), tools=tools,
        runs_dir=OUT / "agent_run", run_id="demo",
    )
    agent_result = loop.run()
    print("agent:", agent_result.status, agent_result.reason)
    agent = {
        "status": agent_result.status,
        "report": agent_result.report,
        "verification": agent_result.verification,
        "reason": agent_result.reason,
        "total_cost_usd": agent_result.total_cost_usd,
        "events": agent_result.events,
        "animations": {"baseline": simulation["animation"]},
    }
    (OUT / "agent.json").write_text(json.dumps(agent), encoding="utf-8")

    bench = default_bench_runner(OUT)
    print("bench:", bench["throughput"])
    manifest = {
        "layout_file": "layout.npz",
        "image": "data/samples/spike_plan_synth.png",
        "simulation_file": "simulation.json",
        "agent_file": "agent.json",
        "bench_file": "bench.json",
        "perception": result.report,
        "note": "Cached demo run. Simulations and verification are real; "
                "the perception read is seeded from the synthetic plan "
                "ground truth so the demo needs no network or API key.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote demo cache to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
