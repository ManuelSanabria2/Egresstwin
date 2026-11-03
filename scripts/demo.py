"""End-to-end demo: image -> layout -> simulation -> verified agent.

Mock mode (no network, no API key, no cost):

    python scripts/demo.py --mock
    python scripts/demo.py --mock --image data/samples/spike_plan_hand.png

Real mode (Meta Model API; needs MODEL_API_KEY in .env):

    python scripts/demo.py --image data/samples/spike_plan_synth.png

In mock mode the perception read comes from the recorded fixtures
and a deterministic scripted driver plays the agent's role, but the
simulations, the N-1 stress test and the verifier all run for real.
In real mode Muse Spark does both jobs through the Responses API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                        help="fixture perception + scripted agent (no network)")
    parser.add_argument("--image", default="data/samples/spike_plan_synth.png")
    parser.add_argument("--people", type=int, default=12)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--out", default=None, help="output directory")
    return parser.parse_args()


ARGS = _parse_args()
if ARGS.mock:
    os.environ["META_MOCK"] = "1"
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import dataclass, field  # noqa: E402

from egresstwin.agent.loop import AgentLoop  # noqa: E402
from egresstwin.agent.tools import AgentTools  # noqa: E402
from egresstwin.layout.schema import SpawnZone  # noqa: E402
from egresstwin.perception.pipeline import image_to_layout  # noqa: E402


@dataclass
class _StubResult:
    data: dict = field(default_factory=dict)
    text: str = ""
    cost_usd: float = 0.0
    cached: bool = False
    model: str = "demo-stub"


class GroundTruthClient:
    """Mock-mode Spark/SAM stub seeded from a sample plan's GT file.

    The bundled sample plans ship exact ground truth
    (spike_plan_synth_gt.json); the hand-style plan shares the same
    geometry. For any other image the regular MetaClient mock
    fixtures apply and may not match, which the error message says.
    """

    def __init__(self, gt_path: Path) -> None:
        self.gt = json.loads(gt_path.read_text(encoding="utf-8"))

    def vision_structured(self, image_path, prompt, schema, schema_name="PlanRead"):  # noqa: ANN001, ANN202
        exits = [
            {"label": d["kind"], "confidence": 0.95, "box_px": d["box"]}
            for d in self.gt["doors"]
        ]
        return _StubResult(
            data={"image_type": "floor_plan", "exits": exits,
                  "dimension_annotations": [
                      {"text": "13.6 m", "box_px": [372, 578, 60, 14]}],
                  "scale_annotation": "13.6 m",
                  "notes": "demo ground-truth read", "confidence": 0.95},
            model="muse-spark-1.3",
        )

    def segment_image_boxes(self, image_path, boxes):  # noqa: ANN001, ANN202
        return _StubResult(
            data={"segments": [
                {"label": "door", "confidence": 0.97,
                 "box": {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}}
                for b in boxes]},
            model="sam-3.1",
        )

    def segment_image(self, image_path, concept):  # noqa: ANN001, ANN202
        return _StubResult(data={"segments": [], "concept": concept},
                           model="sam-3.1")


class ScriptedDemoModel:
    """Deterministic mock-mode driver: explore, prove, then report.

    It reads the real scenario store to build its claims, so every
    number it submits is one the verifier can re-check.
    """

    def __init__(self, tools: AgentTools) -> None:
        self.tools = tools
        self.step = 0
        self.candidate_id: str | None = None

    def _calls(self, specs: list[tuple[str, dict]]) -> SimpleNamespace:
        self.step += 1
        calls = [
            {"name": name, "arguments": json.dumps(args),
             "call_id": f"demo-{self.step}-{i}"}
            for i, (name, args) in enumerate(specs)
        ]
        return SimpleNamespace(
            data={"response_id": f"demo-resp-{self.step}",
                  "function_calls": calls, "text": ""},
            cost_usd=0.0, usage={"input_tokens": 0, "output_tokens": 0},
            model="scripted-demo",
        )

    def _honest(self) -> SimpleNamespace:
        return self._calls([("submit_report", {"report": {
            "baseline_id": "baseline", "objective": "minimize_t95",
            "recommendations": [], "robustness": [],
            "limitations": [
                "Decision support, not a safety certification.",
                "No simulated modification beat the baseline with "
                "statistical significance, so no change is recommended.",
            ],
            "summary": "Explored the layout with simulations; nothing "
                       "demonstrated an improvement, so the honest "
                       "recommendation is to change nothing.",
        }})])

    def responses_turn(self, *, input_items, tools, instructions,
                       previous_response_id=None):  # noqa: ANN001, ANN202
        store = self.tools.store
        if self.step == 0:
            return self._calls([
                ("get_layout_summary", {}),
                ("get_baseline", {"replicas": 3}),
            ])
        if self.step == 1:
            mods: list[dict] = []
            for ex in self.tools.layout.exits:
                if ex.width_m < 2.4 and len(mods) < 2:
                    mods.append({"type": "widen_exit", "exit_id": ex.id,
                                 "width_m": 2.4})
            for obs in self.tools.layout.obstacles:
                if obs.removable and len(mods) < 3:
                    mods.append({"type": "remove_obstacle",
                                 "obstacle_id": obs.id})
                    break
            if not mods:
                return self._honest()
            return self._calls([("run_scenarios", {
                "specs": [{"name": "demo-candidate", "modifications": mods}],
                "replicas": 3,
            })])
        if self.step == 2:
            self.candidate_id = next(
                (sid for sid in store.records if sid != "baseline"), None
            )
            if self.candidate_id is None:
                return self._honest()
            base = store.get("baseline")
            cand = store.get(self.candidate_id)
            assert base is not None and cand is not None
            if cand.aggregate("t95_s")["mean"] < base.aggregate("t95_s")["mean"]:
                return self._calls([
                    ("stress_test", {"scenario_id": self.candidate_id,
                                     "replicas": 3})
                ])
            return self._honest()
        if self.step == 3 and self.candidate_id is not None:
            cand = store.get(self.candidate_id)
            assert cand is not None
            robustness = []
            for exit_id in cand.exit_ids:
                stressed = store.stress.get((self.candidate_id, exit_id))
                if stressed is not None:
                    robustness.append({
                        "scenario_id": self.candidate_id,
                        "blocked_exit_id": exit_id,
                        "claimed_t95_s": stressed.aggregate("t95_s")["mean"],
                    })
            if len(robustness) == len(cand.exit_ids):
                claimed = {
                    k: cand.aggregate(k)["mean"]
                    for k in ("t95_s", "total_time_s", "mean_egress_s")
                }
                mods = [
                    m if isinstance(m, dict) else m
                    for m in cand.modifications
                ]
                return self._calls([("submit_report", {"report": {
                    "baseline_id": "baseline", "objective": "minimize_t95",
                    "recommendations": [{
                        "modifications": mods,
                        "scenario_id": self.candidate_id,
                        "claimed_metrics": claimed,
                        "rationale": "Simulated candidate beats the "
                                     "baseline on T95; see the verifier.",
                    }],
                    "robustness": robustness,
                    "limitations": [
                        "Decision support, not a safety certification."
                    ],
                    "summary": "The candidate improves T95 in simulation "
                               "and survives the N-1 stress test.",
                }})])
            return self._honest()
        return self._honest()


def main() -> int:
    image = (ROOT / ARGS.image).resolve() if not Path(ARGS.image).is_absolute() \
        else Path(ARGS.image)
    if not image.is_file():
        print(f"ERROR: image not found: {image}")
        return 2
    out = Path(ARGS.out) if ARGS.out else ROOT / "runs" / "demo_cli" / image.stem
    out.mkdir(parents=True, exist_ok=True)
    mode = "MOCK (no network, no cost)" if ARGS.mock else "REAL (Meta Model API)"
    print(f"EgressTwin demo - {mode}")
    print(f"image: {image}")
    started = time.perf_counter()

    if not ARGS.mock and not os.environ.get("MODEL_API_KEY"):
        # Also allow the key to live only in .env (settings load it).
        from egresstwin.config import get_settings

        if not get_settings().model_api_key:
            print("ERROR: real mode needs MODEL_API_KEY. Copy "
                  ".env.example to .env and set it, or run with --mock.")
            return 2

    perception_kwargs: dict = {}
    if ARGS.mock:
        gt_path = image.with_name("spike_plan_synth_gt.json")
        if gt_path.is_file() and image.name.startswith("spike_plan_"):
            perception_kwargs = {
                "client": GroundTruthClient(gt_path),
                "scale_two_points": ((60, 572), (740, 572)),
                "scale_distance_m": 13.6,
            }
    try:
        result = image_to_layout(
            image, output_dir=out / "perception", **perception_kwargs
        )
    except Exception as exc:  # noqa: BLE001 - demo must fail visibly
        hint = (
            " In --mock mode the bundled fixtures match the sample "
            "plans in data/samples; other images need real mode."
            if ARGS.mock else ""
        )
        print(f"ERROR: perception failed: {type(exc).__name__}: {exc}{hint}")
        return 1
    layout = result.layout
    if layout.spawn_zones:
        zone = layout.spawn_zones[0]
        layout = layout.model_copy(deep=True)
        layout.spawn_zones = [
            SpawnZone(id=zone.id, polygon=zone.polygon, count=ARGS.people)
        ]
    layout.save_npz(out / "layout.npz")
    print(f"layout: {layout.shape[0]}x{layout.shape[1]} cells, "
          f"exits {[e.id for e in layout.exits]}, "
          f"warnings {len(result.warnings)}")

    tools = AgentTools(layout, default_max_time_s=ARGS.max_time)
    if ARGS.mock:
        loop = AgentLoop(
            layout, objective="minimize_t95",
            model_client=ScriptedDemoModel(tools),
            tools=tools, runs_dir=out, run_id="demo-cli",
        )
    else:
        loop = AgentLoop(
            layout, objective="minimize_t95", tools=tools,
            runs_dir=out, run_id="demo-cli",
        )
    agent = loop.run()
    baseline = tools.store.get("baseline")
    summary = {
        "mode": "mock" if ARGS.mock else "real",
        "image": str(image),
        "layout_shape": list(layout.shape),
        "exits": [e.id for e in layout.exits],
        "baseline_t95_s": (
            baseline.aggregate("t95_s") if baseline else None
        ),
        "agent_status": agent.status,
        "agent_reason": agent.reason,
        "agent_cost_usd": agent.total_cost_usd,
        "scenarios_run": agent.scenarios_run,
        "perception_cost_usd": result.report.get("costs", {}),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"baseline T95: {summary['baseline_t95_s']}")
    print(f"agent: {agent.status} ({agent.reason}), "
          f"scenarios {agent.scenarios_run}, cost USD {agent.total_cost_usd}")
    print(f"done in {summary['elapsed_s']} s. Outputs in {out}")
    return 0 if agent.status in ("VERIFIED", "PARTIALLY_VERIFIED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
