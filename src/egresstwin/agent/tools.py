"""Agent tools with strict JSON schemas, backed by real simulations.

Every number a tool returns comes from the scenario store, which is
filled only by run_sweep over the social-force engine. The agent
cannot invent metrics: the verifier re-checks its claims here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

from egresstwin.agent.report import AgentReport
from egresstwin.layout.schema import Layout
from egresstwin.layout.validate import LayoutValidationError, validate_layout
from egresstwin.parallel.store import METRIC_KEYS, ScenarioStore, welch_compare
from egresstwin.parallel.sweep import run_sweep
from egresstwin.scenarios.spec import Modification as _Modification
from egresstwin.scenarios.spec import ScenarioSpec

_MODS_ADAPTER = TypeAdapter(list[_Modification])


def _mods_of(record) -> list:  # noqa: ANN001, ANN202
    return _MODS_ADAPTER.validate_python(record.modifications)

MAX_BATCH = 8
MAX_REPLICAS = 20


class LayoutSummaryArgs(BaseModel):
    pass


class BaselineArgs(BaseModel):
    replicas: int = Field(default=10, ge=1, le=MAX_REPLICAS)


class RunScenariosArgs(BaseModel):
    specs: list[ScenarioSpec] = Field(min_length=1, max_length=MAX_BATCH)
    replicas: int = Field(default=10, ge=1, le=MAX_REPLICAS)


class ScenarioIdArgs(BaseModel):
    scenario_id: str = Field(min_length=1)


class StressTestArgs(BaseModel):
    scenario_id: str = Field(min_length=1)
    replicas: int = Field(default=5, ge=1, le=MAX_REPLICAS)
    # None (default): blocked from t = 0, as before. A value blocks the
    # exit mid-evacuation at that second instead (P07 timed event).
    block_at_s: float | None = Field(default=None, ge=0)


class CompareArgs(BaseModel):
    a_id: str = Field(min_length=1)
    b_id: str = Field(min_length=1)


class SubmitReportArgs(BaseModel):
    report: AgentReport


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "get_layout_summary": LayoutSummaryArgs,
    "get_baseline": BaselineArgs,
    "run_scenarios": RunScenariosArgs,
    "get_hotspots": ScenarioIdArgs,
    "stress_test": StressTestArgs,
    "compare": CompareArgs,
    "submit_report": SubmitReportArgs,
}


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)
    schema["additionalProperties"] = False
    return schema


def tool_schemas() -> list[dict[str, Any]]:
    """Strict JSON function schemas for the Responses API."""
    schemas = []
    for name, model in TOOL_ARG_MODELS.items():
        schemas.append(
            {
                "type": "function",
                "name": name,
                "description": _DESCRIPTIONS[name],
                "parameters": _strict(model.model_json_schema()),
                "strict": True,
            }
        )
    return schemas


_DESCRIPTIONS = {
    "get_layout_summary": "Exits, widths, obstacles, spawn zones and warnings of the layout.",
    "get_baseline": "Run or retrieve the unmodified baseline scenario with replicas.",
    "run_scenarios": "Simulate up to 8 modification scenarios. Returns metrics with CIs.",
    "get_hotspots": "Highest-density zones and saturated exits of a simulated scenario.",
    "stress_test": "N-1 test: block each exit in turn and report T95 with CI. "
    "Optional block_at_s blocks mid-evacuation at that second instead of from t=0.",
    "compare": "Compare two scenarios: mean difference, Welch test and effect size.",
    "submit_report": "Mandatory closing call with the structured final report.",
}


def _slug(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "scenario"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def _record_summary(record) -> dict[str, Any]:  # noqa: ANN001
    return {
        "scenario_id": record.scenario_id,
        "name": record.name,
        "modifications": record.modifications,
        "metrics": {k: record.aggregate(k) for k in METRIC_KEYS},
        "n_replicas": len(record.replicas),
    }


class AgentTools:
    """Tool implementations bound to one layout and one store."""

    def __init__(
        self,
        layout: Layout,
        *,
        store: ScenarioStore | None = None,
        seed: int = 42,
        max_scenarios: int = 60,
        default_max_time_s: float = 300.0,
        sweep_runner=None,  # noqa: ANN001
    ) -> None:
        self.layout = layout
        self.store = store or ScenarioStore()
        self.seed = seed
        self.max_scenarios = max_scenarios
        self.default_max_time_s = default_max_time_s
        self.scenarios_run = 0
        self._runner = sweep_runner or run_sweep
        self._counter = 0

    # -- budget ------------------------------------------------------

    def _check_budget(self, n_new: int) -> None:
        if self.scenarios_run + n_new > self.max_scenarios:
            raise ValueError(
                f"scenario budget exceeded: {self.scenarios_run + n_new} "
                f"> max_scenarios={self.max_scenarios}"
            )

    def _run_specs(self, specs, replicas, *, ids=None):  # noqa: ANN001, ANN202
        self._check_budget(len(specs))
        if ids is None:
            ids = []
            taken = set(self.store.records)
            for spec in specs:
                self._counter += 1
                ids.append(_slug(spec.name, taken | set(ids)))
        records = self._runner(
            self.layout, specs, replicas=replicas, seed=self.seed,
            store=self.store, scenario_ids=ids,
        )
        self.scenarios_run += len(records)
        return records

    # -- tools --------------------------------------------------------

    def get_layout_summary(self) -> dict[str, Any]:
        warnings: list[str] = []
        try:
            warnings = validate_layout(self.layout)
        except LayoutValidationError as exc:
            warnings = [f"INVALID: {exc}"]
        return {
            "grid": {"rows": self.layout.shape[0], "cols": self.layout.shape[1],
                     "cell_size_m": self.layout.cell_size_m},
            "meters_per_pixel": self.layout.meters_per_pixel,
            "exits": [
                {"id": e.id, "width_m": round(e.width_m, 3),
                 "segment": [[p[0], p[1]] for p in e.segment]}
                for e in self.layout.exits
            ],
            "obstacles": [
                {"id": o.id, "label": o.label, "removable": o.removable}
                for o in self.layout.obstacles
            ],
            "removable_obstacle_ids": [
                o.id for o in self.layout.obstacles if o.removable
            ],
            "spawn_zones": [
                {"id": z.id, "planned_people": z.planned_count()} for z in self.layout.spawn_zones
            ],
            "warnings": warnings,
        }

    def get_baseline(self, replicas: int = 10) -> dict[str, Any]:
        if self.store.baseline_id and self.store.get(self.store.baseline_id):
            record = self.store.get(self.store.baseline_id)
            assert record is not None
            return {**_record_summary(record), "cached": True}
        records = self._run_specs(
            [ScenarioSpec(name="baseline", max_time_s=self.default_max_time_s)],
            replicas,
            ids=["baseline"],
        )
        self.store.baseline_id = "baseline"
        return {**_record_summary(records[0]), "cached": False}

    def run_scenarios(self, specs: list[ScenarioSpec], replicas: int = 10) -> dict[str, Any]:
        specs = [
            s if isinstance(s, ScenarioSpec) else ScenarioSpec.model_validate(s) for s in specs
        ]
        if len(specs) > MAX_BATCH:
            raise ValueError(f"run_scenarios accepts at most {MAX_BATCH} scenarios per batch")
        records = self._run_specs(specs, replicas)
        return {"scenarios": [_record_summary(r) for r in records]}

    def _require(self, scenario_id: str):  # noqa: ANN202
        record = self.store.get(scenario_id)
        if record is None:
            raise ValueError(f"unknown scenario_id {scenario_id!r}: run it first")
        return record

    def get_hotspots(self, scenario_id: str) -> dict[str, Any]:
        record = self._require(scenario_id)
        first = record.replicas[0]
        exits = []
        exit_widths = {}
        # Widths come from the (possibly modified) layout of the record.
        from egresstwin.scenarios.apply import apply_modifications

        modified = apply_modifications(self.layout, _mods_of(record))
        exit_widths = {e.id: e.width_m for e in modified.exits}
        n_total = max(1, sum(f["total_people"] for f in first.flow_per_exit.values()))
        for exit_id, flow in first.flow_per_exit.items():
            width = exit_widths.get(exit_id, 0.0)
            exits.append(
                {
                    "exit_id": exit_id,
                    "share_of_people": round(flow["total_people"] / n_total, 4),
                    "peak_people_per_s": flow["peak_people_per_s"],
                    "peak_flow_per_m_width": (
                        round(flow["peak_people_per_s"] / width, 3) if width > 0 else None
                    ),
                }
            )
        exits.sort(key=lambda e: e["share_of_people"], reverse=True)
        return {
            "scenario_id": scenario_id,
            "hotspots": first.hotspots[:5],
            "exits_by_saturation": exits,
        }

    def stress_test(
        self, scenario_id: str, replicas: int = 5, block_at_s: float | None = None
    ) -> dict[str, Any]:
        record = self._require(scenario_id)
        mods = _mods_of(record)
        results = []
        for exit_id in record.exit_ids:
            # Timed blocks live under a distinct store key so the
            # verifier's t=0 lookup is never polluted by a P07 run.
            stress_key = (
                (scenario_id, exit_id)
                if block_at_s is None
                else (scenario_id, f"{exit_id}@{block_at_s:g}")
            )
            cached = self.store.stress.get(stress_key)
            if cached is not None:
                stressed = cached
            elif len(record.exit_ids) <= 1 and block_at_s is None:
                # Blocking the only exit: nobody can leave, by definition.
                stressed = self._penalty_record(scenario_id, exit_id, replicas)
                self.store.add_stress(scenario_id, exit_id, stressed)
            else:
                if block_at_s is None:
                    stress_id = f"{scenario_id}--no-{exit_id}"
                    spec = ScenarioSpec(
                        name=stress_id,
                        modifications=mods,
                        blocked_exit_ids=[*record.blocked_exit_ids, exit_id],
                        max_time_s=self.default_max_time_s,
                    )
                else:
                    from egresstwin.sim.events import BlockExitEvent

                    stress_id = f"{scenario_id}--no-{exit_id}-at-{block_at_s:g}"
                    spec = ScenarioSpec(
                        name=stress_id,
                        modifications=mods,
                        blocked_exit_ids=list(record.blocked_exit_ids),
                        max_time_s=self.default_max_time_s,
                        events=[BlockExitEvent(exit_id=exit_id, at_s=block_at_s)],
                    )
                try:
                    stressed = self._run_specs([spec], replicas, ids=[stress_id])[0]
                except Exception as exc:  # noqa: BLE001 - reported, not hidden
                    results.append(
                        {"blocked_exit_id": exit_id, "block_at_s": block_at_s,
                         "error": str(exc), "t95_s": None}
                    )
                    continue
                self.store.stress[stress_key] = stressed
            agg = stressed.aggregate("t95_s")
            base = record.aggregate("t95_s")
            results.append(
                {
                    "blocked_exit_id": exit_id,
                    "block_at_s": block_at_s,
                    "scenario_id": stressed.scenario_id,
                    "t95_s": agg,
                    "delta_vs_unblocked_pct": (
                        round(100.0 * (agg["mean"] - base["mean"]) / base["mean"], 2)
                        if base["mean"]
                        else None
                    ),
                }
            )
        return {"scenario_id": scenario_id, "n_minus_1": results}

    def _penalty_record(self, scenario_id: str, blocked_exit_id: str, replicas: int):  # noqa: ANN202
        from egresstwin.parallel.store import ReplicaMetrics, ScenarioRecord

        cap = self.default_max_time_s
        return ScenarioRecord(
            scenario_id=f"{scenario_id}--no-{blocked_exit_id}",
            name=f"{scenario_id}--no-{blocked_exit_id}",
            blocked_exit_ids=[blocked_exit_id],
            exit_ids=[],
            replicas=[
                ReplicaMetrics(
                    seed=self.seed + i, t95_s=cap, total_time_s=cap,
                    mean_egress_s=cap, peak_density_per_m2=0.0,
                    n_people=0, n_exited=0,
                )
                for i in range(replicas)
            ],
        )

    def compare(self, a_id: str, b_id: str) -> dict[str, Any]:
        a, b = self._require(a_id), self._require(b_id)
        out: dict[str, Any] = {"a_id": a_id, "b_id": b_id, "metrics": {}}
        for metric in ("t95_s", "peak_density_per_m2"):
            cmp_ = welch_compare(a.values(metric), b.values(metric))
            cmp_["ci_a"] = a.aggregate(metric)
            cmp_["ci_b"] = b.aggregate(metric)
            cmp_["ci_overlap"] = not (
                cmp_["ci_a"]["ci_high"] < cmp_["ci_b"]["ci_low"]
                or cmp_["ci_b"]["ci_high"] < cmp_["ci_a"]["ci_low"]
            )
            out["metrics"][metric] = cmp_
        return out

    def submit_report(self, report: AgentReport) -> dict[str, Any]:
        """Accept the report shape; verification happens in the loop."""
        return {"accepted": True, "report": report.model_dump(mode="json")}


def default_store_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "scenarios.json"
