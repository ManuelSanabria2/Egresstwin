"""Scenario sweep: apply specs to a layout and simulate replicas.

Serial backend (the multiprocessing/MPI backends share this record
shape). Every replica uses an explicit seed, so results are
reproducible and scenarios share seeds replica by replica.
"""

from __future__ import annotations

from egresstwin.layout.schema import Layout
from egresstwin.parallel.store import ReplicaMetrics, ScenarioRecord, ScenarioStore
from egresstwin.scenarios.apply import apply_modifications
from egresstwin.scenarios.spec import ScenarioSpec
from egresstwin.sim.engine import Scenario, run
from egresstwin.sim.metrics import compute_metrics


def run_sweep(
    layout: Layout,
    specs: list[ScenarioSpec],
    *,
    replicas: int = 10,
    seed: int = 42,
    id_prefix: str = "S",
    store: ScenarioStore | None = None,
    scenario_ids: list[str] | None = None,
) -> list[ScenarioRecord]:
    """Simulate each spec `replicas` times. Returns (and stores) records."""
    if replicas < 1:
        raise ValueError("replicas must be >= 1")
    if scenario_ids is not None and len(scenario_ids) != len(specs):
        raise ValueError("scenario_ids must match specs in length")
    records: list[ScenarioRecord] = []
    for index, spec in enumerate(specs, start=1):
        scenario_id = scenario_ids[index - 1] if scenario_ids else f"{id_prefix}{index}"
        modified = apply_modifications(layout, list(spec.modifications))
        blocked = frozenset(spec.blocked_exit_ids)
        unknown = blocked - {e.id for e in modified.exits}
        if unknown:
            raise ValueError(f"scenario {spec.name!r} blocks unknown exits: {sorted(unknown)}")
        record = ScenarioRecord(
            scenario_id=scenario_id,
            name=spec.name,
            modifications=[m.model_dump(mode="json") for m in spec.modifications],
            blocked_exit_ids=list(spec.blocked_exit_ids),
            exit_ids=[e.id for e in modified.exits if e.id not in blocked],
        )
        for replica in range(replicas):
            replica_seed = seed + replica
            scenario = Scenario(
                max_time_s=spec.max_time_s, blocked_exit_ids=blocked
            )
            result = run(modified, scenario, seed=replica_seed)
            metrics = compute_metrics(result, modified)
            if metrics.n_exited < metrics.n_people:
                # Censored replica: quoting times of the few who left
                # would reward a clogged exit. Penalize with the cap.
                metrics.t95_s = float(spec.max_time_s)
                metrics.total_time_s = float(spec.max_time_s)
                metrics.mean_egress_s = float(spec.max_time_s)
            flow = {}
            for exit_id, per_s in metrics.flow_per_exit_per_s.items():
                flow[exit_id] = {
                    "total_people": float(per_s.sum()),
                    "peak_people_per_s": float(per_s.max()) if len(per_s) else 0.0,
                }
            record.replicas.append(
                ReplicaMetrics(
                    seed=replica_seed,
                    t95_s=metrics.t95_s,
                    total_time_s=metrics.total_time_s,
                    mean_egress_s=metrics.mean_egress_s,
                    peak_density_per_m2=metrics.peak_density_per_m2,
                    n_people=metrics.n_people,
                    n_exited=metrics.n_exited,
                    hotspots=metrics.hotspots,
                    flow_per_exit=flow,
                )
            )
        if store is not None:
            store.add(record)
        records.append(record)
    return records
