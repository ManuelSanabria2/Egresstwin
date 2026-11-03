"""Scenario sweep: apply specs to a layout and simulate replicas.

Two backends share one record shape and one replica function:

- ``serial``: every (scenario, replica) pair runs in this process.
- ``local``: the pairs are spread over a ``ProcessPoolExecutor``
  (spawn start method). Each worker is created with
  ``NUMBA_NUM_THREADS=1`` (and the BLAS/OpenMP thread counts at 1) so
  N processes never oversubscribe the cores with N x threads.

Every replica uses an explicit seed (``seed + replica``) and results
are placed back by (scenario, replica) index, so both backends return
identical records for the same inputs. MPI is not implemented.
"""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Literal

from egresstwin.layout.schema import Layout
from egresstwin.parallel.store import ReplicaMetrics, ScenarioRecord, ScenarioStore
from egresstwin.scenarios.apply import apply_modifications
from egresstwin.scenarios.spec import ScenarioSpec
from egresstwin.sim.engine import Scenario, run
from egresstwin.sim.metrics import compute_metrics
from egresstwin.sim.social_force import SFParams

Backend = Literal["serial", "local"]
BACKENDS: tuple[str, ...] = ("serial", "local")

# One thread per worker process: the process pool is the parallelism.
_WORKER_THREAD_VARS = (
    "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
# ProcessPoolExecutor on Windows accepts at most 61 workers.
_WINDOWS_MAX_WORKERS = 61

_pools: dict[int, ProcessPoolExecutor] = {}
_pools_lock = threading.Lock()
_spawn_env_lock = threading.Lock()


class _PinnedSpawnProcess(SpawnProcess):
    """Spawned worker that inherits the one-thread variables at birth.

    numba and the BLAS libraries read them at import, which in a
    spawned child happens before any initializer runs, so they must
    be in the environment the child is created with.
    """

    def start(self) -> None:
        with _spawn_env_lock:
            saved = {var: os.environ.get(var) for var in _WORKER_THREAD_VARS}
            os.environ.update({var: "1" for var in _WORKER_THREAD_VARS})
            try:
                super().start()
            finally:
                for var, value in saved.items():
                    if value is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = value


class _PinnedSpawnContext(SpawnContext):
    Process = _PinnedSpawnProcess


def default_workers() -> int:
    """Worker processes for the local backend: all logical CPUs."""
    n = max(1, os.cpu_count() or 1)
    return min(n, _WINDOWS_MAX_WORKERS) if sys.platform == "win32" else n


def _init_worker() -> None:
    """Safety net: cap numba at one thread if the pin did not apply."""
    import numba

    if int(numba.config.NUMBA_NUM_THREADS) != 1:
        numba.set_num_threads(1)


def worker_thread_info() -> dict[str, object]:
    """Thread settings seen inside a worker (used by tests)."""
    import numba

    return {
        "pid": os.getpid(),
        "numba_config_threads": int(numba.config.NUMBA_NUM_THREADS),
        "numba_threads": int(numba.get_num_threads()),
        "env": {var: os.environ.get(var) for var in _WORKER_THREAD_VARS},
    }


def get_pool(workers: int) -> ProcessPoolExecutor:
    """Reusable process pool for `workers` processes (created lazily)."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    with _pools_lock:
        pool = _pools.get(workers)
        if pool is None:
            pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=_PinnedSpawnContext(),
                initializer=_init_worker,
            )
            _pools[workers] = pool
        return pool


def shutdown_pools() -> None:
    """Shut down every cached worker pool (benchmarks start cold)."""
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)


def _cancel_pools_at_exit() -> None:
    # Runs in threading shutdown, before concurrent.futures joins its
    # manager threads, so queued replicas are dropped instead of
    # holding interpreter exit until the whole batch has run.
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.shutdown(wait=False, cancel_futures=True)


# Threading exit hooks run in reverse registration order; this module
# imported concurrent.futures.process above, so this one runs first.
threading._register_atexit(_cancel_pools_at_exit)  # type: ignore[attr-defined]


def _discard_pool(workers: int, broken: Executor) -> None:
    """Drop `broken` from the cache unless another thread already did."""
    with _pools_lock:
        if _pools.get(workers) is not broken:
            return
        del _pools[workers]
    broken.shutdown(wait=False, cancel_futures=True)


def simulate_replica(layout: Layout, spec: ScenarioSpec, replica_seed: int) -> ReplicaMetrics:
    """Run one replica of an already-modified layout. Pure in its inputs."""
    blocked = frozenset(spec.blocked_exit_ids)
    params = (
        SFParams(desired_speed_mean=spec.desired_speed_mps, desired_speed_sd=0.0)
        if spec.desired_speed_mps is not None
        else SFParams()
    )
    scenario = Scenario(
        max_time_s=spec.max_time_s, blocked_exit_ids=blocked,
        events=tuple(spec.events),
        pre_movement_min_s=spec.pre_movement_min_s,
        pre_movement_max_s=spec.pre_movement_max_s,
        params=params,
    )
    result = run(layout, scenario, seed=replica_seed)
    metrics = compute_metrics(result, layout)
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
    return ReplicaMetrics(
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


def _run_local(
    tasks: list[tuple[Layout, ScenarioSpec, int]], workers: int, pool: Executor | None
) -> list[ReplicaMetrics]:
    executor = pool if pool is not None else get_pool(workers)
    futures = []
    try:
        futures = [executor.submit(simulate_replica, *task) for task in tasks]
        return [f.result() for f in futures]
    except BrokenProcessPool:
        # A dead worker poisons the pool; drop it so the next call
        # starts a fresh one, then report the failure.
        if pool is None:
            _discard_pool(workers, executor)
        raise
    finally:
        # On any failure, drop this batch's still-queued replicas so a
        # shared pool is not left busy with work nobody will read.
        for future in futures:
            future.cancel()


def run_sweep(
    layout: Layout,
    specs: list[ScenarioSpec],
    *,
    replicas: int = 10,
    seed: int = 42,
    id_prefix: str = "S",
    store: ScenarioStore | None = None,
    scenario_ids: list[str] | None = None,
    backend: Backend = "serial",
    workers: int | None = None,
    pool: Executor | None = None,
) -> list[ScenarioRecord]:
    """Simulate each spec `replicas` times. Returns (and stores) records.

    `backend="local"` spreads the (scenario, replica) pairs over
    `workers` processes (default: all logical CPUs); `pool` lets a
    caller supply its own executor. Results match the serial backend
    exactly for the same seed.
    """
    if replicas < 1:
        raise ValueError("replicas must be >= 1")
    if backend not in BACKENDS:
        raise ValueError(f"unknown sweep backend {backend!r}; expected one of {BACKENDS}")
    if scenario_ids is not None and len(scenario_ids) != len(specs):
        raise ValueError("scenario_ids must match specs in length")
    # Feasibility and exit checks run up front, in this process, so a
    # bad spec fails before any simulation is spent.
    records: list[ScenarioRecord] = []
    modified_layouts: list[Layout] = []
    for index, spec in enumerate(specs, start=1):
        scenario_id = scenario_ids[index - 1] if scenario_ids else f"{id_prefix}{index}"
        modified = apply_modifications(layout, list(spec.modifications))
        blocked = frozenset(spec.blocked_exit_ids)
        unknown = blocked - {e.id for e in modified.exits}
        if unknown:
            raise ValueError(f"scenario {spec.name!r} blocks unknown exits: {sorted(unknown)}")
        records.append(
            ScenarioRecord(
                scenario_id=scenario_id,
                name=spec.name,
                modifications=[m.model_dump(mode="json") for m in spec.modifications],
                blocked_exit_ids=list(spec.blocked_exit_ids),
                exit_ids=[e.id for e in modified.exits if e.id not in blocked],
                events=[e.model_dump(mode="json") for e in spec.events],
            )
        )
        modified_layouts.append(modified)

    tasks = [
        (modified_layouts[i], spec, seed + replica)
        for i, spec in enumerate(specs)
        for replica in range(replicas)
    ]
    if backend == "local":
        results = _run_local(tasks, workers or default_workers(), pool)
    else:
        results = [simulate_replica(*task) for task in tasks]

    for i, record in enumerate(records):
        record.replicas.extend(results[i * replicas:(i + 1) * replicas])
        if store is not None:
            store.add(record)
    return records
