"""Sweep backends: the local process pool must match the serial run."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np
import pytest

from egresstwin.agent.tools import AgentTools
from egresstwin.layout.schema import Exit, Layout, SpawnZone
from egresstwin.parallel.store import ScenarioStore
from egresstwin.parallel.sweep import get_pool, run_sweep, worker_thread_info
from egresstwin.scenarios.spec import ScenarioSpec, WidenExitModification

PARENT_NUMBA = os.environ.get("NUMBA_NUM_THREADS")


def small_room(count: int = 10) -> Layout:
    """8 x 5 m room with two 1 m exits on opposite walls."""
    cell, rows, cols = 0.25, 20, 32
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    return Layout(
        cell_size_m=cell,
        walkable_mask=walkable,
        exits=[
            Exit(id="E", segment=((8.0, 2.0), (8.0, 3.0))),
            Exit(id="W", segment=((0.0, 2.0), (0.0, 3.0))),
        ],
        spawn_zones=[
            SpawnZone(id="z", polygon=[(1.0, 1.0), (7.0, 1.0), (7.0, 4.0), (1.0, 4.0)],
                      count=count)
        ],
    )


SPECS = [
    ScenarioSpec(name="base", max_time_s=60.0),
    ScenarioSpec(
        name="wide-e",
        modifications=[WidenExitModification(exit_id="E", width_m=2.0)],
        max_time_s=60.0,
    ),
    ScenarioSpec(name="no-w", blocked_exit_ids=["W"], max_time_s=60.0),
]


def test_local_backend_matches_serial_exactly() -> None:
    layout = small_room()
    store_serial, store_local = ScenarioStore(), ScenarioStore()
    serial = run_sweep(layout, SPECS, replicas=3, seed=7, store=store_serial)
    local = run_sweep(
        layout, SPECS, replicas=3, seed=7, store=store_local, backend="local", workers=2
    )
    assert [asdict(r) for r in local] == [asdict(r) for r in serial]
    assert list(store_local.records) == list(store_serial.records)
    # Replicas are in seed order inside each scenario.
    assert [r.seed for r in local[0].replicas] == [7, 8, 9]


def test_local_backend_is_independent_of_worker_count() -> None:
    layout = small_room(count=6)
    one = run_sweep(layout, SPECS[:2], replicas=2, seed=3, backend="local", workers=1)
    three = run_sweep(layout, SPECS[:2], replicas=2, seed=3, backend="local", workers=3)
    assert [asdict(r) for r in one] == [asdict(r) for r in three]


def test_workers_run_numba_single_threaded() -> None:
    pool = get_pool(2)
    infos = [pool.submit(worker_thread_info).result() for _ in range(4)]
    for info in infos:
        # Pinned at spawn: numba read NUMBA_NUM_THREADS=1 at import.
        assert info["numba_config_threads"] == 1
        assert info["numba_threads"] == 1
        assert info["env"]["NUMBA_NUM_THREADS"] == "1"
    # The parent's environment is left as it was.
    assert os.environ.get("NUMBA_NUM_THREADS") == PARENT_NUMBA


class RecordingPool(ProcessPoolExecutor):
    """Real worker processes that also count what was submitted."""

    def __init__(self) -> None:
        super().__init__(max_workers=2, mp_context=get_context("spawn"))
        self.submitted = 0

    def submit(self, fn, /, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
        self.submitted += 1
        return super().submit(fn, *args, **kwargs)


def test_local_backend_dispatches_every_replica_to_worker_processes() -> None:
    layout = small_room(count=4)
    with RecordingPool() as pool:
        local = run_sweep(layout, SPECS[:2], replicas=2, seed=5, backend="local", pool=pool)
        pids = {pool.submit(os.getpid).result() for _ in range(4)}
    assert pool.submitted == 4 + 4  # 2 scenarios x 2 replicas, plus the pid probes
    assert os.getpid() not in pids
    serial = run_sweep(layout, SPECS[:2], replicas=2, seed=5)
    assert [asdict(r) for r in local] == [asdict(r) for r in serial]


def test_bad_spec_fails_before_any_simulation() -> None:
    bad = ScenarioSpec(name="ghost", blocked_exit_ids=["Z"], max_time_s=60.0)
    with pytest.raises(ValueError, match="unknown exits"):
        run_sweep(small_room(), [SPECS[0], bad], replicas=1, backend="local", workers=2)
    with pytest.raises(ValueError, match="unknown sweep backend"):
        run_sweep(small_room(), SPECS[:1], replicas=1, backend="mpi")  # type: ignore[arg-type]


def test_agent_tools_default_to_local_backend() -> None:
    tools = AgentTools(small_room())
    assert tools._runner.keywords["backend"] == "local"
