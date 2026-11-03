"""Benchmark the local sweep backend at 1, 2, 4 and 8 worker processes.

The same set (default 8 scenarios x 5 replicas, 200 people) runs at
every process count, each time on a cold pool so process start-up is
included in the wall time. Speedup and efficiency are relative to the
1-process run. The script also checks that every process count
returned identical records, then writes the results into
data/demo/bench.json (key ``sweep_parallel``) and the figure
data/demo/bench_parallel.png.

Usage: .venv/Scripts/python.exe scripts/bench_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from egresstwin.layout.schema import Exit, Layout, SpawnZone  # noqa: E402
from egresstwin.parallel.sweep import get_pool, run_sweep, shutdown_pools  # noqa: E402
from egresstwin.scenarios.spec import (  # noqa: E402
    AddExitModification,
    ScenarioSpec,
    WidenExitModification,
)
from egresstwin.sim.engine import Scenario, run  # noqa: E402

ROOM_W, ROOM_H, CELL = 20.0, 12.0, 0.1


def bench_layout(people: int) -> Layout:
    """20 x 12 m room, exits of 1.2 m (east) and 1.0 m (west)."""
    rows, cols = int(ROOM_H / CELL), int(ROOM_W / CELL)
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    return Layout(
        cell_size_m=CELL,
        walkable_mask=walkable,
        exits=[
            Exit(id="E1", segment=((ROOM_W, 5.4), (ROOM_W, 6.6))),
            Exit(id="E2", segment=((0.0, 5.5), (0.0, 6.5))),
        ],
        spawn_zones=[
            SpawnZone(id="z", polygon=[(2.0, 2.0), (18.0, 2.0), (18.0, 10.0), (2.0, 10.0)],
                      count=people)
        ],
    )


def bench_specs(n: int, max_time_s: float) -> list[ScenarioSpec]:
    """A realistic agent batch: widenings, a new exit, N-1 blocks."""
    north = AddExitModification(segment=((9.4, ROOM_H), (10.6, ROOM_H)), exit_id="N")
    pool = [
        ScenarioSpec(name="baseline"),
        ScenarioSpec(name="widen-E1-2.0",
                     modifications=[WidenExitModification(exit_id="E1", width_m=2.0)]),
        ScenarioSpec(name="widen-E2-2.0",
                     modifications=[WidenExitModification(exit_id="E2", width_m=2.0)]),
        ScenarioSpec(name="widen-both-1.8", modifications=[
            WidenExitModification(exit_id="E1", width_m=1.8),
            WidenExitModification(exit_id="E2", width_m=1.8)]),
        ScenarioSpec(name="add-north-exit", modifications=[north]),
        ScenarioSpec(name="baseline-no-E1", blocked_exit_ids=["E1"]),
        ScenarioSpec(name="baseline-no-E2", blocked_exit_ids=["E2"]),
        ScenarioSpec(name="widen-E1-add-north", modifications=[
            WidenExitModification(exit_id="E1", width_m=2.0), north]),
    ]
    if n > len(pool):
        raise SystemExit(f"at most {len(pool)} scenarios are defined")
    return [s.model_copy(update={"max_time_s": max_time_s}) for s in pool[:n]]


def cpu_name() -> str:
    name = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--people", type=int, default=200)
    parser.add_argument("--max-time", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "data" / "demo"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layout = bench_layout(args.people)
    specs = bench_specs(args.scenarios, args.max_time)
    n_runs = len(specs) * args.replicas
    print(f"set: {len(specs)} scenarios x {args.replicas} replicas = {n_runs} runs, "
          f"{args.people} people, cpu_count={os.cpu_count()}")

    # Compile/load the Numba kernel in this process first (cache=True
    # lets workers load it from disk instead of recompiling).
    run(bench_layout(5), Scenario(max_time_s=5.0), seed=0)

    runs: list[dict] = []
    reference: list[dict] | None = None
    for workers in args.workers:
        shutdown_pools()
        started = time.perf_counter()
        pool = get_pool(workers)
        # Start-up alone: one trivial task per worker on the cold pool.
        list(pool.map(abs, range(workers)))
        startup_s = time.perf_counter() - started
        records = run_sweep(layout, specs, replicas=args.replicas, seed=args.seed,
                            backend="local", workers=workers)
        wall_s = time.perf_counter() - started
        shutdown_pools()
        dumped = [asdict(r) for r in records]
        if reference is None:
            reference = dumped
        identical = dumped == reference
        runs.append({"workers": workers, "wall_s": round(wall_s, 3),
                     "startup_s": round(startup_s, 3),
                     "identical_to_1_process": identical})
        print(f"workers={workers}: wall {wall_s:.2f} s (start-up {startup_s:.2f} s), "
              f"identical={identical}")

    base = runs[0]["wall_s"]
    for entry in runs:
        speedup = base / entry["wall_s"]
        entry["speedup"] = round(speedup, 3)
        entry["efficiency"] = round(speedup / (entry["workers"] / runs[0]["workers"]), 3)
        entry["runs_per_min"] = round(n_runs / entry["wall_s"] * 60.0, 2)

    assert reference is not None
    summary_rows = [
        {"scenario": r["name"],
         "t95_mean_s": round(float(np.mean([x["t95_s"] for x in r["replicas"]])), 2),
         "exited": f"{sum(x['n_exited'] for x in r['replicas'])}/"
                   f"{sum(x['n_people'] for x in r['replicas'])}"}
        for r in reference
    ]
    result = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": {"cpu": cpu_name(), "logical_cpus": os.cpu_count(),
                    "os": platform.platform(), "python": platform.python_version()},
        "set": {"scenarios": len(specs), "replicas": args.replicas, "runs": n_runs,
                "people": args.people, "max_time_s": args.max_time,
                "room_m": [ROOM_W, ROOM_H], "cell_size_m": CELL},
        "backend": "local (ProcessPoolExecutor, spawn, NUMBA_NUM_THREADS=1 per worker)",
        "note": "Wall time includes cold process-pool start-up. Speedup and "
                "efficiency are relative to the 1-process run.",
        "runs": runs,
        "scenario_results": summary_rows,
    }
    for entry in runs:
        print(f"  p={entry['workers']}: speedup {entry['speedup']}, "
              f"efficiency {entry['efficiency']}, {entry['runs_per_min']} runs/min")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    procs = [r["workers"] for r in runs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(procs, [r["wall_s"] for r in runs], marker="o")
    ax1.set_xlabel("worker processes")
    ax1.set_ylabel("wall-clock seconds")
    ax1.set_xticks(procs)
    ax1.set_title(f"Sweep {len(specs)} scenarios x {args.replicas} replicas, "
                  f"{args.people} people")
    ax2.plot(procs, [r["speedup"] for r in runs], marker="o", label="measured")
    ax2.plot(procs, [p / procs[0] for p in procs], linestyle="--", color="gray",
             label="ideal")
    for r in runs:
        ax2.annotate(f"eff {r['efficiency']:.2f}", (r["workers"], r["speedup"]),
                     textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax2.set_xlabel("worker processes")
    ax2.set_ylabel("speedup vs 1 process")
    ax2.set_xticks(procs)
    ax2.legend()
    ax2.set_title(result["machine"]["cpu"] or "speedup")
    fig.tight_layout()
    fig.savefig(out / "bench_parallel.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    bench_file = out / "bench.json"
    bench = json.loads(bench_file.read_text(encoding="utf-8")) if bench_file.is_file() else {}
    bench["sweep_parallel"] = result
    figures = [f for f in bench.get("figures", []) if f != "bench_parallel.png"]
    bench["figures"] = ["bench_parallel.png", *figures]
    bench_file.write_text(json.dumps(bench, indent=2), encoding="utf-8")
    print(f"wrote {bench_file} and {out / 'bench_parallel.png'}")
    return 0 if all(r["identical_to_1_process"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
