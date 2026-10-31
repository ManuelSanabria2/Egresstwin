"""FastAPI backend: image -> layout -> simulation -> verified agent.

Long work (perception, simulation, optimization, benchmarks) runs in
background threads registered in a small in-memory job registry that
also mirrors its status to runs/jobs/. The demo endpoint replays a
complete cached run from data/demo so the demo works with no network
and no API key.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from egresstwin.layout import edit as layout_edit
from egresstwin.layout.schema import Layout
from egresstwin.layout.validate import LayoutValidationError, validate_layout

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / "static"
DEFAULT_DEMO = ROOT / "data" / "demo"


# ---------------------------------------------------------------- jobs


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "queued"  # queued | running | done | error
    result: dict[str, Any] | None = None
    error: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class JobRegistry:
    """In-memory jobs with a JSON mirror on disk."""

    def __init__(self, runs_dir: Path) -> None:
        self.jobs: dict[str, Job] = {}
        self.dir = runs_dir / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def create(self, kind: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind)
        self.jobs[job.job_id] = job
        self._mirror(job)
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id!r}")
        return job

    def run(self, job: Job, fn: Callable[[Job], dict[str, Any]]) -> Job:
        def target() -> None:
            job.status = "running"
            self._mirror(job)
            try:
                job.result = fn(job)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - surfaced in job status
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
            self._mirror(job)

        threading.Thread(target=target, daemon=True).start()
        return job

    def _mirror(self, job: Job) -> None:
        payload = {
            "job_id": job.job_id, "kind": job.kind, "status": job.status,
            "error": job.error, "has_result": job.result is not None,
            "n_events": len(job.events),
        }
        (self.dir / f"{job.job_id}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


# ------------------------------------------------------------ payloads


def _grid_strings(mask: np.ndarray, max_cols: int = 220) -> dict[str, Any]:
    rows, cols = mask.shape
    step = max(1, int(np.ceil(cols / max_cols)))
    small = mask[::step, ::step]
    return {
        "step": step,
        "rows": ["".join("1" if v else "0" for v in row) for row in small],
    }


def _segment_px(segment: Any, scale: float) -> list[list[float]]:
    return [[round(p[0] * scale, 1), round(p[1] * scale, 1)] for p in segment]


@dataclass
class LayoutRecord:
    layout_id: str
    layout: Layout
    directory: Path
    source_image: Path | None = None
    image_size: tuple[int, int] | None = None  # (w, h) px
    perception: dict[str, Any] | None = None
    note: str = ""

    @property
    def px_per_meter(self) -> float:
        if self.layout.meters_per_pixel:
            return 1.0 / self.layout.meters_per_pixel
        return 1.0 / self.layout.cell_size_m  # grid cells as pixels

    def payload(self) -> dict[str, Any]:
        layout = self.layout
        scale = self.px_per_meter
        warnings: list[str] = []
        if self.perception:
            warnings.extend(self.perception.get("warnings", []))
        try:
            warnings.extend(validate_layout(layout))
        except LayoutValidationError as exc:
            warnings.append(f"INVALID: {exc}")
        obstacles = []
        for obs in layout.obstacles:
            mask = obs.rasterized(layout.shape, layout.cell_size_m)
            ys, xs = np.where(mask)
            obstacles.append(
                {
                    "id": obs.id, "label": obs.label, "removable": obs.removable,
                    "cells": int(mask.sum()),
                    "centroid_px": (
                        [round(float(xs.mean() * layout.cell_size_m * scale), 1),
                         round(float(ys.mean() * layout.cell_size_m * scale), 1)]
                        if len(xs) else None
                    ),
                    "grid": _grid_strings(mask),
                }
            )
        return {
            "layout_id": self.layout_id,
            "note": self.note,
            "grid": {"rows": layout.shape[0], "cols": layout.shape[1],
                     "cell_size_m": layout.cell_size_m},
            "meters_per_pixel": layout.meters_per_pixel,
            "px_per_meter": scale,
            "image_size": (
                {"w": self.image_size[0], "h": self.image_size[1]}
                if self.image_size else None
            ),
            "walkable": _grid_strings(layout.effective_walkable()),
            "exits": [
                {"id": e.id, "width_m": round(e.width_m, 3),
                 "segment_m": [[p[0], p[1]] for p in e.segment],
                 "segment_px": _segment_px(e.segment, scale)}
                for e in layout.exits
            ],
            "obstacles": obstacles,
            "spawn_zones": [
                {"id": z.id, "planned_people": z.planned_count(),
                 "polygon_px": [[round(p[0] * scale, 1), round(p[1] * scale, 1)]
                                for p in z.polygon]}
                for z in layout.spawn_zones
            ],
            "warnings": warnings,
            "perception": self.perception,
        }


class EditRequest(BaseModel):
    action: str
    exit_id: str | None = None
    segment_px: list[list[float]] | None = None
    points_px: list[list[float]] | None = None
    obstacle_id: str | None = None
    width_m: float = 0.2


class SimulateRequest(BaseModel):
    replicas: int = Field(default=3, ge=1, le=20)
    seed: int = 42
    max_time_s: float = Field(default=90.0, gt=0)
    modifications: list[dict[str, Any]] = Field(default_factory=list)
    blocked_exit_ids: list[str] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)  # timed events (P07)
    name: str = "simulation"


class OptimizeRequest(BaseModel):
    objective: str = "minimize_t95"
    replicas: int = Field(default=5, ge=1, le=10)
    max_changes: int = Field(default=3, ge=1, le=6)
    constraints: str = ""


# ------------------------------------------------------------- runners


def animation_payload(layout: Layout, result: Any, metrics: Any) -> dict[str, Any]:
    """Frames + density grid + metrics of one simulation, for the canvas."""
    frames = [
        [[round(float(x), 3), round(float(y), 3)] for x, y in frame]
        for frame in result.frames
    ]
    density = result.density_accum / max(result.n_steps, 1) / (result.cell_size_m**2)
    step = max(1, int(np.ceil(density.shape[1] / 90)))
    small = density[::step, ::step]
    return {
        "width_m": layout.width_m,
        "height_m": layout.height_m,
        "frame_dt": result.frame_dt,
        "frames": frames,
        "exits": [
            {"id": e.id, "segment_m": [[p[0], p[1]] for p in e.segment]}
            for e in layout.exits
        ],
        "walkable": _grid_strings(layout.effective_walkable()),
        "density": {
            "step_cells": step,
            "max": round(float(small.max()) if small.size else 0.0, 3),
            "rows": [[round(float(v), 3) for v in row] for row in small],
        },
        "metrics": {
            "total_time_s": metrics.total_time_s, "t95_s": metrics.t95_s,
            "mean_egress_s": metrics.mean_egress_s,
            "peak_density_per_m2": metrics.peak_density_per_m2,
            "n_people": metrics.n_people, "n_exited": metrics.n_exited,
            "hotspots": metrics.hotspots[:5],
        },
        "events": getattr(result, "applied_events", []),
        "warnings": getattr(result, "warnings", []),
    }


def simulate_once(layout: Layout, *, seed: int = 42, max_time_s: float = 90.0,
                  blocked: frozenset[str] = frozenset(),
                  events: tuple = ()) -> dict[str, Any]:
    from egresstwin.sim.engine import Scenario, run
    from egresstwin.sim.metrics import compute_metrics

    result = run(
        layout,
        Scenario(max_time_s=max_time_s, blocked_exit_ids=blocked, events=tuple(events)),
        seed=seed,
    )
    return animation_payload(layout, result, compute_metrics(result, layout))


def default_perception_runner(image_path: Path, output_dir: Path,
                              scale_kwargs: dict[str, Any]) -> Any:
    from egresstwin.perception.pipeline import image_to_layout

    return image_to_layout(image_path, output_dir=output_dir, **scale_kwargs)


def default_bench_runner(out_dir: Path) -> dict[str, Any]:
    """Quick scaling/throughput benchmark with matplotlib figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from egresstwin.layout.schema import Exit, Layout, SpawnZone
    from egresstwin.parallel.sweep import run_sweep
    from egresstwin.scenarios.spec import ScenarioSpec
    from egresstwin.sim.engine import Scenario, run

    def room(n_people: int) -> Layout:
        cell, rows, cols = 0.25, 24, 48
        walkable = np.zeros((rows, cols), dtype=bool)
        walkable[1:-1, 1:-1] = True
        return Layout(
            cell_size_m=cell, walkable_mask=walkable,
            exits=[Exit(id="E", segment=((12.0, 2.5), (12.0, 3.5))),
                   Exit(id="W", segment=((0.0, 2.5), (0.0, 3.5)))],
            spawn_zones=[SpawnZone(id="z", polygon=[(1.0, 1.0), (11.0, 1.0),
                                                    (11.0, 5.0), (1.0, 5.0)],
                                   count=n_people)],
        )

    scaling = []
    for n in (30, 100, 300):
        started = time.perf_counter()
        run(room(n), Scenario(max_time_s=120.0), seed=42)
        scaling.append({"n_people": n, "wall_s": round(time.perf_counter() - started, 3)})
    specs = [
        ScenarioSpec(name=f"bench-{i}", modifications=[], max_time_s=60.0) for i in range(4)
    ]
    started = time.perf_counter()
    run_sweep(room(40), specs, replicas=3, seed=42)
    sweep_s = time.perf_counter() - started
    throughput = {"scenarios": 4, "replicas": 3, "wall_s": round(sweep_s, 3),
                  "scenarios_per_min": round(12 / max(sweep_s, 1e-9) * 60, 2)}

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([s["n_people"] for s in scaling], [s["wall_s"] for s in scaling], marker="o")
    ax.set_xlabel("people per simulation")
    ax.set_ylabel("wall-clock seconds (serial CPU)")
    ax.set_title("Simulation scaling")
    fig.savefig(out_dir / "bench_scaling.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["sweep 4 scenarios x 3 replicas"], [throughput["scenarios_per_min"]])
    ax.set_ylabel("scenarios per minute")
    ax.set_title("Sweep throughput (serial backend)")
    fig.savefig(out_dir / "bench_throughput.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    summary = {"scaling": scaling, "throughput": throughput,
               "figures": ["bench_scaling.png", "bench_throughput.png"]}
    (out_dir / "bench.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def default_agent_factory(layout: Layout, objective: str, on_event, runs_dir: Path,
                          run_id: str, max_changes: int, constraints: str) -> dict[str, Any]:
    """Real Muse Spark loop plus baseline/recommended animations."""
    from egresstwin.agent.loop import AgentLoop
    from egresstwin.scenarios.apply import apply_modifications

    loop = AgentLoop(
        layout, objective=objective, max_changes=max_changes,  # type: ignore[arg-type]
        user_constraints=constraints, runs_dir=runs_dir, run_id=run_id, on_event=on_event,
    )
    result = loop.run()
    animations: dict[str, Any] = {"baseline": simulate_once(layout)}
    report = result.report or {}
    for rec in report.get("recommendations", [])[:1]:
        try:
            modified = apply_modifications(layout, rec.get("modifications", []))
            animations["recommended"] = simulate_once(modified)
            animations["recommended_scenario_id"] = rec.get("scenario_id")
        except Exception as exc:  # noqa: BLE001 - shown in the result
            animations["recommended_error"] = str(exc)
    return {
        "status": result.status, "report": result.report,
        "verification": result.verification, "reason": result.reason,
        "total_cost_usd": result.total_cost_usd, "turns": result.turns,
        "scenarios_run": result.scenarios_run, "animations": animations,
    }


# ---------------------------------------------------------------- app


def create_app(
    *,
    runs_dir: str | Path = "runs",
    demo_dir: str | Path | None = None,
    perception_runner: Callable | None = None,
    agent_factory: Callable | None = None,
    bench_runner: Callable | None = None,
) -> FastAPI:
    runs = Path(runs_dir)
    demo = Path(demo_dir) if demo_dir is not None else DEFAULT_DEMO
    registry = JobRegistry(runs / "app")
    layouts: dict[str, LayoutRecord] = {}
    app = FastAPI(title="EgressTwin")
    perception = perception_runner or default_perception_runner
    bench = bench_runner or default_bench_runner

    def register(record: LayoutRecord) -> LayoutRecord:
        layouts[record.layout_id] = record
        record.directory.mkdir(parents=True, exist_ok=True)
        record.layout.save_npz(record.directory / "layout.npz")
        return record

    def require_layout(layout_id: str) -> LayoutRecord:
        record = layouts.get(layout_id)
        if record is None:
            raise HTTPException(404, f"unknown layout {layout_id!r}")
        return record

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "layouts": len(layouts), "jobs": len(registry.jobs)}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.post("/api/layouts")
    async def create_layout(
        image: UploadFile = File(...),  # noqa: B008 - FastAPI upload declaration
        scale_mode: str = Form("annotation"),
        scale_distance_m: float | None = Form(None),
        scale_points: str | None = Form(None),
        known_door_width_m: float | None = Form(None),
        known_door_box: str | None = Form(None),
    ) -> dict[str, Any]:
        layout_id = uuid.uuid4().hex[:12]
        directory = runs / "app" / "layouts" / layout_id
        directory.mkdir(parents=True, exist_ok=True)
        suffix = Path(image.filename or "plan.png").suffix or ".png"
        image_path = directory / f"source{suffix}"
        image_path.write_bytes(await image.read())
        scale_kwargs: dict[str, Any] = {}
        if scale_mode == "two_points":
            if not scale_points or scale_distance_m is None:
                raise HTTPException(400, "two_points needs scale_points and scale_distance_m")
            pts = json.loads(scale_points)
            scale_kwargs = {"scale_two_points": (tuple(pts[0]), tuple(pts[1])),
                            "scale_distance_m": scale_distance_m}
        elif scale_mode == "known_door":
            if known_door_box is None or known_door_width_m is None:
                raise HTTPException(400, "known_door needs known_door_box and width")
            scale_kwargs = {"known_door": (json.loads(known_door_box), known_door_width_m)}

        def work(_job: Job) -> dict[str, Any]:
            result = perception(image_path, directory / "perception", scale_kwargs)
            layout = result.layout if hasattr(result, "layout") else result
            from PIL import Image

            with Image.open(image_path) as im:
                size = im.size
            record = register(LayoutRecord(
                layout_id=layout_id, layout=layout, directory=directory,
                source_image=image_path, image_size=size,
                perception=getattr(result, "report", None),
            ))
            return record.payload()

        job = registry.run(registry.create("perception"), work)
        return {"job_id": job.job_id, "layout_id": layout_id, "status": job.status}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = registry.get(job_id)
        return {"job_id": job.job_id, "kind": job.kind, "status": job.status,
                "error": job.error, "result": job.result,
                "n_events": len(job.events)}

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str) -> dict[str, Any]:
        return {"events": registry.get(job_id).events}

    @app.get("/api/layouts/{layout_id}")
    def get_layout(layout_id: str) -> dict[str, Any]:
        return require_layout(layout_id).payload()

    @app.get("/api/layouts/{layout_id}/image")
    def get_image(layout_id: str) -> FileResponse:
        record = require_layout(layout_id)
        if record.source_image is None or not record.source_image.is_file():
            raise HTTPException(404, "this layout has no source image")
        return FileResponse(record.source_image)

    @app.post("/api/layouts/{layout_id}/edits")
    def edit_layout(layout_id: str, req: EditRequest) -> dict[str, Any]:
        record = require_layout(layout_id)
        layout = record.layout
        scale = record.px_per_meter

        def to_m(points: list[list[float]] | None) -> list[tuple[float, float]]:
            return [(p[0] / scale, p[1] / scale) for p in (points or [])]

        try:
            if req.action == "add_exit":
                if not req.segment_px or len(req.segment_px) != 2:
                    raise ValueError("add_exit needs segment_px with 2 points")
                seg = to_m(req.segment_px)
                new = layout_edit.add_exit(layout, (seg[0], seg[1]))
            elif req.action == "remove_exit":
                new = layout_edit.remove_exit(layout, req.exit_id or "")
            elif req.action == "mark_walkable":
                new = layout_edit.mark_obstacle_walkable(layout, req.obstacle_id or "")
            elif req.action == "paint_wall":
                pts = to_m(req.points_px)
                if len(pts) < 2:
                    raise ValueError("paint_wall needs at least 2 points")
                new = layout_edit.paint_wall(layout, pts, width_m=req.width_m)
            else:
                raise ValueError(f"unknown edit action {req.action!r}")
            validate_layout(new)
        except (ValueError, LayoutValidationError) as exc:
            raise HTTPException(400, str(exc)) from exc
        record.layout = new
        new.save_npz(record.directory / "layout.npz")
        return record.payload()

    @app.post("/api/layouts/{layout_id}/simulate")
    def simulate(layout_id: str, req: SimulateRequest) -> dict[str, Any]:
        record = require_layout(layout_id)

        def work(_job: Job) -> dict[str, Any]:
            from egresstwin.parallel.store import METRIC_KEYS
            from egresstwin.parallel.sweep import run_sweep
            from egresstwin.scenarios.apply import apply_modifications
            from egresstwin.scenarios.spec import ScenarioSpec

            spec = ScenarioSpec(
                name=req.name, modifications=req.modifications,  # type: ignore[arg-type]
                blocked_exit_ids=req.blocked_exit_ids, max_time_s=req.max_time_s,
                events=req.events,  # type: ignore[arg-type]
            )
            records = run_sweep(
                record.layout, [spec], replicas=req.replicas, seed=req.seed
            )
            rec = records[0]
            modified = apply_modifications(record.layout, list(spec.modifications))
            animation = simulate_once(
                modified, seed=req.seed, max_time_s=req.max_time_s,
                blocked=frozenset(req.blocked_exit_ids),
                events=tuple(spec.events),
            )
            return {
                "name": req.name,
                "aggregates": {k: rec.aggregate(k) for k in METRIC_KEYS},
                "n_replicas": len(rec.replicas),
                "exited_per_replica": [
                    {"exited": r.n_exited, "people": r.n_people} for r in rec.replicas
                ],
                "animation": animation,
            }

        job = registry.run(registry.create("simulation"), work)
        return {"job_id": job.job_id, "status": job.status}

    @app.post("/api/layouts/{layout_id}/optimize")
    def optimize(layout_id: str, req: OptimizeRequest) -> dict[str, Any]:
        record = require_layout(layout_id)
        run_id = f"agent-{uuid.uuid4().hex[:8]}"

        def work(job: Job) -> dict[str, Any]:
            def on_event(event: dict[str, Any]) -> None:
                job.events.append(event)

            if agent_factory is not None:
                return agent_factory(
                    record.layout, req.objective, on_event, runs / "app",
                    run_id, req.max_changes, req.constraints,
                )
            return default_agent_factory(
                record.layout, req.objective, on_event, runs / "app",
                run_id, req.max_changes, req.constraints,
            )

        job = registry.run(registry.create("optimize"), work)
        return {"job_id": job.job_id, "run_id": run_id, "status": job.status}

    @app.get("/api/optimize/{job_id}/stream")
    def optimize_stream(job_id: str) -> StreamingResponse:
        job = registry.get(job_id)

        def gen():  # noqa: ANN202
            sent = 0
            while True:
                while sent < len(job.events):
                    yield f"data: {json.dumps(job.events[sent], default=str)}\n\n"
                    sent += 1
                if job.status in ("done", "error"):
                    yield f"data: {json.dumps({'event': 'stream_end', 'status': job.status})}\n\n"
                    return
                time.sleep(0.25)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/demo")
    def demo_mode() -> dict[str, Any]:
        manifest_path = demo / "manifest.json"
        if not manifest_path.is_file():
            raise HTTPException(404, "demo cache not built; run scripts/build_demo_cache.py")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        layout = Layout.load_npz(demo / manifest["layout_file"])
        image = ROOT / manifest["image"]
        size = None
        source = None
        if image.is_file():
            from PIL import Image

            with Image.open(image) as im:
                size = im.size
            source = image
        record = register(LayoutRecord(
            layout_id="demo", layout=layout,
            directory=runs / "app" / "layouts" / "demo",
            source_image=source, image_size=size,
            perception=manifest.get("perception"), note="demo cache (no network)",
        ))
        def load(name: str) -> dict[str, Any]:
            return json.loads((demo / manifest[name]).read_text(encoding="utf-8"))

        return {
            "layout": record.payload(),
            "simulation": load("simulation_file"),
            "agent": load("agent_file"),
            "bench": load("bench_file"),
        }

    @app.get("/api/bench")
    def bench_summary() -> dict[str, Any]:
        for candidate in (runs / "bench" / "bench.json", demo / "bench.json"):
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        return {"figures": [], "note": "no benchmark run yet"}

    @app.post("/api/bench/run")
    def bench_run() -> dict[str, Any]:
        out_dir = runs / "bench"

        def work(_job: Job) -> dict[str, Any]:
            return bench(out_dir)

        job = registry.run(registry.create("bench"), work)
        return {"job_id": job.job_id, "status": job.status}

    @app.get("/api/bench/figures/{name}")
    def bench_figure(name: str) -> FileResponse:
        safe = Path(name).name
        for candidate in (runs / "bench" / safe, demo / safe):
            if candidate.is_file():
                return FileResponse(candidate, media_type="image/png")
        raise HTTPException(404, f"no bench figure {safe!r}")

    return app


app = create_app()
