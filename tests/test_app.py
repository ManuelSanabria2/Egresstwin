"""FastAPI app tests with injected fast runners (no network)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from egresstwin.app.api import create_app, simulate_once
from egresstwin.layout.schema import Exit, Layout, SpawnZone


def small_layout() -> Layout:
    cell, rows, cols = 0.25, 24, 48
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    return Layout(
        cell_size_m=cell, walkable_mask=walkable,
        exits=[Exit(id="E", segment=((12.0, 2.5), (12.0, 3.5))),
               Exit(id="W", segment=((0.0, 2.5), (0.0, 3.5)))],
        spawn_zones=[SpawnZone(id="z", polygon=[(1.0, 1.0), (5.0, 1.0),
                                                (5.0, 5.0), (1.0, 5.0)], count=3)],
    )


def fake_perception(image_path, output_dir, scale_kwargs):  # noqa: ANN001, ANN202
    return SimpleNamespace(layout=small_layout(), report={"warnings": ["demo warning"]})


def fake_agent(layout, objective, on_event, runs_dir, run_id, max_changes, constraints):  # noqa: ANN001, ANN202
    on_event({"event": "turn", "turn": 1})
    on_event({"event": "tool_call", "turn": 1, "tool": "get_baseline", "arguments": {}})
    animation = simulate_once(layout)
    return {
        "status": "VERIFIED",
        "report": {"baseline_id": "baseline", "objective": objective,
                   "recommendations": [], "robustness": [],
                   "limitations": ["test"], "summary": "honest test report"},
        "verification": {"status": "VERIFIED", "checks": [
            {"claim": "baseline_exists", "passed": True, "detail": "found"}]},
        "reason": "report verified", "total_cost_usd": 0.002, "turns": 1,
        "scenarios_run": 1, "animations": {"baseline": animation},
    }


def fake_bench(out_dir):  # noqa: ANN001, ANN202
    return {"scaling": [{"n_people": 3, "wall_s": 0.1}],
            "throughput": {"scenarios": 1, "replicas": 1, "wall_s": 0.1,
                           "scenarios_per_min": 600.0}, "figures": []}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    demo = tmp_path / "demo"
    demo.mkdir()
    small_layout().save_npz(demo / "layout.npz")
    image = tmp_path / "plan.png"
    Image.new("RGB", (48, 24), "white").save(image)
    animation = simulate_once(small_layout())
    (demo / "simulation.json").write_text(json.dumps(
        {"name": "baseline", "aggregates": {}, "animation": animation}), encoding="utf-8")
    agent_demo = {
        "status": "VERIFIED", "report": {"summary": "demo"},
        "verification": {"status": "VERIFIED", "checks": []},
        "events": [{"event": "run_start"}], "animations": {"baseline": animation},
        "total_cost_usd": 0.0,
    }
    (demo / "agent.json").write_text(json.dumps(agent_demo), encoding="utf-8")
    (demo / "bench.json").write_text(json.dumps(fake_bench(demo)), encoding="utf-8")
    (demo / "manifest.json").write_text(json.dumps({
        "layout_file": "layout.npz", "image": str(image),
        "simulation_file": "simulation.json", "agent_file": "agent.json",
        "bench_file": "bench.json", "perception": {"warnings": []},
    }), encoding="utf-8")
    app = create_app(
        runs_dir=tmp_path / "runs", demo_dir=demo,
        perception_runner=fake_perception, agent_factory=fake_agent,
        bench_runner=fake_bench,
    )
    return TestClient(app)


def wait_job(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def make_layout(client: TestClient) -> str:
    image = Image.new("RGB", (64, 48), "white")
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    started = client.post(
        "/api/layouts", files={"image": ("plan.png", buf.getvalue(), "image/png")}
    ).json()
    job = wait_job(client, started["job_id"])
    assert job["status"] == "done", job
    return started["layout_id"]


def test_health_and_index(client: TestClient) -> None:
    assert client.get("/api/health").json()["ok"] is True
    page = client.get("/")
    assert page.status_code == 200
    assert "EgressTwin" in page.text and "tab-agent" in page.text


def test_layout_flow_and_edits(client: TestClient) -> None:
    layout_id = make_layout(client)
    payload = client.get(f"/api/layouts/{layout_id}").json()
    assert [e["id"] for e in payload["exits"]] == ["E", "W"]
    assert payload["warnings"]
    added = client.post(
        f"/api/layouts/{layout_id}/edits",
        json={"action": "add_exit", "segment_px": [[0, 4], [0, 8]]},
    )
    assert added.status_code == 200
    assert len(added.json()["exits"]) == 3
    new_id = added.json()["exits"][-1]["id"]
    removed = client.post(
        f"/api/layouts/{layout_id}/edits",
        json={"action": "remove_exit", "exit_id": new_id},
    )
    assert removed.status_code == 200
    assert len(removed.json()["exits"]) == 2
    bad = client.post(
        f"/api/layouts/{layout_id}/edits",
        json={"action": "remove_exit", "exit_id": "missing"},
    )
    assert bad.status_code == 400


def test_simulate_and_optimize_sse(client: TestClient) -> None:
    layout_id = make_layout(client)
    started = client.post(
        f"/api/layouts/{layout_id}/simulate", json={"replicas": 1}
    ).json()
    job = wait_job(client, started["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["animation"]["frames"]
    assert job["result"]["aggregates"]["t95_s"]["n"] == 1

    started = client.post(
        f"/api/layouts/{layout_id}/optimize", json={"objective": "minimize_t95"}
    ).json()
    with client.stream("GET", f"/api/optimize/{started['job_id']}/stream") as stream:
        body = "".join(chunk for chunk in stream.iter_text())
    assert "tool_call" in body and "stream_end" in body
    job = wait_job(client, started["job_id"])
    assert job["result"]["status"] == "VERIFIED"
    assert job["result"]["animations"]["baseline"]["frames"]


def test_demo_mode_and_bench(client: TestClient) -> None:
    demo = client.get("/api/demo")
    assert demo.status_code == 200
    data = demo.json()
    assert data["layout"]["layout_id"] == "demo"
    assert data["simulation"]["animation"]["frames"]
    assert data["agent"]["verification"]["status"] == "VERIFIED"
    assert client.get("/api/layouts/demo/image").status_code == 200

    started = client.post("/api/bench/run").json()
    job = wait_job(client, started["job_id"])
    assert job["status"] == "done"
    assert job["result"]["throughput"]["scenarios_per_min"] > 0
    assert client.get("/api/bench").json()["figures"] == []
