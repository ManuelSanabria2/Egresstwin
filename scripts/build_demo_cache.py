"""Build the offline demo cache in data/demo from a recorded real run.

Nothing here calls the API or scripts the agent. The input is the
output directory of a real-mode run of scripts/demo.py (Muse Spark
perception and Muse Spark agent through the Meta Model API, saved
under data/real_runs/<plan>/). The cache reuses that run's layout,
perception report, agent trace (with the real response ids, token
usage and costs), report and verification; it only re-simulates the
baseline and recommended layouts locally to produce the canvas
animations, which are deterministic.

Usage:
    .venv/Scripts/python.exe scripts/build_demo_cache.py \
        --run data/real_runs/test_office
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egresstwin.app.api import simulate_once  # noqa: E402
from egresstwin.layout.schema import Layout  # noqa: E402
from egresstwin.scenarios.apply import apply_modifications  # noqa: E402

OUT = ROOT / "data" / "demo"


def _read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="saved real-run directory")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("mode") != "real":
        print("ERROR: the demo cache is built only from a real-mode run")
        return 2
    agent_dir = run / "demo-cli"
    events = _read_trace(agent_dir / "trace.jsonl")
    turns = [e for e in events if e.get("event") == "turn"]
    models = sorted({str(e.get("model") or "") for e in turns})
    if not turns or any(m.startswith("scripted") or not m for m in models):
        print(f"ERROR: trace is not a model run (models: {models})")
        return 2
    done = next(e for e in reversed(events) if e.get("event") == "run_done")
    report = json.loads((agent_dir / "report.json").read_text(encoding="utf-8")) \
        if (agent_dir / "report.json").is_file() else None
    verification = json.loads((agent_dir / "verification.json").read_text(encoding="utf-8")) \
        if (agent_dir / "verification.json").is_file() else None
    scenarios = json.loads((agent_dir / "scenarios.json").read_text(encoding="utf-8"))
    perception = json.loads(
        (run / "perception" / "layout_report.json").read_text(encoding="utf-8"))

    OUT.mkdir(parents=True, exist_ok=True)
    layout = Layout.load_npz(run / "layout.npz")
    shutil.copyfile(run / "layout.npz", OUT / "layout.npz")
    max_time_s = float(summary["max_time_s"])

    # Simulation tab: the run's own baseline aggregates, plus one
    # deterministic animation of the same layout.
    base = scenarios["baseline"]
    animation = simulate_once(layout, max_time_s=max_time_s)
    simulation = {
        "name": "baseline",
        "aggregates": base["aggregates"],
        "n_replicas": len(base["replicas"]),
        "animation": animation,
    }
    (OUT / "simulation.json").write_text(json.dumps(simulation), encoding="utf-8")

    animations: dict = {"baseline": animation}
    for rec in (report or {}).get("recommendations", [])[:1]:
        modified = apply_modifications(layout, rec.get("modifications", []))
        animations["recommended"] = simulate_once(modified, max_time_s=max_time_s)
        animations["recommended_scenario_id"] = rec.get("scenario_id")
    agent = {
        "status": done.get("status"),
        "report": report,
        "verification": verification,
        "reason": done.get("reason", ""),
        "total_cost_usd": done.get("total_cost_usd"),
        "turns": done.get("turns"),
        "scenarios_run": done.get("scenarios_run"),
        "model": ", ".join(models),
        "events": events,
        "animations": animations,
    }
    (OUT / "agent.json").write_text(json.dumps(agent), encoding="utf-8")

    # Old scripted-run files would contradict the real trace.
    stale = OUT / "agent_run"
    if stale.is_dir():
        shutil.rmtree(stale)

    image = Path(summary["image"])
    try:
        image_rel = image.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        image_rel = image.as_posix()
    manifest = {
        "layout_file": "layout.npz",
        "image": image_rel,
        "simulation_file": "simulation.json",
        "agent_file": "agent.json",
        "bench_file": "bench.json",
        "source_run": run.relative_to(ROOT).as_posix() if run.is_relative_to(ROOT) else str(run),
        "perception": perception,
        "note": (
            f"Recorded real run: perception and agent by {', '.join(models)} through "
            "the Meta Model API (response ids, token usage and costs are in the "
            "agent trace). Replaying it needs no network or API key; only the canvas "
            "animations are re-simulated locally."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"demo cache from {run}: {agent['status']}, model {agent['model']}, "
          f"{len(turns)} turns, USD {agent['total_cost_usd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
