# EgressTwin

EgressTwin is a verifiable AI agent that turns an image of a space (a building
floor plan or an aerial photo of an open space) into an evacuation digital
twin, simulates it in parallel, and only recommends changes it can prove with
simulations.

Status: project scaffold. The Meta Model API client (mock mode, disk cache,
cost tracking and budget guard) is in place; the product pipeline is TBD.

## Quick start (mock mode, no network, no cost)

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use source .venv/bin/activate on Linux
pip install -e ".[dev]"
set META_MOCK=1               # Windows cmd; use export on Linux/macOS
pytest
python scripts/smoke_meta.py
```

With `uv`:

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
```

## Simulator (P02)

Deterministic social-force evacuation simulator: `layout/` holds the
`Layout` model (walkable grid serialised to `.npz`, obstacles, exits,
spawn zones) and its validation; `sim/` holds the geodesic floor field
(per-exit Dijkstra), the precomputed wall-distance field, the Numba
social-force kernel (pure NumPy reference in `sim/reference.py` for
tests), the engine (`run(layout, scenario, seed) -> SimResult`) and
metrics (total/T95/mean egress time, flow per exit, peak density,
time above a density threshold, hotspots).

```bash
python scripts/run_example.py   # 30 x 20 m room, 2 exits, 1000 people
```

### Social-force parameters

Model and parameters follow Helbing, D., Farkas, I. and Vicsek, T.,
"Simulating dynamical features of escape panic", *Nature* 407,
487-490 (2000), doi:10.1038/35035023 (preprint arXiv:cond-mat/0009448),
verified against the paper text:

| Parameter | Value | Source in the paper |
| --- | --- | --- |
| Mass m | 80 kg | "a mass of m_i = 80 kg" |
| Relaxation time tau | 0.5 s | "acceleration time tau_i = 0.5 s" |
| Interaction strength A | 2000 N | "A_i = 2 x 10^3 N" |
| Interaction range B | 0.08 m | "B_i = 0.08 m" |
| Body force k | 1.2e5 kg/s^2 | "k = 1.2 x 10^5 kg/s^2" |
| Sliding friction kappa | 2.4e5 kg/(m s) | "kappa = 2.4 x 10^5 kg/(m s)" |
| Pedestrian diameter | uniform 0.5-0.7 m (radius 0.25-0.35 m) | "diameters 2r_i in [0.5 m, 0.7 m]" |

Desired speeds are drawn from a normal distribution with mean
1.34 m/s and sd 0.26 m/s (clipped to 0.4-2.5 m/s). This distribution
is the conventional normal-walking choice adopted by this project; it
is not from the 2000 paper, which uses scenario desired speeds of
about 0.6 m/s (relaxed), 1.0 m/s (normal) and 1.5 m/s (nervous).

Integration is semi-implicit Euler at dt = 0.01 s with 5 force
substeps per step (the stiff contact spring is unstable at 0.01 s in
dense overlap) and a per-agent speed cap of 1.3 x desired speed.
Same seed, same result: the kernel is serial and deterministic in
P02; parallelism arrives in P03.

## Web app and demo

```bash
.venv/Scripts/python.exe -m uvicorn egresstwin.app.api:app --app-dir src --port 8077
```

Open http://127.0.0.1:8077 in a browser: upload a plan, correct the
layout on the image, watch the evacuation, run the agent and read
its verified report. The "Load demo" button replays a complete
cached run from `data/demo` (perception layout, simulation, agent
trace with a VERIFIED stamp, benchmark figures) with no network
and no API key. Rebuild that cache with
`python scripts/build_demo_cache.py`.

## Real API use

Copy `.env.example` to `.env`, set `MODEL_API_KEY`, set `META_MOCK=0`, and run
`python scripts/smoke_meta.py`. Every charged call is logged to
`runs/costs.jsonl` and the `META_BUDGET_USD` cap raises `BudgetExceeded`.

This tool supports decisions; it is not a safety certification.
