# NEW_WORK.md

Declaration of new work for the Meta Global AI Developer Hackathon
(indie track, 27 Oct - 4 Nov 2026), as required by INSTRUCCIONES.md.

## Statement

All code in this repository was written from 27 October 2026 onwards,
during the hackathon event, except the third-party dependencies and the
published scientific parameters declared below.

No code is copied from the prior C++ prototype of the Parallel and
Distributed Computing course. Algorithms are re-implemented from their
published descriptions where applicable, and any such reuse is declared
here.

## Dates and commit range

- First commit: `9c161b2` - 2026-10-27 20:13:24 -05:00 (Colombia time),
  "docs: project plan and agent instructions".
- Commit range at submission: `9c161b2` .. `HEAD` on branch `main`
  (9 feature/docs commits before the final submission commit
  "docs: final README, NEW_WORK and submission materials").
- Every step is one descriptive commit, so the history itself is the
  evidence that the work was done during the event. Full log:
  `git log --reverse --format="%h %ad %s" --date=iso`.

| Commit | Date (Colombia) | Content |
| --- | --- | --- |
| `9c161b2` | 2026-10-27 | Project plan and agent instructions |
| `2b3ffe4` | 2026-10-27 | Scaffold, Meta API client with cache, cost guard and mock mode |
| `fa399ea` | 2026-10-27 | Perception risk spike with Muse Spark, SAM 3.1 and OpenCV |
| `aca4d1f` | 2026-10-27 | Social-force engine with geodesic floor field and metrics |
| `b676121` | 2026-10-28 | Image to layout with Muse Spark, SAM 3.1, CV fallback and manual edits |
| `4d76d73` | 2026-10-28 | Muse Spark tool-calling optimizer with N-1 stress tests and deterministic verifier |
| `068e24b` | 2026-10-28 | FastAPI backend and single-page UI with live agent trace and demo mode |
| `310ae3e` | 2026-10-30 | Timed live events (blocked exits, hazard zones) with partial floor-field recompute |
| `c763bf0` | 2026-10-31 | Simulator verification against RiMEA standard test cases |
| (final) | 2026-10-31 | Final README, NEW_WORK and submission materials |

## What inspired the project

Two ideas met. First, evacuation planning is still mostly static:
drawings, rules of thumb and expensive one-off simulations, while the
people who manage schools, venues and offices rarely get to ask "what
if this exit is blocked?" and receive a measured answer. Second, AI
agents are usually trusted because they sound confident. EgressTwin
asks the opposite: what if an agent may only recommend what it can
prove with simulations, and independent code - not another LLM -
checks every number before anything is called verified?

The hackathon's Meta models made the combination practical: Muse
Spark reads a floor plan, SAM 3.1 segments it, and the same Spark
drives a tool-calling optimizer whose claims a deterministic verifier
stamps or rejects.

## New modules (all written during the event)

### Package `src/egresstwin/`

| Module | New work |
| --- | --- |
| `config.py` | Settings (model IDs, budget, mock mode) via pydantic-settings |
| `meta/client.py` | Only access point to the Meta Model API: OpenAI SDK wrapper, retries with backoff, disk cache by input hash, cost log, budget guard, mock mode from fixtures |
| `meta/costs.py` | Editable price table and `BudgetExceeded` guard |
| `meta/spark.py`, `meta/sam.py`, `meta/voice.py` | Thin helpers for Muse Spark, SAM 3.1 and Muse Voice |
| `layout/schema.py` | `Layout` model: walkable grid (`.npz`), obstacles, exit segments, spawn zones, explicit scale |
| `layout/validate.py` | Connectivity, exit placement and narrow-passage validation |
| `layout/edit.py` | Pure manual corrections: add/remove exit or obstacle, paint wall, mark walkable |
| `perception/plan_reader.py` | Spark vision read of the plan (doors, exits, scale annotation) with strict structured output |
| `perception/segmenter.py` | SAM 3.1 box-prompt segmentation seeded by Spark, mask decoding |
| `perception/cv_fallback.py` | OpenCV wall extraction and outer-wall gap detection |
| `perception/scale.py` | Scale from two user points, known door, annotation, or a flagged assumption |
| `perception/pipeline.py` | Image to validated `Layout` with per-element confidence, warnings and cost report |
| `sim/floor_field.py` | Per-exit geodesic floor field (Dijkstra) and single-exit recompute for events |
| `sim/wall_field.py` | Precomputed wall-distance field (EDT) with O(1) lookup |
| `sim/social_force.py` | Numba social-force kernel (cell lists; each agent writes only its own force row) |
| `sim/reference.py` | Pure NumPy O(N^2) twin used only to test the kernel |
| `sim/engine.py` | Deterministic engine `run(layout, scenario, seed) -> SimResult`, timed events, per-agent reaction times |
| `sim/events.py` | Timed live events: block/reopen exit, hazard zones, with relocation rule |
| `sim/metrics.py` | Total/T95/mean egress, flow per exit, peak density, hotspots |
| `scenarios/spec.py`, `scenarios/apply.py` | Modification specs (widen/add exit, remove obstacle) and feasibility checks |
| `parallel/sweep.py`, `parallel/store.py` | Scenario x replica sweep; result store with t confidence intervals, Welch test, Cohen d |
| `agent/tools.py` | Strict tool schemas: layout summary, baseline, scenario runs, hotspots, N-1 stress test, compare, submit report |
| `agent/loop.py` | Muse Spark tool-calling optimizer (Responses API, multi-call turns, budgets, report corrections) |
| `agent/verifier.py` | Deterministic verifier: numbers within 1%, significance, N-1 robustness, VERIFIED / PARTIALLY_VERIFIED / FAILED |
| `agent/trace.py`, `agent/report.py` | Run trace (`trace.jsonl`), structured report handling |
| `validation/cases.py`, `validation/status.py` | Six RiMEA/Weidmann verification cases and machine-readable status |
| `app/api.py` | FastAPI backend: uploads, jobs, SSE agent trace, demo replay, verification endpoints |
| `app/static/index.html` | Single-page UI (vanilla JS + canvas): layout correction, simulation, agent trace with VERIFIED stamp, performance tab |

### Tests, scripts and docs (also all new)

- `tests/`: `test_config`, `test_costs`, `test_client`, `test_layout`,
  `test_sim`, `test_perception`, `test_agent`, `test_app`,
  `test_events`, `test_verification`, plus recorded Meta fixtures
  under `tests/fixtures/meta/` for mock mode.
- `scripts/`: `demo.py` (end-to-end demo), `build_demo_cache.py`,
  `run_example.py`, `run_verification.py`, `cost_report.py`,
  `make_screenshots.py`, `make_sample.py`, `smoke_meta.py`, and the
  P01 spike under `scripts/spikes/`.
- `docs/`: perception spike report, verification report and figures,
  cost report, README screenshots, submission materials
  (`docs/submission/`).
- `data/`: synthetic sample plans generated by the project and the
  cached demo run; every asset is registered in `data/ASSETS.md`.

## Reused prior work

| Item | Origin | License / terms | Notes |
| --- | --- | --- | --- |
| Social-force model parameters (m, tau, A, B, k, kappa, diameters) | Helbing, Farkas and Vicsek, "Simulating dynamical features of escape panic", *Nature* 407, 487-490 (2000), doi:10.1038/35035023 | Published scientific parameters, cited | Re-implemented from the paper; no code copied. Verified against the paper text (see README) |
| Verification criteria V1-V5 | RiMEA Guideline 4.1.1, Annex 2 (official site, accessed 2026-10-30) | Standard test definitions, summarised in our own words | Nothing copied into the repo; adapted geometries are labelled in `docs/verification/test_cases.md` |
| Verification criterion V6 | Weidmann (1993), Kladek form as quoted in secondary literature | Published empirical curve, cited | Tolerance is a project choice, documented as such |
| Desired-speed distribution (mean 1.34 m/s, sd 0.26 m/s) | Conventional normal-walking choice adopted by this project | - | Explicitly not from the 2000 paper; documented in README |

No other prior work is reused. In particular, no code comes from the
author's earlier C++ coursework prototype; the engine, kernel and
sweep were written from scratch in Python/Numba for this event.

## Third-party dependencies

Declared in `pyproject.toml` (MIT-compatible use; each keeps its own
license): `openai` SDK (client for the Meta Model API), `pydantic`,
`pydantic-settings`, `numpy`, `numba`, `scipy`,
`opencv-python-headless`, `pillow`, `fastapi`, `uvicorn`,
`sse-starlette`, `python-multipart`, `matplotlib`, and
`meta-sam-parser` (official SAM parser for Python). `mpi4py` is an
optional dependency (`parallel` extra) and needs MS-MPI on Windows.
Dev tools: `pytest`, `ruff`, `httpx`.
