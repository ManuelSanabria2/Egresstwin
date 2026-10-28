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

## Real API use

Copy `.env.example` to `.env`, set `MODEL_API_KEY`, set `META_MOCK=0`, and run
`python scripts/smoke_meta.py`. Every charged call is logged to
`runs/costs.jsonl` and the `META_BUDGET_USD` cap raises `BudgetExceeded`.

This tool supports decisions; it is not a safety certification.
