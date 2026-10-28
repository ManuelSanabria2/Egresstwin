# P01 perception spike

Spike-only code (no production code). See `docs/spike_perception.md`
for results and the decision.

```bash
.venv/Scripts/python.exe scripts/spikes/generate_assets.py
.venv/Scripts/python.exe scripts/spikes/run_spike.py
```

Requires `MODEL_API_KEY` in the environment. SAM mask decoding uses
the official parser (spike-only dependency):
`uv pip install --python .venv/Scripts/python.exe meta-sam-parser`.
API responses are cached in `runs/spikes/cache/`; re-runs cost USD 0.
Set `SPIKE_NO_CACHE=1` to force fresh calls.
