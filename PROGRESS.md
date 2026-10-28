# PROGRESS.md

Bitacora updated at the end of each prompt (see INSTRUCCIONES.md, section 7).

## 2026-10-27 - Scaffold: Meta API client with cache, cost guard and mock mode

- Done: repository structure from INSTRUCCIONES.md section 4, `pyproject.toml`,
  `.gitignore`, MIT `LICENSE`, `.env.example`, `README.md`.
- Done: `src/egresstwin/config.py` (`pydantic-settings`), `meta/costs.py`
  (editable price table, `BudgetExceeded`), `meta/client.py` (OpenAI SDK
  wrapper for `https://api.meta.ai/v1`, retries with backoff, timeout, disk
  cache by input hash, `runs/costs.jsonl` log, mock mode from fixtures).
- Done: thin helpers `meta/spark.py`, `meta/sam.py`, `meta/voice.py`;
  `scripts/smoke_meta.py`; fixtures in `tests/fixtures/meta/`; synthetic
  sample image in `data/samples/`; tests for config, costs, budget cap,
  cache and mock mode; GitHub Actions (`ruff check` + `pytest`, `META_MOCK=1`).
- Decisions: `mpi4py` is an optional `parallel` extra (needs MS-MPI on
  Windows and system MPI in CI). The SAM parser package is not declared
  until its exact PyPI name/version is verified. SAM real-mode transport is
  provisional (see Open issues).
- Open issues: verify against https://dev.meta.ai/docs the real SAM 3.1
  endpoint/payload and the Muse Spark text endpoint; document any
  difference here. Real-key smoke run is not yet recorded.
- Accumulated API cost (from `runs/costs.jsonl`): USD 0.00 (mock only).
