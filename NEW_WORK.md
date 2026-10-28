# NEW_WORK.md

Declaration of new work for the Meta Global AI Developer Hackathon
(indie track, 27 Oct - 4 Nov 2026), as required by INSTRUCCIONES.md.

## Statement

All code in this repository is written from 27 October 2026 onwards, during
the hackathon event, except items explicitly listed below.

No code is copied from the prior C++ prototype of the Parallel and
Distributed Computing course. Algorithms are re-implemented from their
published descriptions where applicable, and any such reuse is declared here.

## Reused prior work

| Item | Origin | License / terms | Notes |
| --- | --- | --- | --- |
| None yet | - | - | Add a row before reusing any prior fragment. |

## Third-party dependencies

Declared in `pyproject.toml`. Notable: `openai` SDK (client for the Meta
Model API), `pydantic`, `numpy`, `numba`, `scipy`, `opencv-python-headless`,
`fastapi`. `mpi4py` is an optional dependency (`parallel` extra) and needs
MS-MPI on Windows.

The SAM parser package name and version cited in INSTRUCCIONES.md
(`meta-sam-parser`) must be verified on PyPI before it is added as a
dependency; it is intentionally not declared yet.
