# Final submission checklist

Deadline (per the team): **today before 6:00 PM Colombia time
(UTC-5)**. States below were checked on 2026-10-31; items marked
MANUAL can only be closed by Manuel.

| # | Item | State | Evidence |
| --- | --- | --- | --- |
| 1 | Public repository | DONE | https://github.com/ManuelSanabria2/Egresstwin loads without sign-in (verified 2026-10-31); remote `origin` points to it and `main` tracks `origin/main` |
| 2 | License | DONE | MIT `LICENSE` at repo root, tracked in git; `license = MIT` in `pyproject.toml` |
| 3 | No secrets in history | DONE (fallback scan) | `gitleaks` is not installed on this machine, so the fallback from the task was run: `git log -p --all` scanned for API-key, token, private-key and password patterns - no matches (only the empty `MODEL_API_KEY=` placeholder in `.env.example`). `.env` is gitignored and untracked. Optional: run `gitleaks detect` once where the binary is available |
| 4 | Assets with rights | DONE | Every image is project-generated (synthetic) and registered with origin and license in `data/ASSETS.md`, including the new `docs/screenshots/` figures |
| 5 | README final, real results, no TBD | DONE | `README.md` rewritten in English: demo T95/CI, verification 5/6 with the V6 failure explained, benchmark table, 70 tests, USD 0.08 API spend, run instructions. `Select-String TBD` over README/NEW_WORK: clean |
| 6 | NEW_WORK.md complete | DONE | First commit date, full commit range/table, every new module listed, inspiration, reused published work declared |
| 7 | Screenshots | DONE | `docs/screenshots/01_image_to_layout.png` and `02_simulation.png`, generated from the real demo cache by `scripts/make_screenshots.py`, plus verification and benchmark figures |
| 8 | Demo video script | DONE | `docs/submission/video_script.md`, timed 0:00-3:00 as specified |
| 9 | Demo video uploaded | MANUAL | Record with the script (cached demo as fallback), upload (YouTube/Vimeo unlisted is fine), paste the URL into `submission_text.md` and the form |
| 10 | Submission texts | DONE | `docs/submission/submission_text.md`, short and long versions |
| 11 | Submission sent before 6:00 PM Colombia | MANUAL | Submit the form with repo URL + video URL; keep a screenshot of the confirmation |
| 12 | Tests green | DONE | 70 passed (64 in sandbox + 6 app tests outside it, which need loopback sockets), `ruff check` clean |
| 13 | Offline demo works | DONE | `python scripts/demo.py --mock` completed end-to-end (VERIFIED) with no network/key |

## Known honest limitations (do not hide in the form)

- Verification is 5/6: V6 (fundamental diagram) fails at crush
  densities; explained in `docs/verification/report.md`.
- Aerial-photo input is spike-tested only; floor plans are the
  supported path.
- The demo agent run recommends changing nothing on the bundled
  plan - that is the verifier working as intended, and the widening
  success case is covered by the acceptance test.

## Out of this commit (working tree, from the previous task)

Modified but intentionally not part of the submission commit:
`src/egresstwin/sim/engine.py`, `src/egresstwin/app/api.py`,
`src/egresstwin/app/static/index.html`, `tests/test_app.py`,
regenerated demo/verification data, and untracked
`scripts/demo.py`, `scripts/cost_report.py`, `docs/cost_report.md`,
`data/demo/agent_run/`. Review and commit them separately.
