"""Export the demo as a static site and (optionally) push it as a Space.

Hugging Face serves static Spaces for free; Docker Spaces on free
hardware need a PRO plan (the API answered 402 for this account), so
the public demo is static. The export asks the demo-only app for every
response the page needs (health, demo replay, bench) and writes them as
JSON next to index.html, plus the plan image, the bench figures and
the verification report. index.html reads them through
window.EGRESSTWIN_STATIC_ROUTES: no server, no API key, no API calls.
The Dockerfile serves the same demo-only app with a server.

    .venv/Scripts/python.exe scripts/deploy_space.py              # export only
    .venv/Scripts/python.exe scripts/deploy_space.py --push USER/egresstwin

Pushing uses the `hf` CLI after `hf auth login`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STAGE = ROOT / "runs" / "hf_space"


def export() -> Path:
    from fastapi.testclient import TestClient

    from egresstwin.app.api import STATIC, create_app

    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "api").mkdir(parents=True)
    (STAGE / "figures").mkdir()
    client = TestClient(create_app(runs_dir=STAGE.parent / "hf_space_runs", demo_only=True))
    routes: dict[str, str] = {}

    def save_json(path: str, name: str) -> dict:
        response = client.get(path)
        response.raise_for_status()
        (STAGE / name).write_text(response.text, encoding="utf-8")
        routes[path] = name
        return response.json()

    save_json("/api/health", "api/health.json")
    demo = save_json("/api/demo", "api/demo.json")
    save_json("/api/bench", "api/bench.json")
    for name in demo["bench"].get("figures", []):
        shutil.copyfile(ROOT / "data" / "demo" / name, STAGE / "figures" / name)
        routes[f"/api/bench/figures/{name}"] = f"figures/{name}"
    manifest = json.loads((ROOT / "data" / "demo" / "manifest.json").read_text(encoding="utf-8"))
    image = ROOT / manifest["image"]
    shutil.copyfile(image, STAGE / ("plan" + image.suffix))
    routes["/api/layouts/demo/image"] = "plan" + image.suffix
    shutil.copyfile(ROOT / "docs" / "verification" / "report.md",
                    STAGE / "verification_report.md")
    routes["/api/verification/report"] = "verification_report.md"

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    inject = ("<script>window.EGRESSTWIN_STATIC_ROUTES = "
              + json.dumps(routes, sort_keys=True) + ";</script>\n")
    if html.count("<script>") != 1:
        raise SystemExit("index.html layout changed: expected exactly one <script> block")
    (STAGE / "index.html").write_text(html.replace("<script>", inject + "<script>", 1),
                                      encoding="utf-8")
    shutil.copyfile(ROOT / "deploy" / "hf_space" / "README.md", STAGE / "README.md")
    shutil.rmtree(STAGE.parent / "hf_space_runs", ignore_errors=True)
    return STAGE


def push(repo_id: str, folder: Path) -> str:
    hf = shutil.which("hf")
    if hf is None:
        raise SystemExit("the `hf` CLI is not on PATH (pip install huggingface_hub)")
    subprocess.run([hf, "repos", "create", repo_id, "--type", "space", "--space-sdk", "static",
                    "--exist-ok"], check=True)
    subprocess.run([hf, "upload", repo_id, str(folder), ".", "--repo-type", "space",
                    "--commit-message", "Deploy EgressTwin static demo"], check=True)
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", metavar="REPO_ID", help="Space id, e.g. user/egresstwin")
    args = parser.parse_args()
    folder = export()
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    print(f"exported {len(files)} files to {folder}: {', '.join(files)}")
    if args.push:
        print("pushed:", push(args.push, folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
