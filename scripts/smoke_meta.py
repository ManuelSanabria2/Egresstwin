"""Smoke test for the Meta Model API client.

Runs one Muse Spark text call and one SAM segmentation (concept 'door')
over an image from data/samples/. Works in mock mode (META_MOCK=1, no
network) and in real mode (MODEL_API_KEY set). Prints a short summary and
the charged cost; every charged call is logged to runs/costs.jsonl.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egresstwin.config import get_settings  # noqa: E402
from egresstwin.meta.client import MetaClient  # noqa: E402
from egresstwin.meta.sam import segment_concept  # noqa: E402
from egresstwin.meta.spark import ask_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


def find_sample() -> Path:
    """Return the first sample image, generating the synthetic one if absent."""
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        found = sorted(SAMPLES.glob(pattern))
        if found:
            return found[0]
    from make_sample import main as make  # scripts/ is on sys.path as script dir

    make()
    return SAMPLES / "door_sample.png"


def main() -> int:
    """Run both smoke calls and print a summary plus total cost."""
    settings = get_settings()
    client = MetaClient(settings)
    mode = "mock" if settings.mock else "real"
    text = ask_text(client, "In one sentence, what exit do you see in this plan?")
    image = find_sample()
    sam = segment_concept(client, image, "door")
    segments = sam.data.get("segments", [])
    total = text.cost_usd + sam.cost_usd
    print(f"mode: {mode}")
    print(f"spark ({text.model}): {text.text}")
    print(f"sam ({sam.model}) on {image.name}: {len(segments)} segment(s) for 'door'")
    for seg in segments:
        print(f"  - {seg.get('label')} confidence={seg.get('confidence')} box={seg.get('box')}")
    print(f"cost this run: USD {total:.6f} (spent total USD {client.tracker.spent_usd:.6f})")
    print("logged to runs/costs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
