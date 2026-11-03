"""Generate README/submission screenshots from the real demo cache.

Reads only committed demo data (data/demo) and the plan it names;
writes PNGs to docs/screenshots/. No network, no API calls.

    python scripts/make_screenshots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
MANIFEST = json.loads((ROOT / "data/demo/manifest.json").read_text(encoding="utf-8"))


def image_to_layout() -> Path:
    image = Path(MANIFEST["image"])
    plan = np.asarray(Image.open(ROOT / image).convert("RGB"))
    data = np.load(ROOT / "data/demo/layout.npz")
    walk = data["walkable"].astype(bool)
    exits = json.loads(str(data["meta"])).get("exits", []) if "meta" in data else []
    exit_ids = "/".join(e.get("id", "?") for e in exits) or "detected"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].imshow(plan)
    axes[0].set_title(f"1. Input image ({image.name})")
    axes[0].axis("off")
    overlay = np.zeros((*walk.shape, 4))
    overlay[walk] = (0.15, 0.65, 0.35, 0.55)
    overlay[~walk] = (0.10, 0.10, 0.10, 0.85)
    axes[1].imshow(overlay, origin="upper")
    axes[1].set_title(f"2. Layout: walkable grid, exits {exit_ids} (correctable)")
    axes[1].axis("off")
    fig.tight_layout()
    path = OUT / "01_image_to_layout.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def simulation_frame() -> Path:
    sim = json.loads((ROOT / "data/demo/simulation.json").read_text())
    anim = sim["animation"]
    frames = [np.array(f) for f in anim["frames"] if f]
    data = np.load(ROOT / "data/demo/layout.npz")
    walk = data["walkable"].astype(bool)
    t95 = sim["aggregates"]["t95_s"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, frame, title in (
        (axes[0], frames[len(frames) // 3], "Evacuation in progress"),
        (axes[1], frames[-1], "Final frame"),
    ):
        ax.imshow(walk, cmap="Greys", origin="upper",
                  extent=[0, anim["width_m"], anim["height_m"], 0], alpha=0.35)
        if len(frame):
            ax.scatter(frame[:, 0], frame[:, 1], s=22, c="#c1272d")
        ax.set_xlim(0, anim["width_m"])
        ax.set_ylim(anim["height_m"], 0)
        ax.set_xlabel("metres")
        ax.set_ylabel("metres")
        ax.set_title(title)
    n_people = anim.get("metrics", {}).get("n_people") or len(frames[0])
    fig.suptitle(
        f"Baseline: {n_people} people, T95 {t95['mean']:.1f} s "
        f"(95% CI {t95['ci_low']:.1f}-{t95['ci_high']:.1f}, {sim['n_replicas']} replicas)"
    )
    fig.tight_layout()
    path = OUT / "02_simulation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (image_to_layout, simulation_frame):
        print(make())
    print("Benchmark figures: data/demo/bench_parallel.png, bench_scaling.png, "
          "bench_throughput.png")
    print("Verification figures: docs/figures/verification/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
