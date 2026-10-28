"""Example run: 30 x 20 m room, 2 exits, 1,000 people.

Prints metrics and saves a density heatmap PNG under runs/example/.
Deterministic for the fixed seed below.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egresstwin.layout.schema import Exit, Layout, SpawnZone  # noqa: E402
from egresstwin.sim.engine import Scenario, run  # noqa: E402
from egresstwin.sim.metrics import compute_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "example"


def build_layout() -> Layout:
    """30 x 20 m room at 0.1 m cells, exits centred on the short walls."""
    width_m, height_m, cell = 30.0, 20.0, 0.1
    rows, cols = int(height_m / cell), int(width_m / cell)
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    return Layout(
        cell_size_m=cell,
        walkable_mask=walkable,
        exits=[
            Exit(id="west", segment=((0.0, 8.8), (0.0, 11.2))),
            Exit(id="east", segment=((width_m, 8.8), (width_m, 11.2))),
        ],
        spawn_zones=[
            SpawnZone(
                id="room",
                polygon=[(0.5, 0.5), (29.5, 0.5), (29.5, 19.5), (0.5, 19.5)],
                count=1000,
            )
        ],
        source_image_hash="script:run_example",
    )


def main() -> int:
    layout = build_layout()
    started = time.perf_counter()
    result = run(layout, Scenario(max_time_s=600.0), seed=42)
    elapsed = time.perf_counter() - started
    metrics = compute_metrics(result, layout)
    thr = metrics.density_threshold_per_m2
    print(f"people: {metrics.n_people} (exited {metrics.n_exited})")
    print(f"total: {metrics.total_time_s:.1f} s | T95: {metrics.t95_s:.1f} s"
          f" | mean: {metrics.mean_egress_s:.1f} s")
    print(f"peak density: {metrics.peak_density_per_m2:.2f} people/m^2")
    print(f"time above {thr:.0f} people/m^2: {metrics.seconds_above_threshold:.1f} s")
    for exit_id, flow in metrics.flow_per_exit_per_s.items():
        print(f"exit {exit_id}: {int(flow.sum())} people, peak {flow.max():.0f} people/s")
    print(f"hotspots (> {thr:.0f} people/m^2 mean): {len(metrics.hotspots)}")
    for spot in metrics.hotspots[:5]:
        print(f"  - {spot['area_m2']:.1f} m^2 at {spot['centroid_m']}, "
              f"peak {spot['peak_density_per_m2']:.1f} people/m^2")
    print(f"wall-clock: {elapsed:.1f} s for {result.n_steps} steps")
    # Heatmap of time-averaged density.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_density = result.density_accum / max(result.n_steps, 1) / (result.cell_size_m**2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    img = ax.imshow(mean_density, origin="upper", cmap="inferno")
    ax.set_title("Mean density (people/m^2) - 30x20 m room, 1000 people, 2 exits")
    ax.set_xlabel("x (cells of 0.1 m)")
    ax.set_ylabel("y (cells of 0.1 m)")
    fig.colorbar(img, ax=ax, label="people/m^2")
    heatmap = OUT / "heatmap.png"
    fig.savefig(heatmap, dpi=130, bbox_inches="tight")
    print(f"heatmap: {heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
