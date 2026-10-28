"""Evacuation metrics from a SimResult.

Densities are people per square meter on the layout grid, smoothed
with a Gaussian kernel (sigma 0.5 m by default). Time above the
density threshold counts frames whose peak smoothed density exceeds
it. Hotspots are connected clusters of high mean density.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Metrics:
    """Headline numbers of one simulation run."""

    total_time_s: float
    t95_s: float
    mean_egress_s: float
    n_people: int
    n_exited: int
    flow_per_exit_per_s: dict[str, np.ndarray]  # people per 1 s bin
    peak_density_per_m2: float
    seconds_above_threshold: float
    density_threshold_per_m2: float
    hotspots: list[dict] = field(default_factory=list)


def _frame_density(
    frame: np.ndarray, grid_shape: tuple[int, int], cell_m: float, sigma_m: float
) -> np.ndarray:
    """Smoothed density (people/m^2) of one recorded frame."""
    pos = frame.astype(np.float64)
    inside = (pos[:, 0] >= 0) & (pos[:, 1] >= 0)
    rows = np.clip((pos[inside, 1] / cell_m).astype(int), 0, grid_shape[0] - 1)
    cols = np.clip((pos[inside, 0] / cell_m).astype(int), 0, grid_shape[1] - 1)
    grid = np.zeros(grid_shape)
    np.add.at(grid, (rows, cols), 1.0)
    density = grid / (cell_m * cell_m)
    if sigma_m > 0:
        density = ndimage.gaussian_filter(density, sigma=sigma_m / cell_m, mode="constant")
    return density


def compute_metrics(
    result, layout, *, density_threshold_per_m2: float = 4.0, smooth_sigma_m: float = 0.5
) -> Metrics:
    """Compute metrics; layout supplies grid geometry (must match result)."""
    times = result.exit_times[np.isfinite(result.exit_times)]
    n = len(result.exit_times)
    total = float(times.max()) if len(times) else result.total_time_s
    t95 = float(np.percentile(times, 95)) if len(times) else float("nan")
    mean = float(times.mean()) if len(times) else float("nan")
    # Flow per exit in 1-second bins from the per-step counts.
    steps_per_s = max(1, round(1.0 / result.dt))
    n_bins = int(np.ceil(result.exits_per_step.shape[0] / steps_per_s))
    flow: dict[str, np.ndarray] = {}
    for k, exit_id in enumerate(result.exit_ids):
        per_step = result.exits_per_step[:, k].astype(np.float64)
        padded = np.pad(per_step, (0, n_bins * steps_per_s - len(per_step)))
        flow[exit_id] = padded.reshape(n_bins, steps_per_s).sum(axis=1)
    # Density over time from the subsampled frames.
    peak = 0.0
    above_frames = 0
    for frame in result.frames:
        density = _frame_density(frame, result.grid_shape, result.cell_size_m, smooth_sigma_m)
        frame_peak = float(density.max()) if density.size else 0.0
        peak = max(peak, frame_peak)
        above_frames += int(frame_peak > density_threshold_per_m2)
    # Hotspots: clusters of high time-averaged density.
    hotspots: list[dict] = []
    if result.n_steps > 0:
        mean_density = result.density_accum / result.n_steps / (result.cell_size_m**2)
        if smooth_sigma_m > 0:
            mean_density = ndimage.gaussian_filter(
                mean_density, sigma=smooth_sigma_m / result.cell_size_m, mode="constant"
            )
        labels, count = ndimage.label(
            mean_density > density_threshold_per_m2, structure=np.ones((3, 3))
        )
        for label_id in range(1, count + 1):
            cells = labels == label_id
            rows, cols = np.argwhere(cells).mean(axis=0)
            hotspots.append(
                {
                    "n_cells": int(cells.sum()),
                    "area_m2": float(cells.sum() * result.cell_size_m**2),
                    "peak_density_per_m2": float(mean_density[cells].max()),
                    "centroid_m": [
                        float(cols * result.cell_size_m),
                        float(rows * result.cell_size_m),
                    ],
                }
            )
        hotspots.sort(key=lambda h: h["peak_density_per_m2"], reverse=True)
    return Metrics(
        total_time_s=total,
        t95_s=t95,
        mean_egress_s=mean,
        n_people=n,
        n_exited=int(len(times)),
        flow_per_exit_per_s=flow,
        peak_density_per_m2=peak,
        seconds_above_threshold=above_frames * result.frame_dt,
        density_threshold_per_m2=density_threshold_per_m2,
        hotspots=hotspots,
    )
