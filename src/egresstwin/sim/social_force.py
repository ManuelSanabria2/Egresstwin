"""Social-force model of Helbing, Farkas and Vicsek (Nature, 2000).

Person-person forces use a cell list (uniform grid, agents sorted by
cell) inside a Numba kernel; each agent writes only its own force row,
so the kernel stays deterministic and parallel-ready (parallelism is
P03). Wall forces are O(1) lookups into the precomputed wall field.
Integration is semi-implicit Euler with a per-agent speed cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class SFParams:
    """Reference parameters, verified against the paper (README table).

    Helbing, Farkas and Vicsek, Nature 407, 487-490 (2000), set
    m = 80 kg, tau = 0.5 s, A = 2e3 N, B = 0.08 m, k = 1.2e5 kg/s^2,
    kappa = 2.4e5 kg/(m s), and pedestrian diameters uniform in
    [0.5, 0.7] m (radius 0.25-0.35 m). The desired-speed distribution
    (mean 1.34 m/s, sd 0.26 m/s) is the conventional normal-walking
    choice adopted here; the paper itself uses scenario speeds of
    about 0.6 (relaxed) to 1.5 m/s (nervous).
    """

    mass: float = 80.0  # kg
    radius_min: float = 0.25  # m
    radius_max: float = 0.35  # m
    desired_speed_mean: float = 1.34  # m/s
    desired_speed_sd: float = 0.26  # m/s
    tau_s: float = 0.5  # s
    a_n: float = 2000.0  # N
    b_m: float = 0.08  # m
    k_kg_s2: float = 1.2e5  # kg/s^2 (body compression)
    kappa: float = 2.4e5  # kg/(m s) (sliding friction)
    cutoff_m: float = 0.6  # interaction ignored beyond r_ij + this (m)
    cell_list_size_m: float = 1.0  # uniform grid cell for the cell list
    max_speed_factor: float = 1.3  # speed cap relative to desired speed


def validate_dt(params: SFParams, dt: float) -> None:
    """Raise if dt is too large for the contact spring (stability)."""
    dt_max = 0.5 * float(np.sqrt(params.mass / params.k_kg_s2))
    if dt > dt_max:
        raise ValueError(
            f"dt={dt:g} s is unstable for the contact spring: "
            f"need dt <= 0.5*sqrt(m/k) = {dt_max:.4f} s "
            f"(m={params.mass:g} kg, k={params.k_kg_s2:g} kg/s^2)"
        )
    if dt > params.tau_s / 5.0:
        raise ValueError(f"dt={dt:g} s too large for tau={params.tau_s:g} s (need dt <= tau/5)")


def build_cell_list(
    pos: np.ndarray, active: np.ndarray, cell_size: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stable sort of active agents by cell.

    Returns (order, cell_start, cell_end, agent_cell): agents
    order[cell_start[c]:cell_end[c]] occupy cell c. Deterministic:
    stable argsort keeps ascending agent index inside each cell.
    """
    rows = np.clip((pos[:, 1] / cell_size).astype(np.int64), 0, shape[0] - 1)
    cols = np.clip((pos[:, 0] / cell_size).astype(np.int64), 0, shape[1] - 1)
    agent_cell = rows * shape[1] + cols
    active_idx = np.flatnonzero(active)
    order = active_idx[np.argsort(agent_cell[active_idx], kind="stable")]
    sorted_cells = agent_cell[order]
    bounds = np.searchsorted(sorted_cells, np.arange(shape[0] * shape[1] + 1))
    return order, bounds[:-1].astype(np.int64), bounds[1:].astype(np.int64), agent_cell


@njit(cache=True)
def _forces_kernel(
    pos,
    vel,
    radius,
    desired_speed,
    desired_dir,
    wall_dist,
    wall_gx,
    wall_gy,
    active,
    order,
    cell_start,
    cell_end,
    agent_cell,
    grid_cols,
    grid_rows,
    cell_size,
    neighbor_radius,
    mass,
    tau_s,
    a_n,
    b_m,
    k_body,
    kappa,
    cutoff_m,
    out,
):
    n = pos.shape[0]
    for i in range(n):
        if not active[i]:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            continue
        fx = mass * (desired_speed[i] * desired_dir[i, 0] - vel[i, 0]) / tau_s
        fy = mass * (desired_speed[i] * desired_dir[i, 1] - vel[i, 1]) / tau_s
        row = agent_cell[i] // grid_cols
        col = agent_cell[i] % grid_cols
        for dr in range(-neighbor_radius, neighbor_radius + 1):
            for dc in range(-neighbor_radius, neighbor_radius + 1):
                nr, nc = row + dr, col + dc
                if nr < 0 or nc < 0 or nr >= grid_rows or nc >= grid_cols:
                    continue
                cell = nr * grid_cols + nc
                for idx in range(cell_start[cell], cell_end[cell]):
                    j = order[idx]
                    if j == i:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    r_ij = radius[i] + radius[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d >= r_ij + cutoff_m:
                        continue
                    if d < 1e-9:
                        nx = 1.0 if i < j else -1.0
                        ny = 0.0
                        d = 1e-9
                    else:
                        nx, ny = dx / d, dy / d
                    tx, ty = -ny, nx
                    overlap = r_ij - d
                    if overlap < 0.0:
                        overlap = 0.0
                    dv_t = (vel[j, 0] - vel[i, 0]) * tx + (vel[j, 1] - vel[i, 1]) * ty
                    mag = a_n * np.exp((r_ij - d) / b_m) + k_body * overlap
                    fx += mag * nx + kappa * overlap * dv_t * tx
                    fy += mag * ny + kappa * overlap * dv_t * ty
        overlap_w = radius[i] - wall_dist[i]
        if overlap_w < 0.0:
            overlap_w = 0.0
        mag_w = a_n * np.exp((radius[i] - wall_dist[i]) / b_m) + k_body * overlap_w
        v_t = vel[i, 0] * (-wall_gy[i]) + vel[i, 1] * wall_gx[i]
        fx += mag_w * wall_gx[i] - kappa * overlap_w * v_t * (-wall_gy[i])
        fy += mag_w * wall_gy[i] - kappa * overlap_w * v_t * wall_gx[i]
        out[i, 0] = fx
        out[i, 1] = fy


def compute_forces(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: np.ndarray,
    desired_speed: np.ndarray,
    desired_dir: np.ndarray,
    wall_dist: np.ndarray,
    wall_gx: np.ndarray,
    wall_gy: np.ndarray,
    active: np.ndarray,
    grid_shape: tuple[int, int],
    layout_cell_size_m: float,
    params: SFParams,
) -> np.ndarray:
    """Total force (N, 2) via cell list + Numba kernel.

    grid_shape is the simulation grid (rows, cols); the cell list is a
    coarser uniform grid (cell_list_size_m) over the same extent.
    """
    cell = params.cell_list_size_m
    grid_rows = max(1, int(np.ceil(grid_shape[0] * layout_cell_size_m / cell)))
    grid_cols = max(1, int(np.ceil(grid_shape[1] * layout_cell_size_m / cell)))
    order, start, end, agent_cell = build_cell_list(pos, active, cell, (grid_rows, grid_cols))
    max_reach = 2.0 * params.radius_max + params.cutoff_m
    neighbor_radius = max(1, int(np.ceil(max_reach / cell)))
    out = np.zeros((len(pos), 2), dtype=np.float64)
    _forces_kernel(
        pos,
        vel,
        radius,
        desired_speed,
        desired_dir,
        wall_dist,
        wall_gx,
        wall_gy,
        active,
        order,
        start,
        end,
        agent_cell,
        grid_cols,
        grid_rows,
        cell,
        neighbor_radius,
        params.mass,
        params.tau_s,
        params.a_n,
        params.b_m,
        params.k_kg_s2,
        params.kappa,
        params.cutoff_m,
        out,
    )
    return out
