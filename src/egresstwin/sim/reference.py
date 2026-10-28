"""Pure NumPy reference implementation of the social-force step.

O(N^2) pairwise loops with no Numba and no cell list. Used only in
tests to validate the Numba kernel on small N. Formulas follow
Helbing, Farkas and Vicsek (Nature 407, 487-490, 2000); see
social_force.SFParams for the parameter values.
"""

from __future__ import annotations

import numpy as np

from egresstwin.sim.social_force import SFParams


def compute_forces_reference(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: np.ndarray,
    desired_speed: np.ndarray,
    desired_dir: np.ndarray,
    wall_dist: np.ndarray,
    wall_gx: np.ndarray,
    wall_gy: np.ndarray,
    params: SFParams,
) -> np.ndarray:
    """Total force (N, 2) on every agent, Helbing et al. (2000)."""
    n = len(pos)
    forces = params.mass * (desired_speed[:, None] * desired_dir - vel) / params.tau_s
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = pos[i] - pos[j]
            d = float(np.hypot(delta[0], delta[1]))
            r_ij = radius[i] + radius[j]
            if d >= r_ij + params.cutoff_m:
                continue
            if d < 1e-9:  # coincident centres: deterministic split direction
                normal = np.array([1.0, 0.0]) if i < j else np.array([-1.0, 0.0])
                d = 1e-9
            else:
                normal = delta / d
            tangent = np.array([-normal[1], normal[0]])
            overlap = max(r_ij - d, 0.0)
            dv_t = float(np.dot(vel[j] - vel[i], tangent))
            mag = params.a_n * np.exp((r_ij - d) / params.b_m) + params.k_kg_s2 * overlap
            forces[i] += mag * normal + params.kappa * overlap * dv_t * tangent
        # Wall force from the precomputed wall field (O(1) lookup upstream).
        overlap_w = max(radius[i] - wall_dist[i], 0.0)
        mag_w = (
            params.a_n * np.exp((radius[i] - wall_dist[i]) / params.b_m)
            + params.k_kg_s2 * overlap_w
        )
        normal_w = np.array([wall_gx[i], wall_gy[i]])
        tangent_w = np.array([-wall_gy[i], wall_gx[i]])
        v_t = float(np.dot(vel[i], tangent_w))
        forces[i] += mag_w * normal_w - params.kappa * overlap_w * v_t * tangent_w
    return forces


def step_reference(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: np.ndarray,
    desired_speed: np.ndarray,
    desired_dir: np.ndarray,
    wall_dist: np.ndarray,
    wall_gx: np.ndarray,
    wall_gy: np.ndarray,
    params: SFParams,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One semi-implicit Euler step (no wall guard; tests only)."""
    forces = compute_forces_reference(
        pos, vel, radius, desired_speed, desired_dir, wall_dist, wall_gx, wall_gy, params
    )
    new_vel = vel + forces / params.mass * dt
    speed = np.hypot(new_vel[:, 0], new_vel[:, 1])
    cap = params.max_speed_factor * desired_speed
    over = speed > cap
    new_vel[over] *= (cap[over] / speed[over])[:, None]
    return pos + new_vel * dt, new_vel
