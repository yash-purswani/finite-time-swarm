"""
metrics.py
===============================================================================
Quantitative diagnostics for the docking-approach ablation (baseline constant
bias vs. proposed normal-guided time-varying bias):

  - interior_intrusion:  does the agent's path ever cut through the payload's
                          own interior en route to its slot?
  - terminal_approach_alignment: at the moment of arrival, is the agent's
                          velocity actually aligned with the boundary's
                          inward normal (a physically valid contact
                          approach), or arbitrary?

Pure post-processing: no dynamics, no optimization.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from shapely.geometry import Polygon

from geometry import containment_margin


def interior_intrusion(traj_local: np.ndarray, polygon_local: Polygon) -> Tuple[float, float]:
    """How much (if at all) agent trajectories cut through the payload body.

    Parameters
    ----------
    traj_local : (T, n, 2) ndarray
        Agent positions *relative to the payload CoG* (i.e. x(t) - r(t)),
        sampled over time -- directly comparable to `polygon_local`.

    Returns
    -------
    max_depth : float
        Worst-case penetration depth (0 if the trajectory never enters the
        polygon's interior at all).
    frac_intruding : float
        Fraction of (time, agent) samples for which the agent is strictly
        inside the payload polygon.
    """
    T, n, _ = traj_local.shape
    max_depth = 0.0
    intruding = 0
    total = T * n
    for t in range(T):
        for i in range(n):
            margin = containment_margin(traj_local[t, i], polygon_local)
            if margin > 0:
                intruding += 1
                max_depth = max(max_depth, margin)
    return max_depth, intruding / total


def terminal_approach_alignment(
    traj_world: np.ndarray,
    t: np.ndarray,
    targets_world: np.ndarray,
    normals: np.ndarray,
    contact_eps: float,
) -> np.ndarray:
    """For each agent, the angle (degrees) between its velocity direction at
    the moment it first comes within `contact_eps` of its slot and the ideal
    inward-normal approach direction (-normals[i]). 0 deg = a clean,
    perpendicular approach into the surface; 90 deg = a sideways/tangential
    approach; NaN = the agent never got within contact_eps.

    Parameters
    ----------
    traj_world : (T, n, 2)  agent positions, world frame.
    t          : (T,)       sample times (for finite-difference velocity).
    targets_world : (n, 2)  each agent's assigned slot, world frame.
    normals    : (n, 2)     each agent's assigned outward unit normal.
    """
    T, n, _ = traj_world.shape
    angles = np.full(n, np.nan)
    for i in range(n):
        d = np.linalg.norm(traj_world[:, i, :] - targets_world[i], axis=1)
        hits = np.where(d < contact_eps)[0]
        if len(hits) == 0:
            continue
        k = hits[0]
        k0 = max(k - 1, 0)
        k1 = min(k + 1, T - 1)
        if k1 == k0:
            continue
        vel = (traj_world[k1, i] - traj_world[k0, i]) / (t[k1] - t[k0] + 1e-12)
        speed = np.linalg.norm(vel)
        if speed < 1e-9:
            continue
        vel_dir = vel / speed
        inward = -normals[i]
        cos_ang = np.clip(vel_dir @ inward, -1.0, 1.0)
        angles[i] = np.degrees(np.arccos(cos_ang))
    return angles
