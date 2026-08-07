"""
optimization.py
===============================================================================
Offline formation-offset optimization engine.

Computes the target offsets P = [p_1, ..., p_n]^T (relative to the payload's
CoG, in the local/body frame) that every agent will hold throughout docking
and transport. This is solved *once*, offline, before the kinematic mission
even starts -- the online finite-time controller in `dynamics.py` never
re-touches P.

    min_{p_1..p_n}  sum_i DistToBoundary(p_i, payload)
                    + w1 * sum_i sum_j 1 / (||p_i - p_j||^2 + delta)

    s.t.  sum_i p_i = 0                          (centering, EQUALITY)
          p_i in Payload (interior or boundary)  (containment, INEQUALITY)
          ||p_i - p_j|| >= d_min  for all i != j  (collision safety, INEQUALITY)

Also provides the Hungarian (linear-sum-assignment) docking assignment that
pairs each scattered starting agent to the formation slot that minimizes
total squared travel distance -- this is what keeps the n rendezvous paths
from crossing each other during Phase 1.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from shapely.geometry import LineString, Polygon

from geometry import containment_margin, dist_to_boundary, boundary_points


def _objective(p_flat: np.ndarray, n: int, polygon: Polygon, w1: float, delta: float) -> float:
    p = p_flat.reshape(n, 2)
    boundary_term = sum(dist_to_boundary(p[i], polygon) for i in range(n))
    sep = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sep += 1.0 / (np.sum((p[i] - p[j]) ** 2) + delta)
    return boundary_term + w1 * sep


def _centering_constraint(p_flat: np.ndarray, n: int) -> np.ndarray:
    p = p_flat.reshape(n, 2)
    return p.sum(axis=0)  # == [0, 0]


def _containment_constraints(p_flat: np.ndarray, n: int, polygon: Polygon) -> np.ndarray:
    p = p_flat.reshape(n, 2)
    return np.array([containment_margin(p[i], polygon) for i in range(n)])


def _separation_constraints(p_flat: np.ndarray, n: int, d_min: float) -> np.ndarray:
    p = p_flat.reshape(n, 2)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(np.linalg.norm(p[i] - p[j]) - d_min)
    return np.array(out)


def optimize_formation_offsets(
    polygon_local: Polygon,
    n: int,
    w1: float = 0.05,
    delta: float = 1e-2,
    d_min: float = 0.35,
    maxiter: int = 500,
) -> Tuple[np.ndarray, bool]:
    """Solve for the n formation offsets p_i (local/body frame, CoG-centered).

    Returns
    -------
    P : (n, 2) ndarray
        The optimized offsets, sum(P, axis=0) == [0, 0] (up to solver tol).
    success : bool
        Whether SLSQP reported a converged, constraint-satisfying solution.
    """
    p0 = boundary_points(polygon_local, n).flatten()

    constraints = [
        {"type": "eq", "fun": _centering_constraint, "args": (n,)},
        {"type": "ineq", "fun": _containment_constraints, "args": (n, polygon_local)},
        {"type": "ineq", "fun": _separation_constraints, "args": (n, d_min)},
    ]

    res = minimize(
        _objective,
        p0,
        args=(n, polygon_local, w1, delta),
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": 1e-9},
    )

    P = res.x.reshape(n, 2)

    # SLSQP satisfies constraints only up to solver tolerance; re-center
    # exactly so the hard centering condition holds to floating-point
    # precision regardless of solver residual (this is what the online
    # controller's zero-net-offset / balanced-load guarantee relies on).
    P = P - P.mean(axis=0)

    success = bool(res.success) and np.all(_containment_constraints(P.flatten(), n, polygon_local) > -1e-6)
    return P, success


def docking_assignment(
    x0: np.ndarray, P: np.ndarray, cog0: np.ndarray, polygon_local: Optional[Polygon] = None
) -> np.ndarray:
    """Hungarian assignment pairing each scattered starting agent x0[i] to the
    formation slot (P[j] + cog0) that minimizes total squared travel distance.

    Returns a permutation `perm` such that agent i should track slot
    P[perm[i]] -- i.e. reorder P (and hence the fixed offsets each agent
    holds for the rest of the mission) as `P[perm]`. Minimizing total squared
    displacement via the assignment problem is the standard, cheap way to
    keep the n rendezvous trajectories from crossing one another (crossing
    paths generally increase, never decrease, total squared displacement for
    points converging to a common local neighborhood).

    If `polygon_local` is given, (agent, slot) pairs whose straight segment
    would pass through the payload's own footprint are heavily penalized, so
    Hungarian prefers any available pairing that does not require an agent to
    approach its assigned edge from the *wrong side* of the object (i.e. cut
    through it to get onto the correct approach corridor). Pure distance
    minimization alone does not know about the payload's shape and can select
    exactly such a crossing pairing when it happens to minimize aggregate
    squared distance -- this was found and diagnosed empirically (see
    paper_notes.md); this is the fix.
    """
    targets = P + cog0
    cost = ((x0[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
    if polygon_local is not None:
        n = x0.shape[0]
        for i in range(n):
            x0_local = x0[i] - cog0
            for j in range(n):
                seg = LineString([x0_local, P[j]])
                if seg.intersects(polygon_local):
                    cost[i, j] += 1e8
    _, col = linear_sum_assignment(cost)
    return col
