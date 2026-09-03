"""
approach.py
===============================================================================
Normal-guided, time-varying formation bias p_i(t).

GAP ADDRESSED
-------------
With a *constant* p_i (the original formulation), the finite-time law pulls
each agent along a roughly straight line from wherever it starts directly to
its final slot on the payload's rim. For a scattered starting position this
routinely drags the agent straight through the payload's interior before it
"pops out" onto the boundary -- physically meaningless for a real end
effector, which must approach a grasp/contact point from *outside*, along
the surface's outward normal (the standard "approach ray" / "standoff pose"
convention in single-robot grasp planning; see e.g. approach-vector and
standoff-distance conventions in pre-grasp pose planning literature). Almost
none of the cooperative-transport / time-varying-formation-tracking
literature (which otherwise studies exactly this kind of moving-target
tracking, e.g. for UAV swarm reshaping) imports this discipline.

FIX
---
Instead of a constant p_i, define a two-point *approach corridor* per agent:

    S_i (standoff point)  = B_i + d_standoff * N_i      (just outside the rim)
    B_i (boundary point)                                  (the true grasp slot)

and blend smoothly between them with a single scalar, time-only schedule
alpha(t) in [0, 1] (zero-derivative at both ends -- a "smoothstep"):

    p_i(t)    = (1 - alpha(t)) * S_i + alpha(t) * B_i
    pdot_i(t) = alpha_dot(t) * (B_i - S_i)

Every agent shares the same alpha(t) -- a *global* approach schedule -- so
the whole swarm transitions from "loiter just outside the object, aligned
with your own edge" to "commit inward along the normal to the exact grasp
point" together. Because B, S are both expressed in the CoG-centered local
frame with sum(B) = 0 (the optimizer's hard centering constraint), and
sum(S) = d_standoff * sum(N) is generally *nonzero*, the swarm's own
centroid is naturally offset from the CoG while alpha < 1 and slides onto
the CoG exactly as alpha -> 1 -- i.e. "the centroid approaches the CoG" is
not a separate mechanism, it falls straight out of this same schedule.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from shapely.geometry import Polygon

from geometry import snap_to_boundary_with_normals, containment_margin


def compute_approach_corridor(
    P: np.ndarray,
    polygon_local: Polygon,
    d_standoff: float,
    min_clearance_frac: float = 0.6,
    max_scale: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Snap optimizer output P onto the rim and build standoff points.

    Near a *reflex* (concave) vertex -- e.g. the inner notch of an L-shape --
    moving a fixed d_standoff along one edge's own outward normal does not
    guarantee that distance of clearance from the polygon overall, because a
    different edge of the same non-convex object can curve back close by.
    Rather than silently allow a near-zero-clearance standoff point there,
    each point's offset is adaptively scaled up (geometrically, capped at
    `max_scale` * d_standoff) until it clears at least
    `min_clearance_frac` * d_standoff from the *whole* polygon boundary, not
    just its own edge. This is a local, cheap patch -- not a substitute for a
    proper medial-axis / free-space treatment of the concave case, which is
    noted as a limitation (see paper_notes.md) and links naturally to the
    free-space extraction already built for the companion obstacle-avoidance
    kinematic model.

    Returns (B, N, S, scale): boundary (grasp) points, outward unit normals,
    standoff points S = B + scale * d_standoff * N, and the per-point scale
    actually used (all ones for a convex payload).
    """
    B, N = snap_to_boundary_with_normals(P, polygon_local)
    n = P.shape[0]
    S = np.zeros((n, 2))
    scale = np.ones(n)
    for i in range(n):
        s = 1.0
        while True:
            cand = B[i] + s * d_standoff * N[i]
            clearance = -containment_margin(cand, polygon_local)  # positive => outside
            reached_cap = s >= max_scale
            if clearance >= min_clearance_frac * d_standoff or reached_cap:
                S[i] = cand
                scale[i] = s
                break
            s = min(s * 1.3, max_scale)
    return B, N, S, scale


def min_standoff_clearance(S: np.ndarray, polygon_local: Polygon) -> float:
    """Diagnostic: the worst-case (smallest) distance from any standoff point
    to the polygon boundary. For a convex payload this always equals
    d_standoff; for a concave (reflex-vertex) payload it can be *smaller*
    near a notch, since moving along one edge's normal does not guarantee
    clearance from a different, nearby edge of the same object. Flagged as a
    known limitation rather than silently handled -- see module docstring.
    """
    margins = np.array([containment_margin(s, polygon_local) for s in S])
    return float(-margins.max())  # margins are negative (outside); clearance is their magnitude


def smoothstep(s: np.ndarray) -> np.ndarray:
    """Cubic smoothstep on [0, 1]: 0 and 1 at the ends with zero derivative
    there (an S-curve blend), clipped outside [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    return 3 * s ** 2 - 2 * s ** 3


def smoothstep_dot(s: np.ndarray, s_dot: float) -> np.ndarray:
    """d/dt smoothstep(s(t)) = (6s - 6s^2) * s_dot, zero outside [0, 1]."""
    s_clipped = np.clip(s, 0.0, 1.0)
    raw = (6 * s_clipped - 6 * s_clipped ** 2) * s_dot
    return np.where((s < 0.0) | (s > 1.0), 0.0, raw)


class NormalApproachBias:
    """The time-varying formation bias p_i(t): standoff -> boundary, blended
    by a single global smoothstep schedule over `T_approach` seconds."""

    def __init__(self, S: np.ndarray, B: np.ndarray, T_approach: float) -> None:
        self.S = S
        self.B = B
        self.T = T_approach

    def p(self, t: float) -> np.ndarray:
        s = t / self.T
        a = float(smoothstep(np.array(s)))
        return (1 - a) * self.S + a * self.B

    def p_dot(self, t: float) -> np.ndarray:
        s = t / self.T
        a_dot = float(smoothstep_dot(np.array(s), 1.0 / self.T))
        return a_dot * (self.B - self.S)

    def as_callables(self) -> Tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
        return self.p, self.p_dot


def constant_bias(P: np.ndarray) -> Tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    """The original (baseline) formulation as a callable pair, for direct
    comparison against NormalApproachBias under the identical ODE machinery."""
    zero = np.zeros_like(P)
    return (lambda t: P), (lambda t: zero)
