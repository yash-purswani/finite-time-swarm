"""Formation shapes, optimal re-labelling, and smooth morph schedules.

Every shape returned here is centred, so the centering condition Eq. (4)
(sum_i p_i = 0, and hence sum_i pdot_i = 0) holds by construction. The one place
it is deliberately violated is the F5 ablation, which calls
:func:`break_centering` explicitly.
"""

from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Vec = np.ndarray


def _centre(P: Vec) -> Vec:
    return P - P.mean(axis=0)


# =========================================================================
# Shapes
# =========================================================================

def hexagon(n: int, L: float = 1.0, d: int = 2) -> Vec:
    """n points equally spaced on a circle of radius L."""
    ang = 2.0 * np.pi * np.arange(n) / n
    P = np.zeros((n, d))
    P[:, 0] = L * np.cos(ang)
    P[:, 1] = L * np.sin(ang)
    return _centre(P)


def line(n: int, L: float = 1.0, d: int = 2, axis: int = 1) -> Vec:
    """n points evenly spread along one axis, total extent 5L."""
    P = np.zeros((n, d))
    P[:, axis] = np.linspace(-2.5 * L, 2.5 * L, n)
    return _centre(P)


def vee(n: int, L: float = 1.0, d: int = 2) -> Vec:
    """A V / chevron with its apex leading along +x."""
    P = np.zeros((n, d))
    for i in range(n):
        arm = (i + 1) // 2                      # 0, 1, 1, 2, 2, 3, 3, ...
        side = 1.0 if i % 2 else -1.0
        P[i, 0] = -0.6 * L * arm
        P[i, 1] = side * L * arm
    return _centre(P)


def grid(n: int, L: float = 1.0, d: int = 2) -> Vec:
    """Roughly square lattice with spacing L."""
    cols = int(np.ceil(np.sqrt(n)))
    P = np.zeros((n, d))
    for i in range(n):
        P[i, 0] = L * (i % cols)
        P[i, 1] = L * (i // cols)
    return _centre(P)


def sphere(n: int, L: float = 1.0) -> Vec:
    """Fibonacci sphere of radius L, used for the d = 3 run in F6."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    P = L * np.column_stack([np.cos(theta) * np.sin(phi),
                             np.sin(theta) * np.sin(phi),
                             np.cos(phi)])
    return _centre(P)


SHAPES = {"hexagon": hexagon, "line": line, "vee": vee, "grid": grid}


def break_centering(P: Vec, offset: Sequence[float]) -> Vec:
    """Shift a formation so that sum_i p_i = n * offset, violating Eq. (4).

    Used only by the F5 ablation, which shows the centroid then locks onto
    r + offset instead of r.
    """
    return P + np.asarray(offset, dtype=float)


# =========================================================================
# Optimal re-labelling
# =========================================================================

def reassign(P_from: Vec, P_to: Vec) -> Vec:
    """Permute the rows of ``P_to`` to minimise total squared travel from ``P_from``.

    Same Hungarian step as the earlier morphing script, but applied slot-to-slot
    rather than agent-to-slot, so it composes with a smooth interpolation.
    """
    cost = ((P_from[:, None, :] - P_to[None, :, :]) ** 2).sum(axis=2)
    _, col = linear_sum_assignment(cost)
    return P_to[col]


def reassign_from_state(X: Vec, P_to: Vec, r: Vec) -> Vec:
    """Permute ``P_to`` to minimise travel from the agents' current positions."""
    targets = P_to + np.asarray(r, dtype=float)
    cost = ((X[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
    _, col = linear_sum_assignment(cost)
    return P_to[col]


# =========================================================================
# Static and morphing schedules
# =========================================================================

def static_schedule(P: Vec) -> Tuple[Callable, Callable]:
    """Constant P(t), zero Pdot(t)."""
    P = np.asarray(P, dtype=float)
    Z = np.zeros_like(P)
    return (lambda t: P), (lambda t: Z)


def _quintic(u: float) -> Tuple[float, float]:
    """Minimum-jerk blend s(u) and ds/du on u in [0, 1]; s(0)=0, s(1)=1, s'=s''=0 at both ends."""
    u = min(max(u, 0.0), 1.0)
    s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    ds = 30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4
    return s, ds


def morph_schedule(shapes: List[Vec],
                   switch_times: Sequence[float],
                   morph_time: float,
                   relabel: bool = True) -> Tuple[Callable, Callable]:
    """Piecewise-quintic interpolation through a sequence of formations.

    ``switch_times[k]`` is when the blend from ``shapes[k]`` to ``shapes[k+1]``
    begins; each blend lasts ``morph_time``. Both endpoints are centred and the
    blend is affine in them, so sum_i p_i(t) = 0 and sum_i pdot_i(t) = 0 hold at
    every instant -- Eq. (4) is respected throughout the morph, not just at the
    endpoints.

    Returning pdot analytically (rather than differencing) matters: it is the
    feedforward term of Eq. (8), and F5 ablates it.
    """
    S = [np.asarray(s, dtype=float) for s in shapes]
    if relabel:
        for k in range(1, len(S)):
            S[k] = reassign(S[k - 1], S[k])
    switch = np.asarray(switch_times, dtype=float)
    if switch.size != len(S) - 1:
        raise ValueError("need one switch time per transition")

    def P_fn(t: float) -> Vec:
        k = int(np.searchsorted(switch, t, side="right"))
        if k == 0:
            return S[0]
        t_start = switch[k - 1]
        u = (t - t_start) / morph_time
        if u >= 1.0:
            return S[k]
        s, _ = _quintic(u)
        return S[k - 1] + s * (S[k] - S[k - 1])

    def Pdot_fn(t: float) -> Vec:
        k = int(np.searchsorted(switch, t, side="right"))
        if k == 0:
            return np.zeros_like(S[0])
        u = (t - switch[k - 1]) / morph_time
        if u >= 1.0:
            return np.zeros_like(S[0])
        _, ds = _quintic(u)
        return (ds / morph_time) * (S[k] - S[k - 1])

    return P_fn, Pdot_fn
