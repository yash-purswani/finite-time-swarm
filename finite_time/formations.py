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


def _arc_sample(curve: Vec, n: int, closed: bool) -> Vec:
    """n points equally spaced by arc length along a polyline or sampled curve.

    Rows come out in outline order, so drawing them as a closed polygon traces
    the shape.
    """
    pts = np.vstack([curve, curve[:1]]) if closed else curve
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    targets = np.linspace(0.0, s[-1], n, endpoint=not closed)
    return np.column_stack([np.interp(targets, s, pts[:, k])
                            for k in range(pts.shape[1])])


def _planar(P2: Vec, d: int) -> Vec:
    P = np.zeros((P2.shape[0], d))
    P[:, :2] = P2
    return _centre(P)


def star(n: int, L: float = 1.0, inner: float = 0.45, points: int = 5,
         d: int = 2) -> Vec:
    """Star outline with ``points`` tips at radius L, sampled at equal arc length.

    With n a multiple of 2 * points every vertex is a slot.
    """
    k = np.arange(2 * points)
    ang = np.pi / 2 + np.pi * k / points
    rad = np.where(k % 2 == 0, L, inner * L)
    outline = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    return _planar(_arc_sample(outline, n, closed=True), d)


def heart(n: int, L: float = 1.0, d: int = 2) -> Vec:
    """n points equally spaced by arc length on the classic heart curve, width 2L.

    Sampling starts at the top notch, so for even n the bottom tip is a slot too.
    """
    t = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
    curve = np.column_stack([16 * np.sin(t) ** 3,
                             13 * np.cos(t) - 5 * np.cos(2 * t)
                             - 2 * np.cos(3 * t) - np.cos(4 * t)]) * (L / 16.0)
    return _planar(_arc_sample(curve, n, closed=True), d)


def arrow(n: int, L: float = 1.0, d: int = 2) -> Vec:
    """Block arrow of length 2L pointing along +x, rows in outline order (n >= 7).

    The seven corners of the outline are always stations; the other n - 7 go in
    pairs along the shaft, plus one at the centre of the tail if n - 7 is odd.
    """
    head, h, w = 0.8 * L, 0.6 * L, 0.3 * L      # head length, head and shaft half-widths
    b = L - head
    xs = np.linspace(b, -L, (n - 7) // 2 + 2)[1:-1]
    top = [[L, 0.0], [b, h], [b, w]] + [[x, w] for x in xs] + [[-L, w]]
    tail = [[-L, 0.0]] if (n - 7) % 2 else []
    bottom = [[-L, -w]] + [[x, -w] for x in xs[::-1]] + [[b, -w], [b, -h]]
    return _planar(np.array(top + tail + bottom), d)


def wedge(n: int, L: float = 1.0, d: int = 2) -> Vec:
    """Filled wedge with its apex leading along +x: rows of 1, 2, 3, ... agents.

    A triangular lattice of spacing L; the last row is centred if n is not a
    triangular number.
    """
    pts, row = [], 0
    while len(pts) < n:
        m = min(row + 1, n - len(pts))
        x = -row * L * np.sqrt(3) / 2
        pts += [[x, (j - (m - 1) / 2) * L] for j in range(m)]
        row += 1
    return _planar(np.array(pts), d)


def sphere(n: int, L: float = 1.0) -> Vec:
    """Fibonacci sphere of radius L, used for the d = 3 run in F6."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    P = L * np.column_stack([np.cos(theta) * np.sin(phi),
                             np.sin(theta) * np.sin(phi),
                             np.cos(phi)])
    return _centre(P)


SHAPES = {"hexagon": hexagon, "line": line, "vee": vee, "grid": grid,
          "star": star, "heart": heart, "arrow": arrow, "wedge": wedge}


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


def rotating_schedule(P: Vec, omega: float) -> Tuple[Callable, Callable]:
    """Rigid rotation of a centred planar formation at constant rate ``omega``.

    Exercises the pdot_i feedforward of the model, which ``static_schedule``
    leaves identically zero. Rotation preserves the centering condition exactly:
    sum_i R(t) p_i = R(t) sum_i p_i = 0, so Eq. (4) holds for every t whenever it
    holds for the template.
    """
    P = np.asarray(P, dtype=float)
    if P.shape[1] != 2:
        raise ValueError("rotating_schedule is planar; got d=%d" % P.shape[1])

    def P_fn(t):
        c, s = np.cos(omega * t), np.sin(omega * t)
        return P @ np.array([[c, -s], [s, c]]).T

    def Pdot_fn(t):
        c, s = np.cos(omega * t), np.sin(omega * t)
        return P @ (omega * np.array([[-s, -c], [c, -s]])).T

    return P_fn, Pdot_fn
