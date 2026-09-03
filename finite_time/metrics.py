"""Error signals and measured quantities extracted from a simulation."""

import numpy as np

Vec = np.ndarray


def reference(t: Vec, cfg) -> Vec:
    """Stacked reference r(t) of shape (T, d)."""
    return np.array([cfg.r_fn(float(ti)) for ti in np.atleast_1d(t)], dtype=float)


def biases(t: Vec, cfg) -> Vec:
    """Stacked formation biases P(t) of shape (T, n, d)."""
    return np.array([cfg.P_fn(float(ti)) for ti in np.atleast_1d(t)], dtype=float)


def centroid_error(t: Vec, X: Vec, cfg) -> Vec:
    """sigma(t) = c(t) - r(t), Eq. (3). ``X`` is (T, n, d); returns (T, d)."""
    return X.mean(axis=1) - reference(t, cfg)


def formation_errors(t: Vec, X: Vec, cfg) -> Vec:
    """delta_i(t) = x_i - p_i - r, Eq. (5). ``X`` is (T, n, d); returns (T, n, d)."""
    return X - biases(t, cfg) - reference(t, cfg)[:, None, :]


def delta_norm(t: Vec, X: Vec, cfg) -> Vec:
    """Collective ||delta(t)|| = sqrt(sum_i ||delta_i||^2), the quantity Prop. 2 bounds."""
    D = formation_errors(t, X, cfg)
    return np.linalg.norm(D.reshape(D.shape[0], -1), axis=1)


def max_agent_error(t: Vec, X: Vec, cfg) -> Vec:
    """max_i ||delta_i(t)||, the quantity Cor. 1 turns into a docking tolerance."""
    D = formation_errors(t, X, cfg)
    return np.linalg.norm(D, axis=2).max(axis=1)


# =========================================================================
# Derived measurements
# =========================================================================

def empirical_rate(t: Vec, y: Vec, lo: float = None, hi: float = None,
                   floor: float = 1e-9) -> float:
    """Least-squares decay rate of ``y`` from the slope of ln y over [lo, hi].

    Returns ``-d(ln y)/dt``, so a value >= 1 confirms the Prop. 2 guarantee.
    Samples at or below ``floor`` are dropped -- they carry integrator noise, not
    dynamics.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y) & (y > floor)
    if lo is not None:
        m &= t >= lo
    if hi is not None:
        m &= t <= hi
    if m.sum() < 3:
        return np.nan
    slope = np.polyfit(t[m], np.log(y[m]), 1)[0]
    return float(-slope)


def settling_time(t: Vec, y: Vec, tol: float) -> float:
    """First time after which ``y`` stays at or below ``tol`` for the rest of the run.

    Interpolates linearly between the bracketing samples so the answer is not
    quantised to the output grid.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    above = np.where(y > tol)[0]
    if above.size == 0:
        return float(t[0])
    k = above[-1]
    if k + 1 >= t.size:
        return np.inf                       # never settles within the horizon
    y0, y1 = y[k], y[k + 1]
    if y0 == y1:
        return float(t[k + 1])
    frac = (y0 - tol) / (y0 - y1)
    return float(t[k] + frac * (t[k + 1] - t[k]))


def ultimate_value(t: Vec, y: Vec, tail_frac: float = 0.25, stat: str = "max") -> float:
    """Steady-state level of ``y``, taken over the last ``tail_frac`` of the horizon."""
    t = np.asarray(t, dtype=float)
    cut = t[0] + (1.0 - tail_frac) * (t[-1] - t[0])
    tail = np.asarray(y, dtype=float)[t >= cut]
    if tail.size == 0:
        return float(np.asarray(y)[-1])
    return float(tail.max() if stat == "max" else np.mean(tail))


def min_pair_distance(X: Vec) -> Vec:
    """Smallest inter-agent distance at each time step; ``X`` is (T, n, d)."""
    diff = X[:, :, None, :] - X[:, None, :, :]
    dist = np.linalg.norm(diff, axis=3)
    n = X.shape[1]
    iu = np.triu_indices(n, k=1)
    return dist[:, iu[0], iu[1]].min(axis=1)


def max_pair_distance(X: Vec) -> Vec:
    """Largest inter-agent distance at each time step; ``X`` is (T, n, d)."""
    diff = X[:, :, None, :] - X[:, None, :, :]
    return np.linalg.norm(diff, axis=3).reshape(X.shape[0], -1).max(axis=1)


def held_commands(t: Vec, X: Vec) -> Vec:
    """Recover the commanded velocity from a sampled trajectory, shape (T-1, n, d).

    Forward differences, not ``np.gradient``: under a zero-order hold the plant is
    an integrator, so (x_{k+1} - x_k)/dt *is* the command u_k exactly. A central
    difference would average consecutive commands and halve precisely the
    step-to-step switching this is meant to measure.
    """
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)[:, None, None]
    return np.diff(np.asarray(X, dtype=float), axis=0) / dt


def control_effort(t: Vec, X: Vec) -> float:
    """Integral of sum_i ||u_i|| dt over the run."""
    t = np.asarray(t, dtype=float)
    u = held_commands(t, X)
    speed = np.linalg.norm(u, axis=2).sum(axis=1)
    return float(np.sum(speed * np.diff(t)))


def chattering_index(t: Vec, X: Vec, tail_frac: float = 0.5) -> float:
    """Total variation of the *commanded* signal over the tail, per unit time.

    Summed over agents and coordinates rather than over ||u_i||. The discrete-time
    sliding motion is a two-cycle that flips the sign of the command each step
    while leaving its magnitude almost unchanged, so a norm-based measure reports
    zero for a trajectory that is in fact switching at every sample.
    """
    t = np.asarray(t, dtype=float)
    u = held_commands(t, X)
    tu = t[:-1]
    cut = tu[0] + (1.0 - tail_frac) * (tu[-1] - tu[0])
    m = tu >= cut
    if m.sum() < 3:
        return 0.0
    span = tu[m][-1] - tu[m][0]
    tv = np.abs(np.diff(u[m], axis=0)).sum()
    return float(tv / max(span, 1e-12))
