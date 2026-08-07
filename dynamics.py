"""
dynamics.py
===============================================================================
Finite-Time Swarm Kinematic Guidance Model with (possibly time-varying)
Formation Biases.

    dx_i/dt = -(x_i - p_i(t)) + r(t) + rdot(t) + pdot_i(t) - zeta(t)
              - (M/n) * sum_j  [ (x_i-p_i(t)) - (x_j-p_j(t)) ]
                              / ( || (x_i-p_i(t)) - (x_j-p_j(t)) ||^nu + eps )

with sigma(t) = c(t) - r(t), c(t) = mean_i x_i(t), and
     zeta(t)  = sign(sigma(t)) * |sigma(t)|^beta   (elementwise), beta in (0,1).

Writing the per-agent tracking error delta_i(t) = x_i(t) - p_i(t) - r(t), the
pairwise differences (x_i-p_i)-(x_j-p_j) collapse to delta_i - delta_j (the
r(t) term cancels), and substituting the law above gives the closed-loop
error equation

    ddelta_i/dt = -delta_i(t) - zeta(t) - coupling_i(t)

which is *independent of how p_i(t) was chosen* -- constant (the original
formulation) or time-varying (see approach.py's normal-guided standoff ->
boundary schedule), as long as pdot_i(t) is fed forward exactly as above.

Note one subtlety versus the constant-bias case: sigma(t) = c(t) - r(t)
expands to mean_i(delta_i(t)) + mean_i(p_i(t)). With a constant bias the
offline optimizer's hard centering constraint sum_i p_i = 0 makes the second
term vanish identically, so sigma(t) = mean_i(delta_i(t)) exactly. With the
time-varying standoff->boundary schedule, sum_i p_i(t) = 0 only once the
schedule completes (mean of the boundary points B is zero by construction,
but mean of the standoff points S generally is not) -- so *before* the
schedule finishes, sigma(t) also carries this mean-offset term. This is not
a bug: it is exactly the mechanism that makes the swarm's centroid genuinely
approach the CoG *as* the per-agent schedule completes, rather than being a
separately-driven quantity (see approach.py's module docstring).

This is the same finite-time coupling structure used in the companion
obstacle-avoidance swarm work; nothing about the coupling law itself is
touched here.

This module is pure dynamics: no geometry, no optimization, no plotting.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def zeta(sigma: np.ndarray, beta: float) -> np.ndarray:
    """Non-smooth sliding-mode signal, elementwise sign(.)|.|^beta."""
    return np.sign(sigma) * np.abs(sigma) ** beta


def coupling_term(delta: np.ndarray, M: np.ndarray, nu: float, eps: float) -> np.ndarray:
    """Per-agent finite-time coupling term, shape (n, 2).

    coupling_i = (M/n) * sum_j (delta_i - delta_j) / (||delta_i-delta_j||^nu + eps)
    """
    n = delta.shape[0]
    diff = delta[:, None, :] - delta[None, :, :]           # (n, n, 2)
    norm = np.linalg.norm(diff, axis=2)                     # (n, n)
    denom = (norm ** nu + eps)[:, :, None]                  # (n, n, 1)
    raw = (diff / denom).sum(axis=1) / n                     # (n, 2)
    return raw @ M


def state_derivative(
    x: np.ndarray,
    p: np.ndarray,
    p_dot: np.ndarray,
    r: np.ndarray,
    r_dot: np.ndarray,
    M: np.ndarray,
    nu: float,
    eps: float,
    beta: float,
) -> np.ndarray:
    """dx/dt for all n agents at once. x, p, p_dot: (n, 2). r, r_dot: (2,).

    p_dot is the feedforward derivative of a (possibly time-varying) formation
    bias p(t) -- see approach.py. Feeding it forward here keeps the resulting
    closed-loop error equation ddelta/dt = -delta - zeta - coupling identical
    in form to the constant-bias case (delta = x - p(t) - r(t)); for a
    constant bias, p_dot == 0 and this reduces exactly to the original law.
    """
    delta = x - p - r
    sigma = delta.mean(axis=0)
    z = zeta(sigma, beta)
    coup = coupling_term(delta, M, nu, eps)
    return -delta + r_dot + p_dot - z - coup   # broadcasts (2,) - (n,2) correctly


def make_ode_rhs(
    p_fn: Callable[[float], np.ndarray],
    p_dot_fn: Callable[[float], np.ndarray],
    r_fn: Callable[[float], np.ndarray],
    r_dot_fn: Callable[[float], np.ndarray],
    M: np.ndarray,
    nu: float,
    eps: float,
    beta: float,
    n: int,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build a `solve_ivp`-compatible RHS: f(t, x_flat) -> xdot_flat.

    p_fn / p_dot_fn are functions of time returning (n, 2) arrays -- for the
    original constant-bias formulation, wrap a fixed P with
    `approach.constant_bias(P)` to get a matching (p_fn, p_dot_fn) pair.
    """

    def rhs(t: float, x_flat: np.ndarray) -> np.ndarray:
        x = x_flat.reshape(n, 2)
        xdot = state_derivative(x, p_fn(t), p_dot_fn(t), r_fn(t), r_dot_fn(t), M, nu, eps, beta)
        return xdot.flatten()

    return rhs


def centroid_error(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """sigma(t) = c(t) - r(t)."""
    return x.mean(axis=0) - r


def tracking_errors(x: np.ndarray, P: np.ndarray, r: np.ndarray) -> np.ndarray:
    """delta_i(t) = x_i(t) - p_i - r(t), returned as per-agent norms, shape (n,)."""
    delta = x - P - r
    return np.linalg.norm(delta, axis=1)


def tracking_errors_from_target(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Same as `tracking_errors`, but takes a precomputed target = p(t) + r(t)
    directly (handy when p(t) is time-varying and was already evaluated)."""
    return np.linalg.norm(x - target, axis=1)


def theoretical_rho(M: np.ndarray, nu: float, eps: float) -> float:
    """Theoretical upper bound radius on the individual tracking errors:

        rho = (||M|| / nu) * ((nu - 1) / eps) ** ((nu - 1) / nu)

    ||M|| is the induced 2-norm (largest singular value) of M.
    """
    M_norm = np.linalg.norm(M, ord=2)
    if nu <= 1:
        # The bound as given is only meaningful for nu > 1 (nu = 2 in our
        # parameterization); guard degenerate configurations defensively.
        return float("nan")
    return (M_norm / nu) * ((nu - 1) / eps) ** ((nu - 1) / nu)
