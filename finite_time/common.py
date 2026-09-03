"""Shared scenario defaults and disturbance generators.

Keeping the nominal parameter set in one place means every figure in the paper is
generated from the same swarm unless it explicitly says otherwise.
"""

from typing import Callable, Tuple

import numpy as np

from .formations import hexagon, static_schedule
from .model import SwarmConfig

Vec = np.ndarray

# --- nominal scenario -----------------------------------------------------
N = 6
D = 2
NU = 2
EPS = 1.0
BETA = 0.5
L = 5.0
M_NOMINAL = 5.0
T_END = 20.0
SEED = 42


def straight_reference() -> Tuple[Callable, Callable]:
    """r(t) = (t, 0): unit-speed translation along +x."""
    return (lambda t: np.array([t, 0.0])), (lambda t: np.array([1.0, 0.0]))


def reference_nd(d: int) -> Tuple[Callable, Callable]:
    """Unit-speed translation along the first axis in d dimensions."""
    def r(t):
        v = np.zeros(d)
        v[0] = t
        return v

    def rdot(t):
        v = np.zeros(d)
        v[0] = 1.0
        return v

    return r, rdot


def initial_state(n: int = N, d: int = D, scale: float = 5.0,
                  seed: int = SEED) -> Vec:
    """Flattened Gaussian initial positions."""
    rng = np.random.default_rng(seed)
    return (scale * rng.standard_normal((n, d))).ravel()


def nominal_config(M: float = M_NOMINAL, n: int = N, d: int = D,
                   beta: float = BETA, nu: int = NU, eps: float = EPS,
                   Lf: float = L, **overrides) -> SwarmConfig:
    """The default swarm: n agents on a hexagon of radius ``Lf``, isotropic M."""
    r_fn, rdot_fn = reference_nd(d)
    P = hexagon(n, Lf, d) if d == 2 else _sphere_or_hex(n, Lf, d)
    P_fn, Pdot_fn = static_schedule(P)
    M_mat = np.asarray(M, dtype=float)
    if M_mat.ndim == 0:
        M_mat = float(M_mat) * np.eye(d)
    cfg = SwarmConfig(n=n, d=d, M=M_mat, nu=nu, eps=eps, beta=beta,
                      r_fn=r_fn, rdot_fn=rdot_fn, P_fn=P_fn, Pdot_fn=Pdot_fn)
    return cfg.with_(**overrides) if overrides else cfg


def _sphere_or_hex(n: int, Lf: float, d: int) -> Vec:
    from .formations import sphere
    if d == 3:
        return sphere(n, Lf)
    P = np.zeros((n, d))
    P[:, :2] = hexagon(n, Lf, 2)
    return P - P.mean(axis=0)


# =========================================================================
# Disturbance generators, all satisfying ||w_i(t)|| <= wbar
# =========================================================================

def w_common(wbar: float, d: int = D, axis: bool = False) -> Callable:
    """Identical constant push on every agent -- the benign, common-mode case.

    The push is spread equally over the coordinates by default, so every
    component of the mean disturbance is wbar/sqrt(d) and ||w_i|| = wbar exactly.
    Putting the whole push on one axis (``axis=True``) leaves the other channels
    undisturbed, and an undisturbed channel drives sigma_s to zero exactly, where
    the ideal sign(.) term chatters and stalls the solver. That matters wherever
    the centroid floor itself is being measured.
    """
    v = np.zeros(d)
    if axis:
        v[0] = wbar
    else:
        v[:] = wbar / np.sqrt(d)
    return lambda t, X: np.broadcast_to(v, X.shape).copy()


def w_random(wbar: float, n: int, d: int, seed: int = 0,
             hold: float = 0.05) -> Callable:
    """Independent directions on the sphere of radius ``wbar``, resampled every ``hold``.

    Piecewise-constant rather than continuously random so the vector field stays
    integrable; ``hold`` is far below the closed-loop time constant of 1 s.
    """
    rng = np.random.default_rng(seed)
    n_slots = 4096
    W = rng.standard_normal((n_slots, n, d))
    W /= np.linalg.norm(W, axis=2, keepdims=True)
    W *= wbar

    def w(t, X):
        k = int(max(t, 0.0) / hold) % n_slots
        return W[k]

    return w


def w_sine(wbar: float, n: int, d: int, omega: float = 2.0) -> Callable:
    """Sinusoids at agent-dependent phase, each of magnitude exactly ``wbar``."""
    phase = 2.0 * np.pi * np.arange(n)[:, None] / max(n, 1)

    def w(t, X):
        ang = omega * t + phase
        out = np.zeros((n, d))
        out[:, 0] = np.cos(ang[:, 0])
        if d > 1:
            out[:, 1] = np.sin(ang[:, 0])
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return wbar * out / np.maximum(norm, 1e-300)

    return w


def w_adversarial(wbar: float, P_fn: Callable, r_fn: Callable,
                  rho_rel: float = 1e-3) -> Callable:
    """Worst case: push each agent along its own error, w_i ~ wbar * delta_i/||delta_i||.

    This is the alignment the Cauchy-Schwarz step of Prop. 3 assumes, so it is the
    disturbance that makes the sqrt(n) wbar ball tight.

    The exact normalisation is discontinuous wherever an individual delta_i passes
    through zero, which happens routinely once n is more than a handful and stalls
    any adaptive solver at the crossing. Softening the denominator to
    sqrt(||delta_i||^2 + rho^2) makes the field Lipschitz everywhere while keeping
    ||w_i|| <= wbar exactly. With rho = 1e-3 * wbar the alignment loses a factor of
    (1 - 5e-7) at the steady-state error scale ||delta_i|| ~ wbar, so the bound it
    probes is no less tight.
    """
    rho = rho_rel * wbar

    def w(t, X):
        delta = X - np.asarray(P_fn(t), dtype=float) - np.asarray(r_fn(t), dtype=float)
        soft = np.sqrt((delta ** 2).sum(axis=1, keepdims=True) + rho ** 2)
        return wbar * delta / soft

    return w
