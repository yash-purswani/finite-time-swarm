"""Hybrid swarm kinematic model, Eq. (8) of the finite-time report.

    xdot_i = -(x_i - p_i - r) + rdot + pdot_i - zeta
             - (M/n) sum_j [ (x_i - p_i) - (x_j - p_j) ] / ( ||x_i - x_j||^nu + eps )
             + w_i

The numerator lives in formation-error coordinates while the denominator uses the
*true* physical inter-agent distance.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

Vec = np.ndarray


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class SwarmConfig:
    """Everything needed to evaluate the right-hand side once."""

    n: int
    d: int = 2

    # --- interaction coupling -------------------------------------------
    M: Vec = None                 # (d, d) symmetric; None -> zeros
    nu: int = 2                   # potential decay exponent, nu in N_+
    eps: float = 1.0              # regulariser preventing singular forces

    # --- finite-time term ------------------------------------------------
    beta: float = 0.5             # fractional exponent, beta in (0, 1)
    zeta_mode: str = "sign"       # "sign" (Eq. 6) or "sat" (boundary layer)
    phi: float = 1e-3             # boundary-layer width, used when zeta_mode == "sat"

    # --- reference trajectory -------------------------------------------
    r_fn: Callable[[float], Vec] = None       # t -> (d,)
    rdot_fn: Callable[[float], Vec] = None    # t -> (d,)

    # --- formation biases -------------------------------------------------
    P_fn: Callable[[float], Vec] = None       # t -> (n, d)
    Pdot_fn: Callable[[float], Vec] = None    # t -> (n, d); None -> zeros

    # --- disturbance ------------------------------------------------------
    # w(t, X) -> (n, d). Takes the state so adversarial disturbances are expressible.
    w_fn: Optional[Callable[[float, Vec], Vec]] = None

    # --- implementation switches -----------------------------------------
    v_max: Optional[float] = None      # per-agent speed saturation, None -> unbounded
    use_true_distance: bool = True     # False reproduces the legacy bias-shifted variant
    use_pdot: bool = True              # False ablates the pdot_i feedforward
    # Components of sigma latched to zero by the equivalent-control switch.
    # Shape (d,) boolean; set by integrate.simulate(), not by the user.
    sigma_latch: Vec = field(default=None, repr=False)

    def __post_init__(self):
        if self.M is None:
            self.M = np.zeros((self.d, self.d))
        self.M = np.atleast_2d(np.asarray(self.M, dtype=float))
        if self.M.shape != (self.d, self.d):
            raise ValueError(f"M must be ({self.d},{self.d}), got {self.M.shape}")
        if not (0.0 < self.beta <= 1.0):
            raise ValueError("beta must lie in (0, 1]; beta=1 is the linear baseline")
        if self.r_fn is None:
            raise ValueError("r_fn is required")
        if self.rdot_fn is None:
            raise ValueError("rdot_fn is required")
        if self.P_fn is None:
            raise ValueError("P_fn is required")
        if self.sigma_latch is None:
            self.sigma_latch = np.zeros(self.d, dtype=bool)

    def with_(self, **kwargs) -> "SwarmConfig":
        """Copy with fields overridden."""
        return replace(self, **kwargs)

    @property
    def lambda_min_M(self) -> float:
        return float(np.min(np.linalg.eigvalsh(self.M)))


# =========================================================================
# Building blocks
# =========================================================================

def zeta(sigma: Vec, beta: float, mode: str = "sign", phi: float = 1e-3) -> Vec:
    """Finite-time sliding-mode term, Eq. (6).

    ``mode="sat"`` replaces sign(.) by a boundary-layer saturation of width phi,
    the standard fix for chattering under zero-order-hold implementation.
    """
    a = np.abs(sigma)
    if mode == "sign":
        return np.sign(sigma) * a ** beta
    if mode == "sat":
        # Continuous outside the layer, linear inside, matched at |sigma| = phi.
        inner = sigma / phi * phi ** beta
        outer = np.sign(sigma) * a ** beta
        return np.where(a <= phi, inner, outer)
    raise ValueError(f"unknown zeta_mode {mode!r}")


def interaction_term(X: Vec, B: Vec, cfg: SwarmConfig) -> Vec:
    """(M/n) sum_j (B_i - B_j) / (||x_i - x_j||^nu + eps), returned as (n, d).

    ``X`` are physical positions, ``B = X - P`` carries the error-coordinate
    numerator (the reference r cancels out of B_i - B_j).
    """
    if not np.any(cfg.M):
        return np.zeros_like(X)

    diff_B = B[:, None, :] - B[None, :, :]              # (n, n, d) numerator source

    # Denominator: true physical distance (Eq. 8) or the legacy bias-shifted one.
    ref = X if cfg.use_true_distance else B
    diff_ref = ref[:, None, :] - ref[None, :, :]
    dist = np.linalg.norm(diff_ref, axis=2)
    denom = dist ** cfg.nu + cfg.eps

    coupled = diff_B @ cfg.M.T                          # (n, n, d)
    contrib = coupled / denom[:, :, None]

    # Drop self-interaction without assuming d == 2.
    idx = np.arange(cfg.n)
    contrib[idx, idx, :] = 0.0

    return contrib.sum(axis=1) / cfg.n


def rhs(t: float, Z: Vec, cfg: SwarmConfig) -> Vec:
    """Flattened right-hand side of Eq. (8) for ``solve_ivp``."""
    X = Z.reshape(cfg.n, cfg.d)

    r = np.asarray(cfg.r_fn(t), dtype=float)
    rdot = np.asarray(cfg.rdot_fn(t), dtype=float)
    P = np.asarray(cfg.P_fn(t), dtype=float)

    if cfg.use_pdot and cfg.Pdot_fn is not None:
        Pdot = np.asarray(cfg.Pdot_fn(t), dtype=float)
    else:
        Pdot = np.zeros_like(P)

    B = X - P                                  # (n, d); delta_i = B_i - r
    # Eq. (3) as the controller actually measures it: centroid minus reference.
    # Deliberately *not* B.mean(axis=0) - r, so that violating the centering
    # condition (Eq. 4) shows up as a real failure in the F5 ablation.
    sigma = X.mean(axis=0) - r

    z = zeta(sigma, cfg.beta, cfg.zeta_mode, cfg.phi)

    # Equivalent control: once component s has reached the sliding surface the
    # nominal solution stays there identically, so freeze it instead of letting
    # the solver chatter across the discontinuity.
    if np.any(cfg.sigma_latch):
        z = np.where(cfg.sigma_latch, -sigma, z)

    coupling = interaction_term(X, B, cfg)

    Xdot = -(B - r) + rdot + Pdot - z - coupling

    if cfg.w_fn is not None:
        Xdot = Xdot + np.asarray(cfg.w_fn(t, X), dtype=float)

    if cfg.v_max is not None:
        speed = np.linalg.norm(Xdot, axis=1, keepdims=True)
        scale = np.minimum(1.0, cfg.v_max / np.maximum(speed, 1e-300))
        Xdot = Xdot * scale

    return Xdot.ravel()


# =========================================================================
# Baseline laws for the comparison figure (F7)
# =========================================================================

def rhs_per_agent_finite_time(t: float, Z: Vec, cfg: SwarmConfig) -> Vec:
    """Decentralised alternative: a finite-time sliding term on *every* agent.

        xdot_i = rdot + pdot_i - delta_i - sign(delta_i)|delta_i|^beta

    Uses n non-smooth channels where Eq. (8) uses one, which is the point of the
    comparison in F7.
    """
    X = Z.reshape(cfg.n, cfg.d)
    r = np.asarray(cfg.r_fn(t), dtype=float)
    rdot = np.asarray(cfg.rdot_fn(t), dtype=float)
    P = np.asarray(cfg.P_fn(t), dtype=float)
    if cfg.use_pdot and cfg.Pdot_fn is not None:
        Pdot = np.asarray(cfg.Pdot_fn(t), dtype=float)
    else:
        Pdot = np.zeros_like(P)

    delta = X - P - r
    Xdot = rdot + Pdot - delta - zeta(delta, cfg.beta, cfg.zeta_mode, cfg.phi)

    if cfg.w_fn is not None:
        Xdot = Xdot + np.asarray(cfg.w_fn(t, X), dtype=float)
    if cfg.v_max is not None:
        speed = np.linalg.norm(Xdot, axis=1, keepdims=True)
        Xdot = Xdot * np.minimum(1.0, cfg.v_max / np.maximum(speed, 1e-300))
    return Xdot.ravel()
