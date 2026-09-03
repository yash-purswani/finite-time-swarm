"""Analytical bounds from the finite-time report, one function per equation.

Every reference curve drawn in the figures comes from here, so the formulas exist
in exactly one place and can be audited against the PDF line by line. Equation
numbers refer to the current revision of ``finite-time_report.tex``.

A note on anchoring. Every formation-channel bound is stated from the initial
time t0, not from the centroid settling time tau_c. This follows from Lemma 1
(Eq. 12-13): under the centering condition sum_i delta_i = n sigma, so the
sliding-mode term contributes -n sum_s |sigma_s|^(1+beta) <= 0 to the formation
Lyapunov derivative whatever the centroid is doing. Retaining zeta can only make
V_delta decrease faster, so no assumption of the form "zeta == 0 for t >= tau_c"
is needed anywhere. This matters under disturbance, where sigma never reaches the
origin (Proposition 4) and tau_c is therefore undefined.

tau_c survives only where it is meaningful: as the nominal centroid settling time
of Proposition 1.
"""

import numpy as np

Vec = np.ndarray


def _scalar_or_array(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


# =========================================================================
# Proposition 1 -- finite-time centroid convergence (nominal)
# =========================================================================

def tau_c(sigma0: Vec, beta: float, t0: float = 0.0) -> float:
    """Nominal centroid settling time: t0 + max_s |sigma_s(t0)|^(1-beta) / (1-beta).

    Upper bound obtained by discarding the -2V_s term of the Lyapunov derivative.
    Valid only for w == 0; under disturbance see :func:`T_sigma`.
    """
    if beta >= 1.0:
        return np.inf                      # linear case: asymptotic only
    a = np.abs(np.asarray(sigma0, dtype=float))
    return float(t0 + np.max(a ** (1.0 - beta)) / (1.0 - beta))


# =========================================================================
# Lemma 1 -- the identity the t0 anchoring rests on
# =========================================================================

def zeta_dissipation(sigma: Vec, beta: float, n: int) -> Vec:
    """Eq. (13): the term Proposition 3 used to assume away.

        -sum_i delta_i^T zeta = -n sum_s |sigma_s|^(1+beta) <= 0

    Returned for verification: it must be non-positive along every trajectory,
    which is exactly why the bounds below hold from t0 rather than from tau_c.
    """
    sig = np.abs(np.asarray(sigma, dtype=float))
    return _scalar_or_array(-n * np.sum(sig ** (1.0 + beta), axis=-1))


# =========================================================================
# Proposition 2 -- global exponential formation convergence (nominal)
# =========================================================================

def delta_envelope(t: Vec, delta_0: float, t0: float = 0.0) -> Vec:
    """Eq. (26): ||delta(t)|| <= ||delta(t0)|| exp(-(t - t0)) for all t >= t0.

    Holds from the initial time -- no centroid hypothesis -- by Lemma 1.
    """
    t = np.asarray(t, dtype=float)
    return delta_0 * np.exp(-np.maximum(t - t0, 0.0))


# =========================================================================
# Corollary 1 -- deterministic docking time (nominal)
# =========================================================================

def T_dock(delta_0: float, eps_tol: float, t0: float = 0.0) -> float:
    """Eq. (35): T_dock = t0 + max(0, ln(||delta(t0)|| / eps_tol)).

    Available immediately; it does not wait on the centroid settling.
    """
    if eps_tol <= 0.0:
        raise ValueError("eps_tol must be positive")
    return float(t0 + max(0.0, np.log(delta_0 / eps_tol)))


# =========================================================================
# Proposition 3 -- ISS under bounded disturbance
# =========================================================================

def B_iss(n, wbar):
    """Eq. (40): radius of the forward-invariant hyper-ball, sqrt(n) * wbar.

    Either argument may be an array, so the radius can be swept over n or over
    wbar without the caller reimplementing it.
    """
    return _scalar_or_array(np.sqrt(np.asarray(n, dtype=float))
                            * np.asarray(wbar, dtype=float))


def iss_envelope(t: Vec, delta_0: float, t0: float, n: int, wbar: float) -> Vec:
    """Eq. (39): ||delta(t0)|| e^{-(t-t0)} + sqrt(n) wbar (1 - e^{-(t-t0)}).

    Anchored at t0 and valid for every t >= t0, covering the whole transient
    rather than only the post-tau_c tail.
    """
    t = np.asarray(t, dtype=float)
    decay = np.exp(-np.maximum(t - t0, 0.0))
    return delta_0 * decay + B_iss(n, wbar) * (1.0 - decay)


# =========================================================================
# Corollary 2 -- finite entrance time into the eps_tol-neighbourhood
# =========================================================================

def T_enter(delta_0: float, eps_tol: float, t0: float, n: int, wbar: float) -> float:
    """Eq. (52): t0 + max(0, ln(max(0, ||delta(t0)|| - sqrt(n) wbar) / eps_tol))."""
    if eps_tol <= 0.0:
        raise ValueError("eps_tol must be positive")
    excess = max(0.0, delta_0 - B_iss(n, wbar))
    if excess == 0.0:
        return float(t0)
    return float(t0 + max(0.0, np.log(excess / eps_tol)))


# =========================================================================
# Proposition 4 -- centroid disturbance ball
# =========================================================================

def sigma_ball(wbar, beta: float):
    """Radius of the forward-invariant centroid box: wbar^(1/beta), Prop. 4(ii).

    From the perturbed centroid dynamics Eq. (55),

        sigmadot_s = -sigma_s - sign(sigma_s)|sigma_s|^beta + wbar_c,s

    decay persists while |sigma_s|^beta > wbar, so the box {|sigma_s| <=
    wbar^(1/beta)} is forward invariant and is the ultimate bound, Prop. 4(iv).

    Since 1/beta > 1 this is *super-linear* attenuation for wbar < 1: for
    beta = 0.5, wbar = 0.1 the centroid is held to 1e-2, against the formation
    channel's linear sqrt(n) wbar. For wbar >= 1 the set is still invariant but
    no longer improves on wbar.
    """
    if beta >= 1.0:
        return _scalar_or_array(wbar)      # linear case: no attenuation
    return _scalar_or_array(np.asarray(wbar, dtype=float) ** (1.0 / beta))


def sigma_ball_norm(wbar, beta: float, d: int):
    """Euclidean-norm version of :func:`sigma_ball`: sqrt(d) * wbar^(1/beta)."""
    return _scalar_or_array(np.sqrt(d) * np.asarray(sigma_ball(wbar, beta)))


def sigma_ball_mu(wbar, beta: float, mu: float):
    """Radius of the set reached in *finite* time, Prop. 4(iii): (wbar/mu)^(1/beta).

    mu in (0,1) trades radius against speed: mu -> 0 recovers the nominal
    settling time with a diverging radius, mu -> 1 tightens the radius to
    :func:`sigma_ball` with a diverging time bound.
    """
    if not 0.0 < mu < 1.0:
        raise ValueError("mu must lie in (0, 1)")
    return sigma_ball(np.asarray(wbar, dtype=float) / mu, beta)


def T_sigma(sigma0: Vec, beta: float, mu: float, t0: float = 0.0) -> float:
    """Eq. (56): finite entrance time into the (wbar/mu)^(1/beta) box.

        T_sigma(mu) = t0 + max_s |sigma_s(t0)|^(1-beta) / ((1-mu)(1-beta))

    This is :func:`tau_c` with alpha replaced by (1-mu)*alpha, and reduces to it
    as mu -> 0. It is the disturbed-case replacement for tau_c, which does not
    exist once w != 0.
    """
    if not 0.0 < mu < 1.0:
        raise ValueError("mu must lie in (0, 1)")
    if beta >= 1.0:
        return np.inf
    a = np.abs(np.asarray(sigma0, dtype=float))
    return float(t0 + np.max(a ** (1.0 - beta)) / ((1.0 - mu) * (1.0 - beta)))


# =========================================================================
# Consequence of violating the centering condition (F5 ablation)
# =========================================================================

def sigma_offset_equilibrium(pbar: Vec, beta: float) -> Vec:
    """Steady centroid error when sum_i p_i = n * pbar instead of 0.

    The centroid dynamics pick up the mean bias, sigmadot_s = -sigma_s + pbar_s
    - sign(sigma_s)|sigma_s|^beta, so sigma settles where

        sigma_s + sign(sigma_s)|sigma_s|^beta = pbar_s

    rather than at zero. Eq. (4) is what removes the pbar_s term, and this is the
    exact size of the error it prevents. It is also where Lemma 1 would fail:
    without centering, sum_i delta_i acquires an offset and the zeta term loses
    its sign.
    """
    from scipy.optimize import brentq

    out = np.zeros_like(np.asarray(pbar, dtype=float))
    for s, b in enumerate(np.atleast_1d(np.asarray(pbar, dtype=float))):
        if b == 0.0:
            continue
        mag = abs(b)
        f = lambda x: x + x ** beta - mag
        out[s] = np.sign(b) * brentq(f, 0.0, max(mag, mag ** (1.0 / beta)) + 1.0)
    return out
