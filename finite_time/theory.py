"""Analytical bounds from the finite-time report, one function per equation.

Every reference curve drawn in the figures comes from here, so the formulas exist
in exactly one place and can be audited against the PDF line by line. Equation
numbers refer to the current revision of ``main.tex``; regenerate them with
``grep newlabel{eq: main.aux`` after any reorganisation of the paper.

A note on anchoring. Every formation-channel bound is stated from the initial
time t0, not from the centroid settling time tau_c. This follows from Lemma 1
(Eqs. 3 and 7): under the centering condition sum_i delta_i = n sigma, so the
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
    """Eq. (9): nominal centroid settling time: t0 + max_s |sigma_s(t0)|^(1-beta) / (1-beta).

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
    """Eq. (7): the term the pre-Lemma-1 argument had to assume away.

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
    """Eq. (10): ||delta(t)|| <= ||delta(t0)|| exp(-(t - t0)) for all t >= t0.

    Holds from the initial time -- no centroid hypothesis -- by Lemma 1.
    """
    t = np.asarray(t, dtype=float)
    return delta_0 * np.exp(-np.maximum(t - t0, 0.0))


# =========================================================================
# Corollary 1 -- deterministic docking time (nominal)
# =========================================================================

def T_dock(delta_0: float, eps_tol: float, t0: float = 0.0) -> float:
    """Eq. (17): T_dock = t0 + max(0, ln(||delta(t0)|| / eps_tol)).

    Available immediately; it does not wait on the centroid settling.
    """
    if eps_tol <= 0.0:
        raise ValueError("eps_tol must be positive")
    return float(t0 + max(0.0, np.log(delta_0 / eps_tol)))


# =========================================================================
# Proposition 3 -- ISS under bounded disturbance
# =========================================================================

def B_iss(n, wbar):
    """Eq. (21): radius of the forward-invariant hyper-ball, sqrt(n) * wbar.

    Either argument may be an array, so the radius can be swept over n or over
    wbar without the caller reimplementing it.
    """
    return _scalar_or_array(np.sqrt(np.asarray(n, dtype=float))
                            * np.asarray(wbar, dtype=float))


def iss_envelope(t: Vec, delta_0: float, t0: float, n: int, wbar: float) -> Vec:
    """Eq. (20): ||delta(t0)|| e^{-(t-t0)} + sqrt(n) wbar (1 - e^{-(t-t0)}).

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
    """Eq. (23): t0 + max(0, ln(max(0, ||delta(t0)|| - sqrt(n) wbar) / eps_tol))."""
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

    From the perturbed centroid dynamics of Prop. 4(i),

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
    """Eq. (24): finite entrance time into the (wbar/mu)^(1/beta) box.

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


# =========================================================================
# Proposition 2(ii) / Remark 1 -- coupling-improved exponential rate
# =========================================================================

def kappa_0(lambda_min_M, D: float, nu: int, eps: float):
    """kappa_0 = lambda_min(M) / (D^nu + eps), the rate the coupling buys in Eq. (11).

    ``D`` is the bound on the swarm diameter used in the a_min estimate of
    Prop. 2(ii). Two choices are meaningful:

    * ``D = D_0 = sqrt(2)||delta(t0)|| + D_p`` -- the a priori bound, valid from
      t0 but very conservative while the swarm is still far from formation;
    * ``D = D_p`` -- the limit obtained by restarting the argument of Prop. 2(ii)
      at a late t_1 (Remark 1): as ||delta(t_1)|| -> 0 the diameter bound
      contracts onto the formation's own diameter, and the guaranteed rate rises
      to 1 + lambda_min(M)/(D_p^nu + eps).

    The second is the quantity a late-time slope fit must be compared against;
    the first is what holds over the whole trajectory.
    """
    lam = np.asarray(lambda_min_M, dtype=float)
    return _scalar_or_array(lam / (float(D) ** nu + eps))


def formation_rate(lambda_min_M, D: float, nu: int, eps: float):
    """Guaranteed post-tau_c decay rate 1 + kappa_0, Eq. (11) with Remark 1.

    A measured late-time rate must lie at or above this; it may exceed it,
    because the bound replaces the algebraic connectivity of the distance-weighted
    Laplacian by its worst-case entry a_min.
    """
    return _scalar_or_array(1.0 + np.asarray(kappa_0(lambda_min_M, D, nu, eps)))


def D_0(delta_0: float, D_p: float) -> float:
    """A priori swarm-diameter bound sqrt(2)||delta(t0)|| + D_p of Prop. 2(ii)."""
    return float(np.sqrt(2.0) * delta_0 + D_p)


def formation_diameter(P: Vec) -> float:
    """D_p, Eq. (2): the largest inter-slot distance of a static formation."""
    P = np.asarray(P, dtype=float)
    diff = P[:, None, :] - P[None, :, :]
    return float(np.linalg.norm(diff, axis=2).max())


def delta_envelope_coupled(t: Vec, delta_0: float, t0: float, tau_c_: float,
                           lambda_min_M: float, D_p: float, nu: int,
                           eps: float) -> Vec:
    """Per-coupling envelope: Eq. (10) up to tau_c, then Prop. 2(ii) restarted.

    Prop. 2(ii), Eq. (11), gives the improved rate only on [tau_c, inf): before
    tau_c one has sum_i delta_i = n sigma != 0, so the coupling term is bounded
    below only by a_min lambda_min (||delta||^2 - n||sigma||^2) and cannot be
    credited. On [t0, tau_c] the envelope is therefore the geometry-free Eq. (10)
    decay, rate 1, whatever M is.

    On [tau_c, inf) Remark 1 allows the argument to be restarted at any later
    t_1, with D_0 replaced by sqrt(2)||delta(t_1)|| + D_p. Restarting at every
    instant turns the piecewise-constant rate into the scalar comparison ODE

        Edot = -E - lambda_min(M) E / ((sqrt(2) E + D_p)^nu + eps),   E(tau_c) = ...

    which is Eq. (16) of the paper.

    whose solution dominates ||delta(t)||: the diameter bound
    ||x_i - x_j|| <= sqrt(2)||delta(t)|| + D_p holds instantaneously, and ||delta||
    is non-increasing, so the comparison lemma applies. The rate it encodes rises
    from 1 + lambda_min/(D(tau_c)^nu + eps) at the restart to the contracted-swarm
    limit 1 + lambda_min/(D_p^nu + eps) of :func:`formation_rate` as E -> 0.

    Integrated in log coordinates y = ln E, which is exact over the many decades
    the panel spans and keeps E positive.
    """
    from scipy.integrate import solve_ivp

    t = np.asarray(t, dtype=float)
    lam = float(lambda_min_M)
    E_tau = float(delta_0) * np.exp(-max(tau_c_ - t0, 0.0))
    out = float(delta_0) * np.exp(-np.maximum(t - t0, 0.0))
    late = t >= tau_c_
    if not late.any():
        return out

    def rhs(_, y):
        E = np.exp(y[0])
        return [-(1.0 + lam / ((np.sqrt(2.0) * E + D_p) ** nu + eps))]

    t_late = t[late]
    sol = solve_ivp(rhs, (tau_c_, max(float(t_late[-1]), tau_c_ + 1e-12)),
                    [np.log(E_tau)], t_eval=t_late, rtol=1e-10, atol=1e-12,
                    max_step=0.05)
    out[late] = np.exp(sol.y[0])
    return out


# =========================================================================
# Corollary 3 -- finite-time reconnection of a limited-range proximity graph
# =========================================================================

def T_conn(delta_0: float, D_p: float, delta_sense: float,
           t0: float = 0.0) -> float:
    """Eq. (30): t0 + max(0, ln(sqrt(2)||delta(t0)|| / (delta_sense - D_p))).

    Time after which the proximity graph of Sec. V is complete. The bound rests
    only on the rate-1 envelope :func:`delta_envelope`, which Prop. 5(ii) shows
    holds under *any* partition, together with the instantaneous diameter bound
    ||x_i - x_j|| <= sqrt(2)||delta(t)|| + D_p of Eq. (15). It is finite exactly
    when the sensing radius exceeds the formation diameter; at or below D_p the
    slots themselves can hold two agents that far apart, so the swarm may stay
    fragmented forever.
    """
    if delta_sense <= D_p:
        return np.inf
    excess = np.sqrt(2.0) * float(delta_0) / (float(delta_sense) - float(D_p))
    return float(t0 + max(0.0, np.log(excess)))
