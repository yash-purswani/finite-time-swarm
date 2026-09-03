"""Integration front-end.

Two things matter for the figures to mean anything:

1. Tight tolerances. The vector field carries a non-smooth sign(.)|.|^beta term;
   at ``solve_ivp`` defaults (rtol 1e-3) the resulting error floor sits around
   1e-5, which is large enough to be mistaken for a real formation-error bound.

2. Event-based detection of the sliding surface. Each component sigma_s reaches
   zero at its own finite time. Rather than searching an output grid for it, we
   stop the solve on the arrival, latch that component (equivalent control:
   sigma_s == 0 is invariant for the nominal system), and continue. This both
   measures tau_c far more finely than an output grid can and removes the
   chattering that would otherwise dominate ||delta(t)|| once the centroid has
   converged.

   The event is ``|sigma_s| = LATCH_TOL``, not a sign change. Finite-time arrival
   is an exact-arithmetic statement: in floating point sigma_s approaches zero
   without ever changing sign, so a sign-change event never fires and the solver
   instead grinds its step size down forever at the crossing. Triggering on a
   small positive threshold fires reliably, and the error it introduces is
   bounded by the remaining travel time from LATCH_TOL to zero,
   ln(1 + tol^(1-beta))/(1-beta) ~ 2e-6 s at the default -- four orders of
   magnitude below the plotting resolution of any figure here, and biased towards
   *under*-reporting tau_c, so it can never manufacture agreement with Eq. (25).

Latching is a *nominal-system* statement. Under disturbance sigma does not stay at
zero -- it settles into a ball of radius wbar^(1/beta) -- so runs with ``w_fn``
set are integrated straight through without latching.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .model import SwarmConfig, rhs

Vec = np.ndarray

RTOL = 1e-10
ATOL = 1e-12
METHOD = "LSODA"
LATCH_TOL = 1e-12      # |sigma_s| at which a component is declared arrived


@dataclass
class SimResult:
    """Sampled trajectory plus the measured sliding-surface arrival times."""

    t: Vec                      # (T,)
    X: Vec                      # (T, n, d)
    cfg: SwarmConfig
    tau_s: Vec                  # (d,) per-component arrival times, inf if never
    segments: List = field(default_factory=list, repr=False)

    @property
    def tau_c_measured(self) -> float:
        """max_s tau_s -- the measured counterpart of Eq. (25)."""
        return float(np.max(self.tau_s))

    @property
    def Z(self) -> Vec:
        """Flattened states, (T, n*d)."""
        return self.X.reshape(self.t.size, -1)

    def at(self, t_query: Vec) -> Vec:
        """Dense evaluation at arbitrary times; returns (Q, n, d)."""
        q = np.atleast_1d(np.asarray(t_query, dtype=float))
        out = np.empty((q.size, self.cfg.n, self.cfg.d))
        for k, tq in enumerate(q):
            dense = self.segments[-1][2]
            for t0, t1, d_ in self.segments:
                if tq <= t1 + 1e-12:
                    dense = d_
                    break
            out[k] = dense(tq).reshape(self.cfg.n, self.cfg.d)
        return out


def _sigma_component(t: float, Z: Vec, cfg: SwarmConfig, s: int) -> float:
    X = Z.reshape(cfg.n, cfg.d)
    return float(X[:, s].mean() - np.asarray(cfg.r_fn(t), dtype=float)[s])


def simulate(cfg: SwarmConfig,
             Z0: Vec,
             t_span: Tuple[float, float],
             t_eval: Optional[Vec] = None,
             latch: Optional[bool] = None,
             rtol: float = RTOL,
             atol: float = ATOL,
             method: str = METHOD,
             max_step: float = np.inf,
             latch_tol: float = LATCH_TOL,
             field_fn: Callable = rhs) -> SimResult:
    """Integrate Eq. (8), detecting and latching each sliding-surface arrival.

    ``latch=None`` selects the behaviour automatically: on for nominal runs with
    the discontinuous sign(.) term, off otherwise. Latching is only meaningful for
    laws whose non-smoothness sits on the centroid channel, so pass ``latch=False``
    alongside a different ``field_fn``.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    if t_eval is None:
        t_eval = np.linspace(t0, t1, 2001)
    t_eval = np.asarray(t_eval, dtype=float)

    if latch is None:
        latch = (cfg.w_fn is None and cfg.zeta_mode == "sign" and cfg.beta < 1.0)

    cfg = cfg.with_(sigma_latch=np.zeros(cfg.d, dtype=bool))
    tau_s = np.full(cfg.d, np.inf)

    # A component that starts on the surface is already there.
    if latch:
        for s in range(cfg.d):
            if abs(_sigma_component(t0, Z0, cfg, s)) <= latch_tol:
                cfg.sigma_latch[s] = True
                tau_s[s] = t0

    segments: List[Tuple[float, float, Callable]] = []
    t_cur, Z_cur = t0, np.asarray(Z0, dtype=float).copy()

    while t_cur < t1 - 1e-12:
        events = []
        if latch:
            for s in range(cfg.d):
                if cfg.sigma_latch[s]:
                    continue

                # solve_ivp appends ``args`` to every event call, so absorb them.
                def ev(t, Z, *_unused, s=s, _cfg=cfg):
                    return abs(_sigma_component(t, Z, _cfg, s)) - latch_tol

                ev.terminal = True
                ev.direction = -1.0          # arriving, not departing
                events.append((s, ev))

        sol = solve_ivp(field_fn, (t_cur, t1), Z_cur, args=(cfg,),
                        method=method, rtol=rtol, atol=atol,
                        dense_output=True, max_step=max_step,
                        events=[e for _, e in events] if events else None)
        if not sol.success:
            raise RuntimeError(f"integration failed at t={t_cur}: {sol.message}")

        segments.append((t_cur, float(sol.t[-1]), sol.sol))

        fired = [(s, sol.t_events[k][0])
                 for k, (s, _) in enumerate(events)
                 if sol.t_events[k].size > 0]
        if not fired:
            break

        # Several components can cross within one step; latch every one that did.
        t_hit = min(th for _, th in fired)
        for s, th in fired:
            if th <= t_hit + 1e-9:
                cfg.sigma_latch[s] = True
                tau_s[s] = th

        t_cur = float(sol.t[-1])
        Z_cur = np.asarray(sol.y[:, -1], dtype=float)

        # Project the latched components exactly onto the surface so the residual
        # left by the event tolerance does not persist through the rest of the run.
        Xc = Z_cur.reshape(cfg.n, cfg.d)
        r_now = np.asarray(cfg.r_fn(t_cur), dtype=float)
        resid = Xc.mean(axis=0) - r_now
        Xc[:, cfg.sigma_latch] -= resid[cfg.sigma_latch]
        Z_cur = Xc.ravel()

    res = SimResult(t=t_eval, X=np.empty((t_eval.size, cfg.n, cfg.d)),
                    cfg=cfg, tau_s=tau_s, segments=segments)
    res.X = res.at(t_eval)
    return res


# =========================================================================
# Fixed-step zero-order-hold stepper (F6, F7)
# =========================================================================

def simulate_zoh(cfg: SwarmConfig,
                 Z0: Vec,
                 t_span: Tuple[float, float],
                 dt: float,
                 field_fn: Callable = rhs) -> SimResult:
    """Sample-and-hold integration at a fixed rate, as a real controller runs.

    The plant is a single integrator, so holding the command constant over a step
    makes ``x_{k+1} = x_k + dt * u_k`` exact -- the only error is the sampling
    itself, which is precisely what F6 is measuring.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    n_steps = int(np.ceil((t1 - t0) / dt))
    t = t0 + dt * np.arange(n_steps + 1)
    Z = np.empty((n_steps + 1, cfg.n * cfg.d))
    Z[0] = np.asarray(Z0, dtype=float)

    cfg = cfg.with_(sigma_latch=np.zeros(cfg.d, dtype=bool))
    for k in range(n_steps):
        Z[k + 1] = Z[k] + dt * field_fn(t[k], Z[k], cfg)

    return SimResult(t=t, X=Z.reshape(-1, cfg.n, cfg.d), cfg=cfg,
                     tau_s=np.full(cfg.d, np.inf), segments=[])
