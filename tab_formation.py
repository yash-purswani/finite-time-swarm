"""Table III -- formation convergence and acceptance time across couplings.

The ring of Table I under several couplings, including a rotating template
(pdot_i != 0) and a non-symmetric M. Each row reports

  * the certified-quantity rate: the largest lambda with
    ||delta(t)|| <= ||delta(t_0)|| e^{-lambda (t - t_0)} while ||delta|| >= FLOOR,
    which Theorem 2 guarantees is at least 1;
  * the instant after which every agent stays within eps_tol, for each tolerance.

T_dock of Corollary 1 depends only on ||delta(t_0)||, which neither the coupling
nor the rotation changes, so one bound row serves the whole table. Prints the
LaTeX rows.
"""

import numpy as np

from finite_time import common, metrics, theory
from finite_time.formations import hexagon, rotating_schedule
from finite_time.integrate import simulate

T_END = 14.0
TOLS = (0.1, 0.01)
FLOOR = 1e-6            # rates are read above this, clear of the solver floor
OMEGA = 0.5             # rad/s, as in Table I
J = np.array([[0.0, -1.0], [1.0, 0.0]])     # skew: 5I + 5J has symmetric part 5I

ROWS = [
    (r"$0$", 0.0, False),
    (r"$5I_2$", 5.0, False),
    (r"$5I_2$, rotating", 5.0, True),
    (r"$25I_2$", 25.0, False),
    (r"$5I_2+5J$", 5.0 * np.eye(2) + 5.0 * J, False),
]


def _entry(t, mx, tol):
    """First instant after which max_i ||delta_i|| stays at or below tol."""
    above = np.flatnonzero(mx > tol)
    return float(t[0]) if above.size == 0 else float(t[min(above[-1] + 1, t.size - 1)])


def main() -> dict:
    Z0 = common.initial_state()
    t = np.linspace(0.0, T_END, 14001)
    rows, summary = [], {}
    for label, M, rotating in ROWS:
        cfg = common.nominal_config(M=M)
        if rotating:
            P_fn, Pdot_fn = rotating_schedule(hexagon(common.N, common.L), OMEGA)
            cfg = cfg.with_(P_fn=P_fn, Pdot_fn=Pdot_fn)
        res = simulate(cfg, Z0, (0.0, T_END), t_eval=t)
        dn = metrics.delta_norm(t, res.X, cfg)
        mx = metrics.max_agent_error(t, res.X, cfg)
        live = (t > 0) & (dn >= FLOOR)
        rate = float((np.log(dn[0] / dn[live]) / t[live]).min())
        entries = [_entry(t, mx, tol) for tol in TOLS]
        rows.append((label, rate, entries))
        summary[label] = (rate, entries)

    d0 = float(metrics.delta_norm(t[:1], Z0.reshape(1, common.N, common.D),
                                  common.nominal_config())[0])
    bounds = [theory.T_dock(d0, tol, 0.0) for tol in TOLS]
    summary["T_dock"] = bounds
    print(f"    ||delta(t_0)|| = {d0:.4f}")
    print(f"    Bound (Thm. 2, Cor. 1) & $1$ & " +
          " & ".join(f"${b:.2f}$" for b in bounds) + r" \\")
    for label, rate, entries in rows:
        print(f"    {label} & ${rate:.2f}$ & " +
              " & ".join(f"${e:.2f}$" for e in entries) + r" \\")
    ok = all(r >= 1.0 - 1e-6 and all(e <= b for e, b in zip(es, bounds))
             for _, r, es in rows)
    print(f"    every rate >= 1 and every entry before T_dock: {ok}")
    return summary


if __name__ == "__main__":
    main()
