"""F7 -- comparison against the natural alternatives.

Three baselines, all tracking the same reference with the same formation:

  (i)   beta = 1        the linear law, which is asymptotic rather than finite-time
  (ii)  M = 0           Eq. (8) with the inter-agent coupling switched off
  (iii) per-agent SMC   xdot_i = rdot + pdot_i - delta_i - sign(delta_i)|delta_i|^beta,
                        the decentralised finite-time law

What the measurements actually say. Control effort and chattering come out the
same for every discontinuous law (panel c): putting zeta on the centroid does not
reduce the number of switching command channels, because that one zeta is applied
to all n agents. And the decentralised law is the faster of the two to formation
(2.5 s vs 4.8 s) and holds a wider inter-agent gap. Neither of those favours
Eq. (8), and both are reported as measured.

What Eq. (8) does give, and no per-agent law can, is panel (d): the centroid
arrival time depends on sigma(0) alone. Change the formation from a hexagon to a
line to a V to a grid and the deadline moves by 1e-11 s, against 0.29 s for the
decentralised law. That is Prop. 1's real content -- the mission-level guarantee
is decoupled from the formation design -- and it comes from acting on the centroid
channel, not from the fractional exponent. The exponent is what makes the deadline
finite at all: beta = 1 is equally formation-independent but never arrives.

(a) Centroid error -- who actually reaches the reference exactly.
(b) Formation error.
(c) Chattering and control effort under a 20 Hz zero-order hold.
(d) The claim that actually separates them. Prop. 1 makes the centroid arrival
    time depend on sigma(0) alone -- not on the formation, not on M. No baseline
    inherits that: run each law over four different formations from one initial
    condition and see how far the centroid settling time moves.
"""

import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style, theory
from finite_time.formations import SHAPES, static_schedule
from finite_time.integrate import simulate, simulate_zoh
from finite_time.model import rhs, rhs_per_agent_finite_time

T_END = 16.0
ZOH_HZ = 20.0
EPS_TOL = 0.05
PHI_BASELINE = 1e-6


def _variants():
    """(label, config, vector field, needs_boundary_layer)."""
    base = common.nominal_config()
    return [
        ("Eq. (8)", base, rhs, False),
        (r"$\beta=1$ (linear)", common.nominal_config(beta=1.0), rhs, False),
        ("$M=0$ (decoupled)", common.nominal_config(M=0.0), rhs, False),
        ("per-agent SMC", base, rhs_per_agent_finite_time, True),
    ]


def main() -> dict:
    style.use_paper_style()
    Z0 = common.initial_state()
    t_eval = np.linspace(0, T_END, 2001)
    summary = {}

    fig = plt.figure(figsize=(style.COL2_W, 4.4))
    gs = fig.add_gridspec(2, 6)
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_c = fig.add_subplot(gs[1, 0:3])
    ax_d = fig.add_subplot(gs[1, 3:6])

    rows = []
    for i, (name, cfg, field, needs_layer) in enumerate(_variants()):
        if not needs_layer:
            res = simulate(cfg, Z0, (0, T_END), t_eval=t_eval, field_fn=field)
        else:
            # The per-agent law puts a discontinuity on each of the n agents, so
            # there is no single surface to latch and the ideal law cannot be
            # integrated through its own arrival. Give it a narrow boundary layer
            # instead -- a generous treatment, since with phi = 1e-6 it still
            # converges to 1e-14 -- and let the zero-order-hold panels below show
            # what n non-smooth channels actually cost.
            res = simulate(cfg.with_(zeta_mode="sat", phi=PHI_BASELINE), Z0,
                           (0, T_END), t_eval=t_eval, latch=False,
                           rtol=1e-8, atol=1e-10, field_fn=field)
        sn = np.linalg.norm(metrics.centroid_error(t_eval, res.X, cfg), axis=1)
        dn = metrics.delta_norm(t_eval, res.X, cfg)
        mx = metrics.max_agent_error(t_eval, res.X, cfg)

        ax_a.semilogy(t_eval, np.maximum(sn, 1e-17), label=name,
                      **style.series_style(i))
        ax_b.semilogy(t_eval, np.maximum(dn, 1e-17), label=name,
                      **style.series_style(i))
        # Same laws under a zero-order hold, this time with the ideal sign(.)
        # term in every case, which is where the number of non-smooth channels
        # starts to matter. Explicit Euler has no trouble stepping through a
        # discontinuity, so all four are on identical footing here.
        zr = simulate_zoh(cfg, Z0, (0, T_END), 1.0 / ZOH_HZ, field_fn=field)
        rows.append({
            "name": name,
            "t_settle": metrics.settling_time(t_eval, mx, EPS_TOL),
            "sigma_floor": metrics.ultimate_value(t_eval, sn),
            "effort": metrics.control_effort(zr.t, zr.X),
            "chatter": metrics.chattering_index(zr.t, zr.X),
            "min_gap": float(metrics.min_pair_distance(res.X).min()),
        })

        # (d) the same law over four formations, from one initial condition.
        spread = []
        for shape_fn in SHAPES.values():
            P_fn, Pdot_fn = static_schedule(shape_fn(common.N, common.L))
            c2 = cfg.with_(P_fn=P_fn, Pdot_fn=Pdot_fn)
            if needs_layer:
                c2 = c2.with_(zeta_mode="sat", phi=PHI_BASELINE)
            r2 = simulate(c2, Z0, (0, T_END), t_eval=t_eval,
                          latch=not needs_layer and None, field_fn=field,
                          rtol=1e-8 if needs_layer else 1e-10,
                          atol=1e-10 if needs_layer else 1e-12)
            s2 = np.linalg.norm(metrics.centroid_error(t_eval, r2.X, c2), axis=1)
            spread.append(metrics.settling_time(t_eval, s2, 1e-3))
        rows[-1]["formation_spread"] = spread
        ax_d.plot(range(len(spread)), spread, marker=style.MARKERS[i], ms=3.2,
                  label=name, **style.series_style(i))

    ax_a.set_xlabel("time  $t$  [s]")
    ax_a.set_ylabel(r"$\Vert\sigma(t)\Vert$")
    ax_a.set_title("(a) centroid tracking", loc="left")
    ax_a.set_xlim(0, T_END)
    ax_a.set_ylim(3e-17, 5)
    ax_a.legend(loc="upper right", fontsize=5.6, ncol=2)

    ax_b.set_xlabel("time  $t$  [s]")
    ax_b.set_ylabel(r"$\Vert\mathbf{\delta}(t)\Vert$")
    ax_b.set_title("(b) formation error", loc="left")
    ax_b.set_xlim(0, T_END)
    ax_b.set_ylim(1e-14, 1e2)
    ax_b.legend(loc="upper right", fontsize=5.6, ncol=2)

    # ---- (c) cost under zero-order hold -----------------------------------
    names = [r["name"] for r in rows]
    chatter = np.array([r["chatter"] for r in rows])
    effort = np.array([r["effort"] for r in rows])
    xs = np.arange(len(rows))
    # Both quantities are normalised to Eq. (8), so they share one dimensionless
    # axis rather than needing a second scale.
    # Linear, not log: these are ratios clustered at 1, and the finding is that
    # they are all the same. A log axis would spend most of its height on empty
    # decades below the only value that differs.
    w = 0.36
    rc, re = chatter / chatter[0], effort / effort[0]
    ax_c.bar(xs - w / 2, rc, width=w - 0.04, color=style.SERIES[0],
             label="chattering index")
    ax_c.bar(xs + w / 2, re, width=w - 0.04, color=style.SERIES[1],
             label="control effort")
    ax_c.axhline(1.0, **style.bound_style(color=style.INK))
    ax_c.plot([], [], label="Eq. (8) baseline", **style.bound_style(color=style.INK))
    ax_c.set_xticks(xs, names, fontsize=5.4)
    ax_c.set_ylim(0, 1.45)
    ax_c.set_ylabel("relative to Eq. (8)")
    ax_c.set_title(f"(c) cost at {ZOH_HZ:g} Hz zero-order hold", loc="left")
    ax_c.legend(loc="upper right", fontsize=5.6, ncol=3)
    for x, v in zip(xs, rc):
        ax_c.text(x - w / 2, v + 0.03, "0" if v < 1e-3 else f"{v:.2f}",
                  ha="center", fontsize=5.2, color=style.INK_2)
    ax_c.text(0.02, 0.06, "no discontinuity $\\Rightarrow$ no chattering",
              transform=ax_c.transAxes, fontsize=5.4, color=style.MUTED)

    ax_d.set_xticks(range(len(SHAPES)), list(SHAPES.keys()), fontsize=5.8)
    ax_d.set_ylabel(r"time to $\Vert\sigma\Vert\leq10^{-3}$  [s]")
    ax_d.set_title("(d) does the formation change the centroid deadline?", loc="left")
    ax_d.legend(loc="center right", fontsize=5.6)
    spreads = {r["name"]: float(np.ptp(r["formation_spread"])) for r in rows}
    best = spreads["Eq. (8)"]
    worst_other = max(v for k, v in spreads.items() if k != "Eq. (8)")
    ax_d.text(0.03, 0.42, f"centroid-channel laws: spread $\\leq${max(v for k, v in spreads.items() if k != 'per-agent SMC'):.0e} s\n"
                          f"per-agent SMC: {spreads['per-agent SMC']:.2f} s",
              transform=ax_d.transAxes, ha="left", va="top", fontsize=5.8,
              color=style.INK_2)

    style.save(fig, "f7_baselines")

    print(f"    {'law':<22}{'t to 0.05':>10}{'|sig| floor':>13}{'effort':>9}"
          f"{'chatter':>10}{'min gap':>9}{'form. spread':>14}")
    for r in rows:
        ts = "never" if not np.isfinite(r["t_settle"]) else f"{r['t_settle']:.2f}s"
        sp = np.ptp(r["formation_spread"])
        print(f"    {r['name']:<22}{ts:>10}{r['sigma_floor']:>13.2e}"
              f"{r['effort']:>9.1f}{r['chatter']:>10.2f}{r['min_gap']:>9.3f}"
              f"{sp:>14.2e}")
    summary["rows"] = rows
    return summary


if __name__ == "__main__":
    main()
