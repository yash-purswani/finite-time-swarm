"""F6 -- scale, dimension, and what happens on real hardware.

(a) tau_c is independent of n and d. Every run is given the same sigma(0), so
    Eq. (25) predicts one number for all of them.
(b) The exponential rate stays in a narrow band and never drops below the
    guaranteed value of 1. Unlike tau_c it is not exactly invariant -- more
    neighbours means a little more dissipation from the coupling term -- but the
    variation is a few percent across a fiftyfold change in n.
(c) Zero-order-hold implementation at rates a ROS node would actually use. The
    sign(.) term chatters in a band around the sliding surface.
(d) The chatter band scales as (dt/2)^(1/(1-beta)) -- the amplitude at which one
    Euler step overshoots the surface -- and the sat(sigma/phi) boundary layer
    removes it.
(e) Actuator saturation ||xdot_i|| <= v_max. The reference itself moves at unit
    speed, so v_max must exceed 1 for any tracking at all; above that, tau_c
    degrades smoothly.
"""

import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style, theory
from finite_time.integrate import simulate, simulate_zoh
from finite_time.model import rhs

NS = [4, 10, 20, 50, 100, 200]
DIMS = [2, 3]
SIGMA0 = {2: np.array([0.5, -1.5]), 3: np.array([0.5, -1.5, 0.8])}
RATES_HZ = [100.0, 50.0, 20.0, 10.0]
DTS = np.array([1 / 200, 1 / 100, 1 / 50, 1 / 20, 1 / 10, 1 / 5])
PHI = 1e-2
VMAXES = [1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
T_END = 16.0


def _state_with_sigma0(n: int, d: int, sigma0, seed: int = 42):
    """Random positions shifted so the initial centroid error is exactly sigma0."""
    rng = np.random.default_rng(seed)
    X = 5.0 * rng.standard_normal((n, d))
    X += np.asarray(sigma0, dtype=float) - X.mean(axis=0)   # r(0) = 0
    return X.ravel()


def main() -> dict:
    style.use_paper_style()
    summary = {}

    fig = plt.figure(figsize=(style.COL2_W, 4.8))
    gs = fig.add_gridspec(2, 6)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    # ---- (a)+(b) independence of n and d ----------------------------------
    t_eval = np.linspace(0, T_END, 1601)
    taus, rates = {}, {}
    for k, d in enumerate(DIMS):
        taus[d], rates[d] = [], []
        for n in NS:
            cfg = common.nominal_config(n=n, d=d)
            Z0 = _state_with_sigma0(n, d, SIGMA0[d])
            res = simulate(cfg, Z0, (0, T_END), t_eval=t_eval)
            dn = metrics.delta_norm(t_eval, res.X, cfg)
            taus[d].append(res.tau_c_measured)
            rates[d].append(metrics.empirical_rate(
                t_eval, dn, lo=res.tau_c_measured + 0.5, hi=T_END, floor=1e-7))

        ax_a.semilogx(NS, taus[d], marker=style.MARKERS[k], ms=3,
                      label=rf"$d={d}$", **style.series_style(k))
        ax_b.semilogx(NS, rates[d], marker=style.MARKERS[k], ms=3,
                      label=rf"$d={d}$", **style.series_style(k))

    pred_tau = theory.tau_c(SIGMA0[2], common.BETA)
    ax_a.axhline(pred_tau, **style.bound_style(color=style.INK))
    ax_a.plot([], [], label=r"predicted $\tau_c$", **style.bound_style(color=style.INK))
    ax_a.set_xlabel("$n$")
    ax_a.set_ylabel(r"$\tau_c$  [s]")
    ax_a.set_title("(a) $\\tau_c$ vs. $n$, $d$", loc="left")
    spread_tau = float(np.ptp(np.concatenate([taus[d] for d in DIMS])))
    lo = min(np.concatenate([taus[d] for d in DIMS]))
    ax_a.set_ylim(lo - 0.25, pred_tau + 0.25)
    ax_a.legend(loc="lower left", fontsize=5.8, ncol=3)
    # The residual spread is the width of the latch threshold, not an n- or
    # d-dependence: every run arrives at the same tau_c to five decimals.
    style.annotate_pass(ax_a, f"spread over $n,d$: {spread_tau:.1e} s",
                        ok=spread_tau < 1e-3, loc="upper left")

    ax_b.axhline(1.0, **style.bound_style(color=style.INK))
    ax_b.plot([], [], label="guaranteed $=1$", **style.bound_style(color=style.INK))
    ax_b.set_xlabel("$n$")
    ax_b.set_ylabel("measured decay rate")
    ax_b.set_title("(b) rate vs. $n$, $d$", loc="left")
    all_rates = np.concatenate([rates[d] for d in DIMS])
    min_rate, max_rate = float(np.nanmin(all_rates)), float(np.nanmax(all_rates))
    ax_b.set_ylim(0.94, max_rate + 0.10)
    ax_b.legend(loc="upper left", fontsize=5.8, ncol=3)
    style.annotate_pass(ax_b, f"all rates in [{min_rate:.3f}, {max_rate:.3f}]",
                        ok=min_rate >= 1.0 - 1e-3, loc="lower right")
    summary["tau_spread"] = spread_tau
    summary["rate_min"] = min_rate

    # ---- (c) zero-order-hold at realistic rates ---------------------------
    cfg = common.nominal_config()
    Z0 = _state_with_sigma0(common.N, common.D, SIGMA0[2])
    bands = {}
    for k, hz in enumerate(RATES_HZ):
        dt = 1.0 / hz
        res = simulate_zoh(cfg, Z0, (0, T_END), dt)
        sn = np.linalg.norm(metrics.centroid_error(res.t, res.X, cfg), axis=1)
        ax_c.semilogy(res.t, np.maximum(sn, 1e-17), label=f"{hz:g} Hz",
                      **style.series_style(k))
        bands[hz] = metrics.ultimate_value(res.t, sn)

    ax_c.set_xlabel("time  $t$  [s]")
    ax_c.set_ylabel(r"$\Vert\sigma(t)\Vert$")
    ax_c.set_title("(c) zero-order hold", loc="left")
    ax_c.set_xlim(0, T_END)
    ax_c.set_ylim(1e-8, 5)
    ax_c.legend(loc="upper right", fontsize=5.8, ncol=2)
    summary["zoh_bands"] = bands

    # ---- (d) chatter band vs step, and the boundary-layer fix -------------
    sign_band, sat_band = [], []
    for dt in DTS:
        r1 = simulate_zoh(cfg, Z0, (0, T_END), dt)
        s1 = np.linalg.norm(metrics.centroid_error(r1.t, r1.X, cfg), axis=1)
        sign_band.append(metrics.ultimate_value(r1.t, s1))

        r2 = simulate_zoh(cfg.with_(zeta_mode="sat", phi=PHI), Z0, (0, T_END), dt)
        s2 = np.linalg.norm(metrics.centroid_error(r2.t, r2.X, cfg), axis=1)
        sat_band.append(metrics.ultimate_value(r2.t, s2))

    # One explicit Euler step overshoots the surface once dt|sigma|^beta > 2|sigma|,
    # i.e. below |sigma| = (dt/2)^(1/(1-beta)). That is the width of the band the
    # discrete-time trajectory ends up rattling around in.
    pred_band = (DTS / 2.0) ** (1.0 / (1.0 - common.BETA))

    ax_d.loglog(DTS, sign_band, marker="o", ms=3, label=r"$\mathrm{sign}(\sigma)$",
                **style.series_style(0))
    ax_d.loglog(DTS, np.maximum(sat_band, 1e-17), marker="s", ms=3,
                label=rf"$\mathrm{{sat}}(\sigma/\phi)$, $\phi={PHI:g}$",
                **style.series_style(1))
    ax_d.loglog(DTS, pred_band, **style.bound_style(color=style.INK))
    ax_d.plot([], [], label=r"$(\Delta t/2)^{1/(1-\beta)}$",
              **style.bound_style(color=style.INK))
    for hz in RATES_HZ:
        ax_d.axvline(1.0 / hz, **style.marker_line_style(alpha=0.5))
    ax_d.set_xlabel(r"sample period $\Delta t$  [s]")
    ax_d.set_ylabel(r"residual $\Vert\sigma\Vert$")
    ax_d.set_title(r"(d) chatter band and the boundary-layer fix", loc="left")
    ax_d.legend(loc="upper left", fontsize=5.8)
    fit = np.polyfit(np.log(DTS), np.log(sign_band), 1)[0]
    ax_d.text(0.97, 0.62, f"measured slope {fit:.2f}\nvs predicted "
                          f"{1/(1-common.BETA):.2f}", transform=ax_d.transAxes,
              ha="right", va="top", fontsize=6, color=style.GOOD)
    # The boundary layer removes the band entirely, but it has its own explicit
    # stability limit: inside the layer the effective gain is 1 + phi^(beta-1),
    # so an Euler step is only stable while dt < 2 / (1 + phi^(beta-1)).
    dt_sat = 2.0 / (1.0 + PHI ** (common.BETA - 1.0))
    ax_d.axvline(dt_sat, **style.marker_line_style(color=style.CRITICAL))
    ax_d.plot([], [], label=r"$\Delta t = 2/(1+\phi^{\beta-1})$",
              **style.marker_line_style(color=style.CRITICAL))
    ax_d.legend(loc="upper left", fontsize=5.4)
    summary["dt_sat_limit"] = float(dt_sat)
    summary["chatter_slope"] = float(fit)
    summary["sat_band"] = sat_band

    # ---- (e) actuator saturation -------------------------------------------
    t_e = np.linspace(0, 30.0, 3001)
    taus_v, unsat = [], None
    for vm in VMAXES:
        res = simulate(cfg.with_(v_max=vm), Z0, (0, 30.0), t_eval=t_e)
        taus_v.append(res.tau_c_measured)
    res = simulate(cfg, Z0, (0, 30.0), t_eval=t_e)
    unsat = res.tau_c_measured

    ax_e.semilogx(VMAXES, taus_v, marker="o", ms=3, label=r"saturated",
                  **style.series_style(0))
    ax_e.axhline(unsat, **style.bound_style(color=style.INK))
    ax_e.plot([], [], label=r"unsaturated $\tau_c$",
              **style.bound_style(color=style.INK))
    ax_e.axvline(1.0, **style.marker_line_style(color=style.CRITICAL))
    ax_e.plot([], [], label=r"$v_{max}=\Vert\dot{r}\Vert=1$: tracking impossible",
              **style.marker_line_style(color=style.CRITICAL))
    ax_e.set_xlabel(r"$v_{max}$")
    ax_e.set_ylabel(r"$\tau_c$  [s]")
    ax_e.set_title("(e) actuator saturation", loc="left")
    lost = [v for v, t in zip(VMAXES, taus_v) if not np.isfinite(t)]
    if lost:
        ax_e.text(0.03, 0.55, "$v_{max}=" + f"{lost[0]:g}$: no arrival\nwithin the horizon",
                  transform=ax_e.transAxes, ha="left", va="top", fontsize=5.8,
                  color=style.CRITICAL)
    ax_e.legend(loc="upper right", fontsize=5.6)
    summary["tau_vs_vmax"] = list(zip(VMAXES, taus_v))
    summary["tau_unsat"] = unsat

    style.save(fig, "f6_scale")

    print(f"    (a) tau_c across n={NS[0]}..{NS[-1]} and d=2,3 : "
          f"spread {spread_tau:.2e} s (predicted {pred_tau:.4f} s)")
    print(f"    (b) min decay rate across all n, d           : {min_rate:.4f}")
    print("    (c) ZOH residual ||sigma||:")
    for hz, v in bands.items():
        print(f"          {hz:5g} Hz -> {v:.3e}")
    print(f"    (d) chatter band slope: measured {fit:.3f}, "
          f"predicted {1/(1-common.BETA):.3f}")
    print(f"        boundary layer at the same steps: "
          f"{min(sat_band):.2e} .. {max(sat_band):.2e}")
    print(f"    (e) tau_c unsaturated {unsat:.3f} s; "
          + ", ".join(f"v_max={v:g}->{t:.3f}s" for v, t in zip(VMAXES, taus_v)))
    return summary


if __name__ == "__main__":
    main()
