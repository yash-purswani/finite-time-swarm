"""F3 -- Corollaries 1 and 2: deterministic docking and entrance times.

(a) Nominal: every agent is inside its acceptance radius by the predicted
    T_dock (Eq. 35) and stays there.
(b) Perturbed: the collective error enters the eps_tol-neighbourhood of the ISS
    ball by the predicted T_enter (Eq. 52).
(c) Predicted vs achieved over 100 initial conditions and two tolerances.

Both times are anchored at t0 rather than at tau_c. By Lemma 1 the sliding-mode
term is dissipative for V_delta regardless of the centroid, so neither corollary
needs to wait for centroid convergence -- and in the perturbed case (b) tau_c does
not exist at all, since sigma settles into the Prop. 4 ball instead of reaching
the origin.
"""

import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style, theory
from finite_time.integrate import simulate

T_END = 26.0
TOLS = [0.1, 0.01]
WBAR = 0.05
N_MC = 100


def main() -> dict:
    style.use_paper_style()
    Z0 = common.initial_state()
    t_eval = np.linspace(0, T_END, 3001)
    summary = {}

    fig, axes = plt.subplots(1, 3, figsize=(style.COL2_W, 2.35))

    # ---- (a) Corollary 1, nominal -----------------------------------------
    ax = axes[0]
    cfg = common.nominal_config()
    res = simulate(cfg, Z0, (0, T_END), t_eval=t_eval)
    dn = metrics.delta_norm(t_eval, res.X, cfg)
    mx = metrics.max_agent_error(t_eval, res.X, cfg)
    d_0 = float(dn[0])

    ax.semilogy(t_eval, np.maximum(mx, 1e-14), label=r"$\max_i\Vert\delta_i(t)\Vert$",
                **style.series_style(0))
    dock_ok = True
    for k, tol in enumerate(TOLS):
        Td = theory.T_dock(d_0, tol, 0.0)
        ax.axhline(tol, **style.bound_style(color=style.SERIES[k + 1], linewidth=0.8))
        ax.axvline(Td, **style.marker_line_style(color=style.SERIES[k + 1]))
        after = t_eval >= Td
        ok = bool(mx[after].max() <= tol)
        dock_ok &= ok
        ax.annotate(rf"$\varepsilon_{{tol}}={tol:g}$", xy=(Td, tol), fontsize=5.8,
                    color=style.SERIES[k + 1], xytext=(2.0, 3.0),
                    textcoords="offset points")
        summary[f"T_dock_{tol}"] = (Td, ok)

    ax.plot([], [], label=r"$\varepsilon_{tol}$ / predicted $T_{dock}$",
            **style.bound_style())
    ax.set_xlabel("time  $t$  [s]")
    ax.set_ylabel(r"$\max_i\Vert\delta_i(t)\Vert$")
    ax.set_title("(a) Cor. 1: docking (nominal)", loc="left")
    ax.set_xlim(0, T_END)
    ax.set_ylim(1e-11, 1e2)
    ax.legend(loc="upper right", fontsize=5.8)
    style.annotate_pass(ax, "no violation after $T_{dock}$" if dock_ok
                        else "VIOLATED", ok=dock_ok, loc="lower left")

    # ---- (b) Corollary 2, perturbed ---------------------------------------
    ax = axes[1]
    cfg_w = common.nominal_config().with_(
        w_fn=common.w_random(WBAR, common.N, common.D, seed=3))
    res_w = simulate(cfg_w, Z0, (0, T_END), t_eval=t_eval)
    dn_w = metrics.delta_norm(t_eval, res_w.X, cfg_w)
    # delta(0) is disturbance-independent, so this is the same anchor as (a).
    d_0_w = float(dn_w[0])
    ball = theory.B_iss(common.N, WBAR)

    ax.semilogy(t_eval, dn_w, label=r"$\Vert\mathbf{\delta}(t)\Vert$",
                **style.series_style(0))
    ax.axhline(ball, **style.bound_style(color=style.INK))
    ax.plot([], [], label=r"$\sqrt{n}\,\bar{w}$", **style.bound_style(color=style.INK))

    enter_ok = True
    for k, tol in enumerate(TOLS):
        Te = theory.T_enter(d_0_w, tol, 0.0, common.N, WBAR)
        ax.axhline(ball + tol, **style.bound_style(color=style.SERIES[k + 1],
                                                   linewidth=0.7))
        ax.axvline(Te, **style.marker_line_style(color=style.SERIES[k + 1]))
        after = t_eval >= Te
        ok = bool(dn_w[after].max() <= ball + tol)
        enter_ok &= ok
        summary[f"T_enter_{tol}"] = (Te, ok)

    ax.plot([], [], label=r"$\sqrt{n}\,\bar{w}+\varepsilon_{tol}$ / $T_{enter}$",
            **style.bound_style())
    ax.set_xlabel("time  $t$  [s]")
    ax.set_ylabel(r"$\Vert\mathbf{\delta}(t)\Vert$")
    ax.set_title(rf"(b) Cor. 2: entrance ($\bar{{w}}={WBAR:g}$)", loc="left")
    ax.set_xlim(0, T_END)
    ax.set_ylim(1e-3, 1e2)
    ax.legend(loc="upper right", fontsize=5.8)
    style.annotate_pass(ax, "entered by $T_{enter}$" if enter_ok else "VIOLATED",
                        ok=enter_ok, loc="lower left")

    # ---- (c) predicted vs achieved over many initial conditions -----------
    ax = axes[2]
    rng = np.random.default_rng(23)
    pts = {tol: ([], []) for tol in TOLS}
    for k in range(N_MC):
        scale = 10.0 ** rng.uniform(-1.0, 2.0)
        X0 = scale * rng.standard_normal((common.N, common.D))
        r = simulate(cfg, X0.ravel(), (0, T_END), t_eval=t_eval)
        m = metrics.max_agent_error(t_eval, r.X, cfg)
        dnk = metrics.delta_norm(t_eval, r.X, cfg)
        for tol in TOLS:
            pred = theory.T_dock(float(dnk[0]), tol, 0.0)
            achieved = metrics.settling_time(t_eval, m, tol)
            pts[tol][0].append(pred)
            pts[tol][1].append(achieved)

    all_pred, all_ach = [], []
    for k, tol in enumerate(TOLS):
        p, a = np.array(pts[tol][0]), np.array(pts[tol][1])
        all_pred.append(p)
        all_ach.append(a)
        ax.scatter(p, a, s=7, facecolor="none", edgecolor=style.SERIES[k],
                   linewidths=0.6, label=rf"$\varepsilon_{{tol}}={tol:g}$")
    lim = [0, float(np.concatenate(all_pred).max()) * 1.06]
    ax.plot(lim, lim, **style.bound_style(color=style.INK))
    ax.plot([], [], label="achieved $=$ predicted", **style.bound_style(color=style.INK))
    ax.set_xlim(lim)
    ax.set_ylim(0, lim[1])
    ax.set_xlabel(r"predicted $T_{dock}$  [s]")
    ax.set_ylabel(r"achieved  [s]")
    ax.set_title(rf"(c) {N_MC} initial conditions", loc="left")
    ax.legend(loc="upper left", fontsize=5.8)

    P, A = np.concatenate(all_pred), np.concatenate(all_ach)
    finite = np.isfinite(A)
    mc_ok = bool((A[finite] <= P[finite] + 1e-6).all())
    ratio = float(np.median(A[finite] / np.maximum(P[finite], 1e-12)))
    style.annotate_pass(ax, f"all below the line; median ratio {ratio:.2f}",
                        ok=mc_ok, loc="lower right")
    summary["mc_ok"] = mc_ok
    summary["mc_median_ratio"] = ratio

    style.save(fig, "f3_docking")

    print(f"    (a) Cor. 1 holds for both tolerances : {dock_ok}")
    for tol in TOLS:
        Td, ok = summary[f"T_dock_{tol}"]
        print(f"          eps_tol={tol:<6g} T_dock={Td:7.3f} s  respected={ok}")
    print(f"    (b) Cor. 2 holds for both tolerances : {enter_ok}")
    for tol in TOLS:
        Te, ok = summary[f"T_enter_{tol}"]
        print(f"          eps_tol={tol:<6g} T_enter={Te:7.3f} s  respected={ok}")
    print(f"    (c) {N_MC} ICs, all achieved <= predicted : {mc_ok}")
    print(f"          median achieved/predicted        : {ratio:.3f}")
    return summary


if __name__ == "__main__":
    main()
