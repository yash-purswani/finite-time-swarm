"""F2 -- Proposition 2 and Corollary 1: exponential formation convergence.

(a) The per-agent error max_i ||delta_i(t)|| against the rate-1 envelope of
    Prop. 2 and the acceptance deadlines T_dock of Cor. 1 for two tolerances.
    Both are anchored at t_0, not at tau_c: by Lemma 1 the sliding-mode term is
    dissipative for V_delta whatever the centroid is doing, so neither statement
    waits for the centroid to settle.
(b) The same envelope against four settings of (M, nu, eps) spanning four decades
    of coupling. Unlike the centroid channel, the formation *trajectory* does
    depend on the interaction -- only the bound does not. What (b) checks is
    therefore the claim Prop. 2 actually makes: whatever the coupling does to the
    path, it never lifts it above the rate-1 envelope.

Runs the same swarm as F1, so the two figures report one experiment.
"""

import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style, theory
from finite_time.formations import hexagon, rotating_schedule
from finite_time.integrate import simulate

T_END = 12.0
TOLS = [0.1, 0.01]
OMEGA = 0.5          # rad/s, rotation rate of the time-varying formation
# (M, nu, eps) -- deliberately spread over orders of magnitude. The formation is
# held fixed so every run shares ||delta(t_0)|| and one envelope serves all four.
# Four, not five: the palette carries four distinguishable series, and a fifth
# would reuse a hue and dash pattern. These four span M in {0..100}, nu in {1,2,3}
# and eps in {0.01..1}; the nominal (5, 2, 1) is already the subject of panel (a).
VARIANTS = [
    (0.0, 2, 1.00),
    (100.0, 2, 1.00),
    (5.0, 1, 0.01),
    (50.0, 3, 0.10),
]


def main() -> dict:
    style.use_paper_style()
    Z0 = common.initial_state()
    t_eval = np.linspace(0, T_END, 3001)
    summary = {}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(style.COL2_W, 2.35))

    # ---- (a) envelope and acceptance deadline -----------------------------
    cfg = common.nominal_config()
    res = simulate(cfg, Z0, (0, T_END), t_eval=t_eval)
    dn = metrics.delta_norm(t_eval, res.X, cfg)
    mx = metrics.max_agent_error(t_eval, res.X, cfg)
    d_0 = float(dn[0])
    env = theory.delta_envelope(t_eval, d_0, 0.0)

    # A rigidly rotating copy of the same hexagon, which drives the pdot_i
    # feedforward that static_schedule leaves at zero. Prop. 2 claims the bound is
    # independent of the offsets p_i(.), so a moving formation must sit under the
    # same envelope as the frozen one -- and shares it, since rotation preserves
    # ||delta(t_0)||.
    P_fn, Pdot_fn = rotating_schedule(hexagon(common.N, common.L), OMEGA)
    cfg_rot = common.nominal_config().with_(P_fn=P_fn, Pdot_fn=Pdot_fn)
    res_rot = simulate(cfg_rot, Z0, (0, T_END), t_eval=t_eval)
    dn_rot = metrics.delta_norm(t_eval, res_rot.X, cfg_rot)

    ax_a.semilogy(t_eval, dn, color=style.SERIES[0], lw=1.2,
                  label=r"$\Vert\delta(t)\Vert$, static")
    ax_a.semilogy(t_eval, dn_rot, color=style.SERIES[1], lw=1.0,
                  dashes=style.DASHES[1],
                  label=rf"$\Vert\delta(t)\Vert$, rotating $\omega={OMEGA:g}$")
    ax_a.semilogy(t_eval, mx, color=style.SERIES[2], lw=1.0,
                  dashes=style.DASHES[2], label=r"$\max_i\Vert\delta_i(t)\Vert$")
    ax_a.semilogy(t_eval, env, **style.bound_style(linewidth=0.9))
    ax_a.plot([], [], label=r"envelope $\Vert\delta(t_0)\Vert e^{-(t-t_0)}$",
              **style.bound_style(linewidth=0.9))

    dock_ok, docks = True, {}
    for k, tol in enumerate(TOLS):
        Td = theory.T_dock(d_0, tol, 0.0)
        ok = bool(mx[t_eval >= Td].max() <= tol)
        dock_ok &= ok
        docks[tol] = (Td, float(t_eval[mx <= tol][0]), ok)
        # Tolerances and deadlines are chrome, not data: muted ink, so the palette
        # stays available for measured series.
        ax_a.axhline(tol, **style.bound_style(color=style.MUTED, linewidth=0.7,
                                              linestyle=(0, (1.5, 1.5))))
        ax_a.axvline(Td, **style.marker_line_style())
        ax_a.annotate(rf"$\varepsilon_{{\mathrm{{tol}}}}={tol:g}$", xy=(Td, tol),
                      fontsize=5.6, color=style.INK_2,
                      xytext=(2.5, 3.0), textcoords="offset points")

    rot_ratio = float((dn_rot / env).max())
    summary["rotating_ratio"] = rot_ratio

    ax_a.set_xlabel("time  $t$  [s]")
    ax_a.set_ylabel(r"formation error")
    ax_a.set_title(r"(a) envelope and acceptance time $T_{\mathrm{dock}}$", loc="left")
    ax_a.set_xlim(0, T_END)
    ax_a.set_ylim(1e-3, 60)
    ax_a.legend(loc="upper right", fontsize=5.8)
    style.annotate_pass(ax_a, f"every agent inside $\\varepsilon_{{tol}}$ by "
                              f"$T_{{dock}}$: {dock_ok}", ok=dock_ok, loc="lower left")
    summary["delta_0"] = d_0
    summary["dock"] = docks

    # ---- (b) the bound is independent of the coupling ---------------------
    worst = 0.0
    for i, (M, nu, eps) in enumerate(VARIANTS):
        cfg_v = common.nominal_config(M=M, nu=nu, eps=eps)
        res_v = simulate(cfg_v, Z0, (0, T_END), t_eval=t_eval)
        dv = metrics.delta_norm(t_eval, res_v.X, cfg_v)
        worst = max(worst, float((dv / env).max()))
        ax_b.semilogy(t_eval, dv, color=style.SERIES[i % 4],
                      dashes=style.DASHES[i % 4], lw=1.0,
                      label=rf"$M={M:g}I,\ \nu={nu},\ \varepsilon={eps:g}$")

    ax_b.semilogy(t_eval, env, **style.bound_style(linewidth=1.0))
    ax_b.plot([], [], label="rate-1 envelope", **style.bound_style(linewidth=1.0))
    ax_b.set_xlabel("time  $t$  [s]")
    ax_b.set_ylabel(r"$\Vert\delta(t)\Vert$")
    ax_b.set_title(r"(b) envelope holds for every $(M,\nu,\varepsilon)$", loc="left")
    ax_b.set_xlim(0, T_END)
    ax_b.set_ylim(1e-3, 60)
    ax_b.legend(loc="upper right", fontsize=5.4)
    style.annotate_pass(ax_b, f"max $\\Vert\\delta\\Vert$/envelope = {worst:.3f}",
                        ok=worst <= 1.0 + 1e-9, loc="lower left")
    summary["envelope_ratio"] = worst

    style.save(fig, "f2_docking")

    print(f"    ||delta(t_0)|| = {d_0:.4f}")
    print(f"    rotating formation: max ||delta||/envelope = {rot_ratio:.4f}")
    for tol, (Td, meas, ok) in docks.items():
        print(f"      eps_tol={tol:<5g} T_dock={Td:.4f}  measured entry={meas:.4f}  "
              f"margin={Td - meas:.4f} ({100 * (Td - meas) / Td:.1f}%)  ok={ok}")
    print(f"    (b) worst ||delta||/envelope over 4 couplings: {worst:.4f}")
    return summary


if __name__ == "__main__":
    main()
