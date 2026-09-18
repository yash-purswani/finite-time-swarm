"""F2 -- Proposition 2: global exponential formation convergence.

(a) ||delta(t)|| (solid) against its own predicted envelope (dashed, same hue)
    for a wide range of M: rate 1 on [t0, tau_c], then the improved rate
    1 + kappa_0 restarted at every instant per Remark 1. The bound is anchored at
    t0, not at tau_c: by Lemma 1 the sliding-mode term is dissipative for V_delta
    whatever the centroid is doing, so Prop. 2 needs no centroid hypothesis and
    the envelope covers the whole trajectory.
(b) Measured rate over the (nu, eps) grid, each cell reporting the measured rate
    over the predicted one.

Two further checks are computed and printed but not plotted, since they add no
shape a reader cannot already read off (a): the rate against lambda_min(M) over
eight decades of coupling including anisotropic and rotated M, and globality over
100 initial conditions spanning five decades of ||delta(0)||.
"""

import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style, theory
from finite_time.integrate import simulate

T_END = 18.0
FIT_FLOOR = 1e-7        # below this ||delta|| is round-off, not dynamics
# The rate checks measure the *asymptotic* rate, so the fit starts at the centroid
# settling time tau_c. That is a theoretical anchor, not a cosmetic one: the
# improved rate 1 + kappa_0 of Prop. 2(ii) is only claimed on [tau_c, inf), where
# sigma == 0 removes the -n sum_s |sigma_s|^(1+beta) term and sum_i delta_i = 0
# makes the coupling term act on the full ||delta||. The rate-1 bound, by
# contrast, holds from t0 and is what panel (a) shows.
FIT_START = None        # set in main() to tau_c
M_CURVES = [0.0, 10.0, 100.0]
LAMBDAS = [0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]
NUS = [1, 2, 3]
EPSILONS = [0.01, 0.1, 1.0]
N_GLOBAL = 100


def _run(cfg, Z0, t_eval, **kw):
    return simulate(cfg, Z0, (0, t_eval[-1]), t_eval=t_eval, **kw)


def _rate(t, dn):
    return metrics.empirical_rate(t, dn, lo=FIT_START, hi=T_END, floor=FIT_FLOOR)


def _predicted(lam, nu, eps, D_p):
    """Prop. 2(ii) rate with the Remark 1 late-time diameter bound D -> D_p."""
    return theory.formation_rate(lam, D_p, nu, eps)


def main() -> dict:
    global FIT_START
    style.use_paper_style()
    Z0 = common.initial_state()
    t_eval = np.linspace(0, T_END, 3001)
    summary = {}

    # The predicted rate needs the formation's own diameter D_p and the fit needs
    # tau_c; both come from the nominal scenario, so neither is a free knob.
    cfg_ref = common.nominal_config()
    D_p = theory.formation_diameter(cfg_ref.P_fn(0.0))
    sigma0 = Z0.reshape(common.N, common.D).mean(axis=0) - cfg_ref.r_fn(0.0)
    tau_c = theory.tau_c(sigma0, cfg_ref.beta)
    FIT_START = tau_c
    summary["D_p"] = D_p
    summary["tau_c"] = tau_c

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(style.COL2_W, 2.6))

    # ---- (a) per-coupling envelope ----------------------------------------
    # Every run here shares Z0 and the same formation, so ||delta(0)|| is identical
    # across M and one normalisation serves all of them. Each M gets its own
    # envelope: the rate-1 branch is only what M = 0 is entitled to, and drawing it
    # for M = 100I hides exactly the improvement Prop. 2(ii) is about.
    worst_ratio = 0.0
    env_ratios = {}
    for i, M in enumerate(M_CURVES):
        cfg = common.nominal_config(M=M)
        res = _run(cfg, Z0, t_eval)
        dn = metrics.delta_norm(t_eval, res.X, cfg)
        env = theory.delta_envelope_coupled(t_eval, dn[0], 0.0, tau_c,
                                            cfg.lambda_min_M, D_p,
                                            cfg.nu, cfg.eps)
        ratio = float((dn / env).max())
        env_ratios[M] = ratio
        worst_ratio = max(worst_ratio, ratio)
        ax_a.semilogy(t_eval, np.maximum(dn / dn[0], 1e-14),
                      color=style.SERIES[i], label=rf"$M={M:g}I$")
        ax_a.semilogy(t_eval, np.maximum(env / dn[0], 1e-14),
                      **style.bound_style(color=style.SERIES[i], linewidth=0.9))

    ax_a.axvline(tau_c, **style.marker_line_style())
    ax_a.text(tau_c + 0.25, 0.9, r"$\tau_c$", transform=ax_a.get_xaxis_transform(),
              fontsize=6, color=style.MUTED, va="top")
    ax_a.plot([], [], label="predicted envelope",
              **style.bound_style(color=style.INK_2, linewidth=0.9))
    ax_a.set_xlabel(r"$t-t_0$  [s]")
    ax_a.set_ylabel(r"$\Vert\mathbf{\delta}(t)\Vert\,/\,\Vert\mathbf{\delta}(t_0)\Vert$")
    ax_a.set_title(r"(a) envelope: rate $1$, then $1+\kappa_0$", loc="left")
    ax_a.set_xlim(0, T_END)
    ax_a.set_ylim(1e-11, 3)
    ax_a.legend(loc="upper right", fontsize=5.6)
    style.annotate_pass(ax_a,
                        rf"never exceeds its own envelope (max {worst_ratio:.3f})",
                        ok=worst_ratio <= 1.0, loc="lower left")
    summary["envelope_ratio"] = worst_ratio
    summary["envelope_ratio_by_M"] = env_ratios

    # ---- (b) rate over the (nu, eps) grid ---------------------------------
    Rg = np.zeros((len(NUS), len(EPSILONS)))
    Pg = np.zeros_like(Rg)
    for i, nu in enumerate(NUS):
        for j, eps in enumerate(EPSILONS):
            cfg = common.nominal_config(M=common.M_NOMINAL, nu=nu, eps=eps)
            res = _run(cfg, Z0, t_eval)
            dn = metrics.delta_norm(t_eval, res.X, cfg)
            Rg[i, j] = _rate(t_eval, dn)
            Pg[i, j] = _predicted(cfg.lambda_min_M, nu, eps, D_p)

    im = ax_b.imshow(Rg, cmap="Blues", vmin=1.0, vmax=max(1.25, Rg.max()),
                     aspect="auto", origin="lower")
    ax_b.set_xticks(range(len(EPSILONS)), [f"{e:g}" for e in EPSILONS])
    ax_b.set_yticks(range(len(NUS)), [str(v) for v in NUS])
    ax_b.set_xlabel(r"$\varepsilon$")
    ax_b.set_ylabel(r"$\nu$")
    ax_b.grid(False)
    for i in range(len(NUS)):
        for j in range(len(EPSILONS)):
            shade = "white" if Rg[i, j] > 1.0 + 0.6 * (Rg.max() - 1.0) else style.INK
            # measured over predicted, so a reader can check the inequality cell
            # by cell rather than trusting the colour.
            ax_b.text(j, i, f"{Rg[i, j]:.2f}", ha="center", va="bottom",
                      fontsize=6, color=shade)
            ax_b.text(j, i, f"({Pg[i, j]:.2f})", ha="center", va="top",
                      fontsize=5.2, color=shade, alpha=0.85)
    cb = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03)
    cb.set_label("measured decay rate", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)
    ax_b.set_title(r"(b) rate over $(\nu,\varepsilon)$:  measured / (predicted)",
                   loc="left")
    summary["grid_min_rate"] = float(Rg.min())
    summary["grid_min_margin"] = float((Rg / Pg).min())

    style.save(fig, "f2_formation")

    # ---- unplotted check 1: rate vs lambda_min(M) --------------------------
    rates = []
    for lam in LAMBDAS:
        cfg = common.nominal_config(M=lam)
        res = _run(cfg, Z0, t_eval)
        dn = metrics.delta_norm(t_eval, res.X, cfg)
        rates.append(_rate(t_eval, dn))

    # Anisotropic and rotated M, same lambda_min, to show only the spectrum matters.
    extras = []
    th = np.deg2rad(37.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for name, M in [("diag(1, 20)", np.diag([1.0, 20.0])),
                    ("rot. diag(1, 20)", R @ np.diag([1.0, 20.0]) @ R.T),
                    ("diag(5, 5)", np.diag([5.0, 5.0]))]:
        cfg = common.nominal_config().with_(M=M)
        res = _run(cfg, Z0, t_eval)
        dn = metrics.delta_norm(t_eval, res.X, cfg)
        extras.append((name, cfg.lambda_min_M, _rate(t_eval, dn)))

    pred_iso = np.atleast_1d(_predicted(np.array(LAMBDAS, dtype=float),
                                        common.NU, common.EPS, D_p))
    pred_extra = np.atleast_1d(_predicted(np.array([l for _, l, _ in extras]),
                                          common.NU, common.EPS, D_p))
    meas_all = np.array(rates + [r for _, _, r in extras], dtype=float)
    pred_all = np.concatenate([pred_iso, pred_extra])
    margin = float(np.nanmin(meas_all / pred_all))
    min_rate = float(np.nanmin(meas_all))
    # The two rotations of diag(1, 20) share lambda_min, so the spread between
    # their measured rates is what "only the spectrum matters" costs numerically.
    aniso = [r for name, _, r in extras if "20)" in name]
    summary["min_rate"] = min_rate
    summary["min_rate_margin"] = margin
    summary["aniso_spread"] = float(max(aniso) - min(aniso))
    summary["extras"] = [(name, lam, r, float(p))
                         for (name, lam, r), p in zip(extras, pred_extra)]

    # ---- unplotted check 2: globality --------------------------------------
    n, d = common.N, common.D
    t_g = np.linspace(0, 14.0, 1401)
    rng = np.random.default_rng(11)
    worst = 0.0
    cfg_g = common.nominal_config()
    for k in range(N_GLOBAL):
        if k == 0:
            X0 = np.tile(np.array([0.0, 0.0]), (n, 1))         # all collocated
        else:
            scale = 10.0 ** rng.uniform(-2.0, 3.0)             # 1e-2 .. 1e3
            X0 = scale * rng.standard_normal((n, d))
        res = _run(cfg_g, X0.ravel(), t_g)
        dn = metrics.delta_norm(t_g, res.X, cfg_g)
        worst = max(worst, float((dn / (dn[0] * np.exp(-t_g))).max()))
    summary["global_worst_ratio"] = worst

    print(f"    D_p = {D_p:.3f}, tau_c = {tau_c:.3f} (fit window [{tau_c:.2f}, {T_END:g}])")
    print(f"    (a) worst ||delta||/envelope over M sweep : {worst_ratio:.4f}")
    for M, rr in env_ratios.items():
        print(f"          M={M:6.1f}I  max ||delta||/envelope = {rr:.4f}")
    print(f"    (b) min rate over (nu, eps) grid         : {Rg.min():.4f}")
    print(f"        min measured/predicted over grid     : {(Rg / Pg).min():.4f}")
    print(f"    [not plotted] min measured decay rate    : {min_rate:.4f}")
    print(f"        min measured/predicted rate          : {margin:.4f}")
    for lam, r, pr in zip(LAMBDAS, rates, pred_iso):
        print(f"          lambda_min={lam:6.2f}  measured={r:.4f}  predicted={pr:.4f}")
    for name, lam, r, pr in summary["extras"]:
        print(f"          {name:<18} lambda_min={lam:6.2f}  "
              f"measured={r:.4f}  predicted={pr:.4f}")
    print(f"    [not plotted] worst ratio over {N_GLOBAL} ICs   : {worst:.4f}")
    return summary


if __name__ == "__main__":
    main()
