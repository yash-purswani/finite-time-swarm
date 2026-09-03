"""F2 -- Proposition 2: global exponential formation convergence.

(a) ||delta(t)|| against the Eq. (26) envelope for a wide range of M. The bound
    is anchored at t0, not at tau_c: by Lemma 1 the sliding-mode term is
    dissipative for V_delta whatever the centroid is doing, so Prop. 2 needs no
    centroid hypothesis and the envelope covers the whole trajectory.
(b) Measured decay rate vs lambda_min(M), including anisotropic and rotated M --
    the guaranteed rate of 1 is never violated.
(c) Measured rate over the (nu, eps) grid.
(d) Globality: 100 initial conditions spanning six orders of magnitude in
    ||delta(0)||, including exactly collocated agents.
"""

import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style, theory
from finite_time.integrate import simulate

T_END = 18.0
FIT_FLOOR = 1e-7        # below this ||delta|| is round-off, not dynamics
# The rate panels measure the *asymptotic* rate, so the fit skips the opening
# transient. That window is a numerical choice, not a theoretical anchor: the
# Eq. (26) bound itself holds from t0. Early on the discarded
# -n sum_s |sigma_s|^(1+beta) term is largest, so the measured rate there is if
# anything faster than in the tail.
FIT_START = 2.5
M_CURVES = [0.0, 10.0, 100.0]
LAMBDAS = [0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]
NUS = [1, 2, 3]
EPSILONS = [0.01, 0.1, 1.0]
N_GLOBAL = 100


def _run(cfg, Z0, t_eval, **kw):
    return simulate(cfg, Z0, (0, t_eval[-1]), t_eval=t_eval, **kw)


def _rate(t, dn):
    return metrics.empirical_rate(t, dn, lo=FIT_START, hi=T_END, floor=FIT_FLOOR)


def main() -> dict:
    style.use_paper_style()
    Z0 = common.initial_state()
    t_eval = np.linspace(0, T_END, 3001)
    summary = {}

    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
        2, 2, figsize=(style.COL2_W, 5.6))

    # ---- (a) envelope ------------------------------------------------------
    # Every run here shares Z0 and the same formation, so ||delta(0)|| is identical
    # across M and one normalised envelope serves all of them. Anchoring at t0
    # rather than tau_c also shows the curves over their entire lifetime,
    # transient included -- which is the regime Prop. 2 now covers.
    worst_ratio = 0.0
    for i, M in enumerate(M_CURVES):
        cfg = common.nominal_config(M=M)
        res = _run(cfg, Z0, t_eval)
        dn = metrics.delta_norm(t_eval, res.X, cfg)
        env = theory.delta_envelope(t_eval, dn[0], 0.0)
        worst_ratio = max(worst_ratio, float((dn / env).max()))
        ax_a.semilogy(t_eval, np.maximum(dn / dn[0], 1e-14),
                      color=style.SERIES[i], label=rf"$M={M:g}I$")

    ax_a.semilogy(t_eval, np.exp(-t_eval), **style.bound_style(color=style.INK))
    ax_a.plot([], [], label=r"envelope $e^{-(t-t_0)}$",
              **style.bound_style(color=style.INK))
    ax_a.set_xlabel(r"$t-t_0$  [s]")
    ax_a.set_ylabel(r"$\Vert\mathbf{\delta}(t)\Vert\,/\,\Vert\mathbf{\delta}(t_0)\Vert$")
    ax_a.set_title("(a) exponential envelope", loc="left")
    ax_a.set_xlim(0, T_END)
    ax_a.set_ylim(1e-11, 3)
    ax_a.legend(loc="upper right", fontsize=5.6)
    style.annotate_pass(ax_a, rf"never exceeds envelope (max {worst_ratio:.3f})",
                        ok=worst_ratio <= 1.0, loc="lower left")
    summary["envelope_ratio"] = worst_ratio

    # ---- (b) rate vs lambda_min(M) ----------------------------------------
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

    ax_b.semilogx(np.maximum(LAMBDAS, 3e-2), rates, marker="o", ms=3,
                  color=style.SERIES[0], dashes=style.DASHES[0],
                  label="isotropic $M=\\lambda I$")
    ax_b.scatter([max(l, 3e-2) for _, l, _ in extras], [r for _, _, r in extras],
                 s=16, marker="^", facecolor="none",
                 edgecolor=style.SERIES[1], linewidths=0.9, zorder=3,
                 label="anisotropic / rotated")
    ax_b.axhline(1.0, **style.bound_style())
    ax_b.plot([], [], label="guaranteed rate $=1$", **style.bound_style())
    ax_b.set_xlabel(r"$\lambda_{\min}(M)$   (leftmost point: $M=0$)")
    ax_b.set_ylabel("measured decay rate")
    ax_b.set_title("(b) rate vs. coupling", loc="left")
    ax_b.set_ylim(0.88, None)
    ax_b.legend(loc="upper left", fontsize=5.6)
    min_rate = float(np.nanmin(rates + [r for _, _, r in extras]))
    ax_b.text(0.97, 0.24, f"min measured rate = {min_rate:.3f}",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=6,
              color=style.GOOD if min_rate >= 1.0 - 1e-3 else style.CRITICAL)
    summary["min_rate"] = min_rate
    summary["extras"] = extras

    # ---- (c) rate over the (nu, eps) grid ---------------------------------
    Rg = np.zeros((len(NUS), len(EPSILONS)))
    for i, nu in enumerate(NUS):
        for j, eps in enumerate(EPSILONS):
            cfg = common.nominal_config(M=common.M_NOMINAL, nu=nu, eps=eps)
            res = _run(cfg, Z0, t_eval)
            dn = metrics.delta_norm(t_eval, res.X, cfg)
            Rg[i, j] = _rate(t_eval, dn)

    im = ax_c.imshow(Rg, cmap="Blues", vmin=1.0, vmax=max(1.25, Rg.max()),
                     aspect="auto", origin="lower")
    ax_c.set_xticks(range(len(EPSILONS)), [f"{e:g}" for e in EPSILONS])
    ax_c.set_yticks(range(len(NUS)), [str(v) for v in NUS])
    ax_c.set_xlabel(r"$\varepsilon$")
    ax_c.set_ylabel(r"$\nu$")
    ax_c.set_title(r"(c) rate over $(\nu,\varepsilon)$", loc="left")
    ax_c.grid(False)
    for i in range(len(NUS)):
        for j in range(len(EPSILONS)):
            shade = "white" if Rg[i, j] > 1.0 + 0.6 * (Rg.max() - 1.0) else style.INK
            ax_c.text(j, i, f"{Rg[i, j]:.2f}", ha="center", va="center",
                      fontsize=6, color=shade)
    cb = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03)
    cb.set_label("decay rate", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)
    summary["grid_min_rate"] = float(Rg.min())

    # ---- (d) globality ------------------------------------------------------
    # Normalised by ||delta(0)|| and plotted from t0, so all 100 runs collapse onto
    # the single envelope e^{-(t-t0)} regardless of how far out they start.
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
        norm = np.maximum(dn / dn[0], 1e-12)
        worst = max(worst, float((norm / np.exp(-t_g)).max()))
        ax_d.semilogy(t_g, norm, color=style.SERIES[0], lw=0.35, alpha=0.16)

    ax_d.semilogy(t_g, np.exp(-t_g),
                  **style.bound_style(color=style.INK, linewidth=1.1))
    ax_d.plot([], [], label=r"envelope $e^{-(t-t_0)}$",
              **style.bound_style(color=style.INK, linewidth=1.1))
    ax_d.plot([], [], color=style.SERIES[0], lw=1.0,
              label=rf"{N_GLOBAL} initial conditions")
    ax_d.set_xlabel(r"$t-t_0$  [s]")
    ax_d.set_ylabel(r"$\Vert\mathbf{\delta}(t)\Vert\,/\,\Vert\mathbf{\delta}(t_0)\Vert$")
    ax_d.set_title(r"(d) globality: $\Vert\mathbf{\delta}(0)\Vert$ over 5 decades",
                   loc="left")
    ax_d.set_xlim(0, 14)
    ax_d.set_ylim(1e-9, 3)
    ax_d.legend(loc="upper right", fontsize=5.8)
    style.annotate_pass(ax_d, f"max ratio to envelope = {worst:.3f}",
                        ok=worst <= 1.0, loc="lower left")
    summary["global_worst_ratio"] = worst

    style.save(fig, "f2_formation")

    print(f"    (a) worst ||delta||/envelope over M sweep : {worst_ratio:.4f}")
    print(f"    (b) min measured decay rate              : {min_rate:.4f}")
    for name, lam, r in extras:
        print(f"          {name:<18} lambda_min={lam:6.2f}  rate={r:.4f}")
    print(f"    (c) min rate over (nu, eps) grid         : {Rg.min():.4f}")
    print(f"    (d) worst ratio over {N_GLOBAL} ICs              : {worst:.4f}")
    return summary


if __name__ == "__main__":
    main()
