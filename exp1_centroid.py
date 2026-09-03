"""F1 -- Proposition 1: finite-time centroid convergence.

(a) |sigma(t)| for several beta against the Eq. (25) settling-time bound. The
    fractional exponents drive the error to machine zero at a finite instant; the
    linear baseline beta = 1 only decays asymptotically.
(b) The same |sigma(t)| under six different (M, nu, eps, shape, L) settings. The
    curves coincide, which is the "irrespective of p_i and M" clause of Prop. 1.
"""

import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style, theory
from finite_time.formations import grid, hexagon, line, static_schedule, vee
from finite_time.integrate import simulate

FLOOR = 1e-16          # plotting floor: below this the values are pure round-off
T_END = 8.0
BETAS = [0.2, 0.5, 0.8, 1.0]

# (M, nu, eps, shape, L) -- deliberately spread over orders of magnitude
VARIANTS = [
    (0.0, 2, 1.00, "hexagon", 5.0),
    (5.0, 2, 1.00, "hexagon", 5.0),
    (100.0, 2, 1.00, "line", 10.0),
    (5.0, 1, 0.01, "vee", 2.0),
    (50.0, 3, 0.10, "grid", 8.0),
    (10.0, 2, 1.00, "grid", 0.5),
]
SHAPES = {"hexagon": hexagon, "line": line, "vee": vee, "grid": grid}


def _clip(y):
    return np.maximum(y, FLOOR)


def main() -> dict:
    style.use_paper_style()
    Z0 = common.initial_state()
    t_eval = np.linspace(0, T_END, 4001)
    sigma0 = Z0.reshape(common.N, common.D).mean(axis=0) - np.array([0.0, 0.0])

    fig, axes = plt.subplots(1, 2, figsize=(style.COL2_W, 2.35))
    summary = {}

    # ---- (a) finite-time arrival vs the Eq. (25) bound --------------------
    ax = axes[0]
    rows = []
    for i, beta in enumerate(BETAS):
        cfg = common.nominal_config(beta=beta)
        res = simulate(cfg, Z0, (0, T_END), t_eval=t_eval)
        sn = np.linalg.norm(metrics.centroid_error(t_eval, res.X, cfg), axis=1)

        pred = theory.tau_c(sigma0, beta)
        meas = res.tau_c_measured
        lbl = (rf"$\beta={beta}$" if beta < 1
               else r"$\beta=1$ (linear)")
        # Solid for the trajectory, dashed in the same hue for its own predicted
        # tau_c: the pairing is then read off the colour, and dashed-vs-solid means
        # predicted-vs-measured rather than merely "another series".
        ax.semilogy(t_eval, _clip(sn), label=lbl, color=style.SERIES[i], lw=1.2)
        if np.isfinite(pred):
            ax.axvline(pred, color=style.SERIES[i], linestyle=(0, (4.0, 2.0)),
                       linewidth=0.9, alpha=0.85, zorder=1.2)
        rows.append((beta, pred, meas))

    # Describe the vertical markers in the legend rather than as floating text,
    # which would collide with the round-off floor.
    ax.plot([], [], label=r"predicted $\tau_c$ (Eq. 25)", color=style.MUTED,
            linestyle=(0, (4.0, 2.0)), linewidth=0.9)

    ax.set_xlabel("time  $t$  [s]")
    ax.set_ylabel(r"$\Vert\sigma(t)\Vert$")
    ax.set_title(r"(a) finite-time arrival vs. bound $\tau_c$", loc="left")
    ax.set_xlim(0, T_END)
    ax.set_ylim(3e-17, 5)
    # One column, hard against the right edge: the beta = 1 curve passes below it
    # there, so the box no longer sits on top of the series it is labelling.
    ax.legend(loc="upper right", ncol=1, fontsize=6.2)
    ok = all(m <= p * (1 + 1e-9) for _, p, m in rows if np.isfinite(p))
    summary["arrival"] = rows
    summary["arrival_ok"] = ok

    # ---- (b) invariance to M, nu, eps and the formation -------------------
    ax = axes[1]
    curves = []
    for M, nu, eps, shape, L in VARIANTS:
        P = SHAPES[shape](common.N, L)
        P_fn, Pdot_fn = static_schedule(P)
        cfg = common.nominal_config(M=M, nu=nu, eps=eps).with_(
            P_fn=P_fn, Pdot_fn=Pdot_fn)
        res = simulate(cfg, Z0, (0, T_END), t_eval=t_eval)
        curves.append(np.linalg.norm(metrics.centroid_error(t_eval, res.X, cfg), axis=1))

    C = np.array(curves)
    for k, (M, nu, eps, shape, L) in enumerate(VARIANTS):
        ax.semilogy(t_eval, _clip(C[k]),
                    color=style.SERIES[0] if k == 0 else style.SERIES[k % 4],
                    lw=2.4 if k == 0 else 0.9,
                    alpha=0.30 if k == 0 else 0.95,
                    dashes=style.DASHES[k % 4],
                    label=(rf"$M={M:g},\ \nu={nu},\ \varepsilon={eps:g}$, {shape}"))

    ax.set_xlabel("time  $t$  [s]")
    ax.set_ylabel(r"$\Vert\sigma(t)\Vert$")
    ax.set_title(r"(b) invariance to $M,\nu,\varepsilon$ and $p_i$", loc="left")
    ax.set_xlim(0, 4)
    ax.set_ylim(3e-17, 5)
    ax.legend(loc="upper right", fontsize=5.2)

    # The six curves are the evidence; the inset that used to quantify their
    # separation is gone, so the number it carried is stamped on the panel instead.
    max_spread = float(np.abs(C - C[0]).max())
    ax.text(0.40, 0.42, f"max spread between\nthe six: {max_spread:.1e}",
            transform=ax.transAxes, ha="left", va="center", fontsize=6,
            color=style.GOOD if max_spread < 1e-8 else style.CRITICAL)
    summary["invariance_spread"] = max_spread

    style.save(fig, "f1_centroid")

    print(f"    Eq. (25) bound holds for every beta: {ok}")
    for beta, pred, meas in rows:
        p = "inf" if not np.isfinite(pred) else f"{pred:.4f}"
        m = "never" if not np.isfinite(meas) else f"{meas:.4f}"
        print(f"      beta={beta}: predicted tau_c={p:>8}  measured={m:>8}")
    print(f"    invariance: max spread across 6 settings = {max_spread:.2e}")
    return summary


if __name__ == "__main__":
    main()
