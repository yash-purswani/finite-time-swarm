"""F4 -- Proposition 3 and the B3 centroid patch, under bounded disturbance.

(a) ||delta(t)|| under four disturbance classes against the Eq. (39) envelope,
    anchored at t0 so it covers the entire trajectory rather than a post-tau_c
    tail. tau_c does not exist here: under disturbance sigma settles into the
    Prop. 4 ball instead of reaching the origin.
(b) Ultimate ||delta|| vs wbar. The sqrt(n) wbar radius is tight only for the
    adversarially aligned disturbance the Cauchy-Schwarz step of Prop. 3 assumes.
(c) B3: the centroid floor scales as wbar^(1/beta), not wbar -- super-linear
    attenuation on the channel that carries the mission.

Panels are lettered (a)-(c) in the figure but keep their original ax_a / ax_c /
ax_e names in the code, so the summary keys and the printed diagnostics still line
up with the earlier five-panel revision.

Prop. 3 no longer carries a "zeta == 0 for t >= tau_c" assumption: by Lemma 1
(Eq. 12-13) that term equals -n sum_s |sigma_s|^(1+beta) <= 0, so retaining it
only makes V_delta decrease faster. That is what lets the ISS bound be stated from
t0, which matters because tau_c is undefined in this regime. Proposition 4 supplies
what the centroid channel actually does instead.
"""

import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style, theory
from finite_time.integrate import simulate

T_END = 25.0
WBAR = 0.05
HOLD = 0.2          # resample interval of the piecewise-constant random class
# Disturbed runs need far less solver precision than nominal ones: the quantities
# of interest sit at 1e-3..1e-1, not at the round-off floor, and there is no
# sliding surface to resolve because sigma settles into the B3 ball instead of
# reaching zero. rtol 1e-8 reproduces the rtol 1e-10 answers to five digits at
# roughly one thirtieth of the cost.
RTOL_W = 1e-8
ATOL_W = 1e-10
# Disturbed runs cannot latch the sliding surface: under disturbance sigma does
# not stay at zero once it arrives. But for the classes whose *mean* disturbance
# nearly cancels -- i.i.d. directions average to O(wbar/sqrt(n)), and the
# adversarial one can cancel outright -- sigma still reaches zero, and the ideal
# sign(.) term then chatters at infinite frequency and stalls the solver. A
# boundary layer of width 1e-7 makes the field Lipschitz there. Everything these
# panels measure lives at 1e-3 or above, four decades clear of it.
PHI_W = 1e-7
WBARS = np.logspace(-3.0, -0.7, 6)
# Panel (c) keeps the exact sign(.) term, so its sweep has to stay where the
# predicted floor wbar^(1/beta) is actually resolvable. beta = 0.3 at wbar = 1e-3
# predicts 1e-10 -- indistinguishable from the surface itself, and the solver
# grinds there. beta in {0.5, 0.7} over wbar in [1e-2, 0.2] puts every floor
# between 1e-4 and 0.1, three clean decades, still far below the linear reference.
BETAS = [0.5, 0.7]
WBARS_E = np.logspace(-2.0, -0.7, 6)


def _classes(wbar, cfg, n=common.N, d=common.D):
    """The four disturbance shapes, all satisfying ||w_i(t)|| <= wbar."""
    return [
        ("common mode", common.w_common(wbar, d)),
        ("i.i.d. on sphere", common.w_random(wbar, n, d, seed=5, hold=HOLD)),
        ("sinusoidal", common.w_sine(wbar, n, d)),
        ("adversarial", common.w_adversarial(wbar, cfg.P_fn, cfg.r_fn)),
    ]


def _run(cfg, Z0, t_eval, rtol=RTOL_W, layer=True):
    if layer:
        cfg = cfg.with_(zeta_mode="sat", phi=PHI_W)
    return simulate(cfg, Z0, (0, t_eval[-1]), t_eval=t_eval,
                    rtol=rtol, atol=rtol * 1e-2)


def main() -> dict:
    style.use_paper_style()
    Z0 = common.initial_state()
    t_eval = np.linspace(0, T_END, 2001)
    summary = {}

    # Three panels, not five: (a) spans the top because it is a time series and
    # needs the aspect, the two log-log sweeps sit side by side beneath it. With
    # the crowding gone the panels get their own breathing room, so the pads are
    # opened up from the package defaults.
    fig = plt.figure(figsize=(style.COL2_W, 4.4), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.10, hspace=0.06, wspace=0.06)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])

    base = common.nominal_config()
    ball = theory.B_iss(common.N, WBAR)

    # ---- (a) four disturbance classes -------------------------------------
    # delta(0) does not depend on the disturbance, so a single Eq. (39) envelope
    # anchored at t0 serves all four classes.
    ults = {}
    for i, (name, w) in enumerate(_classes(WBAR, base)):
        cfg = base.with_(w_fn=w)
        res = _run(cfg, Z0, t_eval)
        dn = metrics.delta_norm(t_eval, res.X, cfg)
        ax_a.semilogy(t_eval, dn, label=name, **style.series_style(i))
        ults[name] = metrics.ultimate_value(t_eval, dn)

    d_0 = float(np.linalg.norm(
        (Z0.reshape(common.N, common.D) - base.P_fn(0.0)
         - np.asarray(base.r_fn(0.0))).ravel()))
    env = theory.iss_envelope(t_eval, d_0, 0.0, common.N, WBAR)
    ax_a.semilogy(t_eval, env, **style.bound_style())
    ax_a.axhline(ball, **style.bound_style(color=style.INK, linewidth=0.9))
    ax_a.plot([], [], label="Eq. (39) envelope", **style.bound_style())
    ax_a.plot([], [], label=r"$\mathcal{B}_{ISS}$: $\sqrt{n}\,\bar{w}$",
              **style.bound_style(color=style.INK, linewidth=0.9))
    ax_a.set_xlabel("time  $t$  [s]")
    ax_a.set_ylabel(r"$\Vert\mathbf{\delta}(t)\Vert$")
    ax_a.set_title(rf"(a) ISS, $\bar{{w}}={WBAR:g}$", loc="left")
    ax_a.set_xlim(0, T_END)
    ax_a.set_ylim(1e-3, 1e2)
    ax_a.legend(loc="upper right", fontsize=6.2, ncol=2)
    worst_env = max(ults.values()) / ball
    # The envelope claim itself, checked over the whole run rather than just the tail.
    env_ratio = 0.0
    for name, w in _classes(WBAR, base):
        r_e = _run(base.with_(w_fn=w), Z0, t_eval)
        env_ratio = max(env_ratio, float(
            (metrics.delta_norm(t_eval, r_e.X, base) / env).max()))
    style.annotate_pass(ax_a, f"worst ultimate / ball = {worst_env:.2f}; "
                              f"envelope ratio {env_ratio:.2f}",
                        ok=worst_env <= 1.0 and env_ratio <= 1.0, loc="lower left")
    summary["classes"] = ults
    summary["env_ratio_from_t0"] = env_ratio

    # ---- (b) tightness vs wbar --------------------------------------------
    t_c = np.linspace(0, 20.0, 1201)
    curves_c = {}
    for i, (label, maker) in enumerate([
            ("adversarial", lambda wb: common.w_adversarial(wb, base.P_fn, base.r_fn)),
            ("i.i.d. on sphere", lambda wb: common.w_random(wb, common.N, common.D,
                                                            seed=5, hold=HOLD))]):
        vals = []
        for wb in WBARS:
            cfg = base.with_(w_fn=maker(wb))
            res = _run(cfg, Z0, t_c)
            vals.append(metrics.ultimate_value(t_c, metrics.delta_norm(t_c, res.X, cfg)))
        curves_c[label] = np.array(vals)
        ax_c.loglog(WBARS, vals, marker=style.MARKERS[i], ms=2.8,
                    **style.series_style(i), label=label)

    ax_c.loglog(WBARS, theory.B_iss(common.N, WBARS), **style.bound_style(color=style.INK))
    ax_c.plot([], [], label=r"$\sqrt{n}\,\bar{w}$", **style.bound_style(color=style.INK))
    ax_c.set_xlabel(r"$\bar{w}$")
    ax_c.set_ylabel(r"ultimate $\Vert\mathbf{\delta}\Vert$")
    ax_c.set_title(r"(b) tightness in $\bar{w}$", loc="left")
    ax_c.legend(loc="upper left", fontsize=6.2)
    adv_ratio = float((curves_c["adversarial"] / theory.B_iss(common.N, WBARS)).max())
    rnd_ratio = float((curves_c["i.i.d. on sphere"] /
                       theory.B_iss(common.N, WBARS)).max())
    style.annotate_pass(ax_c, f"adversarial {adv_ratio:.2f}, random {rnd_ratio:.2f}",
                        ok=adv_ratio <= 1.0, loc="lower right")
    summary["adv_ratio"] = adv_ratio
    summary["rnd_ratio"] = rnd_ratio

    # ---- (c) B3: centroid floor scales as wbar^(1/beta) --------------------
    t_e = np.linspace(0, 20.0, 1201)
    for i, beta in enumerate(BETAS):
        floors = []
        for wb in WBARS_E:
            cfg = common.nominal_config(beta=beta).with_(
                w_fn=common.w_common(wb, common.D))
            # No boundary layer here: this panel measures the centroid floor, and
            # a constant common-mode push holds sigma at wbar^(1/beta) > 0, so the
            # exact sign(.) term never chatters.
            res = _run(cfg, Z0, t_e, layer=False)
            sn = np.abs(metrics.centroid_error(t_e, res.X, cfg)).max(axis=1)
            floors.append(metrics.ultimate_value(t_e, sn))
        floors = np.array(floors)
        # w_common spreads the push over the d coordinates, so the mean
        # disturbance seen by each scalar channel is wbar/sqrt(d). Predicting
        # against that rather than against wbar makes this a tight check of the
        # exponent instead of a loose one.
        pred = np.array([theory.sigma_ball(wb / np.sqrt(common.D), beta)
                         for wb in WBARS_E])
        ax_e.loglog(WBARS_E, floors, marker=style.MARKERS[i], ms=2.8,
                    **style.series_style(i), label=rf"$\beta={beta}$ measured")
        ax_e.loglog(WBARS_E, pred, **style.bound_style(color=style.SERIES[i]))
        summary[f"b3_beta{beta}"] = float((floors / pred).max())
        summary[f"b3_slope{beta}"] = float(
            np.polyfit(np.log(WBARS_E), np.log(floors), 1)[0])

    ax_e.loglog(WBARS_E, WBARS_E, **style.bound_style(color=style.MUTED))
    ax_e.plot([], [], label=r"$\bar{w}^{1/\beta}$ (B3)", **style.bound_style())
    ax_e.plot([], [], label=r"$\bar{w}$ (linear ref.)",
              **style.bound_style(color=style.MUTED))
    ax_e.set_xlabel(r"$\bar{w}$")
    ax_e.set_ylabel(r"ultimate $\max_s|\sigma_s|$")
    ax_e.set_title(r"(c) B3: centroid floor", loc="left")
    ax_e.legend(loc="lower right", fontsize=5.8)
    b3_ok = all(summary[f"b3_beta{b}"] <= 1.0 for b in BETAS)
    slopes = ", ".join(f"{summary[f'b3_slope{b}']:.2f}/{1/b:.2f}" for b in BETAS)
    ax_e.text(0.03, 0.97, f"slope meas./pred.:\n{slopes}", transform=ax_e.transAxes,
              ha="left", va="top", fontsize=5.8,
              color=style.GOOD if b3_ok else style.CRITICAL)

    style.save(fig, "f4_iss")

    print(f"    (a) Eq.(39) envelope from t0, worst ratio : {env_ratio:.4f}")
    print(f"    (a) ultimate ||delta|| vs ball {ball:.4f}:")
    for k, v in ults.items():
        print(f"          {k:<18} {v:.4f}   ({v/ball:.3f} of the ball)")
    print(f"    (b) worst ultimate/ball  adversarial={adv_ratio:.3f}  "
          f"random={rnd_ratio:.3f}")
    for b in BETAS:
        print(f"    (c) beta={b}: worst measured/predicted floor = "
              f"{summary[f'b3_beta{b}']:.3f}, log-log slope "
              f"{summary[f'b3_slope{b}']:.3f} vs 1/beta = {1/b:.3f}")
    return summary


if __name__ == "__main__":
    main()
