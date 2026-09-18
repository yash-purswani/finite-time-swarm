"""F1 -- Proposition 1: finite-time centroid convergence and its control cost.

(a) ||sigma(t)|| for several beta against the predicted settling bound tau_c. The
    fractional exponents drive the error to zero at a finite instant; the linear
    baseline beta = 1 only decays asymptotically.
(b) The effort spent by the averaged controller to achieve that arrival. The
    agents are single integrators, so the command *is* the velocity and
    ||u_avg|| = ||cdot|| is the effort of the averaged loop. Averaging the model
    over i collapses it to

        cdot = -sigma + rdot - zeta(sigma),

    since the interaction term cancels pairwise and the pdot_i cancel under the
    centering condition -- so the averaged effort is a function of the centroid
    error alone, which is what makes (a) and (b) two views of one channel.

Panel (b) reports what that arrival costs. The finding is that it costs almost
nothing. Total corrective action is essentially independent of beta -- every
exponent spends within 0.5% of the same integral, and within 0.5% of the
geometric floor ||sigma(t_0)||, since the centroid must travel that distance
relative to the reference whatever law moves it. The *peak* command meanwhile
falls as beta falls, so the exponent that arrives first is also the gentlest on
the actuator.

That ordering is structural rather than incidental. For ||sigma|| > 1 the
correction |sigma_s|^beta is *smaller* than the linear sigma_s and shrinks
further as beta falls, so a fractional law is less aggressive than a linear one
during the initial transient; only once the error drops below unity does the
ordering invert and the fractional term become the stronger of the two. The law
therefore spends its authority near the origin, which is exactly where a linear
law runs out of gain and gives up converging. Note this peak ordering is tied to
||sigma(t_0)|| > 1, which holds here; for an initial error inside the unit ball
the fractional laws are the more aggressive ones from the outset.

Both panels carry the tau_c markers, which makes the pair readable together:
effort returns to the feedforward floor ||rdot|| exactly when the corresponding
curve in (a) reaches zero.
"""

import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style, theory
from finite_time.integrate import simulate
from finite_time.model import zeta

# Displayed floor for ||sigma||, equal to the arrival threshold LATCH_TOL of the
# integrator: below it a centroid error says nothing physical about a swarm, and
# each arrival reads as the vertical plunge off the bottom of the axis.
PLOT_FLOOR = 1e-8
CLIP = 1e-30           # far below the axis, so arrivals leave the frame cleanly
T_END = 8.0
# Every effort curve is on the feedforward floor by ~1.5 s; past that (b) is flat.
T_EFFORT = 2.0
BETAS = [0.2, 0.5, 0.8, 1.0]


def _clip(y):
    return np.maximum(y, CLIP)


def _avg_effort(t, res, cfg):
    """Averaged single-integrator command: total ||u_avg|| and its corrective part.

    Evaluated in closed form from sigma rather than by differencing c(t): the
    averaged dynamics are exact, so this carries no finite-difference error at
    the arrival instant, which is precisely where the interesting structure is.

    Once component s has reached the sliding surface the equivalent control holds
    sigma_s == 0 identically, and the command on that component is exactly the
    feedforward rdot_s. Reconstructing zeta from the round-off residual instead
    would report a spurious |1e-15|^beta ~ 1e-3 of effort forever after arrival.

    Returns ``(||u||, ||u - rdot||)``. The second is the corrective command
    -sigma - zeta(sigma) taken as a *vector* before norming; integrating it gives
    the arc length the centroid travels relative to the reference, which is
    bounded below by ||sigma(t_0)||. Norming each term first and subtracting
    would not respect that bound.
    """
    sig = metrics.centroid_error(t, res.X, cfg)                  # (T, d)
    rdot = np.array([cfg.rdot_fn(float(ti)) for ti in t])        # (T, d)
    z = zeta(sig, cfg.beta, cfg.zeta_mode, cfg.phi)              # (T, d)

    arrived = t[:, None] >= res.tau_s[None, :]                   # (T, d)
    sig = np.where(arrived, 0.0, sig)
    z = np.where(arrived, 0.0, z)

    corrective = -sig - z                                        # (T, d)
    return (np.linalg.norm(rdot + corrective, axis=1),
            np.linalg.norm(corrective, axis=1))


def main() -> dict:
    style.use_paper_style()
    Z0 = common.initial_state()
    t_eval = np.linspace(0, T_END, 4001)
    cfg_ref = common.nominal_config()
    sigma0 = Z0.reshape(common.N, common.D).mean(axis=0) - cfg_ref.r_fn(0.0)
    sigma0_norm = float(np.linalg.norm(sigma0))
    ff = float(np.linalg.norm(cfg_ref.rdot_fn(0.0)))    # feedforward floor ||rdot||

    fig, axes = plt.subplots(1, 2, figsize=(style.COL2_W, 2.35))
    ax_a, ax_b = axes
    summary = {}

    rows, efforts, lows = [], [], []
    for i, beta in enumerate(BETAS):
        cfg = common.nominal_config(beta=beta)
        res = simulate(cfg, Z0, (0, T_END), t_eval=t_eval)
        sn = np.linalg.norm(metrics.centroid_error(t_eval, res.X, cfg), axis=1)
        u, u_corr = _avg_effort(t_eval, res, cfg)

        pred = theory.tau_c(sigma0, beta)
        meas = res.tau_c_measured
        lbl = rf"$\beta={beta}$" if beta < 1 else r"$\beta=1$ (linear)"

        # Solid for the trajectory, dashed in the same hue for its own predicted
        # tau_c: the pairing is read off the colour, and dashed-vs-solid means
        # predicted-vs-measured rather than merely "another series".
        ax_a.semilogy(t_eval, _clip(sn), label=lbl, color=style.SERIES[i], lw=1.2)
        ax_b.plot(t_eval, u, label=lbl, color=style.SERIES[i], lw=1.2)
        if np.isfinite(pred):
            ax_a.axvline(pred, color=style.SERIES[i], linestyle=(0, (4.0, 2.0)),
                         linewidth=0.9, alpha=0.85, zorder=1.2)

        # Arc length of sigma relative to r: bounded below by ||sigma(t_0)||.
        excess = np.trapz(u_corr, t_eval)
        rows.append((beta, pred, meas))
        efforts.append((beta, float(u.max()), float(excess)))
        lows.append(float(u[t_eval <= T_EFFORT].min()))

    # ---- (a) finite-time arrival vs the predicted bound -------------------
    ax_a.plot([], [], label=r"predicted $\tau_c$", color=style.MUTED,
              linestyle=(0, (4.0, 2.0)), linewidth=0.9)
    ax_a.set_xlabel("time  $t$  [s]")
    ax_a.set_ylabel(r"$\Vert\sigma(t)\Vert$")
    ax_a.set_title(r"(a) finite-time arrival vs. bound $\tau_c$", loc="left")
    ax_a.set_xlim(0, T_END)
    ax_a.set_ylim(PLOT_FLOOR, 5)
    ax_a.legend(loc="upper right", ncol=1, fontsize=6.2)
    ok = all(m <= p * (1 + 1e-9) for _, p, m in rows if np.isfinite(p))
    summary["arrival"] = rows
    summary["arrival_ok"] = ok

    # ---- (b) effort of the averaged controller ----------------------------
    # The feedforward floor is chrome, not data: every law must pay ||rdot|| just
    # to keep station on a moving reference, so it is the baseline against which
    # the corrective effort should be read.
    ax_b.axhline(ff, **style.bound_style(linewidth=0.9))
    ax_b.plot([], [], label=r"feedforward $\Vert\dot r\Vert$",
              **style.bound_style(linewidth=0.9))
    ax_b.set_xlabel("time  $t$  [s]")
    ax_b.set_ylabel(r"$\Vert u_{\mathrm{avg}}(t)\Vert=\Vert\dot c(t)\Vert$")
    ax_b.set_title(r"(b) averaged control effort", loc="left")
    ax_b.set_xlim(0, T_EFFORT)
    # Fit the axis to the curves; anchoring it at 0 would spend half the panel
    # on empty space. The command dips below ||rdot|| while the corrector opposes
    # the reference velocity, so the feedforward level is not the axis floor.
    top = max(p for _, p, _ in efforts)
    ax_b.set_ylim(min(lows) - 0.08, top + 0.12)
    ax_b.legend(loc="upper right", ncol=1, fontsize=6.2)
    summary["effort"] = efforts
    summary["feedforward"] = ff

    style.save(fig, "f1_centroid")

    print(f"    settling bound holds for every beta: {ok}")
    for (beta, pred, meas), (_, pk, ex) in zip(rows, efforts):
        p = "inf" if not np.isfinite(pred) else f"{pred:.4f}"
        m = "never" if not np.isfinite(meas) else f"{meas:.4f}"
        print(f"      beta={beta}: tau_c pred={p:>8} meas={m:>8} | "
              f"peak effort={pk:.4f}  corrective action={ex:.4f}")
    print(f"    feedforward floor ||rdot|| = {ff:.4f}")
    print(f"    ||sigma(t_0)|| = {sigma0_norm:.4f}  (lower bound on corrective action)")
    return summary


if __name__ == "__main__":
    main()
