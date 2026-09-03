"""F5 -- time-varying formations, as an animation.

The static five-panel figure has been replaced by a movie, and the movie is the
only output: the morph is a *temporal* claim, and nothing about it survives a
still. ``figures/f5_morphing.mp4`` plays the swarm through hexagon -> line -> V ->
grid while it tracks r(t), with the two error channels drawn live alongside:

  * left, in a camera that follows r(t): agent trails, the current formation slots
    p_i(t) + r(t), and the reference itself;
  * right, top: ||sigma(t)|| -- the centroid never notices a morph. Because every
    shape is centred and the blend between two centred shapes is affine,
    sum_i pdot_i(t) = 0 holds *throughout* the morph, not merely at its endpoints,
    so reshaping is invisible to the centroid channel;
  * right, bottom: ||delta(t)|| -- exponential re-convergence, undisturbed by any
    morph, because the pdot_i feedforward of Eq. (8) makes reshaping free.

Morph windows are shaded in both traces.

Two quantities a movie cannot show are measured and printed instead: the peak
||sigma|| after tau_c across all three morphs, and the peak ||delta|| during a
single morph as a function of its duration, with and without the feedforward.

The two necessity ablations live in ``verify.py`` now that they have no panel:
``check_centering_condition`` and ``check_centering_offset`` for Eq. (4), and the
feedforward ablation in the printed sweep below.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style
from finite_time.formations import grid, hexagon, line, morph_schedule, vee
from finite_time.integrate import simulate

L = 4.0
MORPH_TIME = 4.0
SWITCH = [8.0, 16.0, 24.0]
T_END = 34.0
NAMES = ["hexagon", "line", "V", "grid"]

FPS = 25
SPEEDUP = 2.0           # 34 s of simulation in 17 s of video
TRAIL = 2.5             # seconds of trail behind each agent
CAM_HALF = 9.0          # half-width of the window that follows r(t)


def _schedule():
    shapes = [hexagon(common.N, L), line(common.N, L),
              vee(common.N, L), grid(common.N, L)]
    return morph_schedule(shapes, SWITCH, MORPH_TIME)


def _shade(ax):
    for ts in SWITCH:
        ax.axvspan(ts, ts + MORPH_TIME, color=style.GRID, alpha=0.9, lw=0, zorder=0)


def _phase_name(t: float) -> str:
    """Which shape the swarm is in, or which morph it is midway through."""
    for k, ts in enumerate(SWITCH):
        if t < ts:
            return NAMES[k]
        if t < ts + MORPH_TIME:
            return f"{NAMES[k]} $\\to$ {NAMES[k + 1]}"
    return NAMES[-1]


def _animate(t_eval, X, sn, dn, tau, P_fn, r_fn):
    """Render the morph movie. Returns the path written."""
    fig = plt.figure(figsize=(style.COL2_W, 3.4), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08, w_pad=0.10, hspace=0.05, wspace=0.06)
    gs = fig.add_gridspec(2, 5)
    ax_p = fig.add_subplot(gs[:, 0:3])
    ax_s = fig.add_subplot(gs[0, 3:5])
    ax_d = fig.add_subplot(gs[1, 3:5], sharex=ax_s)

    # ---- static furniture --------------------------------------------------
    ax_p.set_xlabel("$x$")
    ax_p.set_ylabel("$y$")
    ax_p.set_ylim(-9, 9)
    ax_p.set_aspect("equal", adjustable="box")

    # The full trace is laid down in grid ink first, so the movie reveals a path
    # against its own future rather than growing into blank space.
    for ax, series, lab, lo in ((ax_s, sn, r"$\Vert\sigma(t)\Vert$", 3e-17),
                                (ax_d, dn, r"$\Vert\mathbf{\delta}(t)\Vert$", 1e-13)):
        _shade(ax)
        ax.plot(t_eval, np.maximum(series, lo), color=style.GRID, lw=1.0, zorder=1)
        ax.set_xlim(0, T_END)
        ax.set_ylabel(lab)
        ax.set_yscale("log")
    ax_s.axvline(tau, **style.marker_line_style())
    ax_s.set_ylim(3e-17, 30)
    ax_s.tick_params(labelbottom=False)
    ax_s.set_title(r"centroid: $\tau_c$ dotted, morphs shaded", loc="left", fontsize=7)
    ax_d.set_ylim(1e-13, 1e2)
    ax_d.set_xlabel("time  $t$  [s]")
    ax_d.set_title("formation error", loc="left", fontsize=7)

    # ---- animated artists --------------------------------------------------
    trails = [ax_p.plot([], [], color=style.SERIES[0], lw=0.7, alpha=0.55)[0]
              for _ in range(common.N)]
    dots = ax_p.scatter(np.zeros(common.N), np.zeros(common.N), s=22,
                        color=style.SERIES[0], zorder=5, linewidths=0)
    slots = ax_p.scatter(np.zeros(common.N), np.zeros(common.N), s=26,
                         facecolors="none", edgecolors=style.SERIES[1],
                         linewidths=0.8, zorder=4)
    ref = ax_p.scatter([0], [0], marker="+", s=44, color=style.INK,
                       linewidths=1.0, zorder=6)
    caption = ax_p.set_title("", loc="left")

    ax_p.plot([], [], color=style.SERIES[0], lw=1.0, label="agents")
    ax_p.scatter([], [], s=26, facecolors="none", edgecolors=style.SERIES[1],
                 linewidths=0.8, label=r"slots $p_i(t)+r(t)$")
    ax_p.scatter([], [], marker="+", s=44, color=style.INK, linewidths=1.0,
                 label="reference $r(t)$")
    ax_p.legend(loc="lower left", fontsize=6, ncol=3)

    live_s, = ax_s.plot([], [], **style.series_style(0))
    live_d, = ax_d.plot([], [], **style.series_style(0))
    cursors = [ax.axvline(0.0, color=style.INK, lw=0.7, zorder=6)
               for ax in (ax_s, ax_d)]
    for ax in (ax_p, ax_s, ax_d):
        leg = ax.get_legend()
        if leg is not None:
            leg.set_in_layout(False)

    frame_t = np.arange(0.0, T_END, SPEEDUP / FPS)
    idx = np.searchsorted(t_eval, frame_t).clip(0, t_eval.size - 1)

    def draw(k):
        j = idx[k]
        t = t_eval[j]
        j0 = max(0, j - int(TRAIL / (t_eval[1] - t_eval[0])))
        for i, ln in enumerate(trails):
            ln.set_data(X[j0:j + 1, i, 0], X[j0:j + 1, i, 1])
        dots.set_offsets(X[j])
        r_now = np.asarray(r_fn(float(t)), dtype=float)
        slots.set_offsets(np.asarray(P_fn(float(t))) + r_now)
        ref.set_offsets(r_now[None, :])
        ax_p.set_xlim(r_now[0] - CAM_HALF, r_now[0] + CAM_HALF)
        caption.set_text(f"$t={t:5.1f}$ s    {_phase_name(float(t))}")

        live_s.set_data(t_eval[:j + 1], np.maximum(sn[:j + 1], 3e-17))
        live_d.set_data(t_eval[:j + 1], np.maximum(dn[:j + 1], 1e-13))
        for cur in cursors:
            cur.set_xdata([t, t])
        return (*trails, dots, slots, ref, live_s, live_d, *cursors)

    anim = animation.FuncAnimation(fig, draw, frames=frame_t.size,
                                   interval=1000 / FPS, blit=False)
    style.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = style.FIG_DIR / "f5_morphing.mp4"
    anim.save(out, writer=animation.FFMpegWriter(fps=FPS, bitrate=3200),
              dpi=180)
    plt.close(fig)
    print(f"    wrote figures/{out.name} ({frame_t.size} frames, "
          f"{frame_t.size / FPS:.0f}s at {SPEEDUP:g}x)")
    return out


def main() -> dict:
    style.use_paper_style()
    P_fn, Pdot_fn = _schedule()
    cfg = common.nominal_config().with_(P_fn=P_fn, Pdot_fn=Pdot_fn)
    Z0 = common.initial_state()
    t_eval = np.linspace(0, T_END, 3401)
    summary = {}

    res = simulate(cfg, Z0, (0, T_END), t_eval=t_eval)
    tau = res.tau_c_measured
    sn = np.linalg.norm(metrics.centroid_error(t_eval, res.X, cfg), axis=1)
    dn = metrics.delta_norm(t_eval, res.X, cfg)

    # Ablation 1, drawn live in the movie: without the feedforward the
    # quasi-steady solution of deltadot = -delta - pdot is delta ~ -pdot, so
    # ||delta|| tracks ||Pdot||_F through every morph instead of staying at zero.
    cfg_np = cfg.with_(use_pdot=False)
    res_np = simulate(cfg_np, Z0, (0, T_END), t_eval=t_eval)
    dn_np = metrics.delta_norm(t_eval, res_np.X, cfg_np)
    pdot_norm = np.array([np.linalg.norm(Pdot_fn(float(t))) for t in t_eval])

    post = t_eval >= tau + 1e-9
    peak = float(sn[post].max())
    summary["sigma_peak_after_tau"] = peak

    during = (t_eval > SWITCH[0] + 1.0) & (t_eval < SWITCH[0] + MORPH_TIME - 1.0)
    summary["pdot_ablation_ratio"] = float(
        (dn_np[during] / np.maximum(pdot_norm[during], 1e-30)).mean())
    summary["pdot_ablation_floor"] = float(dn_np[during].max())
    summary["pdot_baseline_floor"] = float(dn[during].max())

    _animate(t_eval, res.X, sn, dn, tau, P_fn, cfg.r_fn)

    # ---- cost of a morph vs its duration (printed; no panel) ---------------
    # Start every agent exactly on its slot, so delta(0) = 0 and sigma(0) = 0 and
    # the whole excursion is attributable to the morph rather than to leftover
    # transient. With the feedforward the excursion should be zero exactly.
    MORPH_TIMES = [0.5, 1.0, 2.0, 4.0, 8.0]
    peaks = {True: [], False: []}
    quasi = []
    for Tm in MORPH_TIMES:
        Pm_fn, Pmd_fn = morph_schedule(
            [hexagon(common.N, L), line(common.N, L)], [1.0], Tm)
        c0 = common.nominal_config().with_(P_fn=Pm_fn, Pdot_fn=Pmd_fn)
        X0 = Pm_fn(0.0) + np.asarray(c0.r_fn(0.0))
        t_m = np.linspace(0, 1.0 + Tm + 4.0, 1200)
        for use in (True, False):
            rm = simulate(c0.with_(use_pdot=use), X0.ravel(), (0, t_m[-1]),
                          t_eval=t_m)
            peaks[use].append(float(metrics.delta_norm(t_m, rm.X, c0).max()))
        # The quasi-steady prediction holds while the morph is slow compared with
        # the unit closed-loop time constant; for a fast morph the error saturates
        # instead at the total shape change it never has time to track.
        sweep = np.linspace(1.0, 1.0 + Tm, 400)
        max_pdot = max(float(np.linalg.norm(Pmd_fn(float(t)))) for t in sweep)
        total = float(np.linalg.norm(Pm_fn(1.0 + Tm) - Pm_fn(1.0)))
        quasi.append(min(max_pdot, total))
    summary["morph_peaks_with_pdot"] = peaks[True]
    summary["morph_peaks_without_pdot"] = peaks[False]

    print(f"    peak ||sigma|| after tau_c through 3 morphs : {peak:.2e}")
    print("    peak ||delta|| during a single morph, by duration:")
    for Tm, a, b, q in zip(MORPH_TIMES, peaks[True], peaks[False], quasi):
        print(f"          T_morph={Tm:4.1f}s   with pdot={a:.2e}   "
              f"without={b:.2e}   predicted={q:.2e}")
    print(f"    no-pdot error during morph : {summary['pdot_ablation_floor']:.3e} "
          f"(with pdot: {summary['pdot_baseline_floor']:.3e})")
    print(f"        mean ||delta|| / ||Pdot||_F during morph : "
          f"{summary['pdot_ablation_ratio']:.3f}")
    return summary


if __name__ == "__main__":
    main()
