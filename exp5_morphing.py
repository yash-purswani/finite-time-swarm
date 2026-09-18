"""F2 -- formation reconfiguration: ring -> wedge -> star -> arrow.

The swarm tracks r(t) while its template changes three times. The paper figure
``figures/f2_morphing.png`` draws every agent's trajectory dashed, with the agents
(filled) and their stations p_i(t) + r(t) (hollow) marked at one instant in each
formation.

Which agent takes which slot is free, so it is fixed by the Hungarian algorithm
(linear_sum_assignment): from the initial positions onto the first template, then
slot to slot at every switch. The blend between two templates is affine and
synchronised, so relative to r(t) the stations move on straight lines, and the
minimum-squared-travel assignment keeps those paths apart. It is a heuristic for
the agents: they follow their stations only up to delta_i, and the coupling acts
on delta_i - delta_j, so it supplies no repulsion. The minimum inter-agent
distance is printed with and without the assignment.

Also printed: the peak ||sigma|| after tau_c across the three morphs, and the peak
||delta|| during a single morph as a function of its duration, with and without
the pdot feedforward. ``--movie`` also renders the animation
``figures/f5_morphing.mp4``.
"""

import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from finite_time import common, metrics, style
from scipy.spatial import ConvexHull

from finite_time.formations import (arrow, hexagon, line, morph_schedule,
                                    reassign_from_state, star, wedge)
from finite_time.integrate import simulate

N_M = common.N          # the same swarm as every other experiment
L = 5.0                 # the ring is the nominal template of Table I
MORPH_TIME = 4.0
SWITCH = [14.0, 26.0, 38.0]
T_END = 52.0
NAMES = ["ring", "wedge", "star", "arrow"]
FILLED = {"wedge"}      # outlined by its hull rather than through its rows
DT = 0.002              # output step; fine enough for the separation scan

# Start, then one settled instant per shape; the start in ink, one hue per shape.
SNAPS = [(0.0, "initial"), (13.0, NAMES[0]), (25.0, NAMES[1]),
         (37.0, NAMES[2]), (50.0, NAMES[3])]
TRAJ = style.AXIS       # light, so the snapshots carry the figure
SNAP_COLORS = [style.INK_2] + style.SERIES

FPS = 25
SPEEDUP = 2.0           # 34 s of simulation in 17 s of video
TRAIL = 2.5             # seconds of trail behind each agent
CAM_HALF = 9.0          # half-width of the window that follows r(t)


def _templates():
    """The four shapes; hexagon() with n points is the ring.

    Rows are in outline order except for the filled wedge.
    """
    return [hexagon(N_M, L), wedge(N_M, 0.6 * L), star(N_M, 1.1 * L),
            arrow(N_M, L)]


def _schedule(X0=None, relabel=True):
    """Morph schedule; with ``X0`` the first template is also assigned to the agents."""
    shapes = _templates()
    if X0 is not None:
        r0 = common.nominal_config(n=N_M).r_fn(0.0)
        shapes[0] = reassign_from_state(X0, shapes[0], r0)
    return morph_schedule(shapes, SWITCH, MORPH_TIME, relabel=relabel)


def _min_separation(t, X, t_from=0.0):
    """Smallest ||x_i - x_j|| over all pairs and all t >= t_from, and when."""
    iu = np.triu_indices(X.shape[1], 1)
    D = np.linalg.norm(X[:, iu[0]] - X[:, iu[1]], axis=2).min(axis=1)
    D = np.where(t >= t_from, D, np.inf)
    k = int(np.argmin(D))
    return float(D[k]), float(t[k])


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


def _figure(t, X, cfg):
    """Trajectories (dashed) with agents and stations, coloured by formation."""
    # Height matched to the equal-aspect data box, so no slack opens above it.
    fig, ax = plt.subplots(figsize=(style.COL2_W, 2.5))
    R = np.array([cfg.r_fn(float(s)) for s in t])
    ax.plot(R[:, 0], R[:, 1], color=style.INK, lw=0.6, zorder=1)
    for i in range(cfg.n):
        ax.plot(X[:, i, 0], X[:, i, 1], color=TRAJ, lw=0.6,
                linestyle=(0, (3.0, 1.8)), zorder=2)

    templates = _templates()
    for k, ((ts, name), col) in enumerate(zip(SNAPS, SNAP_COLORS)):
        j = int(np.argmin(np.abs(t - ts)))
        r = np.asarray(cfg.r_fn(ts), dtype=float)
        S = np.asarray(cfg.P_fn(ts)) + r
        # At t_0 the agents are still far from their stations, whose rings would
        # only collide with the first settled snapshot; the agents alone are drawn.
        # The assignment permutes the slots, so the outline is traced through the
        # template in its own (outline) order rather than through S.
        if ts > t[0]:
            P = templates[k - 1]
            if name in FILLED:
                P = P[ConvexHull(P).vertices]
            O = np.vstack([P, P[:1]]) + r
            ax.plot(O[:, 0], O[:, 1], color=col, lw=0.7, alpha=0.6, zorder=2.5)
            ax.scatter(S[:, 0], S[:, 1], s=40, facecolors="none",
                       edgecolors=col, linewidths=0.8, zorder=3)
        ax.scatter(X[j, :, 0], X[j, :, 1], s=16, color=col, linewidths=0,
                   zorder=4, label=f"$t={ts:g}$ s, {name}")
        ax.scatter([r[0]], [r[1]], marker="+", s=34, color=style.INK,
                   linewidths=0.9, zorder=5)

    ax.plot([], [], color=TRAJ, lw=0.6, linestyle=(0, (3.0, 1.8)),
            label="agent trajectories")
    ax.scatter([], [], s=46, facecolors="none", edgecolors=style.INK_2,
               linewidths=0.8, label=r"stations $p_i(t)+r(t)$")
    ax.plot([], [], color=style.INK, lw=0.6, marker="+", ms=5,
            label="reference $r(t)$")
    # A figure legend, not an axes one: style.save drops axes legends from the
    # layout, and the tight crop would then cut one placed outside the axes.
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    y_max = float(np.abs(X[:, :, 1]).max()) + 1.0
    ax.set_ylim(-y_max, y_max)
    ax.set_aspect("equal", adjustable="box")
    # Height matched to the equal-aspect data box (plus the x label), so no slack
    # opens between the legend and the axes.
    xr, yr = np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())
    fig.set_size_inches(style.COL2_W, 0.5 + (style.COL2_W - 0.5) * yr / xr)
    style.save(fig, "f2_morphing")


def _animate(t_eval, X, sn, dn, tau, P_fn, r_fn):
    """Render the morph movie. Returns the path written."""
    n = X.shape[1]
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
              for _ in range(n)]
    dots = ax_p.scatter(np.zeros(n), np.zeros(n), s=22,
                        color=style.SERIES[0], zorder=5, linewidths=0)
    slots = ax_p.scatter(np.zeros(n), np.zeros(n), s=26,
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


def main(movie: bool = False) -> dict:
    style.use_paper_style()
    Z0 = common.initial_state()
    P_fn, Pdot_fn = _schedule(Z0.reshape(N_M, common.D))
    cfg = common.nominal_config().with_(P_fn=P_fn, Pdot_fn=Pdot_fn)
    t_eval = np.arange(0.0, T_END + 0.5 * DT, DT)
    summary = {}

    res = simulate(cfg, Z0, (0, T_END), t_eval=t_eval)
    tau = res.tau_c_measured
    sn = np.linalg.norm(metrics.centroid_error(t_eval, res.X, cfg), axis=1)
    dn = metrics.delta_norm(t_eval, res.X, cfg)
    _figure(t_eval, res.X, cfg)

    # ---- what the assignment buys: separation with and without it ----------
    P_na, Pdot_na = _schedule(relabel=False)
    cfg_na = cfg.with_(P_fn=P_na, Pdot_fn=Pdot_na)
    res_na = simulate(cfg_na, Z0, (0, T_END), t_eval=t_eval)
    for key, X in (("assigned", res.X), ("unassigned", res_na.X)):
        summary[f"sep_{key}"] = _min_separation(t_eval, X)
        summary[f"sep_{key}_morphs"] = _min_separation(t_eval, X, SWITCH[0])
    # Compare only while the envelope sits above the solver floor (rtol 1e-10,
    # atol 1e-12); below it the ratio measures integration error, not the bound.
    env = dn[0] * np.exp(-t_eval)
    live = env >= 1e-8
    summary["envelope_ratio"] = float((dn[live] / env[live]).max())

    # Ablation 1: without the feedforward the quasi-steady solution of
    # deltadot = -delta - pdot is delta ~ -pdot, so ||delta|| tracks ||Pdot||_F
    # through every morph instead of staying at zero.
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

    if movie:
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

    for key in ("assigned", "unassigned"):
        (d_all, t_all), (d_m, t_m_) = summary[f"sep_{key}"], summary[f"sep_{key}_morphs"]
        print(f"    min ||x_i - x_j||, {key:>10}: {d_all:.3f} at t={t_all:.2f} s "
              f"(t >= {SWITCH[0]:g} s: {d_m:.3f} at t={t_m_:.2f} s)")
    print(f"    max ||delta|| / envelope (envelope >= 1e-8) : "
          f"{summary['envelope_ratio']:.4f}")
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
    main(movie="--movie" in sys.argv)
