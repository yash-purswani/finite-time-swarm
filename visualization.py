"""
visualization.py
===============================================================================
Output artifacts:
  - workshop_transport_animation.mp4 : animated 2D mission (docking + transport)
  - kinematic_convergence_plots.png  : centroid error + per-agent tracking error
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MplPolygon

from geometry import Payload, containment_margin
import dynamics as dyn


def animate_mission(
    t: np.ndarray,
    x: np.ndarray,
    targets: np.ndarray,
    r: np.ndarray,
    payload: Payload,
    t_dock: float,
    full_ref_path: np.ndarray,
    workshop_bounds: tuple,
    video_path: Optional[str] = None,
    standoff_world: Optional[np.ndarray] = None,
    fps: int = 30,
    dpi: int = 120,
    show: bool = False,
) -> None:
    """
    Parameters
    ----------
    t : (T,) time samples
    x : (T, n, 2) agent positions
    targets : (T, n, 2) each agent's *instantaneous* held target p_i(t) + r(t)
        (time-varying during the normal-guided docking approach, then
        constant B_i + r(t) once docked and during transport).
    r : (T, 2) reference / payload CoG position at each sampled time
    payload : Payload
    t_dock : time at which Phase 1 -> Phase 2 switch happened
    full_ref_path : (M, 2) the whole Phase-2 reference path (for the dashed
        target-trajectory overlay)
    workshop_bounds : (xmin, xmax, ymin, ymax)
    standoff_world : (n, 2) optional -- each agent's standoff point (world
        frame, at t=0) drawn as small markers for context.
    """
    n = targets.shape[1]
    T = len(t)
    frame_skip = max(1, T // 500)
    frame_idx = np.arange(0, T, frame_skip)

    xmin, xmax, ymin, ymax = workshop_bounds
    # Size the figure to match the data's aspect ratio -- otherwise
    # set_aspect("equal") pads the square figure with blank margins to
    # preserve a 1:1 data aspect within a mismatched canvas shape.
    data_aspect = (ymax - ymin) / (xmax - xmin)
    fig_w = 9.0
    fig_h = float(np.clip(fig_w * data_aspect, 4.0, 12.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor("#141414")
    ax.set_title("Cooperative swarm payload pickup & transport")

    ax.plot(full_ref_path[:, 0], full_ref_path[:, 1], "c--", lw=1.2, alpha=0.6,
            label="reference path r(t)")
    ax.plot(payload.cog0[0], payload.cog0[1], "wx", ms=9, mew=2, label="pickup point")
    ax.plot(full_ref_path[-1, 0], full_ref_path[-1, 1], "y*", ms=16, label="drop-off")
    if standoff_world is not None:
        ax.scatter(standoff_world[:, 0], standoff_world[:, 1], marker="+", c="#7fffd4",
                   s=60, linewidths=1.3, zorder=4, label="standoff (approach) points")

    payload_patch = MplPolygon(payload.world_vertices(r[0]), closed=True,
                                facecolor="#c98a3a", edgecolor="white", alpha=0.55, zorder=3)
    ax.add_patch(payload_patch)

    colors = plt.cm.tab10(np.arange(n) % 10)
    dots, trails, offset_lines = [], [], []
    for i in range(n):
        c = colors[i]
        dots.append(ax.plot([], [], "o", color=c, mec="white", mew=1.0, ms=9, zorder=6)[0])
        trails.append(ax.plot([], [], "-", color=c, lw=0.9, alpha=0.45, zorder=2)[0])
        offset_lines.append(ax.plot([], [], "-", color=c, lw=1.3, alpha=0.85, zorder=5)[0])

    phase_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", color="white",
                         fontsize=10, bbox=dict(facecolor="black", alpha=0.55))
    ax.legend(loc="lower right", fontsize=8, facecolor="#222222", labelcolor="white")
    fig.tight_layout()

    def update(f):
        k = frame_idx[f]
        tk = t[k]
        rk = r[k]
        payload_patch.set_xy(payload.world_vertices(rk))

        for i in range(n):
            xi, yi = x[k, i]
            dots[i].set_data([xi], [yi])
            trails[i].set_data(x[:k + 1, i, 0], x[:k + 1, i, 1])
            # vector from the agent's *currently held* target to the agent itself
            slot = targets[k, i]
            offset_lines[i].set_data([slot[0], xi], [slot[1], yi])

        phase = "PHASE 1: rendezvous & docking" if tk < t_dock else "PHASE 2: transport"
        phase_txt.set_text(f"t={tk:5.1f}s   {phase}")
        return [payload_patch, phase_txt, *dots, *trails, *offset_lines]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_idx), interval=1000 / fps, blit=False)

    if video_path:
        if video_path.lower().endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
        ani.save(video_path, writer=writer, dpi=dpi)
        print(f"saved animation -> {video_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def plot_convergence(
    t: np.ndarray,
    x: np.ndarray,
    targets: np.ndarray,
    r: np.ndarray,
    rho: float,
    t_dock: float,
    tau: Optional[float],
    png_path: str,
) -> None:
    n = targets.shape[1]
    sigma_norm = np.array([np.linalg.norm(dyn.centroid_error(x[k], r[k])) for k in range(len(t))])
    delta_norms = np.array([dyn.tracking_errors_from_target(x[k], targets[k]) for k in range(len(t))])  # (T, n)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax0 = axes[0]
    ax0.plot(t, sigma_norm, color="white", lw=1.6)
    ax0.axvline(t_dock, color="cyan", ls="--", lw=1, alpha=0.7, label="dock -> transport switch")
    if tau is not None:
        ax0.axvline(tau, color="lime", ls=":", lw=1.5, label=f"finite-time convergence  tau={tau:.2f}s")
    ax0.set_ylabel(r"$\|\sigma(t)\| = \|c(t) - r(t)\|$")
    ax0.set_title("Centroid tracking error (finite-time convergence)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_facecolor("#141414")
    ax0.grid(alpha=0.2)

    ax1 = axes[1]
    colors = plt.cm.tab10(np.arange(n) % 10)
    for i in range(n):
        ax1.plot(t, delta_norms[:, i], color=colors[i], lw=1.0, alpha=0.85, label=f"agent {i}")
    ax1.axhline(rho, color="red", ls="--", lw=1.3, label=rf"theoretical bound $\rho$ = {rho:.3g}")
    ax1.axvline(t_dock, color="cyan", ls="--", lw=1, alpha=0.7)
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel(r"$\|\delta_i(t)\| = \|x_i(t) - p_i(t) - r(t)\|$")
    ax1.set_title("Individual agent tracking errors vs. theoretical bound")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.set_facecolor("#141414")
    ax1.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130, facecolor="#1a1a1a")
    plt.close(fig)
    print(f"saved convergence plots -> {png_path}")


def plot_docking_comparison(
    payload: Payload,
    x_baseline: np.ndarray,
    x_proposed: np.ndarray,
    B_world: np.ndarray,
    S_world: np.ndarray,
    normals: np.ndarray,
    png_path: str,
    zoom_margin: float = 1.8,
) -> None:
    """Side-by-side ablation figure: baseline constant-bias docking (agents
    cut straight to their slot -- may slice through the payload interior)
    vs. proposed normal-guided docking (standoff -> boundary along the
    normal). Both panels share the same initial scatter and payload.

    Zoomed to the payload's own footprint (+ `zoom_margin`) rather than the
    full trajectory extent: on a whole-workshop scale the far-field
    convergence behaviour (governed by the *same* finite-time law in both
    cases) dominates the picture and the actually-relevant difference --
    whether the final approach clips the object -- is only a few tenths of a
    unit and invisible unless zoomed in.
    """
    n = B_world.shape[0]
    colors = plt.cm.tab10(np.arange(n) % 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    titles = ["Baseline: constant bias (straight to slot)",
              "Proposed: normal-guided (standoff -> boundary)"]
    trajs = [x_baseline, x_proposed]

    local_v = payload.local_vertices
    half_extent = np.abs(local_v).max() + zoom_margin
    xmin, xmax = payload.cog0[0] - half_extent, payload.cog0[0] + half_extent
    ymin, ymax = payload.cog0[1] - half_extent, payload.cog0[1] + half_extent

    for ax, title, xt in zip(axes, titles, trajs):
        ax.set_facecolor("#141414")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        payload_patch = MplPolygon(payload.world_vertices(payload.cog0), closed=True,
                                    facecolor="#c98a3a", edgecolor="white", alpha=0.6, zorder=3)
        ax.add_patch(payload_patch)

        intrusion_pts = []
        for i in range(n):
            c = colors[i]
            # only the portion of the trajectory that's actually within the
            # zoomed window has any bearing on the comparison
            in_view = (np.abs(xt[:, i, 0] - payload.cog0[0]) < half_extent) & \
                      (np.abs(xt[:, i, 1] - payload.cog0[1]) < half_extent)
            ax.plot(xt[in_view, i, 0], xt[in_view, i, 1], "-", color=c, lw=1.6, alpha=0.9, zorder=5)
            ax.plot(xt[-1, i, 0], xt[-1, i, 1], "s", color=c, mec="white", mew=0.8, ms=7, zorder=6)
            ax.plot(B_world[i, 0], B_world[i, 1], "x", color="white", ms=7, mew=1.5, zorder=7)
            # explicitly flag every sample where the agent is literally inside
            # the payload polygon -- the subtle "did it graze the object"
            # question is otherwise easy to miss by eye at this scale
            for k in np.where(in_view)[0]:
                if containment_margin(xt[k, i] - payload.cog0, payload.polygon_local) > 0:
                    intrusion_pts.append(xt[k, i])
        if intrusion_pts:
            ip = np.array(intrusion_pts)
            ax.scatter(ip[:, 0], ip[:, 1], marker="x", c="red", s=45, linewidths=2.0,
                      zorder=9, label=f"intrusion ({len(ip)} samples)")
            ax.legend(loc="upper left", fontsize=8, facecolor="#222222", labelcolor="white")

        ax.quiver(B_world[:, 0], B_world[:, 1], normals[:, 0], normals[:, 1],
                  color="#7fffd4", scale=12, width=0.005, zorder=8)
        ax.scatter(S_world[:, 0], S_world[:, 1], marker="+", c="#7fffd4", s=70,
                  linewidths=1.6, zorder=8)
        ax.grid(alpha=0.15)

    fig.suptitle("Docking-approach ablation, zoomed to the payload: square = agent's final "
                 "docked position, x = assigned grasp slot, + = standoff point, "
                 "arrow = outward normal", color="white", fontsize=10)
    fig.patch.set_facecolor("#1a1a1a")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, facecolor="#1a1a1a")
    plt.close(fig)
    print(f"saved docking comparison -> {png_path}")
