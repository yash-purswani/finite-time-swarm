"""
run_transport_sim.py
===============================================================================
Cooperative swarm payload pickup & transport -- full pipeline.

    python run_transport_sim.py

Runs, end to end, with no extra setup:
  1. Defines an asymmetric L-shaped payload and computes its CoG (shapely).
  2. Offline formation-offset optimization (SLSQP) for n agents around the
     payload's rim, subject to a hard zero-net-offset centering constraint,
     containment, and pairwise separation.
  3. Builds each slot's *approach corridor*: an outward-normal standoff point
     just outside the payload, mitred correctly even at reflex (concave)
     vertices.
  4. Scatters n agents across the workshop floor and computes a Hungarian 
     docking assignment against the standoff points.
  5. PHASE 1 (rendezvous & docking): uses a time-varying, normal-guided bias 
     p_i(t) (standoff -> boundary) to guide the agents to their slots.
  6. PHASE 2 (transport): continues from the docked state, sweeping a smooth 
     S-curve to a drop-off zone.
  7. Shows animation output on screen and saves workshop_transport_animation.mp4 
     and kinematic_convergence_plots.png.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from shapely.geometry import Point

import geometry as geo
import optimization as opt
import approach as ap
import dynamics as dyn
import trajectory as traj
import visualization as viz


# =============================================================================
# MISSION CONFIGURATION
# =============================================================================
@dataclass
class MissionConfig:
    n_agents: int = 8

    # -- payload --
    pickup_point: np.ndarray = field(default_factory=lambda: np.array([6.0, 6.0]))
    dropoff_point: np.ndarray = field(default_factory=lambda: np.array([22.0, 16.0]))

    # -- offline formation optimizer --
    w1: float = 0.05          # pairwise-repulsion weight
    opt_delta: float = 1e-2   # repulsion softening term
    d_min: float = 0.35       # minimum inter-agent slot separation

    # -- normal-guided approach corridor --
    d_standoff: float = 0.4   # nominal standoff distance outside the rim
    t_approach: float = 8.0   # duration of the standoff -> boundary schedule
    contact_eps: float = 0.08 # "arrived at slot" radius

    # -- finite-time kinematic dynamics --
    k_m: float = 6.0          # scalar gain -> M = k_m * I_2
    nu: float = 2.0
    eps: float = 1e-2
    beta: float = 0.5

    # -- mission timing --
    t_dock_max: float = 30.0     # safety cap on Phase 1 duration
    sigma_eps: float = 1e-3      # convergence threshold on ||sigma(t)||
    t_transport: float = 26.0    # Phase 2 duration
    s_curve_bow: float = 2.5
    s_curve_controls: int = 4

    # -- workshop floor (for agent scatter + plotting bounds) --
    floor_bounds: Tuple[float, float, float, float] = (-4.0, 30.0, -4.0, 22.0)

    # -- misc --
    seed: int = 0
    sample_dt: float = 0.08   # dense-output sampling step for animation/plots

    @property
    def M(self) -> np.ndarray:
        return self.k_m * np.eye(2)


# =============================================================================
# PIPELINE STEPS
# =============================================================================
def build_payload(cfg: MissionConfig) -> geo.Payload:
    verts = geo.make_l_shape(arm_long=3.4, arm_short=2.0, thickness=1.0)
    verts = verts - geo.polygon_centroid(verts) + cfg.pickup_point
    return geo.make_payload(verts)


def scatter_agents(cfg: MissionConfig, rng: np.random.Generator, payload: geo.Payload) -> np.ndarray:
    xmin, xmax, ymin, ymax = cfg.floor_bounds
    poly_local = payload.polygon_local
    pts = []
    attempts = 0
    while len(pts) < cfg.n_agents and attempts < 20000:
        p = rng.uniform([-2.0, -2.0], [2.0, 2.0])
        if not poly_local.contains(Point(p - payload.cog0)):
            pts.append(p)
        attempts += 1
    if len(pts) < cfg.n_agents:
        raise RuntimeError("Could not scatter agents outside the payload footprint")
    return np.array(pts)


def run_phase1_docking(
    cfg: MissionConfig,
    p_fn: Callable[[float], np.ndarray],
    p_dot_fn: Callable[[float], np.ndarray],
    x0: np.ndarray,
    cog0: np.ndarray,
):
    r_fn = lambda t: cog0
    rdot_fn = lambda t: np.zeros(2)
    rhs = dyn.make_ode_rhs(p_fn, p_dot_fn, r_fn, rdot_fn, cfg.M, cfg.nu, cfg.eps, cfg.beta, cfg.n_agents)

    def sigma_event(t: float, x_flat: np.ndarray) -> float:
        x = x_flat.reshape(cfg.n_agents, 2)
        return np.linalg.norm(dyn.centroid_error(x, cog0)) - cfg.sigma_eps

    sigma_event.terminal = True
    sigma_event.direction = -1

    sol = solve_ivp(
        rhs, [0.0, cfg.t_dock_max], x0.flatten(),
        dense_output=True, max_step=0.05, events=sigma_event, rtol=1e-8, atol=1e-9,
    )

    converged = len(sol.t_events[0]) > 0
    tau = float(sol.t_events[0][0]) if converged else None
    t_end = tau if converged else cfg.t_dock_max

    if not converged:
        print(f"    WARNING: centroid error did not reach {cfg.sigma_eps} "
              f"within t_dock_max={cfg.t_dock_max}s; docking with best-effort state.")
    else:
        print(f"    finite-time convergence: tau = {tau:.3f}s")

    ts = np.arange(0.0, t_end + 1e-9, cfg.sample_dt)
    ts[-1] = t_end
    xs = np.array([sol.sol(t).reshape(cfg.n_agents, 2) for t in ts])
    rs = np.tile(cog0, (len(ts), 1))
    targets = np.array([p_fn(t) for t in ts]) + rs[:, None, :]
    return ts, xs, rs, targets, tau, xs[-1]


def run_phase2_transport(cfg: MissionConfig, B: np.ndarray, x_start: np.ndarray, cog0: np.ndarray):
    waypoints = traj.s_curve_waypoints(cog0, cfg.dropoff_point, bow=cfg.s_curve_bow,
                                        n_control=cfg.s_curve_controls)
    ref = traj.ReferenceTrajectory(waypoints, T=cfg.t_transport)
    r_fn, rdot_fn = ref.as_callables()
    p_fn, p_dot_fn = ap.constant_bias(B)

    rhs = dyn.make_ode_rhs(p_fn, p_dot_fn, r_fn, rdot_fn, cfg.M, cfg.nu, cfg.eps, cfg.beta, cfg.n_agents)
    sol = solve_ivp(
        rhs, [0.0, cfg.t_transport], x_start.flatten(),
        dense_output=True, max_step=0.05, rtol=1e-8, atol=1e-9,
    )

    ts = np.arange(0.0, cfg.t_transport + 1e-9, cfg.sample_dt)
    ts[-1] = cfg.t_transport
    xs = np.array([sol.sol(t).reshape(cfg.n_agents, 2) for t in ts])
    rs = np.array([r_fn(t) for t in ts])
    targets = B[None, :, :] + rs[:, None, :]

    full_path_t = np.linspace(0.0, cfg.t_transport, 300)
    full_path = np.array([r_fn(t) for t in full_path_t])
    return ts, xs, rs, targets, full_path


# =============================================================================
# MAIN
# =============================================================================
def run_mission(cfg: MissionConfig, out_dir: str = "outputs") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    payload = build_payload(cfg)
    cog0 = payload.cog0
    poly_local = payload.polygon_local

    P_opt, opt_ok = opt.optimize_formation_offsets(
        poly_local, cfg.n_agents, w1=cfg.w1, delta=cfg.opt_delta, d_min=cfg.d_min,
    )
    print(f"[formation optimizer] success={opt_ok}  sum(P)={P_opt.sum(axis=0)}")

    B_slots, N_slots, S_slots, scale = ap.compute_approach_corridor(P_opt, poly_local, cfg.d_standoff)
    min_clear = ap.min_standoff_clearance(S_slots, poly_local)
    print(f"[approach corridor] d_standoff={cfg.d_standoff}  "
          f"min achieved clearance={min_clear:.4f}  "
          f"(per-slot escalation scale: {np.round(scale, 2)})")

    x0 = scatter_agents(cfg, rng, payload)
    perm = opt.docking_assignment(x0, S_slots, cog0, polygon_local=poly_local)
    B, N, S = B_slots[perm], N_slots[perm], S_slots[perm]

    # --- Phase 1: normal-guided time-varying bias ---
    print("[phase 1 | docking: normal-guided standoff -> boundary]")
    bias = ap.NormalApproachBias(S, B, T_approach=cfg.t_approach)
    p_fn_p, pdot_fn_p = bias.as_callables()
    t1, x1, r1, targets1, tau, x_docked = run_phase1_docking(cfg, p_fn_p, pdot_fn_p, x0, cog0)

    # --- Phase 2: transport ---
    print("[phase 2 | transport]")
    t2, x2, r2, targets2, full_ref_path = run_phase2_transport(cfg, B, x_docked, cog0)

    t_dock = t1[-1]
    t_all = np.concatenate([t1, t2 + t_dock])
    x_all = np.concatenate([x1, x2], axis=0)
    r_all = np.concatenate([r1, r2], axis=0)
    targets_all = np.concatenate([targets1, targets2], axis=0)

    # --- verification ---
    rho = dyn.theoretical_rho(cfg.M, cfg.nu, cfg.eps)
    final_delta = dyn.tracking_errors(x_docked, B, cog0)
    print(f"\n[verification] theoretical bound rho = {rho:.4f}")
    print(f"[verification] post-dock individual errors ||delta_i||: "
          f"min={final_delta.min():.4f} max={final_delta.max():.4f}")
    print(f"[verification] all agents within rho at docking: {bool(np.all(final_delta <= rho))}")

    workshop_bounds = cfg.floor_bounds
    standoff_world = S + cog0

    viz.animate_mission(
        t_all, x_all, targets_all, r_all, payload, t_dock, full_ref_path, workshop_bounds,
        video_path=os.path.join(out_dir, "transport_animation.mp4"),
        standoff_world=standoff_world,
        show=True,
    )
    viz.plot_convergence(
        t_all, x_all, targets_all, r_all, rho, t_dock, tau,
        png_path=os.path.join(out_dir, "kinematic_convergence_plots.png"),
    )

    return dict(t=t_all, x=x_all, r=r_all, targets=targets_all, B=B, N=N, S=S,
                payload=payload, tau=tau, t_dock=t_dock, rho=rho, config=cfg)


if __name__ == "__main__":
    cfg = MissionConfig()
    run_mission(cfg)
