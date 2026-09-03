# Swarm Payload Transport Simulation

This repository contains a full pipeline for cooperative multi-robot payload pickup and transport, built around a finite-time kinematic guidance model.

The codebase specifically addresses a common gap in offset-based cooperative transport: driving agents to grasp points along a straight-line path often causes them to cut through the payload's interior. This repository implements a **normal-guided, time-varying formation bias** that constructs an *approach corridor* (standoff -> boundary) for each agent. Agents first rendezvous at standoff points outside the payload, then approach their assigned grasp slots along the surface's outward normal, preserving the finite-time convergence properties of the original swarm model.

Theoretical grounding for this approach is detailed in `Report.pdf` and `paper_notes.md`.

## Running the Simulation

You can run the end-to-end simulation using:

```bash
python3 run_transport_sim.py
```
This runs the simulation using the proposed normal-guided time-varying bias, displays the animation on screen, and saves the output artifacts to the `outputs/` directory.

To run the ablation study (which compares the baseline constant-bias approach directly against the proposed normal-guided approach), use:
```bash
python3 experiments.py
```

## Repository Structure

### Core Execution
- **`run_transport_sim.py`**: The main execution script. It defines the payload, optimizes formation slots, assigns scattered agents to those slots, and simulates Phase 1 (rendezvous/docking) and Phase 2 (transport) using the normal-guided approach. Displays and saves animations.
- **`experiments.py`**: An ablation study script. Runs both the baseline (constant bias, straight to slot) and proposed (time-varying, normal-guided) methods side-by-side from the same initial scatter, scoring them on interior intrusion and approach alignment.

### Swarm Dynamics & Control
- **`dynamics.py`**: The core finite-time swarm kinematic guidance model. Defines the differential equations (ODE right-hand side for `scipy.integrate.solve_ivp`), the sliding-mode finite-time term ($\zeta$), and the inter-agent coupling.
- **`approach.py`**: Defines the normal-guided, time-varying formation bias $p_i(t)$. Handles the construction of the standoff points and provides the global smoothstep schedule ($\alpha(t)$) used to transition agents from standoff to boundary.
- **`trajectory.py`**: Defines the smooth reference trajectory $r(t)$ for the Phase 2 transport leg. Generates an S-curve path using a clamped cubic spline, providing both the reference position and its exact analytic derivative.

### Geometry & Optimization
- **`geometry.py`**: Payload geometry utilities using `shapely`. Constructs example payloads (e.g., asymmetric L-shapes), computes centers of gravity, and handles boundary/containment math (including proper mitred normals at reflex vertices).
- **`optimization.py`**: Offline formation-offset optimization. Uses SLSQP to find an optimal set of grasp points on the payload boundary that satisfy a strict zero-net-offset (centering) constraint and inter-agent separation. Also includes a line-of-sight-aware Hungarian assignment to optimally pair scattered agents with target slots.

### Analysis & Output
- **`metrics.py`**: Quantitative diagnostics for the docking approach. Calculates the maximum penetration depth/fraction of interior intrusion and terminal approach alignment angles.
- **`visualization.py`**: Handles all plotting and animation using `matplotlib`. Generates the MP4 videos of the simulation, convergence plots over time, and the side-by-side docking ablation figures.

### Documentation
- **`Report.pdf`**: Theoretical report detailing the extension of the swarm kinematic model to include formation biases, proving finite-time centroid convergence and strictly bounded formation tracking errors.
- **`paper_notes.md`**: Technical notes structured for an academic paper (ICRA-style) detailing the method, the specific gap addressed in related work, theoretical properties, ablation results, and limitations.
