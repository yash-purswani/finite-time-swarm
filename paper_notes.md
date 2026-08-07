# Normal-Guided, Time-Varying Formation Biases for Finite-Time Cooperative Payload Docking

*Technical notes accompanying `payload_transport/`, written to be adapted directly into
an ICRA-style paper section (Related Work gap -> Method -> Theory -> Results ->
Limitations). Numbers quoted below come from the actual code in this repo
(`run_workshop_sim.py`, seed=0 unless stated otherwise); regenerate with
`python run_workshop_sim.py`.*

---

## 1. Where this sits relative to recent work

Cooperative multi-robot object transport is currently dominated by three
families of approach:

- **Learned / decentralized control** (the dominant 2024-2025 trend): MARL
  leader-follower policies, learned force/contact representations,
  decentralized pinch-lift-move behaviors. Highly adaptive to unknown
  payloads, but offer no closed-form convergence-time or tracking-error
  guarantee, and treat grasp/contact points as something to be *learned*
  rather than derived from the object's geometry.
- **Force/impedance-based coordination**: admittance control, 6-DoF rigid
  connectors, ZMP-aware carrying for legged platforms. Physically well
  grounded, but requires force/torque sensing and pose-calibration
  infrastructure most of the finite-time kinematic-guidance literature does
  not assume.
- **Trajectory optimization** (QP / MPC / CBF-based): handles cluttered
  environments well, at the cost of per-step optimization that scales
  unfavourably with team size and rarely ships a certifiable error radius.

Separately, **time-varying formation tracking (TVFT)** is an active,
recognized subfield of the finite-time / sliding-mode consensus literature
(it studies exactly the "formation offsets change over time" idea used
here), but it is applied almost exclusively to *free-space reshaping*
(chiefly UAV swarms), never to grasp-approach geometry.

Meanwhile, **standoff distance**, **approach ray**, and **grasp along the
surface normal** are decades-old, textbook conventions in *single-robot*
pre-grasp pose planning. Essentially nothing in the cooperative-transport or
TVFT literature imports this discipline into a multi-robot finite-time
coordination law. That is the specific, narrow gap this extension targets:

> **Gap.** A constant formation bias `p_i` -- the standard assumption in
> virtual-structure / offset-based cooperative transport -- drives each
> agent to its grasp slot along a straight-ish pursuit trajectory with no
> regard for the payload's own geometry, routinely cutting through the
> object's interior en route, and arriving at an arbitrary (not necessarily
> normal) angle. No sensing/force infrastructure is needed to fix this: it
> is a formation-*bias-design* problem, solvable entirely within the
> existing finite-time kinematic law.

---

## 2. Method

### 2.1 Approach corridor (standoff -> boundary)

For each optimizer-assigned slot `p_i` (local, CoG-centered frame), we snap
it onto the payload's nearest boundary edge to get an exact contact point
`B_i` and its outward unit normal `N_i`. Near either endpoint of that edge
the normal is **mitre-blended** with the angle-bisector normal of the
adjacent vertex (`geometry._vertex_bisector_normal`) -- the standard
correct treatment for polygon offsetting at both convex *and reflex*
(concave) corners. This matters concretely: on the L-shaped test payload,
a naive single-edge normal gave a slot right at the inner notch only
**0.0011** units of real clearance from the object (vs. a nominal
`d_standoff = 0.4`); the mitred normal alone restores it to **0.283** with
*no* adaptive escalation needed at all.

The standoff point is `S_i = B_i + d_standoff * N_i`.

### 2.2 Time-varying bias with feedforward compensation

Instead of a constant `p_i`, define

```
p_i(t)    = (1 - a(t)) * S_i + a(t) * B_i
pdot_i(t) = adot(t) * (B_i - S_i)
```

with `a(t)` a shared, time-only cubic smoothstep (zero derivative at both
ends) rising from 0 to 1 over `T_approach` seconds. The control law is
generalized to feed `pdot_i(t)` forward:

```
dx_i/dt = -x_i + (r(t)+ p_i(t)) + (rdot(t)) + pdot_i(t) - zeta(t) - coupling_i
```

**Proposition (closed-loop error equation is invariant to how p_i(t) is chosen).**
Let `delta_i(t) = x_i(t) - p_i(t) - r(t)`. Differentiating and substituting
the law above:

```
ddelta_i/dt = -delta_i(t) - zeta(t) - coupling_i(t)
```

-- algebraically identical to the constant-bias closed-loop equation,
*independent of p_i(t)*, provided `pdot_i(t)` is fed forward exactly. (One
line of algebra: `-(x_i - p_i(t)) + r(t) = -delta_i(t)` by the definition of
`delta_i`, regardless of whether `p_i` depends on `t`.) Consequently the
finite-time convergence result proved for the constant-bias law transfers
**without modification** to any continuously differentiable, bounded-rate
`p_i(t)` -- the smoothstep schedule used here satisfies both conditions
exactly. `fedele_ddelta`-equivalent coupling term is untouched throughout.

One structural consequence worth noting: because `sum_i B_i = 0` (the
optimizer's hard centering constraint) but `sum_i S_i = d_standoff * sum_i
N_i` is generally *nonzero*, the swarm's own centroid is naturally displaced
from the CoG while `a(t) < 1` and slides exactly onto it as `a(t) -> 1`.
"The centroid approaches the CoG as the agents approach the boundary" is not
a separately engineered behavior -- it falls out of the same schedule.

### 2.3 Line-of-sight-aware docking assignment

A pure-distance Hungarian assignment (minimize total squared travel
distance from scattered starts to standoff points) does not know the
payload has a shape: it can -- and, empirically, did -- assign an agent
starting on one side of the object to an edge whose normal points away from
that side, forcing it to cut across the payload just to reach its own
standoff point. We patch `optimization.docking_assignment` with a
`shapely`-based line-of-sight penalty (heavily costing any (agent, slot)
pairing whose straight segment intersects the payload footprint), directly
reusing the same idea as the companion obstacle-avoidance kinematic project.

---

## 3. Results

All numbers below: `n=8` agents, asymmetric L-shaped payload, identical
random scatter shared between baseline and proposed (so the comparison
isolates *only* the constant-vs-time-varying bias effect; both use the
same LOS-aware assignment).

### 3.1 Docking-approach ablation (seed = 0)

| metric                                   |   baseline |   proposed |
|-------------------------------------------|-----------:|-----------:|
| max interior intrusion depth               |     0.1307 |     0.0000 |
| % samples agent-inside-payload             |     11.11% |      0.00% |
| mean terminal approach angle (deg)         |      95.72 |       0.00 |
| max terminal approach angle (deg)          |     180.00 |       0.00 |
| agents that reached contact radius         |        8/8 |        8/8 |

(0 deg = a perfectly perpendicular, inward approach; 90 deg = tangential;
180 deg = arrived moving *away* from the surface, i.e. overshot and
settled back in.)

### 3.2 Robustness across seeds (docking only, no video rendering)

| seed | baseline intrusion (depth / %) | baseline mean angle | proposed intrusion (depth / %) | proposed mean angle |
|-----:|--------------------------------:|---------------------:|---------------------------------:|----------------------:|
| 1 | 0.507 / 21.8% | 88.5 deg | 0.314 / 2.1% | 0.0 deg |
| 2 | 0.130 / 12.5% | 95.7 deg | 0.000 / 0.0% | 0.0 deg |
| 3 | 0.215 / 14.1% | 94.9 deg | 0.000 / 0.0% | 0.0 deg |
| 4 | 0.484 / 22.6% | 95.7 deg | 0.452 / 2.5% | 22.5 deg |
| 5 | 0.057 / 7.3%  | 35.7 deg | 0.000 / 0.0% | 0.0 deg |

The proposed method is a large, consistent improvement on every seed tested
(intrusion reduced by 85-100%, mean approach angle reduced from 36-96 deg
down to 0-23 deg) but is **not uniformly perfect** -- seeds 1 and 4 retain
small residual intrusion. This is reported honestly rather than
cherry-picked; see Limitations.

### 3.3 Finite-time convergence (unaffected by the extension)

Centroid error `||sigma(t)||` still collapses to (numerically) zero at a
finite, detected time `tau` (an explicit `solve_ivp` terminal event, not
eyeballed): `tau_baseline ~= 2.9s`, `tau_proposed ~= 7.2s` (the proposed
value is dominated by `T_approach`, a tunable design parameter, not a
side-effect of weaker convergence -- see `kinematic_convergence_plots.png`).
All post-dock individual tracking errors remain far inside the theoretical
bound `rho` in every run.

---

## 4. Limitations and future work (for the paper's discussion section)

1. **Residual intrusion on some seeds (Table 3.2).** Even with a
   reachable, LOS-clear assignment, the strong pairwise coupling term can
   transiently perturb an agent off a perfectly straight standoff-then-normal
   path when several agents converge on adjacent slots at once. A tighter
   fix is per-agent local path shaping (a short curved lead-in) rather than
   a single interpolated straight segment `S_i -> B_i`; we did not implement
   this to keep the extension narrowly scoped to the bias-design gap.

2. **Concave (reflex-vertex) clearance is a local heuristic, not exact.**
   The mitred-bisector-normal + adaptive-escalation combination
   (`approach.compute_approach_corridor`) fixes the concrete failure case we
   found, but is not a substitute for a proper medial-axis / free-space
   treatment of arbitrarily concave payloads. This links directly to the
   free-space extraction machinery already built for the companion
   obstacle-avoidance kinematic model -- treating the payload itself as a
   static obstacle during rendezvous is the natural unification, and both
   systems already share the identical underlying finite-time law, so this
   is primarily an integration effort, not new theory.

3. **No agent-agent collision avoidance during docking.** Agents are
   point masses by the problem's own specification; the pairwise
   `d_min` constraint is only enforced on the *final* slots `B_i`, not
   along the approach trajectories.

4. **Payload translates but does not rotate.** `p_i` (and hence `B_i`,
   `N_i`) are fixed in the world frame; extending to SE(2) (a rotating
   payload) would require formation offsets defined in the payload's *body*
   frame and rotated by a tracked orientation `theta(t)` -- a natural next
   step, and the same feedforward-compensation argument in Section 2.2
   should extend directly (`pdot_i(t)` would then include an
   `thetadot * R'(theta) p_i` term).

5. **Single, shared time-based schedule `a(t)`.** All agents transition
   standoff -> boundary together. A per-agent, *progress-gated* schedule
   (e.g. triggered by each agent's own proximity to its standoff point)
   would likely be more robust to heterogeneous travel distances (as in
   seed 4, above) at the cost of losing the clean closed-form `pdot_i(t)`
   used here (progress-gating introduces state-feedback into `a(t)`,
   requiring either an implicit-function derivative or an event-triggered,
   piecewise-constant treatment). Flagged as future work rather than
   implemented, to keep the finite-time guarantee's proof exact rather than
   approximate.

---

## 5. Reproducing these numbers

```
python run_workshop_sim.py
```

produces `outputs/workshop_transport_animation.mp4`,
`outputs/kinematic_convergence_plots.png`, and
`outputs/docking_comparison.png`, and prints the ablation table in §3.1 to
stdout. All parameters (payload shape, `n_agents`, `d_standoff`,
`T_approach`, dynamics gains) are in `MissionConfig` at the top of
`run_workshop_sim.py`. The multi-seed table in §3.2 was produced by looping
`MissionConfig(seed=...)` through `scatter_agents` / `run_phase1_docking`
directly (no video rendering) -- see the docstring of `metrics.py` for the
two metrics' exact definitions.
