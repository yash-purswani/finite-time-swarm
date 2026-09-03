# Numerical suite for the finite-time swarm report

Validates every proposition and corollary of `finite-time_report.pdf`, plus one
patch to Prop. 3 (B3, below), and emits IEEE two-column figures as 300 dpi PNGs
(plus one MP4 for the morphing sequence, which has no still).

```
python run_all.py               # verification, then all seven figures -> figures/
python run_all.py f2 f4         # just those
python run_all.py --verify      # correctness checks only
```

## What changed relative to `finite_time_validation.py`

That script validated an **earlier draft**. Three things had to be fixed before
any figure meant anything.

1. **The denominator.** Eq. (8) divides by the *true physical distance*
   `||x_i - x_j||^nu + eps`. The old script used the bias-shifted distance
   `||(x_i-p_i) - (x_j-p_j)||^nu + eps`, so it was integrating a different vector
   field than the one the propositions are proved for. The numerator was already
   right — only `norms` changes. Switching it moves the trajectory by 0.66 in the
   nominal scenario, so it was not a cosmetic difference.
2. **`compute_rho()` is gone.** It implemented a formation-error ball from the
   superseded draft. The current Prop. 2 proves `delta -> 0` exponentially, and
   there is no `rho`.
3. **Solver tolerances.** The old run used `solve_ivp` defaults (rtol 1e-3) on a
   non-smooth field. The resulting 1e-5 error floor — the reason the old plots
   needed a moving average — was integrator noise, not a property of the system.
   At the tolerances used here `||delta||` decays cleanly to 1e-12.

Also added, none of which existed before: the `pdot_i` feedforward and
time-varying `p_i(t)`, matrix-valued `M`, disturbances `w_i`, `d`-generic code
(the old diagonal removal hardcoded `d = 2`), zero-order-hold and actuator
saturation.

## Two theory patches, both now in the report

**Lemma 1 — everything is anchored at `t0`, not `tau_c`.** The report used to
assume `zeta == 0` for `t >= tau_c`. Under the centering condition
`sum_i delta_i = n*sigma`, so that term is worth
`-n sum_s |sigma_s|^(1+beta) <= 0` — sign-definite negative. Retaining it only
makes `V_delta` decrease faster, so no centroid hypothesis is needed anywhere:
Props. 2 and 3 and Cors. 1 and 2 all hold from `t0`. This matters because under
disturbance `tau_c` does not exist (see below), so anchoring there would reference
an undefined quantity in exactly the regime those results describe. `tau_c` now
appears only in Prop. 1, where it means something.

A side effect worth knowing: Prop. 2 no longer invokes Prop. 1. Centroid and
formation are two independent guarantees on two decoupled channels.

**Prop. 4 — the centroid is super-linearly robust.** Re-deriving the centroid
channel with the mean disturbance gives a forward-invariant box

>   `|sigma_s| <= wbar^(1/beta)`,  entered in finite time `T_sigma(mu)`

— `1e-2` at `beta = 0.5, wbar = 0.1`, against the formation channel's linear
`sqrt(n) wbar ~ 0.245` for `n = 6`. The tight radius is invariant and approached
asymptotically; any strictly larger radius `(wbar/mu)^(1/beta)` is reached in
finite time, which is what `theory.T_sigma` computes. Validated independently on
the isolated scalar channel before any figure uses it.

Props. 1 and 2 are otherwise validated exactly as written — no re-derivation of
`tau_c`, no tightening of the exponential rate.

## Figures

| | Claim | Headline measurement |
|---|---|---|
| `f1_centroid` | Prop. 1 | arrival at or before Eq. (25) for every `beta`; `sigma(t)` identical to 2e-10 across six `(M, nu, eps, shape, L)` settings |
| `f2_formation` | Prop. 2 | envelope (from `t0`) never exceeded; min rate 1.000 over `lambda_min(M)` in [0, 100] and over the `(nu, eps)` grid |
| `f3_docking` | Cor. 1, Cor. 2 | no violation after `T_dock`/`T_enter`, both anchored at `t0`; 100 ICs all below the predicted line |
| `f4_iss` | Prop. 3, Prop. 4 | three panels: Eq. (39) envelope holds from `t0` for all four disturbance classes; the `sqrt(n) wbar` radius is tight only under adversarial alignment; centroid floor follows `wbar^(1/beta)`. Forward invariance of `B_ISS` moved to `verify.py` when the figure was cut to three panels |
| `f5_morphing.mp4` | time-varying `p(t)` | *animation only*, no still (17 s at 2x): hexagon -> line -> V -> grid with `||sigma||` and `||delta||` drawn live. `sigma` holds to 4e-14 through three morphs and the `pdot` feedforward makes a morph exactly free. Both necessity ablations moved to `verify.py` / the printed sweep |
| `f6_scale` | scale and hardware | `tau_c` invariant over `n = 4..200` and `d = 2, 3`; chatter band follows `(dt/2)^(1/(1-beta))` |
| `f7_baselines` | comparison | the centroid deadline moves by 1e-11 s across four formations, against 0.29 s for a per-agent law |

## Reading the code

| module | holds |
|---|---|
| `model.py` | Eq. (8), and the per-agent baseline law |
| `theory.py` | every analytical bound, one function per equation — the single place a formula appears |
| `integrate.py` | tight-tolerance solve, sliding-surface event detection, ZOH stepper |
| `metrics.py` | error signals and measured quantities |
| `formations.py` | shapes, Hungarian re-labelling, smooth morph schedules |
| `common.py` | nominal scenario, disturbance generators |
| `style.py` | IEEE two-column styling; `save()` writes 300 dpi PNG |
| `verify.py` | correctness checks — run before trusting a figure, including Lemma 1, the `t0` anchoring of Props. 2 and 3, and `T_sigma` |

## Two numerical points a reviewer may ask about

**Why the sliding surface is detected at `|sigma_s| = 1e-12` rather than at a sign
change.** Finite-time arrival is an exact-arithmetic statement. In floating point
`sigma_s` approaches zero without ever changing sign, so a sign-change event never
fires and the solver instead grinds its step size down forever at the crossing. The
threshold costs at most the remaining travel time from 1e-12 to zero, about 2e-6 s,
and it biases `tau_c` *downward* — it cannot manufacture agreement with Eq. (25).
Once a component arrives it is latched to its equivalent control, which is what
removes the chattering that would otherwise dominate `||delta(t)||`.

**Where boundary layers are used, and where they are not.** Disturbed runs cannot
latch, because under disturbance `sigma` does not stay at zero once it arrives. For
the classes whose mean disturbance nearly cancels — i.i.d. directions average to
`O(wbar/sqrt(n))`, adversarial can cancel outright — `sigma` still reaches zero and
the ideal `sign(.)` term stalls the solver. Those runs use a `1e-7` layer, four
decades below anything they measure. The panel that measures the centroid floor
itself (`f4` panel c) keeps the exact `sign(.)`: a constant common-mode push holds
`sigma` at `wbar^(1/beta) > 0`, so nothing chatters there.
