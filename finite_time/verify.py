"""Correctness checks that must pass before any figure is trusted.

These are deliberately separate from the figure scripts: they test the *model and
the integrator*, not the propositions. Run with ``python -m finite_time.verify``.
"""

import numpy as np
from scipy.integrate import solve_ivp

from . import common, formations, metrics, theory
from .integrate import simulate
from .model import SwarmConfig, rhs, zeta


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


# =========================================================================

def check_legacy_regression() -> bool:
    """The rewrite must change only what it was meant to change.

    Running with ``use_true_distance=False`` reproduces the bias-shifted
    denominator of the superseded script, so with the same seed, M and static
    hexagon the centroid trajectory must match its output. sigma(t) is
    denominator-independent (Prop. 1), so this really tests the rest of the
    right-hand side; the formation error is compared separately against a direct
    transcription of the old dynamics.
    """
    n, d, L = 6, 2, 5.0
    rng = np.random.default_rng(42)
    Z0 = (5.0 * rng.standard_normal((n, d))).ravel()

    P = np.zeros((n, d))
    for i in range(n):
        P[i] = [np.cos((i + 1) * np.pi / 3), np.sin((i + 1) * np.pi / 3)]
    P = L * (P - P.mean(axis=0))

    M_val, eps, nu, beta = 5.0, 1.0, 2, 0.5

    def old_dynamics(t, Z):
        """Verbatim transcription of the superseded script's right-hand side."""
        Zm = Z.reshape(n, d)
        c = Zm.mean(axis=0)
        r_val = np.array([t, 0.0])
        rd = np.array([1.0, 0.0])
        sig = c - r_val
        z = np.sign(sig) * np.abs(sig) ** beta
        B = Zm - P
        diff = B[:, None, :] - B[None, :, :]
        denom = np.linalg.norm(diff, axis=2) ** nu + eps
        Md = diff @ (M_val * np.eye(d)).T
        inter = Md / denom[:, :, None]
        np.fill_diagonal(inter[:, :, 0], 0)
        np.fill_diagonal(inter[:, :, 1], 0)
        return (-Zm - inter.sum(axis=1) / n + r_val + rd - z + P).ravel()

    # Stop before the first component reaches the sliding surface at t ~ 0.495.
    # The old script has no equivalent-control switch, so integrating it past that
    # point just measures how its solver chatters -- which is the artefact this
    # rewrite exists to remove, not something to reproduce.
    t_eval = np.linspace(0, 0.4, 201)
    ref = solve_ivp(old_dynamics, (0, 0.4), Z0, t_eval=t_eval,
                    rtol=1e-11, atol=1e-13, method="LSODA")

    # Use the old script's exact vertex ordering rather than formations.hexagon(),
    # which starts at angle 0 instead of pi/3. Both describe the same hexagon, but
    # they hand different slots to different agents; this check is about the
    # dynamics, not the shape convention.
    r_fn, rdot_fn = common.straight_reference()
    P_fn, Pdot_fn = formations.static_schedule(P)
    cfg = SwarmConfig(n=n, d=d, M=M_val * np.eye(d), nu=nu, eps=eps, beta=beta,
                      r_fn=r_fn, rdot_fn=rdot_fn, P_fn=P_fn, Pdot_fn=Pdot_fn,
                      use_true_distance=False)
    new = simulate(cfg, Z0, (0, 0.4), t_eval=t_eval, latch=False)

    err = np.abs(new.X.reshape(t_eval.size, -1) - ref.y.T).max()
    return _report("legacy regression", err < 1e-8, f"max |X_new - X_old| = {err:.2e}")


def check_true_distance_is_active() -> bool:
    """The denominator fix must actually change the trajectory.

    Guards against the fix being silently inert -- if these two agreed, the model
    change would not have taken effect.
    """
    Z0 = common.initial_state()
    a = simulate(common.nominal_config(), Z0, (0, 5), t_eval=np.linspace(0, 5, 201))
    b = simulate(common.nominal_config().with_(use_true_distance=False), Z0,
                 (0, 5), t_eval=np.linspace(0, 5, 201))
    gap = np.abs(a.X - b.X).max()
    return _report("true-distance denominator is live", gap > 1e-3,
                   f"max trajectory difference = {gap:.3e}")


def check_centroid_invariance() -> bool:
    """Prop. 1: sigma(t) must not depend on M, nu, eps or the formation."""
    Z0 = common.initial_state()
    t_eval = np.linspace(0, 6, 601)
    curves = []
    for M, nu, eps, Lf in [(0.0, 2, 1.0, 5.0), (10.0, 2, 1.0, 5.0),
                           (5.0, 3, 0.01, 2.0), (100.0, 1, 1.0, 10.0)]:
        cfg = common.nominal_config(M=M, nu=nu, eps=eps, Lf=Lf)
        res = simulate(cfg, Z0, (0, 6), t_eval=t_eval)
        curves.append(np.linalg.norm(metrics.centroid_error(t_eval, res.X, cfg), axis=1))
    spread = np.abs(np.array(curves) - curves[0]).max()
    return _report("centroid invariance (Prop. 1)", spread < 1e-9,
                   f"max spread over (M, nu, eps, L) = {spread:.2e}")


def check_b3_scalar() -> bool:
    """B3: the centroid floor under disturbance must match wbar^(1/beta).

    Checked on the isolated scalar channel at very tight tolerance, so the claim
    is validated independently of the full swarm simulation before it is used.
    """
    ok = True
    for beta in (0.3, 0.5, 0.7):
        for wbar in (1e-1, 1e-2, 1e-3):
            def f(t, y):
                return -y - np.sign(y) * np.abs(y) ** beta + wbar

            sol = solve_ivp(f, (0, 400.0), [5.0], rtol=1e-12, atol=1e-14,
                            method="LSODA", t_eval=np.linspace(300, 400, 200))
            measured = float(np.abs(sol.y[0]).max())
            predicted = theory.sigma_ball(wbar, beta)
            # Predicted is an upper bound: it drops the -sigma term.
            good = measured <= predicted * (1.0 + 1e-6)
            tight = measured >= 0.4 * predicted
            ok &= good and tight
            if not (good and tight):
                print(f"      beta={beta} wbar={wbar:g}: measured={measured:.3e} "
                      f"predicted={predicted:.3e}")
    return _report("B3 centroid ball (scalar channel)", ok,
                   "measured floor within [0.4x, 1.0x] of wbar^(1/beta) for all cases")


def check_high_dimension() -> bool:
    """d = 3 and a large swarm must run clean -- exercises the d-generic code paths."""
    ok = True
    cfg3 = common.nominal_config(n=8, d=3)
    Z3 = common.initial_state(n=8, d=3)
    r3 = simulate(cfg3, Z3, (0, 12), t_eval=np.linspace(0, 12, 201))
    ok &= np.isfinite(r3.X).all()

    cfg_big = common.nominal_config(n=200)
    Zb = common.initial_state(n=200)
    rb = simulate(cfg_big, Zb, (0, 12), t_eval=np.linspace(0, 12, 101))
    ok &= np.isfinite(rb.X).all()

    d3 = metrics.delta_norm(r3.t, r3.X, cfg3)[-1]
    db = metrics.delta_norm(rb.t, rb.X, cfg_big)[-1]
    return _report("d=3 and n=200 run clean", ok,
                   f"final ||delta||: d=3 -> {d3:.2e}, n=200 -> {db:.2e}")


def check_centering_condition() -> bool:
    """Every shipped formation must satisfy Eq. (4), including mid-morph."""
    ok = True
    worst = 0.0
    for name, fn in formations.SHAPES.items():
        worst = max(worst, float(np.abs(fn(7, 3.0).sum(axis=0)).max()))
    shapes = [formations.hexagon(6, 5.0), formations.line(6, 5.0),
              formations.vee(6, 5.0), formations.grid(6, 5.0)]
    P_fn, Pdot_fn = formations.morph_schedule(shapes, [5.0, 10.0, 15.0], 3.0)
    for t in np.linspace(0, 20, 401):
        worst = max(worst, float(np.abs(P_fn(t).sum(axis=0)).max()))
        worst = max(worst, float(np.abs(Pdot_fn(t).sum(axis=0)).max()))
    ok = worst < 1e-10
    return _report("centering condition (Eq. 4)", ok,
                   f"max |sum_i p_i| and |sum_i pdot_i| = {worst:.2e}")


def check_centering_offset() -> bool:
    """What Eq. (4) buys: violating it parks the centroid at a predictable offset.

    This was the F5 centering-ablation figure. With sum_i p_i = n pbar the centroid
    dynamics pick up the mean bias, so sigma settles where sigma + sign(sigma)
    |sigma|^beta = pbar rather than at zero. The shapes are shifted before the
    schedule is built, so Eq. (4) is broken throughout the morphs and not merely at
    their endpoints, while sum_i pdot_i = 0 still holds.
    """
    pbar = np.array([0.0, 2.0])
    L = 4.0
    shapes = [formations.break_centering(s, pbar)
              for s in (formations.hexagon(common.N, L), formations.line(common.N, L),
                        formations.vee(common.N, L), formations.grid(common.N, L))]
    P_fn, Pdot_fn = formations.morph_schedule(shapes, [4.0, 8.0, 12.0], 3.0)
    cfg = common.nominal_config().with_(P_fn=P_fn, Pdot_fn=Pdot_fn)
    t = np.linspace(0, 18.0, 1801)
    res = simulate(cfg, common.initial_state(), (0, t[-1]), t_eval=t)
    sn = np.linalg.norm(metrics.centroid_error(t, res.X, cfg), axis=1)
    pred = float(np.linalg.norm(theory.sigma_offset_equilibrium(pbar, common.BETA)))
    achieved = metrics.ultimate_value(t, sn, stat="mean")
    err = abs(achieved - pred) / pred
    return _report("centering ablation: predicted centroid offset", err < 1e-3,
                   f"settles at {achieved:.5f} vs predicted {pred:.5f} "
                   f"(rel. err {err:.1e})")


def check_pdot_analytic() -> bool:
    """The analytic Pdot must match a finite difference of P -- it is a feedforward term."""
    shapes = [formations.hexagon(6, 5.0), formations.vee(6, 5.0)]
    P_fn, Pdot_fn = formations.morph_schedule(shapes, [4.0], 3.0)
    h = 1e-6
    worst = 0.0
    for t in np.linspace(4.2, 6.8, 60):
        fd = (P_fn(t + h) - P_fn(t - h)) / (2 * h)
        worst = max(worst, float(np.abs(fd - Pdot_fn(t)).max()))
    return _report("analytic Pdot", worst < 1e-5, f"max |Pdot - dP/dt| = {worst:.2e}")


def check_zeta_continuity() -> bool:
    """The boundary-layer variant must be continuous at the layer edge."""
    phi, beta = 1e-3, 0.5
    lo = zeta(np.array([phi * (1 - 1e-9)]), beta, "sat", phi)[0]
    hi = zeta(np.array([phi * (1 + 1e-9)]), beta, "sat", phi)[0]
    return _report("sat() boundary-layer continuity", abs(lo - hi) < 1e-9,
                   f"jump at |sigma| = phi is {abs(lo - hi):.2e}")


# =========================================================================
# Checks backing the t0-anchoring of the report (Lemma 1 and its consequences)
# =========================================================================

def check_lemma1_identity() -> bool:
    """Lemma 1, Eq. (12): sum_i delta_i = n*sigma under the centering condition."""
    cfg = common.nominal_config()
    t = np.linspace(0, 20.0, 1001)
    res = simulate(cfg, common.initial_state(), (0, 20.0), t_eval=t)
    D = metrics.formation_errors(t, res.X, cfg)
    sig = metrics.centroid_error(t, res.X, cfg)
    err = float(np.abs(D.sum(axis=1) - cfg.n * sig).max())
    return _report("Lemma 1 identity (Eq. 12)", err < 1e-9,
                   f"max |sum_i delta_i - n*sigma| = {err:.2e}")


def check_zeta_dissipative() -> bool:
    """Lemma 1, Eq. (13): the term Prop. 3 used to assume away is never positive.

    Checked along both a nominal and a disturbed trajectory, since the whole point
    is that the sign holds without any hypothesis on sigma.
    """
    t = np.linspace(0, 20.0, 1001)
    base = common.nominal_config()
    worst = -np.inf
    for w in (None, common.w_adversarial(0.05, base.P_fn, base.r_fn)):
        cfg = base.with_(w_fn=w)
        if w is not None:
            cfg = cfg.with_(zeta_mode="sat", phi=1e-7)
        res = simulate(cfg, common.initial_state(), (0, 20.0), t_eval=t,
                       rtol=1e-8, atol=1e-10)
        sig = metrics.centroid_error(t, res.X, cfg)
        worst = max(worst, float(theory.zeta_dissipation(sig, cfg.beta, cfg.n).max()))
    return _report("zeta dissipativity (Eq. 13)", worst <= 0.0,
                   f"max over nominal and disturbed runs = {worst:+.2e}")


def check_prop2_from_t0() -> bool:
    """Prop. 2, Eq. (26): the nominal bound holds from t0, not merely from tau_c."""
    cfg = common.nominal_config()
    t = np.linspace(0, 20.0, 2001)
    worst = 0.0
    rng = np.random.default_rng(3)
    for k in range(12):
        X0 = (10.0 ** rng.uniform(-1, 2)) * rng.standard_normal((cfg.n, cfg.d))
        res = simulate(cfg, X0.ravel(), (0, 20.0), t_eval=t)
        dn = metrics.delta_norm(t, res.X, cfg)
        worst = max(worst, float((dn / theory.delta_envelope(t, dn[0], 0.0)).max()))
    return _report("Prop. 2 from t0 (Eq. 26)", worst <= 1.0 + 1e-9,
                   f"max ||delta||/envelope over 12 ICs = {worst:.6f}")


def check_prop3_from_t0() -> bool:
    """Prop. 3, Eq. (39): the ISS bound holds from t0 under every disturbance class."""
    base = common.nominal_config()
    t = np.linspace(0, 20.0, 1501)
    wbar = 0.05
    Z0 = common.initial_state()
    d0 = float(np.linalg.norm((Z0.reshape(base.n, base.d) - base.P_fn(0.0)
                               - np.asarray(base.r_fn(0.0))).ravel()))
    env = theory.iss_envelope(t, d0, 0.0, base.n, wbar)
    worst = 0.0
    for w in (common.w_common(wbar, base.d),
              common.w_random(wbar, base.n, base.d, seed=5, hold=0.2),
              common.w_sine(wbar, base.n, base.d),
              common.w_adversarial(wbar, base.P_fn, base.r_fn)):
        cfg = base.with_(w_fn=w, zeta_mode="sat", phi=1e-7)
        res = simulate(cfg, Z0, (0, 20.0), t_eval=t, rtol=1e-8, atol=1e-10)
        worst = max(worst, float((metrics.delta_norm(t, res.X, cfg) / env).max()))
    return _report("Prop. 3 from t0 (Eq. 39)", worst <= 1.0 + 1e-9,
                   f"max ||delta||/envelope over 4 classes = {worst:.6f}")


def check_T_sigma() -> bool:
    """Prop. 4(iii), Eq. (56): finite entry into the (wbar/mu)^(1/beta) box.

    Uses the isotropic common-mode push, so each scalar channel sees exactly
    wbar/sqrt(d) and the predicted radius is tight rather than loose.
    """
    beta, wbar = common.BETA, 0.05
    cfg = common.nominal_config(beta=beta).with_(
        w_fn=common.w_common(wbar, common.D))
    t = np.linspace(0, 40.0, 2001)
    Z0 = common.initial_state()
    res = simulate(cfg, Z0, (0, 40.0), t_eval=t, rtol=1e-9, atol=1e-11)
    sn = np.abs(metrics.centroid_error(t, res.X, cfg)).max(axis=1)
    sigma0 = Z0.reshape(cfg.n, cfg.d).mean(axis=0) - np.asarray(cfg.r_fn(0.0))
    ok = True
    for mu in (0.3, 0.6, 0.9):
        radius = theory.sigma_ball_mu(wbar / np.sqrt(common.D), beta, mu)
        pred = theory.T_sigma(sigma0, beta, mu)
        meas = metrics.settling_time(t, sn, radius)
        ok &= bool(meas <= pred)
        if meas > pred:
            print(f"      mu={mu}: measured {meas:.3f}s > predicted {pred:.3f}s")
    return _report("Prop. 4(iii) entrance time (Eq. 56)", ok,
                   "measured entry <= T_sigma(mu) for mu in {0.3, 0.6, 0.9}")


def check_b_iss_invariance() -> bool:
    """Prop. 3, Eq. (40): B_ISS is forward invariant -- started inside, never left.

    This was F4 panel (b) before the figure was cut to three panels. The claim is
    worth keeping, so it lives here instead: eight starts strictly inside the ball,
    driven by the adversarial disturbance that aligns itself with delta, and the
    trajectory must never exceed sqrt(n) wbar.
    """
    base = common.nominal_config()
    wbar = 0.05
    ball = theory.B_iss(base.n, wbar)
    cfg = base.with_(w_fn=common.w_adversarial(wbar, base.P_fn, base.r_fn),
                     zeta_mode="sat", phi=1e-7)
    t = np.linspace(0, 20.0, 1201)
    rng = np.random.default_rng(17)
    worst = 0.0
    for _ in range(8):
        seed = rng.standard_normal((base.n, base.d))
        seed -= seed.mean(axis=0)
        seed *= (0.85 * ball * rng.uniform(0.2, 1.0)) / np.linalg.norm(seed)
        X0 = base.P_fn(0.0) + np.asarray(base.r_fn(0.0)) + seed
        res = simulate(cfg, X0.ravel(), (0, t[-1]), t_eval=t, rtol=1e-8, atol=1e-10)
        worst = max(worst, float(metrics.delta_norm(t, res.X, cfg).max() / ball))
    return _report("Prop. 3 forward invariance of B_ISS (Eq. 40)",
                   worst <= 1.0 + 1e-9,
                   f"worst peak / ball over 8 interior starts = {worst:.6f}")


CHECKS = [check_legacy_regression, check_true_distance_is_active,
          check_centroid_invariance, check_b3_scalar, check_high_dimension,
          check_centering_condition, check_centering_offset, check_pdot_analytic,
          check_zeta_continuity,
          check_lemma1_identity, check_zeta_dissipative, check_prop2_from_t0,
          check_prop3_from_t0, check_b_iss_invariance, check_T_sigma]


def main() -> int:
    print("Verification")
    results = [c() for c in CHECKS]
    n_ok = sum(results)
    print(f"  {n_ok}/{len(results)} checks passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
