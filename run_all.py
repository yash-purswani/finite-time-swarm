"""Regenerate every figure in the numerical section.

    python run_all.py            # verification, then all seven figures
    python run_all.py f1 f4      # just those
    python run_all.py --verify   # verification only

Deterministic: every experiment seeds its own generators, so re-running
reproduces the figures and the printed numbers exactly.
"""

import sys
import time

EXPERIMENTS = [
    ("f1", "exp1_centroid", "Prop. 1  finite-time centroid convergence"),
    ("f2", "exp2_formation", "Prop. 2  global exponential formation convergence"),
    ("f3", "exp3_docking", "Cor. 1/2 deterministic docking and entrance times"),
    ("f4", "exp4_iss", "Prop. 3  ISS under bounded disturbance, and B3"),
    ("f5", "exp5_morphing", "time-varying formations and necessity ablations"),
    ("f6", "exp6_scale", "scale, dimension, discretization"),
    ("f7", "exp7_baselines", "comparison against the alternatives"),
]


def main(argv) -> int:
    only = [a.lower() for a in argv if not a.startswith("-")]
    verify_only = "--verify" in argv
    skip_verify = "--no-verify" in argv

    if not skip_verify:
        from finite_time.verify import main as verify_main
        if verify_main() != 0:
            print("\nVerification failed -- not generating figures.")
            return 1
        print()
    if verify_only:
        return 0

    import importlib
    t_start = time.time()
    failures = []
    for key, module, blurb in EXPERIMENTS:
        if only and key not in only:
            continue
        print(f"{key.upper()}: {blurb}")
        t0 = time.time()
        try:
            importlib.import_module(module).main()
        except Exception as exc:              # keep going; report at the end
            failures.append((key, exc))
            print(f"    FAILED: {type(exc).__name__}: {exc}")
        print(f"    [{time.time() - t0:.1f}s]\n")

    print(f"Total {time.time() - t_start:.1f}s")
    if failures:
        print("Failed: " + ", ".join(k for k, _ in failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
