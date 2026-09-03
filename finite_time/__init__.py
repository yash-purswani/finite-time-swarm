"""Numerical validation suite for the finite-time swarm report.

Modules
-------
model       Eq. (8) right-hand side and the F7 baseline laws
theory      every analytical bound, one function per equation
integrate   tight-tolerance solve with sliding-surface event detection
metrics     error signals and measured quantities
formations  shapes, Hungarian re-labelling, smooth morph schedules
common      nominal scenario defaults and disturbance generators
style       IEEE two-column figure styling
"""

from . import common, formations, integrate, metrics, model, theory  # noqa: F401
