"""
trajectory.py
===============================================================================
Smooth reference trajectory r(t) for the Phase-2 transport leg.

Built from a handful of spatial waypoints (start -> S-curve control points ->
drop-off) using a clamped cubic spline in *time*, so both r(t) and its exact
analytic derivative rdot(t) (needed by the dynamics, see `dynamics.py`) are
available directly from `CubicSpline.derivative()` -- no finite differencing.
Clamped (zero-velocity) boundary conditions give the classic smooth
accelerate/cruise/decelerate "S-curve" time profile at the endpoints.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


class ReferenceTrajectory:
    """r(t) for t in [0, T], with exact r(t) and rdot(t)."""

    def __init__(self, waypoints: np.ndarray, T: float) -> None:
        """
        Parameters
        ----------
        waypoints : (K, 2) ndarray
            Spatial waypoints, in order, from the payload's docked CoG to the
            drop-off point. K >= 2.
        T : float
            Total duration to traverse all waypoints.
        """
        K = waypoints.shape[0]
        times = np.linspace(0.0, T, K)
        self.T = T
        self._spline = CubicSpline(times, waypoints, bc_type="clamped")
        self._dspline = self._spline.derivative()

    def r(self, t: float) -> np.ndarray:
        tc = np.clip(t, 0.0, self.T)
        return np.asarray(self._spline(tc), dtype=float)

    def r_dot(self, t: float) -> np.ndarray:
        # Outside [0, T] the reference is held fixed, so its velocity is zero
        # (avoids feeding a nonzero rdot from spline extrapolation beyond the
        # segment the swarm is actually meant to be tracking).
        if t < 0.0 or t > self.T:
            return np.zeros(2)
        return np.asarray(self._dspline(t), dtype=float)

    def as_callables(self) -> Tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
        return self.r, self.r_dot


def s_curve_waypoints(start: np.ndarray, end: np.ndarray, bow: float = 1.5, n_control: int = 4) -> np.ndarray:
    """Generate waypoints for a gentle S-curve from `start` to `end`: a
    straight line displaced by a sinusoidal lateral "bow" perpendicular to the
    direct path, sampled at `n_control` extra interior points."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    unit = direction / length if length > 1e-9 else np.array([1.0, 0.0])
    normal = np.array([-unit[1], unit[0]])

    s = np.linspace(0.0, 1.0, n_control + 2)  # includes endpoints
    lateral = bow * np.sin(2 * np.pi * s)      # one full S wiggle, zero at both ends
    pts = start[None, :] + s[:, None] * direction[None, :] + lateral[:, None] * normal[None, :]
    return pts
