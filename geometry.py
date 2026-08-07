"""
geometry.py
===============================================================================
Payload geometry utilities.

A payload is represented as an arbitrary 2D polygon. This module handles:
  - constructing example (asymmetric) payload shapes,
  - computing the polygon's center of gravity (CoG) via `shapely`,
  - re-expressing the polygon in a CoG-centered *local/body frame*, which is
    the frame every downstream formation-offset p_i lives in,
  - continuous "how far is this point from the boundary / is it contained"
    helpers used as the objective and constraints of the offline formation
    optimizer in `optimization.py`.

All of these are pure geometry: nothing here depends on the swarm dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from shapely.geometry import Point, Polygon


@dataclass
class Payload:
    """A rigid 2D polygonal payload.

    Attributes
    ----------
    local_vertices : (V, 2) ndarray
        Polygon vertices expressed in the CoG-centered *local/body frame*,
        i.e. ``world_vertices - cog0 == local_vertices``. This is the frame
        every formation offset p_i is computed and expressed in.
    cog0 : (2,) ndarray
        The payload's center of gravity in world coordinates at t = 0.
    """

    local_vertices: np.ndarray
    cog0: np.ndarray

    @property
    def polygon_local(self) -> Polygon:
        """Shapely polygon in the CoG-centered local frame."""
        return Polygon(self.local_vertices)

    def world_vertices(self, centroid_now: np.ndarray) -> np.ndarray:
        """Vertices translated so the payload's CoG sits at `centroid_now`.

        The payload only *translates* with the swarm centroid (per the
        transport-mission spec) -- it does not additionally rotate, since the
        formation offsets p_i are themselves fixed in the world frame.
        """
        return self.local_vertices + centroid_now


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """Center of gravity (area centroid) of a simple polygon, via shapely."""
    return np.array(Polygon(vertices).centroid.coords[0])


def make_payload(vertices_world: np.ndarray) -> Payload:
    """Build a Payload from world-frame vertices, auto-centering on its CoG."""
    cog0 = polygon_centroid(vertices_world)
    return Payload(local_vertices=vertices_world - cog0, cog0=cog0)


# =============================================================================
# EXAMPLE PAYLOAD SHAPES
# =============================================================================
def make_l_shape(arm_long: float = 3.0, arm_short: float = 1.8, thickness: float = 1.0) -> np.ndarray:
    """An asymmetric L-shaped crate (world-frame vertices, CCW)."""
    a, b, t = arm_long, arm_short, thickness
    return np.array([
        [0.0, 0.0],
        [a,   0.0],
        [a,   t],
        [t,   t],
        [t,   b],
        [0.0, b],
    ])


def make_t_shape(width: float = 3.2, height: float = 2.4, stem_width: float = 1.0) -> np.ndarray:
    """A T-shaped table-like payload (world-frame vertices, CCW), stem down."""
    w, h, sw = width, height, stem_width
    top_h = h * 0.4
    x0 = (w - sw) / 2.0
    x1 = x0 + sw
    # Trace the outline as a single simple (non-self-intersecting) loop:
    # up the stem's right side, along the bar's underside, up and across the
    # bar's top, back down its left side, then along the underside back to
    # the stem's left side, closing at the bottom of the stem.
    return np.array([
        [x0, 0.0],
        [x1, 0.0],
        [x1, h - top_h],
        [w,  h - top_h],
        [w,  h],
        [0.0, h],
        [0.0, h - top_h],
        [x0, h - top_h],
    ])


# =============================================================================
# BOUNDARY / CONTAINMENT HELPERS (used by the offline formation optimizer)
# =============================================================================
def dist_to_boundary(point_xy: np.ndarray, polygon: Polygon) -> float:
    """Euclidean distance from `point_xy` to the polygon's boundary (rim)."""
    return polygon.exterior.distance(Point(point_xy))


def containment_margin(point_xy: np.ndarray, polygon: Polygon) -> float:
    """Signed containment margin: >= 0 iff point_xy is inside-or-on the
    polygon (distance to the rim); < 0 if outside (magnitude = violation).
    Used as an inequality constraint g(p) >= 0 for the SLSQP optimizer.
    """
    pt = Point(point_xy)
    d = polygon.exterior.distance(pt)
    inside = polygon.contains(pt) or polygon.exterior.distance(pt) < 1e-12
    return d if inside else -d


def boundary_points(polygon: Polygon, n: int) -> np.ndarray:
    """`n` points evenly spaced (by arc length) around the polygon boundary.
    Used only as a good initial guess for the formation optimizer -- points
    already on the rim, close to feasible, so SLSQP mostly just has to fix up
    spacing and the zero-sum centering constraint.
    """
    fracs = np.arange(n) / n
    pts = np.array([polygon.exterior.interpolate(f, normalized=True).coords[0] for f in fracs])
    return pts


def _vertex_bisector_normal(coords: np.ndarray, k: int) -> np.ndarray:
    """Outward angle-bisector normal at vertex k of a closed CCW ring
    (coords[0] == coords[-1], m = len(coords)-1 vertices). This is the
    standard "mitred offset" direction used in polygon buffering: the
    normalized sum of the two adjacent edges' outward normals. Unlike a
    single edge's normal, this is well-defined and geometrically correct at
    *both* convex and reflex (concave) vertices -- at a reflex vertex it
    correctly bisects the notch's opening rather than pointing along either
    wall.
    """
    m = len(coords) - 1

    def edge_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        e = b - a
        e = e / np.linalg.norm(e)
        return np.array([e[1], -e[0]])

    n_in = edge_normal(coords[(k - 1) % m], coords[k])
    n_out = edge_normal(coords[k], coords[(k + 1) % m])
    b = n_in + n_out
    norm = np.linalg.norm(b)
    return b / norm if norm > 1e-9 else n_in  # 1e-9 guard: back-to-back antiparallel edges


def nearest_edge_normal(
    point_xy: np.ndarray, polygon: Polygon, corner_blend_frac: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Snap `point_xy` onto the polygon's nearest boundary edge and return the
    outward unit normal there.

    This is what turns an SLSQP-optimized (approximately-on-the-rim) slot into
    a physically meaningful *grasp contact point + approach direction*: real
    end effectors / pushers approach a flat surface along its outward normal
    (the standard "approach ray" convention in grasp planning), not along an
    arbitrary bearing from wherever the agent happens to start.

    Near either endpoint of the nearest edge (within `corner_blend_frac` of
    its length), the normal is smoothly blended toward that vertex's
    angle-bisector normal (`_vertex_bisector_normal`) rather than used raw.
    This matters specifically for *reflex* (concave) vertices -- e.g. the
    inner notch of an L-shape -- where a point that snaps very close to the
    corner would otherwise get a single edge's normal that provides almost no
    real clearance from the *other* wall of the same notch (found and
    diagnosed empirically; see `approach.compute_approach_corridor`).

    Returns
    -------
    normal : (2,) unit outward normal at the (possibly corner-blended) point.
    proj   : (2,) the exact closest point on the nearest edge (the snapped
             contact point -- used in place of the raw optimizer output so
             the normal and the contact point are mutually consistent).
    dist   : distance from point_xy to proj.

    Assumes `polygon`'s exterior ring is wound counter-clockwise (true for
    every shape constructed in this module).
    """
    coords = np.array(polygon.exterior.coords)  # closed ring: coords[0] == coords[-1]
    m = len(coords) - 1
    best = (np.inf, None, None, None, None)  # dist, normal, proj, edge_idx, s
    for i in range(m):
        a, b = coords[i], coords[i + 1]
        edge = b - a
        edge_len2 = float(edge @ edge)
        if edge_len2 < 1e-12:
            continue
        s = np.clip(((point_xy - a) @ edge) / edge_len2, 0.0, 1.0)
        proj = a + s * edge
        d = float(np.linalg.norm(point_xy - proj))
        if d < best[0]:
            edge_dir = edge / np.sqrt(edge_len2)
            edge_normal = np.array([edge_dir[1], -edge_dir[0]])  # CCW polygon -> outward normal
            best = (d, edge_normal, proj, i, float(s))

    dist, edge_normal, proj, edge_idx, s = best

    if s < corner_blend_frac:
        t = s / corner_blend_frac
        nb = _vertex_bisector_normal(coords, edge_idx)
        normal = (1 - t) * nb + t * edge_normal
        normal = normal / np.linalg.norm(normal)
    elif s > 1.0 - corner_blend_frac:
        t = (1.0 - s) / corner_blend_frac
        nb = _vertex_bisector_normal(coords, edge_idx + 1)
        normal = (1 - t) * nb + t * edge_normal
        normal = normal / np.linalg.norm(normal)
    else:
        normal = edge_normal

    return normal, proj, dist


def snap_to_boundary_with_normals(P: np.ndarray, polygon: Polygon) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized `nearest_edge_normal` over all n formation slots.

    Returns (B, N): B[i] is P[i] snapped exactly onto the nearest edge, N[i]
    is the outward unit normal there.
    """
    n = P.shape[0]
    B = np.zeros((n, 2))
    N = np.zeros((n, 2))
    for i in range(n):
        normal, proj, _ = nearest_edge_normal(P[i], polygon)
        B[i] = proj
        N[i] = normal
    return B, N
