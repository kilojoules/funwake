"""Constraint penalties for layout optimization (plain numpy, license-clean).

These implement TopFarm's published penalty formulations directly:

- Boundary: for a convex CCW polygon, each point's signed distance to the
  polygon is the minimum over edges of the signed distance to the edge's
  infinite line (positive inside).  The penalty is the sum of squared
  outside-distances: ``sum(min(0, d_i)^2)``.
- Spacing: TopFarm's DistanceConstraintAggregation — the sum over violated
  pairs of ``(min_spacing^2 - d_ij^2)`` for pairs closer than min_spacing.

Analytic gradients are provided for use in gradient-based optimizers.
"""

import numpy as np


def _edge_geometry(boundary_vertices):
    """Per-edge unit inward normals for a CCW polygon.

    Returns (x1, y1, nx, ny) arrays of shape (n_edges,): edge start points
    and inward unit normals (90-degree CCW rotation of the edge direction).
    """
    v = np.asarray(boundary_vertices, dtype=float)
    x1, y1 = v[:, 0], v[:, 1]
    x2, y2 = np.roll(v[:, 0], -1), np.roll(v[:, 1], -1)
    ex, ey = x2 - x1, y2 - y1
    el = np.sqrt(ex**2 + ey**2) + 1e-10
    return x1, y1, -ey / el, ex / el


def _signed_distances(x, y, boundary_vertices):
    """Signed distance of each point to each edge's infinite line.

    Positive = inside (left of a CCW edge).  Shape (n_edges, n_points).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x1, y1, nx, ny = _edge_geometry(boundary_vertices)
    d = ((x[None, :] - x1[:, None]) * nx[:, None]
         + (y[None, :] - y1[:, None]) * ny[:, None])
    return d, nx, ny


def boundary_penalty(x, y, boundary_vertices, rho=100.0):
    """Sum of squared distances outside a convex CCW polygon.

    Args:
        x, y: Turbine positions, shape (n,).
        boundary_vertices: Polygon vertices (CCW), shape (n_vertices, 2).
        rho: Unused, kept for API compatibility.

    Returns:
        Scalar penalty (0 when all points are inside).
    """
    d, _, _ = _signed_distances(x, y, boundary_vertices)
    violations = np.minimum(0.0, d.min(axis=0))
    return np.sum(violations**2)


def boundary_penalty_grad(x, y, boundary_vertices):
    """Analytic gradient of boundary_penalty w.r.t. x and y.

    For each outside point, the active constraint is the nearest edge
    (argmin of the signed distances); d(v^2)/dp = 2 v * n_hat of that edge.
    """
    d, nx, ny = _signed_distances(x, y, boundary_vertices)
    e = np.argmin(d, axis=0)
    idx = np.arange(d.shape[1])
    violations = np.minimum(0.0, d[e, idx])
    return 2.0 * violations * nx[e], 2.0 * violations * ny[e]


def spacing_penalty(x, y, min_spacing, rho=100.0):
    """Sum over violated pairs of (min_spacing^2 - d^2).

    Args:
        x, y: Turbine positions, shape (n,).
        min_spacing: Minimum allowed pairwise distance.
        rho: Unused, kept for API compatibility.

    Returns:
        Scalar penalty (0 when all pairs are at least min_spacing apart).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if n < 2:
        return 0.0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    iu, ju = np.triu_indices(n, k=1)
    violations = np.maximum(0.0, min_spacing**2 - dist_sq[iu, ju])
    return np.sum(violations)


def spacing_penalty_grad(x, y, min_spacing):
    """Analytic gradient of spacing_penalty w.r.t. x and y.

    Each violated pair (i, j) contributes -2*(x_i - x_j) to grad_x_i
    (and symmetrically to grad_x_j).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if n < 2:
        return np.zeros_like(x), np.zeros_like(y)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    violated = (dist_sq < min_spacing**2) & ~np.eye(n, dtype=bool)
    gx = -2.0 * np.sum(violated * dx, axis=1)
    gy = -2.0 * np.sum(violated * dy, axis=1)
    return gx, gy
