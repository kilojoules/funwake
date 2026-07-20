"""General polygon boundary primitives for wind farm layout optimization.

Provides signed distance fields, containment penalties, and exclusion penalties
for both convex and concave polygons. All functions are JAX-native and
differentiable.

Convention:
    - Polygon vertices are in CCW (counter-clockwise) order.
    - Signed distance field: **negative inside, positive outside** (standard SDF).
    - ``_signed_distance_to_edge_line``: positive = inside (left of CCW edge).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# =============================================================================
# Low-level primitives
# =============================================================================


def _signed_distance_to_edge_line(
    px: jnp.ndarray,
    py: jnp.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> jnp.ndarray:
    """Signed distance from points to the *infinite line* through an edge.

    Positive = inside (to the left of a CCW edge).
    Negative = outside.

    Args:
        px, py: Point coordinates, shape ``(n,)``.
        x1, y1: Edge start point.
        x2, y2: Edge end point.

    Returns:
        Signed distances, shape ``(n,)``.
    """
    edge_x = x2 - x1
    edge_y = y2 - y1
    edge_len = jnp.sqrt(edge_x**2 + edge_y**2) + 1e-10

    # Inward normal (90-deg CCW rotation of edge direction)
    normal_x = -edge_y / edge_len
    normal_y = edge_x / edge_len

    ap_x = px - x1
    ap_y = py - y1

    return ap_x * normal_x + ap_y * normal_y


def _nearest_distance_to_segment(
    px: jnp.ndarray,
    py: jnp.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> jnp.ndarray:
    """Unsigned distance from points to a line *segment*.

    Clamps the projection parameter ``t`` to ``[0, 1]`` so that the closest
    feature may be a vertex rather than the edge interior.  This is necessary
    for correct SDF on concave polygons.

    Args:
        px, py: Point coordinates, shape ``(n,)``.
        x1, y1: Segment start.
        x2, y2: Segment end.

    Returns:
        Unsigned distances, shape ``(n,)``.
    """
    edge_x = x2 - x1
    edge_y = y2 - y1
    edge_len_sq = edge_x**2 + edge_y**2 + 1e-20

    # Projection parameter, clamped to [0, 1]
    t = ((px - x1) * edge_x + (py - y1) * edge_y) / edge_len_sq
    t = jnp.clip(t, 0.0, 1.0)

    # Closest point on segment
    closest_x = x1 + t * edge_x
    closest_y = y1 + t * edge_y

    return jnp.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2 + 1e-20)


def _point_in_polygon(
    px: jnp.ndarray,
    py: jnp.ndarray,
    vertices: jnp.ndarray,
) -> jnp.ndarray:
    """Ray-casting point-in-polygon test.

    Returns 1.0 if inside, 0.0 if outside.  Adapted from
    ``geometry.py:_compute_winding_sign``.

    Args:
        px, py: Scalar point coordinates.
        vertices: Polygon vertices, shape ``(n_vertices, 2)``, CCW order.

    Returns:
        Scalar, 1.0 (inside) or 0.0 (outside).
    """
    v1 = vertices
    v2 = jnp.roll(vertices, -1, axis=0)

    y1 = v1[:, 1]
    y2 = v2[:, 1]
    x1 = v1[:, 0]
    x2 = v2[:, 0]

    # Edge straddles the horizontal ray?
    cond1 = (y1 > py) != (y2 > py)

    # Intersection x-coordinate
    dy = y2 - y1
    t = jnp.where(jnp.abs(dy) > 1e-10, (py - y1) / dy, 0.5)
    x_intersect = x1 + t * (x2 - x1)

    crossings = jnp.sum(cond1 & (x_intersect > px))
    return jnp.where(crossings % 2 == 1, 1.0, 0.0)


# =============================================================================
# Signed distance field
# =============================================================================


def polygon_sdf(
    x: jnp.ndarray,
    y: jnp.ndarray,
    vertices: jnp.ndarray,
) -> jnp.ndarray:
    """General polygon signed distance field.

    Negative inside, positive outside (standard SDF convention).

    Args:
        x, y: Point coordinates, shape ``(n,)``.
        vertices: Polygon vertices, shape ``(n_vertices, 2)``, CCW order.

    Returns:
        Signed distances, shape ``(n,)``.
    """
    n_vertices = vertices.shape[0]

    def segment_dist(i: int) -> jnp.ndarray:
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n_vertices]
        return _nearest_distance_to_segment(x, y, x1, y1, x2, y2)

    # Unsigned distance = min over all edges
    all_dists = jax.vmap(segment_dist)(jnp.arange(n_vertices))  # (n_edges, n_pts)
    unsigned = jnp.min(all_dists, axis=0)  # (n_pts,)

    # Sign via ray casting (vectorized over points)
    inside = jax.vmap(lambda px, py: _point_in_polygon(px, py, vertices))(x, y)
    sign = jnp.where(inside > 0.5, -1.0, 1.0)

    return sign * unsigned


# =============================================================================
# Penalty functions
# =============================================================================


def containment_penalty(
    x: jnp.ndarray,
    y: jnp.ndarray,
    vertices: jnp.ndarray,
    convex: bool = True,
) -> jnp.ndarray:
    """Penalty for points *outside* the polygon ("stay inside").

    Args:
        x, y: Point coordinates, shape ``(n,)``.
        vertices: Polygon vertices, shape ``(n_vertices, 2)``, CCW order.
        convex: If ``True`` (default), use the fast half-plane approach
            (identical to the original ``sgd.py:boundary_penalty``).
            If ``False``, use ``polygon_sdf`` for general (concave) polygons.

    Returns:
        Scalar penalty (0 when all points are inside).
    """
    if convex:
        n_vertices = vertices.shape[0]

        def edge_distances(i: int) -> jnp.ndarray:
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n_vertices]
            return _signed_distance_to_edge_line(x, y, x1, y1, x2, y2)

        all_distances = jax.vmap(edge_distances)(jnp.arange(n_vertices))
        min_distances = jnp.min(all_distances, axis=0)
        violations = jnp.minimum(0.0, min_distances)
        return jnp.sum(violations**2)

    # General (concave) path
    sdf = polygon_sdf(x, y, vertices)
    violations = jnp.maximum(0.0, sdf)  # positive SDF = outside
    return jnp.sum(violations**2)


def exclusion_penalty(
    x: jnp.ndarray,
    y: jnp.ndarray,
    vertices: jnp.ndarray,
    buffer: float = 0.0,
    convex: bool = True,
) -> jnp.ndarray:
    """Penalty for points *inside* the polygon + buffer ("stay outside").

    Args:
        x, y: Point coordinates, shape ``(n,)``.
        vertices: Polygon vertices, shape ``(n_vertices, 2)``, CCW order.
        buffer: Additional keep-out distance beyond the polygon boundary.
        convex: If ``True`` (default), use the fast half-plane approach
            (matches the hand-rolled logic in ``run_dei_ift_bilevel.py``).
            If ``False``, use ``polygon_sdf`` for general (concave) polygons.

    Returns:
        Scalar penalty (0 when all points are far enough outside).
    """
    if convex:
        # Half-plane approach: min_distance is the signed distance to the
        # nearest edge (positive = inside).  Penalize when
        # min_distance + buffer > 0, i.e. point is inside boundary + buffer.
        n_vertices = vertices.shape[0]

        def edge_distances(i: int) -> jnp.ndarray:
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n_vertices]
            return _signed_distance_to_edge_line(x, y, x1, y1, x2, y2)

        all_distances = jax.vmap(edge_distances)(jnp.arange(n_vertices))
        min_distances = jnp.min(all_distances, axis=0)
        violations = jnp.maximum(0.0, min_distances + buffer)
        return jnp.sum(violations**2)

    # General (concave) path: SDF < 0 inside polygon.
    # Penalize when sdf < buffer  (i.e. too close or inside).
    sdf = polygon_sdf(x, y, vertices)
    violations = jnp.maximum(0.0, buffer - sdf)
    return jnp.sum(violations**2)
