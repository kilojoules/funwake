"""Differentiable geometry primitives for adversarial optimization.

This module provides JAX-native implementations of:
- Closed B-splines for morphable boundary representation
- Signed Distance Field (SDF) computation
- Soft turbine packing based on SDF masks

These primitives enable gradient-based discovery of critical neighbor
configurations that maximize design regret.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from pixwake.jax_utils import ssqrt


@dataclass
class BSplineBoundary:
    """Differentiable closed B-spline boundary.

    Represents a morphable neighbor farm boundary as a closed B-spline curve.
    The control points are the design variables for adversarial optimization.

    Attributes:
        control_points: Control points of shape (n_control, 2).
        degree: B-spline degree (default 3 for cubic).
    """

    control_points: jnp.ndarray
    degree: int = 3

    @property
    def n_control(self) -> int:
        """Number of control points."""
        return self.control_points.shape[0]

    def evaluate(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the closed B-spline at parameters t ∈ [0, 1].

        Args:
            t: Parameter values in [0, 1], shape (n_points,).

        Returns:
            Points on the spline, shape (n_points, 2).
        """
        return evaluate_closed_bspline(self.control_points, t, self.degree)

    def signed_distance(self, points: jnp.ndarray) -> jnp.ndarray:
        """Compute signed distance from points to the boundary.

        Negative values indicate points inside the boundary.

        Args:
            points: Query points of shape (n_points, 2).

        Returns:
            Signed distances of shape (n_points,).
        """
        return compute_sdf_to_spline(points, self.control_points, self.degree)

    def contains(self, points: jnp.ndarray, temperature: float = 10.0) -> jnp.ndarray:
        """Soft containment test using sigmoid of SDF.

        Args:
            points: Query points of shape (n_points, 2).
            temperature: Sigmoid sharpness (higher = sharper boundary).

        Returns:
            Soft containment values in (0, 1), shape (n_points,).
        """
        sdf = self.signed_distance(points)
        # Negative SDF = inside, so negate for sigmoid
        return jax.nn.sigmoid(-temperature * sdf)

    def area(self, n_samples: int = 100) -> jnp.ndarray:
        """Compute approximate area enclosed by the spline (Shoelace formula)."""
        t = jnp.linspace(0, 1, n_samples, endpoint=False)
        pts = self.evaluate(t)
        x, y = pts[:, 0], pts[:, 1]
        # Shoelace formula
        return 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y))

    def centroid(self, n_samples: int = 100) -> jnp.ndarray:
        """Compute approximate centroid of the enclosed region."""
        t = jnp.linspace(0, 1, n_samples, endpoint=False)
        pts = self.evaluate(t)
        return jnp.mean(pts, axis=0)


def evaluate_closed_bspline(
    control_points: jnp.ndarray,
    t: jnp.ndarray,
    degree: int = 3,
) -> jnp.ndarray:
    """Evaluate a closed (periodic) B-spline using De Boor's algorithm.

    Args:
        control_points: Control points of shape (n_control, 2).
        t: Parameter values in [0, 1], shape (n_points,).
        degree: B-spline degree.

    Returns:
        Points on the spline, shape (n_points, 2).
    """
    n = control_points.shape[0]

    # For closed spline, wrap control points
    # Add 'degree' points at the end that wrap around
    wrapped_cp = jnp.concatenate([
        control_points,
        control_points[:degree]
    ], axis=0)

    # Uniform knot vector for closed spline
    n_wrapped = wrapped_cp.shape[0]
    n_knots = n_wrapped + degree + 1
    knots = jnp.linspace(0, 1, n_knots)

    # Scale t to the valid parameter range
    t_min = knots[degree]
    t_max = knots[n_wrapped]
    t_scaled = t_min + t * (t_max - t_min)

    # Evaluate using De Boor's algorithm
    return _de_boor_vectorized(wrapped_cp, knots, t_scaled, degree)


def _de_boor_vectorized(
    control_points: jnp.ndarray,
    knots: jnp.ndarray,
    t: jnp.ndarray,
    degree: int,
) -> jnp.ndarray:
    """Vectorized De Boor's algorithm for B-spline evaluation.

    Args:
        control_points: Control points of shape (n_control, 2).
        knots: Knot vector of shape (n_knots,).
        t: Parameter values of shape (n_points,).
        degree: B-spline degree.

    Returns:
        Points on the spline, shape (n_points, 2).
    """
    n_points = t.shape[0]
    n_control = control_points.shape[0]

    # Find knot span for each t value
    # k is the index such that knots[k] <= t < knots[k+1]
    k = jnp.searchsorted(knots, t, side='right') - 1
    # Clamp to valid range
    k = jnp.clip(k, degree, n_control - 1)

    # Initialize with control points in the affected span
    # d[j] = control_points[k - degree + j] for j in 0..degree
    def get_initial_d(ki):
        indices = ki - degree + jnp.arange(degree + 1)
        indices = jnp.clip(indices, 0, n_control - 1)
        return control_points[indices]

    d = jax.vmap(get_initial_d)(k)  # (n_points, degree+1, 2)

    # De Boor recursion
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            # Compute alpha
            left_idx = k - degree + j
            right_idx = left_idx + degree - r + 1

            # Safe indexing
            left_idx = jnp.clip(left_idx, 0, len(knots) - 1)
            right_idx = jnp.clip(right_idx, 0, len(knots) - 1)

            left_knot = knots[left_idx]
            right_knot = knots[right_idx]

            denom = right_knot - left_knot
            # Avoid division by zero
            alpha = jnp.where(
                denom > 1e-10,
                (t - left_knot) / denom,
                0.5
            )

            # Update d
            d = d.at[:, j].set(
                (1 - alpha[:, None]) * d[:, j - 1] + alpha[:, None] * d[:, j]
            )

    return d[:, degree]


def compute_sdf_to_spline(
    points: jnp.ndarray,
    control_points: jnp.ndarray,
    degree: int = 3,
    n_samples: int = 100,
) -> jnp.ndarray:
    """Compute signed distance from points to a closed B-spline boundary.

    Uses sampling-based approximation for efficiency and differentiability.
    Negative values indicate points inside the boundary.

    Args:
        points: Query points of shape (n_points, 2).
        control_points: Spline control points of shape (n_control, 2).
        degree: B-spline degree.
        n_samples: Number of samples on the spline for distance computation.

    Returns:
        Signed distances of shape (n_points,).
    """
    # Sample points on the spline
    t = jnp.linspace(0, 1, n_samples, endpoint=False)
    spline_pts = evaluate_closed_bspline(control_points, t, degree)

    # Compute unsigned distance to nearest spline point
    # points: (n_query, 2), spline_pts: (n_samples, 2)
    diff = points[:, None, :] - spline_pts[None, :, :]  # (n_query, n_samples, 2)
    dist_sq = jnp.sum(diff ** 2, axis=-1)  # (n_query, n_samples)
    unsigned_dist = ssqrt(jnp.min(dist_sq, axis=-1))  # (n_query,)

    # Determine sign using winding number (ray casting)
    sign = _compute_winding_sign(points, spline_pts)

    return sign * unsigned_dist


def _compute_winding_sign(
    points: jnp.ndarray,
    polygon: jnp.ndarray,
) -> jnp.ndarray:
    """Compute sign based on winding number (inside = negative).

    Uses ray casting: count crossings with polygon edges.

    Args:
        points: Query points of shape (n_points, 2).
        polygon: Polygon vertices of shape (n_vertices, 2).

    Returns:
        Sign array: -1 for inside, +1 for outside, shape (n_points,).
    """
    n_vertices = polygon.shape[0]

    # Get edges: (start, end) pairs
    v1 = polygon
    v2 = jnp.roll(polygon, -1, axis=0)

    def count_crossings(point):
        """Count ray crossings for a single point."""
        px, py = point[0], point[1]

        # Ray goes in +x direction from point
        # Edge crosses ray if:
        # 1. One vertex is above ray, one below (or on)
        # 2. Intersection x-coordinate is > px

        y1, y2 = v1[:, 1], v2[:, 1]
        x1, x2 = v1[:, 0], v2[:, 0]

        # Check if edge straddles the ray
        cond1 = (y1 > py) != (y2 > py)

        # Compute x-coordinate of intersection
        # x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        dy = y2 - y1
        # Safe division
        t = jnp.where(jnp.abs(dy) > 1e-10, (py - y1) / dy, 0.5)
        x_intersect = x1 + t * (x2 - x1)

        # Count crossings where intersection is to the right of point
        crossings = jnp.sum(cond1 & (x_intersect > px))

        # Odd crossings = inside = negative sign
        return jnp.where(crossings % 2 == 1, -1.0, 1.0)

    return jax.vmap(count_crossings)(points)


def create_ellipse_control_points(
    center: tuple[float, float],
    semi_major: float,
    semi_minor: float,
    rotation: float = 0.0,
    n_control: int = 8,
) -> jnp.ndarray:
    """Create control points for an ellipse-like B-spline.

    Useful for initializing blob geometries.

    Args:
        center: (x, y) center of the ellipse.
        semi_major: Semi-major axis length.
        semi_minor: Semi-minor axis length.
        rotation: Rotation angle in radians.
        n_control: Number of control points.

    Returns:
        Control points of shape (n_control, 2).
    """
    angles = jnp.linspace(0, 2 * jnp.pi, n_control, endpoint=False)

    # Points on ellipse
    x = semi_major * jnp.cos(angles)
    y = semi_minor * jnp.sin(angles)

    # Apply rotation
    cos_r, sin_r = jnp.cos(rotation), jnp.sin(rotation)
    x_rot = cos_r * x - sin_r * y
    y_rot = sin_r * x + cos_r * y

    # Translate to center
    x_final = x_rot + center[0]
    y_final = y_rot + center[1]

    return jnp.stack([x_final, y_final], axis=-1)


def phi_to_control_points(
    phi: np.ndarray,
    target_center: tuple[float, float],
    fixed_area: float,
    n_control: int = 8,
) -> jnp.ndarray:
    """Map a 4D design vector to B-spline control points for an elliptical blob.

    The design vector uses polar coordinates relative to the target farm center.

    Args:
        phi: Design vector (4,): [bearing_rad, distance_m, rotation_rad, aspect_ratio].
            bearing uses meteorological convention (0=N, clockwise).
        target_center: (x, y) center of the target farm.
        fixed_area: Fixed blob area in m^2.
        n_control: Number of B-spline control points.

    Returns:
        Control points of shape (n_control, 2).
    """
    bearing, distance, rotation, aspect_ratio = phi

    # Blob center from polar coords (meteorological: 0=N, clockwise)
    cx = target_center[0] + distance * np.sin(bearing)
    cy = target_center[1] + distance * np.cos(bearing)

    # Semi-axes from fixed area and aspect ratio
    # area = pi * semi_major * semi_minor, aspect_ratio = semi_major / semi_minor
    semi_minor = np.sqrt(fixed_area / (np.pi * aspect_ratio))
    semi_major = aspect_ratio * semi_minor

    return create_ellipse_control_points(
        (float(cx), float(cy)),
        float(semi_major),
        float(semi_minor),
        float(rotation),
        n_control,
    )


def sample_random_blob(
    key: jnp.ndarray,
    center_bounds: tuple[tuple[float, float], tuple[float, float]],
    size_bounds: tuple[float, float],
    aspect_ratio_bounds: tuple[float, float] = (0.5, 2.0),
    n_control: int = 8,
) -> jnp.ndarray:
    """Sample random initial blob control points.

    Args:
        key: JAX random key.
        center_bounds: ((x_min, x_max), (y_min, y_max)) for center position.
        size_bounds: (min_size, max_size) for semi-major axis.
        aspect_ratio_bounds: (min_ratio, max_ratio) for semi-minor/semi-major.
        n_control: Number of control points.

    Returns:
        Control points of shape (n_control, 2).
    """
    keys = jax.random.split(key, 5)

    # Sample center
    cx = jax.random.uniform(keys[0], minval=center_bounds[0][0], maxval=center_bounds[0][1])
    cy = jax.random.uniform(keys[1], minval=center_bounds[1][0], maxval=center_bounds[1][1])

    # Sample size
    semi_major = jax.random.uniform(keys[2], minval=size_bounds[0], maxval=size_bounds[1])

    # Sample aspect ratio
    aspect = jax.random.uniform(
        keys[3],
        minval=aspect_ratio_bounds[0],
        maxval=aspect_ratio_bounds[1]
    )
    semi_minor = semi_major * aspect

    # Sample rotation
    rotation = jax.random.uniform(keys[4], minval=0, maxval=2 * jnp.pi)

    return create_ellipse_control_points(
        (float(cx), float(cy)),
        float(semi_major),
        float(semi_minor),
        float(rotation),
        n_control,
    )
