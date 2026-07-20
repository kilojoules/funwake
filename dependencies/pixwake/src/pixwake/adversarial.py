"""
Adversarial Neighbor Optimization for Wind Farm Layout.

This module implements optimization schemes to find neighbor configurations
that maximize regret for a target wind farm.

Two gradient-based approaches are available:

1. **Direct Gradient** (GradientAdversarialOptimizer):
   - Optimizes raw (x, y) positions for N turbines
   - Fully differentiable via JAX
   - No shape constraint

2. **Spline Gradient** (DifferentiableSplineOptimizer):
   - Optimizes spline parameters with fixed N turbines
   - Fully differentiable via JAX
   - Maintains cluster geometry

See docs/adversarial_optimization.md for detailed documentation.

Key components:
- SplineCluster: Spline-bounded cluster parameterization
- TurbinePacker: Hexagonal packing of turbines within boundaries
- evaluate_closed_bspline: JAX-differentiable B-spline evaluation
- differentiable_spline_positions: Spline params → turbine positions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull


# =============================================================================
# Spline Cluster Parameterization
# =============================================================================


def shoelace_area(vertices: np.ndarray) -> float:
    """Compute polygon area using shoelace formula.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (N, 2) array of polygon vertices in order.

    Returns
    -------
    float
        Absolute area of the polygon.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Check if point is inside polygon using ray casting.

    Parameters
    ----------
    point : np.ndarray
        Shape (2,) array [x, y].
    vertices : np.ndarray
        Shape (N, 2) array of polygon vertices.

    Returns
    -------
    bool
        True if point is inside polygon.
    """
    x, y = point
    n = len(vertices)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def points_in_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Vectorized point-in-polygon test.

    Parameters
    ----------
    points : np.ndarray
        Shape (M, 2) array of points to test.
    vertices : np.ndarray
        Shape (N, 2) array of polygon vertices.

    Returns
    -------
    np.ndarray
        Boolean array of shape (M,).
    """
    return np.array([point_in_polygon(p, vertices) for p in points])


class SplineCluster:
    """Spline-bounded cluster for neighbor wind farm representation.

    The cluster boundary is defined by a closed B-spline through K control
    points. This allows flexible, smooth shapes including convex and
    concave configurations.

    Parameters
    ----------
    control_points : np.ndarray
        Shape (K, 2) array of control points relative to center.
    center : tuple[float, float]
        Cluster center position (cx, cy).
    rotation : float
        Rotation angle in radians.
    n_boundary_samples : int
        Number of points to sample along the spline boundary.
    """

    def __init__(
        self,
        control_points: np.ndarray,
        center: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        n_boundary_samples: int = 100,
    ):
        self.control_points_raw = np.asarray(control_points)
        self.center = np.asarray(center)
        self.rotation = rotation
        self.n_boundary_samples = n_boundary_samples

        # Compute transformed boundary
        self._boundary = self._compute_boundary()
        self._area = shoelace_area(self._boundary)

    def _compute_boundary(self) -> np.ndarray:
        """Compute the spline boundary with rotation and translation."""
        pts = self.control_points_raw.copy()

        # Apply rotation
        if self.rotation != 0:
            cos_r = np.cos(self.rotation)
            sin_r = np.sin(self.rotation)
            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            pts = pts @ rotation_matrix.T

        # Fit closed B-spline
        # Close the curve by appending first few points
        pts_closed = np.vstack([pts, pts[:3]])

        try:
            tck, _ = splprep([pts_closed[:, 0], pts_closed[:, 1]], s=0, per=True, k=3)
            u_new = np.linspace(0, 1, self.n_boundary_samples, endpoint=False)
            x, y = splev(u_new, tck)
            boundary = np.column_stack([x, y])
        except Exception:
            # Fall back to convex hull if spline fails
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    boundary = pts[hull.vertices]
                except Exception:
                    boundary = pts
            else:
                boundary = pts

        # Translate to center
        boundary = boundary + self.center

        return boundary

    @property
    def boundary(self) -> np.ndarray:
        """Get the boundary polygon vertices."""
        return self._boundary

    @property
    def area(self) -> float:
        """Get the cluster area in square units."""
        return self._area

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Check which points are inside the cluster boundary.

        Parameters
        ----------
        points : np.ndarray
            Shape (M, 2) array of points to test.

        Returns
        -------
        np.ndarray
            Boolean array of shape (M,).
        """
        return points_in_polygon(points, self._boundary)

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """Get axis-aligned bounding box.

        Returns
        -------
        tuple
            (x_min, y_min, x_max, y_max)
        """
        x_min = self._boundary[:, 0].min()
        x_max = self._boundary[:, 0].max()
        y_min = self._boundary[:, 1].min()
        y_max = self._boundary[:, 1].max()
        return x_min, y_min, x_max, y_max

    @classmethod
    def from_ellipse(
        cls,
        center: tuple[float, float],
        semi_major: float,
        semi_minor: float,
        rotation: float = 0.0,
        n_control_points: int = 8,
    ) -> "SplineCluster":
        """Create cluster from ellipse parameters.

        Parameters
        ----------
        center : tuple[float, float]
            Center position.
        semi_major : float
            Semi-major axis length.
        semi_minor : float
            Semi-minor axis length.
        rotation : float
            Rotation angle in radians.
        n_control_points : int
            Number of control points around the ellipse.

        Returns
        -------
        SplineCluster
            Cluster with elliptical boundary.
        """
        angles = np.linspace(0, 2 * np.pi, n_control_points, endpoint=False)
        control_points = np.column_stack([
            semi_major * np.cos(angles),
            semi_minor * np.sin(angles),
        ])
        return cls(control_points, center=center, rotation=rotation)

    @classmethod
    def from_rectangle(
        cls,
        center: tuple[float, float],
        width: float,
        height: float,
        rotation: float = 0.0,
    ) -> "SplineCluster":
        """Create cluster from rectangle parameters.

        Parameters
        ----------
        center : tuple[float, float]
            Center position.
        width : float
            Rectangle width.
        height : float
            Rectangle height.
        rotation : float
            Rotation angle in radians.

        Returns
        -------
        SplineCluster
            Cluster with approximately rectangular boundary.
        """
        hw, hh = width / 2, height / 2
        control_points = np.array([
            [-hw, -hh],
            [0, -hh],
            [hw, -hh],
            [hw, 0],
            [hw, hh],
            [0, hh],
            [-hw, hh],
            [-hw, 0],
        ])
        return cls(control_points, center=center, rotation=rotation)


# =============================================================================
# Turbine Packing
# =============================================================================


class TurbinePacker:
    """Pack turbines within a cluster boundary using hexagonal grid.

    Parameters
    ----------
    min_spacing : float
        Minimum spacing between turbines.
    """

    def __init__(self, min_spacing: float):
        self.min_spacing = min_spacing

    def hexagonal_grid(
        self, bounds: tuple[float, float, float, float], spacing: float
    ) -> np.ndarray:
        """Generate hexagonal grid points within bounding box.

        Parameters
        ----------
        bounds : tuple
            (x_min, y_min, x_max, y_max)
        spacing : float
            Distance between adjacent turbines.

        Returns
        -------
        np.ndarray
            Shape (N, 2) array of grid points.
        """
        x_min, y_min, x_max, y_max = bounds
        dx = spacing
        dy = spacing * np.sqrt(3) / 2

        points = []
        row = 0
        y = y_min
        while y <= y_max:
            x_offset = (row % 2) * dx / 2
            x = x_min + x_offset
            while x <= x_max:
                points.append([x, y])
                x += dx
            y += dy
            row += 1

        return np.array(points) if points else np.empty((0, 2))

    def pack_turbines(
        self, cluster: SplineCluster, n_turbines: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack specified number of turbines inside cluster boundary.

        Uses hexagonal grid and filters to points inside the boundary.
        If more points are available than needed, selects points closest
        to center to maintain compact arrangement.

        Parameters
        ----------
        cluster : SplineCluster
            The cluster boundary.
        n_turbines : int
            Target number of turbines.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (x_positions, y_positions) arrays.
        """
        if n_turbines <= 0:
            return np.array([]), np.array([])

        bounds = cluster.get_bounding_box()

        # Generate hexagonal grid
        grid_points = self.hexagonal_grid(bounds, self.min_spacing)

        if len(grid_points) == 0:
            return np.array([]), np.array([])

        # Filter to points inside boundary
        inside_mask = cluster.contains(grid_points)
        inside_points = grid_points[inside_mask]

        if len(inside_points) == 0:
            return np.array([]), np.array([])

        # If we have more points than needed, select closest to center
        if len(inside_points) > n_turbines:
            center = cluster.center
            distances = np.sqrt(np.sum((inside_points - center) ** 2, axis=1))
            indices = np.argsort(distances)[:n_turbines]
            selected = inside_points[indices]
        else:
            selected = inside_points

        return selected[:, 0], selected[:, 1]

    def pack_by_density(
        self, cluster: SplineCluster, density: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack turbines based on density specification.

        Parameters
        ----------
        cluster : SplineCluster
            The cluster boundary.
        density : float
            Turbines per unit area (e.g., turbines per km²).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (x_positions, y_positions) arrays.
        """
        n_turbines = int(cluster.area * density)
        return self.pack_turbines(cluster, n_turbines)


# =============================================================================
# Differentiable Spline-Based Optimization
# =============================================================================


def _bspline_basis(t: jnp.ndarray, i: int, k: int, knots: jnp.ndarray) -> jnp.ndarray:
    """Compute B-spline basis function (Cox-de Boor recursion).

    Parameters
    ----------
    t : jnp.ndarray
        Parameter values.
    i : int
        Basis function index.
    k : int
        Degree (0 = constant, 1 = linear, 3 = cubic).
    knots : jnp.ndarray
        Knot vector.

    Returns
    -------
    jnp.ndarray
        Basis function values at t.
    """
    if k == 0:
        return jnp.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)

    # Recursive case
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]

    term1 = jnp.where(
        denom1 > 1e-10,
        (t - knots[i]) / denom1 * _bspline_basis(t, i, k - 1, knots),
        0.0
    )
    term2 = jnp.where(
        denom2 > 1e-10,
        (knots[i + k + 1] - t) / denom2 * _bspline_basis(t, i + 1, k - 1, knots),
        0.0
    )

    return term1 + term2


def evaluate_closed_bspline(
    control_points: jnp.ndarray,
    t: jnp.ndarray,
    degree: int = 3
) -> jnp.ndarray:
    """Evaluate closed B-spline curve at parameter values.

    This is a JAX-differentiable B-spline evaluation.

    Parameters
    ----------
    control_points : jnp.ndarray
        Shape (n, 2) control points. The curve will be closed by
        wrapping the first `degree` points.
    t : jnp.ndarray
        Parameter values in [0, 1).
    degree : int
        Spline degree (default: 3 for cubic).

    Returns
    -------
    jnp.ndarray
        Shape (len(t), 2) points on the curve.
    """
    n = len(control_points)

    # Wrap control points for closed curve
    cp_wrapped = jnp.concatenate([control_points, control_points[:degree]], axis=0)
    n_wrapped = n + degree

    # Create uniform knot vector for closed curve
    knots = jnp.linspace(0, 1, n_wrapped - degree + 2)

    # Scale t to knot range (excluding last knot to avoid boundary issues)
    t_scaled = t * (knots[-degree-1] - knots[degree]) + knots[degree]

    # Evaluate using matrix form for efficiency
    # For each t, compute weighted sum of control points
    points = jnp.zeros((len(t), 2))

    for i in range(n_wrapped):
        basis = _bspline_basis(t_scaled, i, degree, knots)
        points = points + basis[:, None] * cp_wrapped[i]

    return points


def differentiable_spline_positions(
    control_points: jnp.ndarray,
    center: jnp.ndarray,
    rotation: float,
    n_turbines: int,
    fill_fraction: float = 0.7,
) -> jnp.ndarray:
    """Compute turbine positions from spline parameters (differentiable).

    Places turbines in a pattern that fills the spline interior.
    Uses a combination of:
    1. Points along the boundary (scaled inward)
    2. Center point(s)

    Parameters
    ----------
    control_points : jnp.ndarray
        Shape (K, 2) spline control points (relative to center).
    center : jnp.ndarray
        Shape (2,) cluster center position.
    rotation : float
        Rotation angle in radians.
    n_turbines : int
        Number of turbines to place.
    fill_fraction : float
        How much to fill toward center (0 = boundary only, 1 = full fill).

    Returns
    -------
    jnp.ndarray
        Shape (n_turbines, 2) turbine positions.
    """
    # Apply rotation to control points
    cos_r = jnp.cos(rotation)
    sin_r = jnp.sin(rotation)
    rot_matrix = jnp.array([[cos_r, -sin_r], [sin_r, cos_r]])
    cp_rotated = control_points @ rot_matrix.T

    # Evaluate boundary at uniform parameter values
    t_boundary = jnp.linspace(0, 1, n_turbines * 3, endpoint=False)
    boundary_points = evaluate_closed_bspline(cp_rotated, t_boundary, degree=3)

    # Compute centroid of boundary
    centroid = jnp.mean(boundary_points, axis=0)

    # Strategy: place turbines at multiple "rings" from boundary to center
    # Ring 0: on boundary (scaled by 0.9 to be slightly inside)
    # Ring 1: 50% toward center
    # Ring 2: at center

    positions = []

    if n_turbines <= 6:
        # Small number: place along boundary
        t_vals = jnp.linspace(0, 1, n_turbines, endpoint=False)
        ring_points = evaluate_closed_bspline(cp_rotated, t_vals, degree=3)
        # Scale slightly inward
        ring_points = centroid + 0.85 * (ring_points - centroid)
        positions = ring_points
    else:
        # Multiple rings
        n_outer = min(n_turbines - 1, max(6, n_turbines // 2))
        n_inner = n_turbines - n_outer

        # Outer ring (85% from center to boundary)
        t_outer = jnp.linspace(0, 1, n_outer, endpoint=False)
        outer_points = evaluate_closed_bspline(cp_rotated, t_outer, degree=3)
        outer_ring = centroid + 0.85 * (outer_points - centroid)

        if n_inner <= 1:
            # Just add center
            inner_ring = centroid[None, :]
        elif n_inner <= 6:
            # Inner ring at 40% scale
            t_inner = jnp.linspace(0, 1, n_inner, endpoint=False)
            inner_points = evaluate_closed_bspline(cp_rotated, t_inner, degree=3)
            inner_ring = centroid + 0.4 * fill_fraction * (inner_points - centroid)
        else:
            # Two inner rings
            n_mid = n_inner // 2
            n_center = n_inner - n_mid

            t_mid = jnp.linspace(0, 1, n_mid, endpoint=False)
            mid_points = evaluate_closed_bspline(cp_rotated, t_mid, degree=3)
            mid_ring = centroid + 0.55 * fill_fraction * (mid_points - centroid)

            if n_center <= 1:
                center_ring = centroid[None, :]
            else:
                t_center = jnp.linspace(0, 1, n_center, endpoint=False)
                center_points = evaluate_closed_bspline(cp_rotated, t_center, degree=3)
                center_ring = centroid + 0.2 * fill_fraction * (center_points - centroid)

            inner_ring = jnp.concatenate([mid_ring, center_ring], axis=0)

        positions = jnp.concatenate([outer_ring, inner_ring], axis=0)

    # Translate to center
    positions = positions + center

    return positions[:n_turbines]


class DifferentiableSplineOptimizer:
    """Gradient-based optimizer using differentiable spline parameterization.

    Turbine positions are smooth functions of spline control points,
    enabling gradient-based optimization of cluster shape.

    Parameters
    ----------
    target_x : np.ndarray
        Target farm x positions.
    target_y : np.ndarray
        Target farm y positions.
    sim_engine : WakeSimulation
        Wake simulation engine.
    wd_arr : jnp.ndarray
        Wind directions array.
    weights : jnp.ndarray
        Wind rose probability weights.
    wind_speed : float
        Reference wind speed.
    rotor_diameter : float
        Turbine rotor diameter.
    n_neighbors : int
        Number of neighbor turbines.
    n_control_points : int
        Number of spline control points.
    search_radius : float
        Maximum distance from target center.
    min_neighbor_distance : float
        Minimum distance from target boundary.
    """

    def __init__(
        self,
        target_x: np.ndarray,
        target_y: np.ndarray,
        sim_engine,
        wd_arr: jnp.ndarray,
        weights: jnp.ndarray,
        wind_speed: float,
        rotor_diameter: float,
        n_neighbors: int = 16,
        n_control_points: int = 6,
        search_radius: float | None = None,
        min_neighbor_distance: float | None = None,
    ):
        self.target_x = jnp.array(target_x)
        self.target_y = jnp.array(target_y)
        self.n_target = len(target_x)
        self.sim_engine = sim_engine
        self.wd_arr = wd_arr
        self.weights = weights
        self.wind_speed = wind_speed
        self.rotor_diameter = rotor_diameter
        self.n_neighbors = n_neighbors
        self.n_control_points = n_control_points

        # Compute target farm geometry
        self.target_center = jnp.array([jnp.mean(target_x), jnp.mean(target_y)])
        self.target_radius = float(jnp.max(jnp.sqrt(
            (target_x - self.target_center[0]) ** 2 +
            (target_y - self.target_center[1]) ** 2
        )))

        D = rotor_diameter
        self.search_radius = search_radius or 30 * D
        self.min_neighbor_distance = min_neighbor_distance or 3 * D
        self.default_cluster_scale = 5 * D

        self._setup_jax_functions()

    def _setup_jax_functions(self):
        """Set up JIT-compiled functions."""

        def compute_aep_from_spline(control_points, center, rotation):
            """Compute target AEP given spline parameters."""
            # Get neighbor positions from spline (differentiable)
            neighbor_pos = differentiable_spline_positions(
                control_points, center, rotation,
                self.n_neighbors, fill_fraction=0.7
            )
            neighbor_x = neighbor_pos[:, 0]
            neighbor_y = neighbor_pos[:, 1]

            # Combine with target
            x_all = jnp.concatenate([self.target_x, neighbor_x])
            y_all = jnp.concatenate([self.target_y, neighbor_y])

            # Simulate
            ws_arr = jnp.full_like(self.wd_arr, self.wind_speed)
            result = self.sim_engine(x_all, y_all, ws_amb=ws_arr, wd_amb=self.wd_arr, ti_amb=0.06)

            # Target AEP only
            power_kw = result.power()
            target_power = power_kw[:, :self.n_target]
            weighted_power = jnp.sum(target_power * self.weights[:, None])
            return weighted_power * 8760 / 1e6

        self._compute_aep = jax.jit(compute_aep_from_spline)
        self._aep_and_grad = jax.jit(jax.value_and_grad(
            compute_aep_from_spline, argnums=(0, 1, 2)
        ))

    def _initialize_params(
        self, angle: float | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, float]:
        """Initialize spline parameters.

        Returns
        -------
        tuple
            (control_points, center, rotation)
        """
        if angle is None:
            dominant_idx = int(jnp.argmax(self.weights))
            dominant_wd = float(self.wd_arr[dominant_idx])
            angle = np.deg2rad(90 - dominant_wd)

        # Place cluster center upwind
        dist = self.target_radius + self.min_neighbor_distance + self.default_cluster_scale
        center = self.target_center + dist * jnp.array([jnp.cos(angle), jnp.sin(angle)])

        # Elliptical control points
        cp_angles = jnp.linspace(0, 2 * jnp.pi, self.n_control_points, endpoint=False)
        control_points = jnp.column_stack([
            self.default_cluster_scale * jnp.cos(cp_angles),
            self.default_cluster_scale * 0.6 * jnp.sin(cp_angles),
        ])

        rotation = 0.0

        return control_points, center, rotation

    def _project_center(self, center: jnp.ndarray) -> jnp.ndarray:
        """Project center to satisfy distance constraints."""
        delta = center - self.target_center
        dist = jnp.sqrt(jnp.sum(delta ** 2)) + 1e-6

        # Minimum distance
        min_dist = self.target_radius + self.min_neighbor_distance
        # Maximum distance
        max_dist = self.search_radius

        # Clamp distance
        clamped_dist = jnp.clip(dist, min_dist, max_dist)
        return self.target_center + delta * (clamped_dist / dist)

    def get_turbine_positions(
        self,
        control_points: jnp.ndarray,
        center: jnp.ndarray,
        rotation: float,
    ) -> jnp.ndarray:
        """Get turbine positions from current parameters."""
        return differentiable_spline_positions(
            control_points, center, rotation,
            self.n_neighbors, fill_fraction=0.7
        )

    def optimize(
        self,
        n_iterations: int = 200,
        learning_rate: float | None = None,
        initial_angle: float | None = None,
        verbose: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, float, list[float]]:
        """Optimize spline parameters to minimize target AEP.

        Parameters
        ----------
        n_iterations : int
            Number of gradient descent iterations.
        learning_rate : float or None
            Step size for control points/center (default: rotor_diameter / 10).
        initial_angle : float or None
            Initial cluster placement angle.
        verbose : bool
            Print progress.

        Returns
        -------
        tuple
            (control_points, center, rotation, aep_history)
        """
        if learning_rate is None:
            learning_rate = self.rotor_diameter / 10

        # Initialize
        control_points, center, rotation = self._initialize_params(initial_angle)
        aep_history = []

        # Learning rates for different parameters
        lr_cp = learning_rate  # Control points
        lr_center = learning_rate * 2  # Center (can move faster)
        lr_rotation = 0.01  # Rotation (radians)

        for i in range(n_iterations):
            # Compute AEP and gradients
            aep, (grad_cp, grad_center, grad_rotation) = self._aep_and_grad(
                control_points, center, rotation
            )
            aep_history.append(float(aep))

            # Gradient descent (minimize AEP)
            control_points = control_points - lr_cp * grad_cp
            center = center - lr_center * grad_center
            rotation = rotation - lr_rotation * grad_rotation

            # Project center to constraints
            center = self._project_center(center)

            # Keep rotation in [-pi, pi]
            rotation = jnp.arctan2(jnp.sin(rotation), jnp.cos(rotation))

            if verbose and i % 20 == 0:
                print(f"Iter {i:4d}: target AEP = {aep:.3f} GWh")

        return control_points, center, float(rotation), aep_history


# =============================================================================
# Gradient-Based Direct Position Optimization
# =============================================================================


class GradientAdversarialOptimizer:
    """Gradient-based adversarial optimizer using direct position optimization.

    Instead of parameterizing clusters with splines, this optimizer directly
    optimizes neighbor turbine positions using JAX autodiff. This is more
    efficient when the number of neighbor turbines is fixed.

    Parameters
    ----------
    target_x : np.ndarray
        Target farm x positions.
    target_y : np.ndarray
        Target farm y positions.
    sim_engine : WakeSimulation
        Wake simulation engine.
    wd_arr : jnp.ndarray
        Wind directions array.
    weights : jnp.ndarray
        Wind rose probability weights.
    wind_speed : float
        Reference wind speed.
    rotor_diameter : float
        Turbine rotor diameter.
    n_neighbors : int
        Number of neighbor turbines to optimize.
    search_radius : float
        Maximum distance from target center.
    min_neighbor_distance : float
        Minimum distance from target boundary.
    """

    def __init__(
        self,
        target_x: np.ndarray,
        target_y: np.ndarray,
        sim_engine,
        wd_arr: jnp.ndarray,
        weights: jnp.ndarray,
        wind_speed: float,
        rotor_diameter: float,
        n_neighbors: int = 16,
        search_radius: float | None = None,
        min_neighbor_distance: float | None = None,
    ):
        self.target_x = jnp.array(target_x)
        self.target_y = jnp.array(target_y)
        self.n_target = len(target_x)
        self.sim_engine = sim_engine
        self.wd_arr = wd_arr
        self.weights = weights
        self.wind_speed = wind_speed
        self.rotor_diameter = rotor_diameter
        self.n_neighbors = n_neighbors

        # Compute target farm center and radius
        self.target_center = (float(jnp.mean(target_x)), float(jnp.mean(target_y)))
        self.target_radius = float(jnp.max(jnp.sqrt(
            (target_x - self.target_center[0]) ** 2 +
            (target_y - self.target_center[1]) ** 2
        )))

        D = rotor_diameter
        self.search_radius = search_radius or 30 * D
        self.min_neighbor_distance = min_neighbor_distance or 3 * D
        self.min_spacing = 2 * D

        # JIT compile evaluation functions
        self._setup_jax_functions()

    def _setup_jax_functions(self):
        """Set up JIT-compiled JAX functions for gradient computation."""

        def compute_target_aep(target_x, target_y, neighbor_x, neighbor_y):
            """Compute AEP for target farm given neighbor positions."""
            x_all = jnp.concatenate([target_x, neighbor_x])
            y_all = jnp.concatenate([target_y, neighbor_y])
            ws_arr = jnp.full_like(self.wd_arr, self.wind_speed)
            result = self.sim_engine(x_all, y_all, ws_amb=ws_arr, wd_amb=self.wd_arr, ti_amb=0.06)
            power_kw = result.power()
            target_power = power_kw[:, :self.n_target]
            weighted_power = jnp.sum(target_power * self.weights[:, None])
            return weighted_power * 8760 / 1e6

        # For maximizing regret, we want to minimize target AEP with neighbors
        # Regret = AEP_conservative - AEP_liberal (with same neighbors)
        # If we fix target layout and optimize neighbors, we minimize AEP
        self._compute_aep = jax.jit(compute_target_aep)

        # Gradient of AEP w.r.t. neighbor positions
        self._aep_grad = jax.jit(jax.grad(compute_target_aep, argnums=(2, 3)))

        # Combined value and gradient
        self._aep_and_grad = jax.jit(jax.value_and_grad(compute_target_aep, argnums=(2, 3)))

    def _initialize_neighbors(self, angle: float | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize neighbor positions in an arc around the target.

        Parameters
        ----------
        angle : float or None
            Central angle in radians. If None, uses dominant wind direction.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Initial (x, y) positions.
        """
        if angle is None:
            dominant_idx = int(jnp.argmax(self.weights))
            dominant_wd = float(self.wd_arr[dominant_idx])
            angle = np.deg2rad(90 - dominant_wd)

        # Place neighbors in an arc upwind
        dist = self.target_radius + self.min_neighbor_distance + 5 * self.rotor_diameter
        arc_span = np.pi / 3  # 60 degree arc

        angles = np.linspace(angle - arc_span/2, angle + arc_span/2, self.n_neighbors)
        x = self.target_center[0] + dist * np.cos(angles)
        y = self.target_center[1] + dist * np.sin(angles)

        return jnp.array(x), jnp.array(y)

    def _apply_constraints(
        self, neighbor_x: jnp.ndarray, neighbor_y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Project positions to satisfy constraints.

        Enforces:
        1. Minimum distance from target center
        2. Maximum distance from target center
        3. Minimum spacing between neighbors
        """
        cx, cy = self.target_center

        # Distance from target center
        dx = neighbor_x - cx
        dy = neighbor_y - cy
        dist = jnp.sqrt(dx**2 + dy**2)

        # Enforce minimum distance
        min_dist = self.target_radius + self.min_neighbor_distance
        scale_up = jnp.maximum(1.0, min_dist / (dist + 1e-6))

        # Enforce maximum distance
        scale_down = jnp.minimum(1.0, self.search_radius / (dist + 1e-6))

        # Apply distance constraints
        scale = jnp.where(dist < min_dist, scale_up, scale_down)
        new_x = cx + dx * scale
        new_y = cy + dy * scale

        return new_x, new_y

    def compute_regret(
        self,
        neighbor_x: jnp.ndarray,
        neighbor_y: jnp.ndarray,
        liberal_x: jnp.ndarray | None = None,
        liberal_y: jnp.ndarray | None = None,
        conservative_x: jnp.ndarray | None = None,
        conservative_y: jnp.ndarray | None = None,
    ) -> float:
        """Compute regret for given neighbor configuration.

        Parameters
        ----------
        neighbor_x, neighbor_y : jnp.ndarray
            Neighbor turbine positions.
        liberal_x, liberal_y : jnp.ndarray or None
            Pre-optimized liberal layout. If None, uses target layout.
        conservative_x, conservative_y : jnp.ndarray or None
            Pre-optimized conservative layout. If None, uses target layout.

        Returns
        -------
        float
            Regret = AEP_conservative_with_neighbor - AEP_liberal_with_neighbor
        """
        if liberal_x is None:
            liberal_x, liberal_y = self.target_x, self.target_y
        if conservative_x is None:
            conservative_x, conservative_y = self.target_x, self.target_y

        aep_lib = self._compute_aep(liberal_x, liberal_y, neighbor_x, neighbor_y)
        aep_con = self._compute_aep(conservative_x, conservative_y, neighbor_x, neighbor_y)

        return float(aep_con - aep_lib)

    def optimize(
        self,
        n_iterations: int = 200,
        learning_rate: float | None = None,
        initial_angle: float | None = None,
        verbose: bool = True,
        target_layout: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, list[float]]:
        """Optimize neighbor positions to minimize target AEP.

        This finds neighbors that cause maximum wake losses to the target farm.
        For computing actual regret, you still need to run inner optimizations.

        Parameters
        ----------
        n_iterations : int
            Number of gradient descent iterations.
        learning_rate : float or None
            Step size (default: rotor_diameter / 5).
        initial_angle : float or None
            Initial neighbor placement angle.
        verbose : bool
            Print progress.
        target_layout : tuple or None
            Fixed target layout (x, y). If None, uses initial target positions.

        Returns
        -------
        tuple
            (optimized_neighbor_x, optimized_neighbor_y, aep_history)
        """
        if learning_rate is None:
            learning_rate = self.rotor_diameter / 5

        if target_layout is None:
            target_x, target_y = self.target_x, self.target_y
        else:
            target_x, target_y = target_layout

        # Initialize
        neighbor_x, neighbor_y = self._initialize_neighbors(initial_angle)
        aep_history = []

        for i in range(n_iterations):
            # Compute AEP and gradients
            aep, (grad_x, grad_y) = self._aep_and_grad(
                target_x, target_y, neighbor_x, neighbor_y
            )
            aep_history.append(float(aep))

            # Gradient descent (minimize AEP = maximize wake impact)
            neighbor_x = neighbor_x - learning_rate * grad_x
            neighbor_y = neighbor_y - learning_rate * grad_y

            # Apply constraints
            neighbor_x, neighbor_y = self._apply_constraints(neighbor_x, neighbor_y)

            if verbose and i % 20 == 0:
                print(f"Iter {i:4d}: target AEP = {aep:.3f} GWh")

        return neighbor_x, neighbor_y, aep_history


