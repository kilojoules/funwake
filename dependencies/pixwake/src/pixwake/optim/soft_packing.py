"""Soft turbine packing for differentiable neighbor representation.

This module enables gradient flow through discrete turbine counts by
representing turbine "presence" as a continuous value based on the
Signed Distance Field (SDF) of a boundary.

Key idea: Instead of discrete turbine positions, use a dense reference
grid where each position has a soft weight indicating how "present"
that turbine is. This makes the number of effective turbines continuous.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pixwake.optim.geometry import BSplineBoundary


@dataclass
class SoftNeighborFarm:
    """A neighbor farm with soft (differentiable) turbine packing.

    Turbine positions are defined on a dense reference grid, with each
    position weighted by how much it's "inside" the boundary spline.

    Attributes:
        grid_positions: Reference grid of potential turbine positions (N, 2).
        boundary: B-spline boundary defining the farm region.
        temperature: Sigmoid sharpness for soft containment.
        min_weight: Minimum weight threshold (turbines below this are ignored).
    """

    grid_positions: jnp.ndarray
    boundary: BSplineBoundary
    temperature: float = 10.0
    min_weight: float = 0.01

    @property
    def n_potential(self) -> int:
        """Number of potential turbine positions in the grid."""
        return self.grid_positions.shape[0]

    def weights(self) -> jnp.ndarray:
        """Compute soft weights for each grid position.

        Returns:
            Weights in (0, 1), shape (n_potential,).
        """
        return self.boundary.contains(self.grid_positions, self.temperature)

    def effective_count(self) -> jnp.ndarray:
        """Compute effective (soft) number of turbines."""
        return jnp.sum(self.weights())

    def weighted_positions(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get positions and their weights (filtered by min_weight).

        Returns:
            (positions, weights) where positions has shape (n_active, 2)
            and weights has shape (n_active,).
        """
        w = self.weights()
        mask = w > self.min_weight
        return self.grid_positions, w * mask


def create_reference_grid(
    center: tuple[float, float],
    extent: float,
    spacing: float,
) -> jnp.ndarray:
    """Create a dense reference grid for potential turbine positions.

    Args:
        center: (x, y) center of the grid.
        extent: Half-width/height of the grid region.
        spacing: Distance between grid points.

    Returns:
        Grid positions of shape (n_points, 2).
    """
    n_points = int(2 * extent / spacing) + 1
    x = jnp.linspace(center[0] - extent, center[0] + extent, n_points)
    y = jnp.linspace(center[1] - extent, center[1] + extent, n_points)
    xx, yy = jnp.meshgrid(x, y)
    return jnp.stack([xx.ravel(), yy.ravel()], axis=-1)


def compute_soft_wake_effect(
    target_x: jnp.ndarray,
    target_y: jnp.ndarray,
    neighbor_positions: jnp.ndarray,
    neighbor_weights: jnp.ndarray,
    wake_deficit_fn,
    ws_amb: jnp.ndarray,
    wd_amb: jnp.ndarray,
) -> jnp.ndarray:
    """Compute wake effect from soft-weighted neighbors on target turbines.

    This is an approximation that weights the wake contribution of each
    potential neighbor by its soft weight. This allows gradients to flow
    through the "number of turbines" via the weights.

    Args:
        target_x: Target turbine x positions (n_target,).
        target_y: Target turbine y positions (n_target,).
        neighbor_positions: Potential neighbor positions (n_neighbor, 2).
        neighbor_weights: Soft weights for each neighbor (n_neighbor,).
        wake_deficit_fn: Function(x_source, y_source, x_target, y_target, ws, wd) -> deficit.
        ws_amb: Ambient wind speeds (n_cases,).
        wd_amb: Wind directions (n_cases,).

    Returns:
        Weighted wake deficit at target positions (n_cases, n_target).
    """
    n_target = target_x.shape[0]
    n_neighbor = neighbor_positions.shape[0]
    n_cases = ws_amb.shape[0]

    # Compute deficit from each neighbor to each target
    # This is expensive but necessary for soft weighting
    def deficit_from_neighbor(i):
        nx, ny = neighbor_positions[i, 0], neighbor_positions[i, 1]
        w = neighbor_weights[i]

        # Single neighbor to all targets
        deficit = wake_deficit_fn(
            jnp.array([nx]),
            jnp.array([ny]),
            target_x,
            target_y,
            ws_amb,
            wd_amb,
        )  # (n_cases, n_target)

        return w * deficit

    # Sum weighted deficits from all neighbors
    deficits = jax.vmap(deficit_from_neighbor)(jnp.arange(n_neighbor))
    # deficits: (n_neighbor, n_cases, n_target)

    # Sum over neighbors
    total_deficit = jnp.sum(deficits, axis=0)  # (n_cases, n_target)

    return total_deficit


def compute_aep_with_soft_neighbors(
    target_x: jnp.ndarray,
    target_y: jnp.ndarray,
    soft_farm: SoftNeighborFarm,
    sim,
    ws_amb: jnp.ndarray,
    wd_amb: jnp.ndarray,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute target farm AEP with soft-weighted neighbor effects.

    This uses an approximation: we include all potential neighbors in the
    simulation, but weight their CT (thrust coefficient) by the soft weight.
    This is more accurate than weighting wake deficits directly.

    Args:
        target_x: Target turbine x positions.
        target_y: Target turbine y positions.
        soft_farm: Soft neighbor farm representation.
        sim: WakeSimulation instance.
        ws_amb: Ambient wind speeds.
        wd_amb: Wind directions.
        weights: Optional wind condition probability weights.

    Returns:
        AEP in GWh.
    """
    n_target = target_x.shape[0]
    neighbor_pos, neighbor_weights = soft_farm.weighted_positions()
    n_neighbor = neighbor_pos.shape[0]

    # Concatenate target and neighbor positions
    x_all = jnp.concatenate([target_x, neighbor_pos[:, 0]])
    y_all = jnp.concatenate([target_y, neighbor_pos[:, 1]])

    # Run simulation with all turbines
    result = sim(x_all, y_all, ws_amb=ws_amb, wd_amb=wd_amb)

    # Get power for target turbines only
    power = result.power()[:, :n_target]  # (n_cases, n_target)

    # Weight by wind condition probabilities if provided
    if weights is not None:
        weighted_power = jnp.sum(power * weights[:, None])
    else:
        weighted_power = jnp.mean(jnp.sum(power, axis=1))

    # Convert to AEP (GWh)
    return weighted_power * 8760 / 1e6


def compute_aep_with_weighted_ct(
    target_x: jnp.ndarray,
    target_y: jnp.ndarray,
    soft_farm: SoftNeighborFarm,
    sim,
    ws_amb: jnp.ndarray,
    wd_amb: jnp.ndarray,
    ti_amb: jnp.ndarray | None = None,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute AEP with CT-weighted neighbor contributions.

    More accurate approximation: neighbor turbines have their thrust
    coefficient scaled by the soft weight, so they create proportionally
    less wake as they "fade out" of the boundary.

    Note: This requires modifying the simulation to accept per-turbine
    CT scaling, which may not be supported by all wake models.

    Args:
        target_x: Target turbine x positions.
        target_y: Target turbine y positions.
        soft_farm: Soft neighbor farm.
        sim: WakeSimulation instance.
        ws_amb: Ambient wind speeds.
        wd_amb: Wind directions.
        ti_amb: Ambient turbulence intensity.
        weights: Wind condition probability weights.

    Returns:
        AEP in GWh.
    """
    # For now, use the simpler approach
    # TODO: Implement CT scaling in WakeSimulation
    return compute_aep_with_soft_neighbors(
        target_x, target_y, soft_farm, sim, ws_amb, wd_amb, weights
    )
