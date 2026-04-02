import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty

def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # --- Initial layout generation: Wind-direction-aware grid ---
    x_min_orig, y_min_orig = jnp.min(boundary, axis=0)
    x_max_orig, y_max_orig = jnp.max(boundary, axis=0)

    wd_rad = jnp.deg2rad(wd)
    sin_sum = jnp.sum(weights * jnp.sin(wd_rad))
    cos_sum = jnp.sum(weights * jnp.cos(wd_rad))
    dominant_wd_rad = jnp.arctan2(sin_sum, cos_sum)
    grid_rotation_angle = dominant_wd_rad + jnp.pi / 2.0

    cos_theta = jnp.cos(grid_rotation_angle)
    sin_theta = jnp.sin(grid_rotation_angle)

    centroid_x = jnp.mean(boundary[:, 0])
    centroid_y = jnp.mean(boundary[:, 1])

    translated_boundary = boundary - jnp.array([centroid_x, centroid_y])
    rotation_matrix = jnp.array([[cos_theta, -sin_theta],
                                 [sin_theta,  cos_theta]])
    rotated_translated_boundary = (rotation_matrix @ translated_boundary.T).T

    x_min_rot, y_min_rot = jnp.min(rotated_translated_boundary, axis=0)
    x_max_rot, y_max_rot = jnp.max(rotated_translated_boundary, axis=0)

    nx_rot = int(jnp.ceil((x_max_rot - x_min_rot) / min_spacing))
    ny_rot = int(jnp.ceil((y_max_rot - y_min_rot) / min_spacing))

    gx_rot, gy_rot = jnp.meshgrid(
        jnp.linspace(x_min_rot + min_spacing/2, x_max_rot - min_spacing/2, nx_rot),
        jnp.linspace(y_min_rot + min_spacing/2, y_max_rot - min_spacing/2, ny_rot))

    rotated_grid_points = jnp.stack([gx_rot.flatten(), gy_rot.flatten()], axis=-1)

    inverse_rotation_matrix = jnp.array([[cos_theta,  sin_theta],
                                         [-sin_theta, cos_theta]])
    original_grid_points = (inverse_rotation_matrix @ rotated_grid_points.T).T + jnp.array([centroid_x, centroid_y])

    candidates_x_wind_aware = original_grid_points[:, 0]
    candidates_y_wind_aware = original_grid_points[:, 1]

    n_verts = boundary.shape[0]
    def edge_dist_orig(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        edge_x, edge_y = x2 - x1, y2 - y1
        edge_len = jnp.sqrt(edge_x**2 + edge_y**2) + 1e-10
        nx_norm, ny_norm = -edge_y / edge_len, edge_x / edge_len
        return (candidates_x_wind_aware - x1) * nx_norm + (candidates_y_wind_aware - y1) * ny_norm
    all_dists_wind_aware = jax.vmap(edge_dist_orig)(jnp.arange(n_verts))
    inside_wind_aware = jnp.min(all_dists_wind_aware, axis=0) > 0

    # Filter to points inside the boundary
    inside_x_wind_aware = candidates_x_wind_aware[inside_wind_aware]
    inside_y_wind_aware = candidates_y_wind_aware[inside_wind_aware]

    init_x = jnp.array([])
    init_y = jnp.array([])

    if len(inside_x_wind_aware) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(inside_x_wind_aware) - 1, n_target)).astype(int)
        init_x = inside_x_wind_aware[idx]
        init_y = inside_y_wind_aware[idx]
    else:
        # Fallback if not enough points from wind-aware grid: random inside bounding box
        key_grid_fallback = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key_grid_fallback, (n_target,), minval=float(x_min_orig), maxval=float(x_max_orig))
        key_grid_fallback, _ = jax.random.split(key_grid_fallback)
        init_y = jax.random.uniform(key_grid_fallback, (n_target,), minval=float(y_min_orig), maxval=float(y_max_orig))

    # --- Penalized objective function for custom SGD (to minimize) ---
    def objective_penalized(x, y, alpha_spacing, alpha_boundary, ks_rho_val):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6 # Negative AEP for minimization

        sp = spacing_penalty(x, y, min_spacing, ks_rho_val)
        bp = boundary_penalty(x, y, boundary, ks_rho_val)
        
        return aep + alpha_spacing * sp + alpha_boundary * bp

    grad_fn = jax.grad(objective_penalized, argnums=(0, 1))

    # --- Custom SGD Optimizer Parameters ---
    initial_learning_rate = 250.0 # High initial learning rate
    learning_rate_decay_rate = 0.999 # Exponential decay per iteration

    # --- Two-Stage Penalty Annealing Strategy ---
    total_iterations = 12000 # Increased total iterations for deeper convergence
    stage1_iterations = 4000 # Iterations for aggressive feasibility (approx 33% of total)

    # Stage 1 Penalty Weights (Aggressive Feasibility)
    alpha_spacing_s1 = 250.0
    alpha_boundary_s1 = 250.0
    ks_rho_s1 = 150.0

    # Stage 2 Penalty Weights (AEP Refinement)
    alpha_spacing_s2 = 3.0
    alpha_boundary_s2 = 3.0
    ks_rho_s2 = 50.0

    # Initial state for SGD
    current_x, current_y = init_x, init_y

    @jax.jit
    def sgd_update(i, state):
        x, y = state

        # Determine current learning rate with exponential decay
        current_lr = initial_learning_rate * (learning_rate_decay_rate ** i)

        # Determine current penalty weights and ks_rho based on stage
        current_alpha_spacing = jnp.where(i < stage1_iterations, alpha_spacing_s1, alpha_spacing_s2)
        current_alpha_boundary = jnp.where(i < stage1_iterations, alpha_boundary_s1, alpha_boundary_s2)
        current_ks_rho = jnp.where(i < stage1_iterations, ks_rho_s1, ks_rho_s2)

        # Compute gradients
        grads_x, grads_y = grad_fn(x, y, current_alpha_spacing, current_alpha_boundary, current_ks_rho)

        # SGD update
        x = x - current_lr * grads_x
        y = y - current_lr * grads_y

        return x, y

    # Run the custom SGD optimization loop
    final_x, final_y = jax.lax.fori_loop(0, total_iterations, sgd_update, 
                                                              (current_x, current_y))

    return final_x, final_y