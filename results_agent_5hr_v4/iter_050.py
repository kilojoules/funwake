import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # Objective function (AEP maximization - negative for minimization)
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # --- Single-Stage Optimization Settings for each multi-start ---
    # Tuned for a balance of AEP maximization and robust feasibility across the entire run.
    settings_single_stage = SGDSettings(learning_rate=150.0, # Moderately high learning rate
                                        max_iter=4000,       # Ample iterations per start
                                        additional_constant_lr_iterations=2000, # Long constant LR phase
                                        tol=1e-6,
                                        spacing_weight=75.0, # Significant penalty for spacing
                                        boundary_weight=75.0, # Significant penalty for boundary
                                        ks_rho=80.0)          # Fairly sharp penalty function

    best_overall_aep = -jnp.inf
    best_overall_x, best_overall_y = None, None

    # --- Generate Diverse Initial Layouts for Multi-start ---
    initial_layouts_pool = []

    x_min_orig, y_min_orig = jnp.min(boundary, axis=0)
    x_max_orig, y_max_orig = jnp.max(boundary, axis=0)

    # Helper function to check if points are inside the boundary
    n_verts = boundary.shape[0]
    def is_inside_boundary(coords_x, coords_y):
        def edge_dist_calc(i, x_cand, y_cand):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            edge_x, edge_y = x2 - x1, y2 - y1
            edge_len = jnp.sqrt(edge_x**2 + edge_y**2) + 1e-10
            nx_norm, ny_norm = -edge_y / edge_len, edge_x / edge_len
            return (x_cand - x1) * nx_norm + (y_cand - y1) * ny_norm
        all_dists = jax.vmap(edge_dist_calc, in_axes=(0, None, None))(jnp.arange(n_verts), coords_x, coords_y)
        return jnp.min(all_dists) > 0.0 # Strictly inside

    # --- 1. Wind-direction-aware grid initialization ---
    wd_rad = jnp.deg2rad(wd)
    sin_sum = jnp.sum(weights * jnp.sin(wd_rad))
    cos_sum = jnp.sum(weights * jnp.cos(wd_rad))
    dominant_wd_rad = jnp.arctan2(sin_sum, cos_sum)
    grid_rotation_angle = dominant_wd_rad + jnp.pi / 2.0 # Rotate grid perpendicular to dominant wind

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

    # Use min_spacing for grid density
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

    inside_wind_aware_mask = jax.vmap(is_inside_boundary)(candidates_x_wind_aware, candidates_y_wind_aware)
    inside_x_wind_aware = candidates_x_wind_aware[inside_wind_aware_mask]
    inside_y_wind_aware = candidates_y_wind_aware[inside_wind_aware_mask]

    init_x_wind_aware, init_y_wind_aware = jnp.array([]), jnp.array([])
    if len(inside_x_wind_aware) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(inside_x_wind_aware) - 1, n_target)).astype(int)
        init_x_wind_aware = inside_x_wind_aware[idx]
        init_y_wind_aware = inside_y_wind_aware[idx]
    else:
        key_fallback = jax.random.PRNGKey(1001)
        init_x_wind_aware = jax.random.uniform(key_fallback, (n_target,), minval=float(x_min_orig), maxval=float(x_max_orig))
        key_fallback, _ = jax.random.split(key_fallback)
        init_y_wind_aware = jax.random.uniform(key_fallback, (n_target,), minval=float(y_min_orig), maxval=float(y_max_orig))
    initial_layouts_pool.append((init_x_wind_aware, init_y_wind_aware))

    # --- 2. Standard rectangular grid initialization ---
    # Use min_spacing for grid density
    nx_std = int(jnp.ceil((x_max_orig - x_min_orig) / min_spacing))
    ny_std = int(jnp.ceil((y_max_orig - y_min_orig) / min_spacing))
    gx_std, gy_std = jnp.meshgrid(
        jnp.linspace(x_min_orig + min_spacing/2, x_max_orig - min_spacing/2, nx_std),
        jnp.linspace(y_min_orig + min_spacing/2, y_max_orig - min_spacing/2, ny_std))
    candidates_x_std = gx_std.flatten()
    candidates_y_std = gy_std.flatten()

    inside_std_mask = jax.vmap(is_inside_boundary)(candidates_x_std, candidates_y_std)
    inside_x_std = candidates_x_std[inside_std_mask]
    inside_y_std = candidates_y_std[inside_std_mask]

    init_x_std, init_y_std = jnp.array([]), jnp.array([])
    if len(inside_x_std) >= n_target:
        idx_std = jnp.round(jnp.linspace(0, len(inside_x_std) - 1, n_target)).astype(int)
        init_x_std = inside_x_std[idx_std]
        init_y_std = inside_y_std[idx_std]
    else:
        key_fallback = jax.random.PRNGKey(1002)
        init_x_std = jax.random.uniform(key_fallback, (n_target,), minval=float(x_min_orig), maxval=float(x_max_orig))
        key_fallback, _ = jax.random.split(key_fallback)
        init_y_std = jax.random.uniform(key_fallback, (n_target,), minval=float(y_min_orig), maxval=float(y_max_orig))
    initial_layouts_pool.append((init_x_std, init_y_std))

    # --- 3. Purely random initialization with a slight perturbation ---
    key_random_pure = jax.random.PRNGKey(1003)
    rand_x_pure = jax.random.uniform(key_random_pure, (n_target,), minval=float(x_min_orig), maxval=float(x_max_orig))
    key_random_pure, _ = jax.random.split(key_random_pure)
    rand_y_pure = jax.random.uniform(key_random_pure, (n_target,), minval=float(y_min_orig), maxval=float(y_max_orig))
    
    perturbation_scale = min_spacing * 0.1 # Small perturbation
    key_perturb = jax.random.PRNGKey(1004)
    dx = jax.random.uniform(key_perturb, (n_target,), minval=-perturbation_scale, maxval=perturbation_scale)
    key_perturb, _ = jax.random.split(key_perturb)
    dy = jax.random.uniform(key_perturb, (n_target,), minval=-perturbation_scale, maxval=perturbation_scale)

    initial_layouts_pool.append((rand_x_pure + dx, rand_y_pure + dy))


    # --- Run Single-Stage Optimization for each initial layout in the pool ---
    # This loop will run a fully-fledged topfarm_sgd_solve for each initial layout.
    # The parameters for settings_single_stage are chosen to be robust enough to handle
    # diverse initializations and converge to a good feasible solution within a single stage.
    for init_x, init_y in initial_layouts_pool:
        # Run a single, robust topfarm_sgd_solve instance
        opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y,
                                         boundary, min_spacing, settings_single_stage)

        # Calculate AEP for the final optimized layout from this multi-start
        current_aep = -objective(opt_x, opt_y) # objective returns negative AEP

        if current_aep > best_overall_aep:
            best_overall_aep = current_aep
            best_overall_x, best_overall_y = opt_x, opt_y

    return best_overall_x, best_overall_y