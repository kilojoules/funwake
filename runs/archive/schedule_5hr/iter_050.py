"""Custom SGD like iter_032 but with constant LR enforcement phase.

iter_032 hit 5600 GWh but infeasible because LR decayed to 0.019 by
enforcement start (iter 9500), making enforcement ineffective.

Fix: Use constant LR (30.0) during enforcement phase so turbines
can actually move to satisfy constraints. Also increase alpha to 300.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # --- Wind-aware grid initialization ---
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
                                 [sin_theta, cos_theta]])
    rotated_translated_boundary = (rotation_matrix @ translated_boundary.T).T

    x_min_rot, y_min_rot = jnp.min(rotated_translated_boundary, axis=0)
    x_max_rot, y_max_rot = jnp.max(rotated_translated_boundary, axis=0)

    nx_rot = int(jnp.ceil((x_max_rot - x_min_rot) / min_spacing))
    ny_rot = int(jnp.ceil((y_max_rot - y_min_rot) / min_spacing))

    gx_rot, gy_rot = jnp.meshgrid(
        jnp.linspace(x_min_rot + min_spacing / 2, x_max_rot - min_spacing / 2, nx_rot),
        jnp.linspace(y_min_rot + min_spacing / 2, y_max_rot - min_spacing / 2, ny_rot))

    rotated_grid_points = jnp.stack([gx_rot.flatten(), gy_rot.flatten()], axis=-1)
    inverse_rotation_matrix = jnp.array([[cos_theta, sin_theta],
                                          [-sin_theta, cos_theta]])
    original_grid_points = (inverse_rotation_matrix @ rotated_grid_points.T).T + jnp.array([centroid_x, centroid_y])

    cand_x = original_grid_points[:, 0]
    cand_y = original_grid_points[:, 1]

    n_verts = boundary.shape[0]

    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        edge_x, edge_y = x2 - x1, y2 - y1
        edge_len = jnp.sqrt(edge_x ** 2 + edge_y ** 2) + 1e-10
        nx_norm = -edge_y / edge_len
        ny_norm = edge_x / edge_len
        return (cand_x - x1) * nx_norm + (cand_y - y1) * ny_norm

    all_dists = jax.vmap(edge_dist)(jnp.arange(n_verts))
    inside = jnp.min(all_dists, axis=0) > 0

    inside_x = cand_x[inside]
    inside_y = cand_y[inside]

    if len(inside_x) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(inside_x) - 1, n_target)).astype(int)
        init_x = inside_x[idx]
        init_y = inside_y[idx]
    else:
        key = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min_orig), maxval=float(x_max_orig))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min_orig), maxval=float(y_max_orig))

    # --- Penalized objective ---
    def penalized_objective(x, y, alpha_spacing, alpha_boundary):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing)
        bp = boundary_penalty(x, y, boundary)
        return aep + alpha_spacing * sp + alpha_boundary * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    # --- Hyperparameters ---
    total_iterations = 14000
    initial_lr = 250.0
    lr_decay_rate = 0.999
    max_step = min_spacing

    # Phase boundaries
    phase1_end = 3500   # feasibility
    phase2_end = 9500   # AEP exploration
    # phase 3: 9500-14000 enforcement with CONSTANT LR

    alpha_sp_1 = 250.0
    alpha_bp_1 = 250.0

    alpha_sp_2 = 3.0
    alpha_bp_2 = 3.0

    alpha_sp_3 = 300.0
    alpha_bp_3 = 300.0

    enforce_lr = 30.0  # constant LR for enforcement phase

    @jax.jit
    def sgd_step(i, state):
        x, y = state

        # Piecewise LR: exponential decay for phases 1-2, constant for phase 3
        in_enforce = i >= phase2_end
        decayed_lr = initial_lr * (lr_decay_rate ** i)
        lr = jnp.where(in_enforce, enforce_lr, decayed_lr)

        in_p1 = i < phase1_end
        in_p2 = (i >= phase1_end) & (i < phase2_end)
        alpha_sp = jnp.where(in_p1, alpha_sp_1, jnp.where(in_p2, alpha_sp_2, alpha_sp_3))
        alpha_bp = jnp.where(in_p1, alpha_bp_1, jnp.where(in_p2, alpha_bp_2, alpha_bp_3))

        gx, gy = grad_fn(x, y, alpha_sp, alpha_bp)

        # Per-turbine step clipping
        step_x = lr * gx
        step_y = lr * gy
        per_turbine_dist = jnp.sqrt(step_x ** 2 + step_y ** 2 + 1e-12)
        scale = jnp.minimum(1.0, max_step / per_turbine_dist)
        step_x = step_x * scale
        step_y = step_y * scale

        x = x - step_x
        y = y - step_y

        return (x, y)

    final_x, final_y = jax.lax.fori_loop(
        0, total_iterations, sgd_step, (init_x, init_y))

    return final_x, final_y
