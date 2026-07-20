"""Hybrid: vanilla SGD fori_loop for AEP + topfarm_sgd_solve enforcement.

Phase 1 (custom SGD, 12000 iters): maximize AEP with 2-phase penalties.
Phase 2 (topfarm_sgd_solve, ~1500 iters): fix feasibility with high
constraint weights and low LR to preserve AEP gains.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
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

    # === PHASE 1: Custom SGD for AEP maximization ===
    def penalized_objective(x, y, alpha_spacing, alpha_boundary, ks_rho_val):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing, ks_rho_val)
        bp = boundary_penalty(x, y, boundary, ks_rho_val)
        return aep + alpha_spacing * sp + alpha_boundary * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    total_iterations = 10000
    initial_lr = 250.0
    lr_decay_rate = 0.999
    max_step = min_spacing

    stage1_iters = 4000

    alpha_s1 = 250.0
    ks_rho_s1 = 150.0
    alpha_s2 = 3.0
    ks_rho_s2 = 50.0

    @jax.jit
    def sgd_step(i, state):
        x, y = state

        lr = initial_lr * (lr_decay_rate ** i)

        in_s1 = i < stage1_iters
        alpha = jnp.where(in_s1, alpha_s1, alpha_s2)
        ks_rho = jnp.where(in_s1, ks_rho_s1, ks_rho_s2)

        gx, gy = grad_fn(x, y, alpha, alpha, ks_rho)

        step_x = lr * gx
        step_y = lr * gy
        per_turbine_dist = jnp.sqrt(step_x ** 2 + step_y ** 2 + 1e-12)
        scale = jnp.minimum(1.0, max_step / per_turbine_dist)

        x = x - step_x * scale
        y = y - step_y * scale

        return (x, y)

    sgd_x, sgd_y = jax.lax.fori_loop(
        0, total_iterations, sgd_step, (init_x, init_y))

    # === PHASE 2: topfarm_sgd_solve to enforce feasibility ===
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    enforce_settings = SGDSettings(
        learning_rate=30.0,
        max_iter=1500,
        additional_constant_lr_iterations=500,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=300.0,
        boundary_weight=300.0,
        ks_rho=150.0,
        gamma_min_factor=0.01,
    )

    opt_x, opt_y = topfarm_sgd_solve(objective, sgd_x, sgd_y,
                                      boundary, min_spacing, enforce_settings)

    return opt_x, opt_y
