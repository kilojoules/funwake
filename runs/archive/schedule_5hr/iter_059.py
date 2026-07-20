"""Custom ADAM with 3-phase cosine schedule + topfarm polish.

Builds on schedule_fn concept: a smooth function controls lr, alpha,
beta1, beta2 per step. Uses alpha0 (gradient-scaled) for generalization.

Phase 1 (0-10%): High alpha feasibility, constant LR, low momentum
Phase 2 (10-55%): Low alpha exploration, gradual LR decay, high momentum
Phase 3 (55-85%): Ramping alpha enforcement, moderate LR, low momentum
Then topfarm polish for residual constraint cleanup.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # --- Wind-aware grid initialization ---
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
    angle = dominant + jnp.pi / 2

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)
    nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
    ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing / 2, rx_max - min_spacing / 2, nx),
        jnp.linspace(ry_min + min_spacing / 2, ry_max - min_spacing / 2, ny))
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
    cand_x, cand_y = orig_pts[:, 0], orig_pts[:, 1]

    n_verts = boundary.shape[0]
    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)
    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
    ix, iy = cand_x[inside], cand_y[inside]

    if len(ix) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(ix) - 1, n_target)).astype(int)
        init_x, init_y = ix[idx], iy[idx]
    else:
        key = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # === Compute alpha0 from gradient magnitude ===
    grad_obj_fn = jax.grad(objective, argnums=(0, 1))
    gox, goy = grad_obj_fn(init_x, init_y)
    lr0 = 150.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0

    # === Custom ADAM with cosine schedule ===
    def con_penalty(x, y):
        return boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)

    grad_con_fn = jax.grad(con_penalty, argnums=(0, 1))

    total_iters = 10000
    eps = 1e-12
    max_step = min_spacing

    # Phase boundaries (normalized)
    t1 = 0.10   # end feasibility
    t2 = 0.55   # end exploration

    # Penalty multipliers
    alpha_high = alpha0 * 6.0
    alpha_low = alpha0 * 0.03
    alpha_final = alpha0 * 8.0

    def adam_step(i, state):
        x, y, mx, my, vx, vy = state
        t = i / total_iters

        in_p1 = t < t1
        in_p2 = (t >= t1) & (t < t2)

        # LR: cosine decay from lr0 to 0.03*lr0
        lr = lr0 * (0.03 + 0.97 * 0.5 * (1.0 + jnp.cos(jnp.pi * t)))

        # Alpha: high → low → high (cosine)
        p2_frac = (t - t1) / (t2 - t1)
        p2_alpha = alpha_low + (alpha_high - alpha_low) * 0.5 * (1.0 + jnp.cos(jnp.pi * p2_frac))
        p3_frac = (t - t2) / (1.0 - t2)
        p3_alpha = alpha_low + (alpha_final - alpha_low) * 0.5 * (1.0 - jnp.cos(jnp.pi * p3_frac))
        alpha = jnp.where(in_p1, alpha_high,
                          jnp.where(in_p2, p2_alpha, p3_alpha))

        # Momentum: high during exploration
        b1 = jnp.where(in_p2, 0.8, 0.1)
        b2 = jnp.where(in_p2, 0.95, 0.2)

        # Separate gradients
        gox, goy = grad_obj_fn(x, y)
        gcx, gcy = grad_con_fn(x, y)
        jx = gox + alpha * gcx
        jy = goy + alpha * gcy

        # ADAM update
        mx = b1 * mx + (1.0 - b1) * jx
        my = b1 * my + (1.0 - b1) * jy
        vx = b2 * vx + (1.0 - b2) * jx ** 2
        vy = b2 * vy + (1.0 - b2) * jy ** 2

        it = (i + 1).astype(jnp.float64)
        bc1 = 1.0 / (1.0 - b1 ** it)
        bc2 = 1.0 / (1.0 - b2 ** it)

        sx = lr * (mx * bc1) / (jnp.sqrt(vx * bc2) + eps)
        sy = lr * (my * bc1) / (jnp.sqrt(vy * bc2) + eps)

        # Per-turbine step clipping
        d = jnp.sqrt(sx ** 2 + sy ** 2 + 1e-12)
        scale = jnp.minimum(1.0, max_step / d)
        sx = sx * scale
        sy = sy * scale

        return (x - sx, y - sy, mx, my, vx, vy)

    init_state = (init_x, init_y,
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y),
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y))

    explored_x, explored_y, _, _, _, _ = jax.lax.fori_loop(
        0, total_iters, adam_step, init_state)

    # === Topfarm polish ===
    polish = SGDSettings(
        learning_rate=25.0,
        max_iter=1200,
        additional_constant_lr_iterations=600,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=300.0,
        boundary_weight=300.0,
        ks_rho=150.0,
        gamma_min_factor=0.01,
    )

    opt_x, opt_y = topfarm_sgd_solve(objective, explored_x, explored_y,
                                      boundary, min_spacing, polish)
    return opt_x, opt_y
