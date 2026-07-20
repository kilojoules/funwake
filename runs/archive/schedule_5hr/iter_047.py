"""Custom ADAM fori_loop with smooth cosine penalty annealing + topfarm polish.

Previous approaches used discrete penalty phase jumps that either:
- Destroy AEP (3-stage topfarm: sw=0.5 → sw=300, ~5557)
- Fail feasibility (iter_032: max alpha=100, 5600 infeasible)

This uses a smooth cosine schedule for penalty weight:
- Iters 0-2000: alpha=300 (feasibility establishment)
- Iters 2000-7000: alpha cosine-decays 300→1 (smooth exploration)
- Iters 7000-12000: alpha cosine-ramps 1→400 (smooth enforcement)

Then a very brief topfarm_sgd_solve polish to fix residual violations.
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

    # === CUSTOM ADAM with smooth cosine penalty annealing ===
    def penalized_objective(x, y, alpha_sp, alpha_bp):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing)
        bp = boundary_penalty(x, y, boundary)
        return aep + alpha_sp * sp + alpha_bp * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    total_iters = 12000
    initial_lr = 200.0
    lr_decay = 0.9995

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    max_step = min_spacing

    # Penalty schedule breakpoints
    phase1_end = 2000    # feasibility
    phase2_end = 7000    # exploration (cosine decay)
    # phase3: 7000-12000  enforcement (cosine ramp)

    alpha_high = 300.0   # feasibility/enforcement peak
    alpha_low = 1.0      # exploration trough
    alpha_final = 400.0  # strong final enforcement

    def adam_step(i, state):
        x, y, m_x, m_y, v_x, v_y = state

        lr = initial_lr * (lr_decay ** i)

        # Smooth penalty schedule
        in_p1 = i < phase1_end
        in_p2 = (i >= phase1_end) & (i < phase2_end)

        # Phase 1: constant high alpha
        # Phase 2: cosine decay from alpha_high to alpha_low
        p2_progress = (i - phase1_end) / (phase2_end - phase1_end)
        p2_alpha = alpha_low + (alpha_high - alpha_low) * 0.5 * (1.0 + jnp.cos(jnp.pi * p2_progress))

        # Phase 3: cosine ramp from alpha_low to alpha_final
        p3_progress = (i - phase2_end) / (total_iters - phase2_end)
        p3_alpha = alpha_low + (alpha_final - alpha_low) * 0.5 * (1.0 - jnp.cos(jnp.pi * p3_progress))

        alpha = jnp.where(in_p1, alpha_high,
                          jnp.where(in_p2, p2_alpha, p3_alpha))

        gx, gy = grad_fn(x, y, alpha, alpha)

        m_x = beta1 * m_x + (1.0 - beta1) * gx
        m_y = beta1 * m_y + (1.0 - beta1) * gy
        v_x = beta2 * v_x + (1.0 - beta2) * gx ** 2
        v_y = beta2 * v_y + (1.0 - beta2) * gy ** 2

        step = (i + 1).astype(jnp.float64)
        bc1 = 1.0 / (1.0 - beta1 ** step)
        bc2 = 1.0 / (1.0 - beta2 ** step)

        step_x = lr * (m_x * bc1) / (jnp.sqrt(v_x * bc2) + eps)
        step_y = lr * (m_y * bc1) / (jnp.sqrt(v_y * bc2) + eps)

        # Per-turbine step clipping
        per_turbine_dist = jnp.sqrt(step_x ** 2 + step_y ** 2 + 1e-12)
        scale = jnp.minimum(1.0, max_step / per_turbine_dist)
        step_x = step_x * scale
        step_y = step_y * scale

        x = x - step_x
        y = y - step_y

        return (x, y, m_x, m_y, v_x, v_y)

    init_state = (init_x, init_y,
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y),
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y))

    explored_x, explored_y, _, _, _, _ = jax.lax.fori_loop(
        0, total_iters, adam_step, init_state)

    # === Gentle topfarm polish ===
    polish_settings = SGDSettings(
        learning_rate=25.0,
        max_iter=1200,
        additional_constant_lr_iterations=600,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=250.0,
        boundary_weight=250.0,
        ks_rho=120.0,
        gamma_min_factor=0.01,
    )

    opt_x, opt_y = topfarm_sgd_solve(objective, explored_x, explored_y,
                                      boundary, min_spacing, polish_settings)

    return opt_x, opt_y
