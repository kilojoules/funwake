"""Tuned iter_058 variant: lower exploration alpha + extended exploration.

iter_058 got 5566.35 with alpha=2 in exploration, 13000 total iters.
This tries:
- Lower exploration alpha=1 (more AEP freedom)
- Longer exploration: 7000 iters (vs 6000)
- Higher enforcement peak: 700 (vs 600)
- Exploration LR starts at 200 (vs 150 in iter_058)
- Separate sp/bp: sp=1, bp=3 during exploration (slight decoupling)
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

    # === CUSTOM ADAM with momentum restarts ===
    def penalized_objective(x, y, alpha_sp, alpha_bp):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing)
        bp = boundary_penalty(x, y, boundary)
        return aep + alpha_sp * sp + alpha_bp * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    total_iters = 14000
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    max_step = min_spacing

    # Phase boundaries
    p1_end = 2000   # feasibility
    p2_end = 9000   # exploration (7000 iters)
    # p3: 9000-14000 enforcement (5000 iters)

    def adam_step(i, state):
        x, y, m_x, m_y, v_x, v_y = state

        # === Momentum warm restarts at phase boundaries ===
        at_restart = (i == p1_end) | (i == p2_end)
        m_x = jnp.where(at_restart, 0.0, m_x)
        m_y = jnp.where(at_restart, 0.0, m_y)
        v_x = jnp.where(at_restart, 1e-8, v_x)
        v_y = jnp.where(at_restart, 1e-8, v_y)

        in_p1 = i < p1_end
        in_p2 = (i >= p1_end) & (i < p2_end)

        # === Per-phase learning rate ===
        # Phase 1: LR 200 -> 50
        p1_lr = 200.0 * (0.9993 ** i)

        # Phase 2: LR 200 -> 25 (higher start for more exploration)
        p2_lr = 200.0 * (0.99970 ** (i - p1_end))

        # Phase 3: LR 80 -> 3
        p3_lr = 80.0 * (0.99935 ** (i - p2_end))

        lr = jnp.where(in_p1, p1_lr, jnp.where(in_p2, p2_lr, p3_lr))

        # === Decoupled penalty weights ===
        # Phase 1: High both
        alpha_sp = jnp.where(in_p1, 200.0,
                   jnp.where(in_p2, 1.0,
                   # Phase 3: cosine ramp 5 -> 700
                   5.0 + 695.0 * 0.5 * (1.0 - jnp.cos(
                       jnp.pi * (i - p2_end) / (total_iters - p2_end)))))

        alpha_bp = jnp.where(in_p1, 200.0,
                   jnp.where(in_p2, 3.0,
                   # Phase 3: cosine ramp 5 -> 700
                   5.0 + 695.0 * 0.5 * (1.0 - jnp.cos(
                       jnp.pi * (i - p2_end) / (total_iters - p2_end)))))

        gx, gy = grad_fn(x, y, alpha_sp, alpha_bp)

        m_x = beta1 * m_x + (1.0 - beta1) * gx
        m_y = beta1 * m_y + (1.0 - beta1) * gy
        v_x = beta2 * v_x + (1.0 - beta2) * gx ** 2
        v_y = beta2 * v_y + (1.0 - beta2) * gy ** 2

        # Phase-relative step count for bias correction
        step_in_phase = jnp.where(
            in_p1, i + 1,
            jnp.where(in_p2, i - p1_end + 1,
                       i - p2_end + 1)).astype(jnp.float64)

        bc1 = 1.0 / (1.0 - beta1 ** step_in_phase)
        bc2 = 1.0 / (1.0 - beta2 ** step_in_phase)

        step_x = lr * (m_x * bc1) / (jnp.sqrt(v_x * bc2) + eps)
        step_y = lr * (m_y * bc1) / (jnp.sqrt(v_y * bc2) + eps)

        # Per-turbine step clipping
        per_d = jnp.sqrt(step_x ** 2 + step_y ** 2 + 1e-12)
        scale = jnp.minimum(1.0, max_step / per_d)
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

    # === Topfarm polish for robust feasibility ===
    polish_settings = SGDSettings(
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
                                      boundary, min_spacing, polish_settings)

    return opt_x, opt_y
