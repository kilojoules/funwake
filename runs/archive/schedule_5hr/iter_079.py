"""2-start ADAM: diverse initializations, pick best feasible.

All previous 68+ attempts used single-start from the same wind-aligned
grid. This explores 2 different basins of attraction with different
grid orientations. Takes the best feasible result.

Start 1: Wind-aligned grid (perpendicular to dominant wind, proven init)
Start 2: Grid at dominant wind angle +30 degrees (different basin)

Each start: condensed custom ADAM (8000 iters) + quick topfarm polish.
Target ~25s per start, 50s total.
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

    # --- Grid initialization helper ---
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    n_verts = boundary.shape[0]

    def make_grid_init(grid_angle):
        cos_a, sin_a = jnp.cos(grid_angle), jnp.sin(grid_angle)
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
            return ix[idx], iy[idx]
        else:
            key = jax.random.PRNGKey(42)
            rx = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
            key, _ = jax.random.split(key)
            ry = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))
            return rx, ry

    # Dominant wind direction
    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))

    # Two grid orientations
    angle1 = dominant + jnp.pi / 2       # perpendicular to wind (proven)
    angle2 = dominant + jnp.pi / 2 + jnp.pi / 6  # +30 degrees offset

    init_x1, init_y1 = make_grid_init(angle1)
    init_x2, init_y2 = make_grid_init(angle2)

    # === Custom ADAM (condensed for 2-start) ===
    def penalized_objective(x, y, alpha_sp, alpha_bp):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing)
        bp = boundary_penalty(x, y, boundary)
        return aep + alpha_sp * sp + alpha_bp * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    total_iters = 8000
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    max_step = min_spacing

    # Phase schedule: feasibility → explore → enforce
    p1_end = 1500   # feasibility
    p2_end = 5000   # exploration
    # p3: 5000-8000 enforcement

    def adam_step(i, state):
        x, y, m_x, m_y, v_x, v_y = state

        # Learning rate: exponential decay
        lr = 200.0 * (0.9994 ** i)  # ~200 → ~15

        in_p1 = i < p1_end
        in_p2 = (i >= p1_end) & (i < p2_end)

        # Penalty schedule
        # Phase 1: high (feasibility push)
        # Phase 2: low (AEP exploration)
        # Phase 3: cosine ramp to high (enforcement)
        p3_progress = (i - p2_end) / (total_iters - p2_end)
        p3_alpha = 2.0 + 398.0 * 0.5 * (1.0 - jnp.cos(jnp.pi * p3_progress))

        alpha_sp = jnp.where(in_p1, 200.0, jnp.where(in_p2, 0.5, p3_alpha))
        alpha_bp = jnp.where(in_p1, 200.0, jnp.where(in_p2, 15.0, p3_alpha))

        gx, gy = grad_fn(x, y, alpha_sp, alpha_bp)

        m_x = beta1 * m_x + (1.0 - beta1) * gx
        m_y = beta1 * m_y + (1.0 - beta1) * gy
        v_x = beta2 * v_x + (1.0 - beta2) * gx ** 2
        v_y = beta2 * v_y + (1.0 - beta2) * gy ** 2

        it = (i + 1).astype(jnp.float64)
        bc1 = 1.0 / (1.0 - beta1 ** it)
        bc2 = 1.0 / (1.0 - beta2 ** it)

        step_x = lr * (m_x * bc1) / (jnp.sqrt(v_x * bc2) + eps)
        step_y = lr * (m_y * bc1) / (jnp.sqrt(v_y * bc2) + eps)

        per_d = jnp.sqrt(step_x ** 2 + step_y ** 2 + 1e-12)
        scale = jnp.minimum(1.0, max_step / per_d)
        step_x = step_x * scale
        step_y = step_y * scale

        return (x - step_x, y - step_y, m_x, m_y, v_x, v_y)

    def run_adam(init_x, init_y):
        init_state = (init_x, init_y,
                      jnp.zeros_like(init_x), jnp.zeros_like(init_y),
                      jnp.zeros_like(init_x), jnp.zeros_like(init_y))
        return jax.lax.fori_loop(0, total_iters, adam_step, init_state)[:2]

    # Run start 1
    ex1, ey1 = run_adam(init_x1, init_y1)

    # Run start 2 (JIT cache warm → faster)
    ex2, ey2 = run_adam(init_x2, init_y2)

    # === Topfarm polish (same for both) ===
    polish = SGDSettings(
        learning_rate=30.0,
        max_iter=1000,
        additional_constant_lr_iterations=500,
        tol=1e-6,
        beta1=0.1, beta2=0.2,
        spacing_weight=300.0, boundary_weight=300.0,
        ks_rho=150.0, gamma_min_factor=0.01,
    )

    ox1, oy1 = topfarm_sgd_solve(objective, ex1, ey1,
                                   boundary, min_spacing, polish)
    ox2, oy2 = topfarm_sgd_solve(objective, ex2, ey2,
                                   boundary, min_spacing, polish)

    # Pick the start with better AEP
    aep1 = objective(ox1, oy1)
    aep2 = objective(ox2, oy2)

    # Lower is better (objective is negative AEP)
    use_first = aep1 < aep2
    opt_x = jnp.where(use_first, ox1, ox2)
    opt_y = jnp.where(use_first, oy1, oy2)

    return opt_x, opt_y
