"""Hybrid: custom ADAM exploration (8000 iters) → topfarm_sgd_solve repair.

Strategy: Custom ADAM fori_loop finds high-AEP regions with relaxed
constraints, then topfarm_sgd_solve with moderate enforcement repairs
feasibility violations without losing much AEP. Previous best feasible
was 5557.2 GWh; infeasible best was 5600.0 GWh. This aims to close
the gap by starting from the high-AEP region and gently repairing.
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

    # --- Wind-aware grid initialization with jitter ---
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

    # Add small jitter to break grid symmetry
    key = jax.random.PRNGKey(7)
    jitter_scale = min_spacing * 0.15
    init_x = init_x + jax.random.normal(key, init_x.shape) * jitter_scale
    key, _ = jax.random.split(key)
    init_y = init_y + jax.random.normal(key, init_y.shape) * jitter_scale

    # ============================================================
    # PHASE 1: Custom ADAM exploration (8000 iterations)
    # ============================================================
    def penalized_objective(x, y, alpha_sp, alpha_bp):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing)
        bp = boundary_penalty(x, y, boundary)
        return aep + alpha_sp * sp + alpha_bp * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    total_iters = 8000
    initial_lr = 200.0
    lr_decay = 0.9994  # decays to ~200 * 0.9994^8000 ≈ 1.6

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # 3-phase penalty schedule
    phase1_end = 2500   # feasibility
    phase2_end = 6500   # AEP exploration
    # phase3: 6500-8000  moderate enforcement

    alpha_p1 = 200.0    # strong feasibility
    alpha_p2 = 2.0      # relaxed for AEP exploration
    alpha_p3 = 80.0     # moderate re-enforcement

    def adam_step(i, state):
        x, y, m_x, m_y, v_x, v_y = state

        lr = initial_lr * (lr_decay ** i)

        in_p1 = i < phase1_end
        in_p2 = (i >= phase1_end) & (i < phase2_end)
        alpha = jnp.where(in_p1, alpha_p1,
                          jnp.where(in_p2, alpha_p2, alpha_p3))

        gx, gy = grad_fn(x, y, alpha, alpha)

        m_x = beta1 * m_x + (1.0 - beta1) * gx
        m_y = beta1 * m_y + (1.0 - beta1) * gy
        v_x = beta2 * v_x + (1.0 - beta2) * gx ** 2
        v_y = beta2 * v_y + (1.0 - beta2) * gy ** 2

        step = (i + 1).astype(jnp.float64)
        bc1 = 1.0 / (1.0 - beta1 ** step)
        bc2 = 1.0 / (1.0 - beta2 ** step)

        x = x - lr * (m_x * bc1) / (jnp.sqrt(v_x * bc2) + eps)
        y = y - lr * (m_y * bc1) / (jnp.sqrt(v_y * bc2) + eps)

        return (x, y, m_x, m_y, v_x, v_y)

    init_state = (init_x, init_y,
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y),
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y))

    explored_x, explored_y, _, _, _, _ = jax.lax.fori_loop(
        0, total_iters, adam_step, init_state)

    # ============================================================
    # PHASE 2: topfarm_sgd_solve constraint repair
    # ============================================================
    repair_settings = SGDSettings(
        learning_rate=30.0,
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=200.0,
        boundary_weight=200.0,
        ks_rho=100.0,
        gamma_min_factor=0.01,
    )

    opt_x, opt_y = topfarm_sgd_solve(objective, explored_x, explored_y,
                                      boundary, min_spacing, repair_settings)

    return opt_x, opt_y
