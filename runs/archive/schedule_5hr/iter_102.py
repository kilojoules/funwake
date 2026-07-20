"""Enhanced ADAM with gradient noise + separate sp/bp penalties.

Building on iter_058 (proven 5566.35 GWh). Key differences:
1. Gradient noise injection in exploration phase (Langevin-style)
   helps escape shallow local optima — genuinely new approach
2. Separate spacing vs boundary penalty weights (boundary harder to fix)
3. Gradual alpha ramp in phase 2 (sp: 2->10, bp: 8->30) to avoid
   wasted work on deeply infeasible configurations
4. Higher final enforcement (alpha_sp->700, alpha_bp->900)
5. 14000 total iterations (vs 13000) with rebalanced phases

Phase structure:
- Phase 1 (0-1500): Feasibility push, sp=200, bp=250, LR 200->80
- Phase 2 (1500-8000): Exploration with noise, sp=2->10, bp=8->30, LR 160->25
- Phase 3 (8000-14000): Enforcement, sp cosine 10->700, bp cosine 30->900, LR 100->3
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

    # --- Wind-aware grid initialization (proven) ---
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

    # === Custom ADAM with gradient noise + separate penalties ===
    def penalized_objective(x, y, alpha_sp, alpha_bp):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing)
        bp = boundary_penalty(x, y, boundary)
        return aep + alpha_sp * sp + alpha_bp * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    total_iters = 14000
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    max_step = min_spacing

    p1_end = 1500
    p2_end = 8000

    def adam_step(i, state):
        x, y, m_x, m_y, v_x, v_y, rng_key = state

        # Momentum resets at phase boundaries
        at_restart = (i == p1_end) | (i == p2_end)
        m_x = jnp.where(at_restart, 0.0, m_x)
        m_y = jnp.where(at_restart, 0.0, m_y)
        v_x = jnp.where(at_restart, 1e-8, v_x)
        v_y = jnp.where(at_restart, 1e-8, v_y)

        in_p1 = i < p1_end
        in_p2 = (i >= p1_end) & (i < p2_end)

        # === Per-phase learning rate ===
        p1_lr = 200.0 * (0.9994 ** i)               # ~80 at i=1500
        p2_lr = 160.0 * (0.99971 ** (i - p1_end))   # ~25 at i=8000
        p3_lr = 100.0 * (0.99942 ** (i - p2_end))   # ~3 at i=14000
        lr = jnp.where(in_p1, p1_lr, jnp.where(in_p2, p2_lr, p3_lr))

        # === Separate spacing/boundary penalty weights ===
        # Phase 1: high for feasibility
        p1_alpha_sp = 200.0
        p1_alpha_bp = 250.0

        # Phase 2: low but gradually increasing (2->10 for sp, 8->30 for bp)
        p2_frac = (i - p1_end) / (p2_end - p1_end)
        p2_alpha_sp = 2.0 + 8.0 * p2_frac
        p2_alpha_bp = 8.0 + 22.0 * p2_frac

        # Phase 3: cosine ramp to high enforcement
        p3_frac = (i - p2_end) / (total_iters - p2_end)
        p3_cos = 0.5 * (1.0 - jnp.cos(jnp.pi * p3_frac))
        p3_alpha_sp = 10.0 + 690.0 * p3_cos   # 10 -> 700
        p3_alpha_bp = 30.0 + 870.0 * p3_cos   # 30 -> 900

        alpha_sp = jnp.where(in_p1, p1_alpha_sp,
                   jnp.where(in_p2, p2_alpha_sp, p3_alpha_sp))
        alpha_bp = jnp.where(in_p1, p1_alpha_bp,
                   jnp.where(in_p2, p2_alpha_bp, p3_alpha_bp))

        g_x, g_y = grad_fn(x, y, alpha_sp, alpha_bp)

        # === Gradient noise injection in exploration phase ===
        # Langevin-style: noise ~ N(0, sigma^2) where sigma decays with progress
        # Only active in phase 2 (exploration)
        rng_key, k1, k2 = jax.random.split(rng_key, 3)
        noise_scale = jnp.where(in_p2,
            lr * 0.08 * (1.0 - p2_frac),  # decays from 0.08*lr to 0
            0.0)
        g_x = g_x + noise_scale * jax.random.normal(k1, shape=x.shape)
        g_y = g_y + noise_scale * jax.random.normal(k2, shape=y.shape)

        # ADAM update
        m_x = beta1 * m_x + (1.0 - beta1) * g_x
        m_y = beta1 * m_y + (1.0 - beta1) * g_y
        v_x = beta2 * v_x + (1.0 - beta2) * g_x ** 2
        v_y = beta2 * v_y + (1.0 - beta2) * g_y ** 2

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

        return (x - step_x * scale, y - step_y * scale,
                m_x, m_y, v_x, v_y, rng_key)

    rng_key = jax.random.PRNGKey(42)
    init_state = (init_x, init_y,
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y),
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y),
                  rng_key)

    explored_x, explored_y, _, _, _, _, _ = jax.lax.fori_loop(
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
