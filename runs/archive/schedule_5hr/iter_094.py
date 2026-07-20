"""iter_058 ADAM schedule + 2-stage topfarm polish.

Hypothesis: iter_058's single polish (sw=300,bw=300,lr=25) may lock positions
too early. A 2-stage polish allows:
1. Stage 1: moderate penalty (sw=50,bw=50) + high LR (80) - allows turbines
   to keep moving and improving AEP while gradually enforcing constraints
2. Stage 2: strong penalty (sw=500,bw=500) + low LR (15) - strict feasibility

Everything else identical to iter_058 (proven 5566.35 GWh).
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

    # --- Wind-aware grid initialization (identical to iter_058) ---
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

    # === Custom ADAM - identical to iter_058 ===
    def penalized_objective(x, y, alpha_sp, alpha_bp):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing)
        bp = boundary_penalty(x, y, boundary)
        return aep + alpha_sp * sp + alpha_bp * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    total_iters = 13000
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    max_step = min_spacing

    p1_end = 2000
    p2_end = 8000

    def adam_step(i, state):
        x, y, m_x, m_y, v_x, v_y = state

        at_restart = (i == p1_end) | (i == p2_end)
        m_x = jnp.where(at_restart, 0.0, m_x)
        m_y = jnp.where(at_restart, 0.0, m_y)
        v_x = jnp.where(at_restart, 1e-8, v_x)
        v_y = jnp.where(at_restart, 1e-8, v_y)

        in_p1 = i < p1_end
        in_p2 = (i >= p1_end) & (i < p2_end)

        p1_lr = 200.0 * (0.9993 ** i)
        p2_lr = 150.0 * (0.99973 ** (i - p1_end))
        p3_lr = 80.0 * (0.99944 ** (i - p2_end))
        lr = jnp.where(in_p1, p1_lr, jnp.where(in_p2, p2_lr, p3_lr))

        alpha = jnp.where(in_p1, 200.0,
                jnp.where(in_p2, 2.0,
                5.0 + 595.0 * 0.5 * (1.0 - jnp.cos(
                    jnp.pi * (i - p2_end) / (total_iters - p2_end)))))

        g_x, g_y = grad_fn(x, y, alpha, alpha)

        m_x = beta1 * m_x + (1.0 - beta1) * g_x
        m_y = beta1 * m_y + (1.0 - beta1) * g_y
        v_x = beta2 * v_x + (1.0 - beta2) * g_x ** 2
        v_y = beta2 * v_y + (1.0 - beta2) * g_y ** 2

        step_in_phase = jnp.where(
            in_p1, i + 1,
            jnp.where(in_p2, i - p1_end + 1,
                      i - p2_end + 1)).astype(jnp.float64)

        bc1 = 1.0 / (1.0 - beta1 ** step_in_phase)
        bc2 = 1.0 / (1.0 - beta2 ** step_in_phase)

        step_x = lr * (m_x * bc1) / (jnp.sqrt(v_x * bc2) + eps)
        step_y = lr * (m_y * bc1) / (jnp.sqrt(v_y * bc2) + eps)

        per_d = jnp.sqrt(step_x ** 2 + step_y ** 2 + 1e-12)
        scale = jnp.minimum(1.0, max_step / per_d)

        return (x - step_x * scale, y - step_y * scale,
                m_x, m_y, v_x, v_y)

    init_state = (init_x, init_y,
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y),
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y))

    explored_x, explored_y, _, _, _, _ = jax.lax.fori_loop(
        0, total_iters, adam_step, init_state)

    # === 2-Stage topfarm polish ===

    # Stage 1: Moderate penalty, high LR - still improves AEP
    polish1 = SGDSettings(
        learning_rate=80.0,
        max_iter=1500,
        additional_constant_lr_iterations=800,
        tol=1e-6,
        beta1=0.1, beta2=0.2,
        spacing_weight=50.0, boundary_weight=50.0,
        ks_rho=100.0, gamma_min_factor=0.01,
    )

    x1, y1 = topfarm_sgd_solve(objective, explored_x, explored_y,
                                 boundary, min_spacing, polish1)

    # Stage 2: Strong penalty, low LR - strict feasibility lock
    polish2 = SGDSettings(
        learning_rate=15.0,
        max_iter=800,
        additional_constant_lr_iterations=400,
        tol=1e-6,
        beta1=0.1, beta2=0.2,
        spacing_weight=500.0, boundary_weight=500.0,
        ks_rho=200.0, gamma_min_factor=0.01,
    )

    opt_x, opt_y = topfarm_sgd_solve(objective, x1, y1,
                                      boundary, min_spacing, polish2)

    return opt_x, opt_y
