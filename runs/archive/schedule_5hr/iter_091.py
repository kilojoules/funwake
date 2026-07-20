"""Anisotropic wind-aligned init + decoupled penalty schedules.

Key differences from iter_058 (5566.35 GWh best):
1. Anisotropic grid: min_spacing cross-wind, 3x along-wind (wake-aware)
2. Separate boundary & spacing alphas (boundary stays moderate, spacing very
   low during exploration phase to allow turbines to "slide past" each other)
3. 4-phase ADAM with longer exploration (15000 total iters)
4. Cosine LR annealing within each phase (smoother than exponential)
5. Phase 2 spacing alpha = 0.1 (vs 2.0 in iter_058) for deeper exploration

Phase structure:
- Phase 1 (0-1000):   Feasibility push, sp=200 bp=200, LR cosine 200→60
- Phase 2 (1000-7000): Deep exploration, sp=0.1 bp=5, LR cosine 180→15
  [MOMENTUM RESET at i=1000]
- Phase 3 (7000-12000): Progressive enforcement, sp cosine 1→500 bp cosine 5→500,
  LR cosine 80→8    [MOMENTUM RESET at i=7000]
- Phase 4 (12000-15000): Strict enforcement, sp=600 bp=600, LR cosine 20→2
  [MOMENTUM RESET at i=12000]

Then topfarm polish (1200+600 iters).
"""
import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # --- Anisotropic wind-aligned grid initialization ---
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
    angle = dominant + jnp.pi / 2  # perpendicular to dominant wind

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)

    # Anisotropic spacing: tight cross-wind, wide along-wind
    cross_wind_spacing = min_spacing
    along_wind_spacing = min_spacing * 3.0

    nx = int(jnp.ceil((rx_max - rx_min) / cross_wind_spacing))
    ny = int(jnp.ceil((ry_max - ry_min) / along_wind_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + cross_wind_spacing / 2, rx_max - cross_wind_spacing / 2, nx),
        jnp.linspace(ry_min + along_wind_spacing / 2, ry_max - along_wind_spacing / 2, ny))
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
        # Farthest-point sampling for max spatial diversity
        pts_x = np.array(ix)
        pts_y = np.array(iy)
        n_pts = len(pts_x)

        cx_f, cy_f = float(cx), float(cy)
        dists_to_center = (pts_x - cx_f)**2 + (pts_y - cy_f)**2
        first_idx = int(np.argmin(dists_to_center))

        selected = [first_idx]
        min_dist_sq = np.full(n_pts, np.inf)
        min_dist_sq[first_idx] = -1.0

        dx = pts_x - pts_x[first_idx]
        dy = pts_y - pts_y[first_idx]
        d_sq = dx**2 + dy**2
        min_dist_sq = np.minimum(min_dist_sq, d_sq)
        min_dist_sq[first_idx] = -1.0

        for _ in range(n_target - 1):
            next_idx = int(np.argmax(min_dist_sq))
            selected.append(next_idx)
            dx = pts_x - pts_x[next_idx]
            dy = pts_y - pts_y[next_idx]
            d_sq = dx**2 + dy**2
            min_dist_sq = np.minimum(min_dist_sq, d_sq)
            min_dist_sq[next_idx] = -1.0

        selected = np.array(selected)
        init_x = jnp.array(pts_x[selected])
        init_y = jnp.array(pts_y[selected])
    else:
        key = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # === Custom ADAM with 4-phase decoupled penalty schedule ===
    def penalized_objective(x, y, alpha_sp, alpha_bp):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing)
        bp = boundary_penalty(x, y, boundary)
        return aep + alpha_sp * sp + alpha_bp * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    total_iters = 15000
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    max_step = min_spacing

    # Phase boundaries
    p1_end = 1000
    p2_end = 7000
    p3_end = 12000
    # Phase 4: 12000-15000

    def cosine_anneal(i, start_i, end_i, high, low):
        """Cosine annealing from high to low over [start_i, end_i)."""
        progress = (i - start_i) / (end_i - start_i)
        return low + 0.5 * (high - low) * (1.0 + jnp.cos(jnp.pi * progress))

    def adam_step(i, state):
        x, y, m_x, m_y, v_x, v_y = state

        # Momentum warm restarts at phase boundaries
        at_restart = (i == p1_end) | (i == p2_end) | (i == p3_end)
        m_x = jnp.where(at_restart, 0.0, m_x)
        m_y = jnp.where(at_restart, 0.0, m_y)
        v_x = jnp.where(at_restart, 1e-8, v_x)
        v_y = jnp.where(at_restart, 1e-8, v_y)

        in_p1 = i < p1_end
        in_p2 = (i >= p1_end) & (i < p2_end)
        in_p3 = (i >= p2_end) & (i < p3_end)

        # Per-phase learning rate (cosine annealing within phase)
        p1_lr = cosine_anneal(i, 0, p1_end, 200.0, 60.0)
        p2_lr = cosine_anneal(i, p1_end, p2_end, 180.0, 15.0)
        p3_lr = cosine_anneal(i, p2_end, p3_end, 80.0, 8.0)
        p4_lr = cosine_anneal(i, p3_end, total_iters, 20.0, 2.0)
        lr = jnp.where(in_p1, p1_lr,
             jnp.where(in_p2, p2_lr,
             jnp.where(in_p3, p3_lr, p4_lr)))

        # Decoupled spacing and boundary penalty schedules
        # Spacing: very low during exploration, ramps up in phases 3-4
        sp_alpha = jnp.where(in_p1, 200.0,
                   jnp.where(in_p2, 0.1,
                   jnp.where(in_p3,
                       cosine_anneal(i, p2_end, p3_end, 500.0, 1.0),  # ramp up 1→500 (reversed cosine)
                       600.0)))
        # Fix: cosine_anneal goes high→low, so for ramping UP we swap args
        sp_alpha = jnp.where(in_p3,
                       1.0 + 499.0 * 0.5 * (1.0 - jnp.cos(jnp.pi * (i - p2_end) / (p3_end - p2_end))),
                       sp_alpha)

        # Boundary: always moderate (never too low to avoid turbines exiting polygon)
        bp_alpha = jnp.where(in_p1, 200.0,
                   jnp.where(in_p2, 5.0,
                   jnp.where(in_p3,
                       5.0 + 495.0 * 0.5 * (1.0 - jnp.cos(jnp.pi * (i - p2_end) / (p3_end - p2_end))),
                       600.0)))

        g_x, g_y = grad_fn(x, y, sp_alpha, bp_alpha)

        m_x = beta1 * m_x + (1.0 - beta1) * g_x
        m_y = beta1 * m_y + (1.0 - beta1) * g_y
        v_x = beta2 * v_x + (1.0 - beta2) * g_x ** 2
        v_y = beta2 * v_y + (1.0 - beta2) * g_y ** 2

        # Phase-relative step count for bias correction
        step_in_phase = jnp.where(
            in_p1, i + 1,
            jnp.where(in_p2, i - p1_end + 1,
            jnp.where(in_p3, i - p2_end + 1,
                      i - p3_end + 1))).astype(jnp.float64)

        bc1 = 1.0 / (1.0 - beta1 ** step_in_phase)
        bc2 = 1.0 / (1.0 - beta2 ** step_in_phase)

        step_x = lr * (m_x * bc1) / (jnp.sqrt(v_x * bc2) + eps)
        step_y = lr * (m_y * bc1) / (jnp.sqrt(v_y * bc2) + eps)

        # Per-turbine step clipping
        per_d = jnp.sqrt(step_x ** 2 + step_y ** 2 + 1e-12)
        scale = jnp.minimum(1.0, max_step / per_d)

        return (x - step_x * scale, y - step_y * scale,
                m_x, m_y, v_x, v_y)

    init_state = (init_x, init_y,
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y),
                  jnp.zeros_like(init_x), jnp.zeros_like(init_y))

    explored_x, explored_y, _, _, _, _ = jax.lax.fori_loop(
        0, total_iters, adam_step, init_state)

    # === Topfarm polish for robust feasibility ===
    polish = SGDSettings(
        learning_rate=25.0,
        max_iter=1200,
        additional_constant_lr_iterations=600,
        tol=1e-6,
        beta1=0.1, beta2=0.2,
        spacing_weight=300.0, boundary_weight=300.0,
        ks_rho=150.0, gamma_min_factor=0.01,
    )

    opt_x, opt_y = topfarm_sgd_solve(objective, explored_x, explored_y,
                                      boundary, min_spacing, polish)

    return opt_x, opt_y
