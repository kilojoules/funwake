"""2-start best-of: perpendicular vs 60-degree-offset grid orientations.

Both starts use compact 2-stage topfarm_sgd_solve pipeline:
  Stage 1: Moderate exploration + enforcement (balanced)
  Stage 2: Final enforcement at low lr

Previous 3-stage pipelines took 24-33s each. This 2-stage is ~16-20s,
allowing 2 starts within the 60s budget.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # Stage 1: Balanced exploration + moderate enforcement
    settings_s1 = SGDSettings(
        learning_rate=150.0,
        max_iter=3000,
        additional_constant_lr_iterations=1500,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=50.0,
        boundary_weight=50.0,
        ks_rho=100.0,
        gamma_min_factor=0.01,
    )

    # Stage 2: Final enforcement + refinement
    settings_s2 = SGDSettings(
        learning_rate=50.0,
        max_iter=2500,
        additional_constant_lr_iterations=1250,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=200.0,
        boundary_weight=200.0,
        ks_rho=100.0,
        gamma_min_factor=0.01,
    )

    # --- Grid initialization helper ---
    def make_grid_init(angle):
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        centroid_x, centroid_y = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
        translated = boundary - jnp.array([centroid_x, centroid_y])
        rot_mat = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_bnd = (rot_mat @ translated.T).T

        rx_min, ry_min = jnp.min(rot_bnd, axis=0)
        rx_max, ry_max = jnp.max(rot_bnd, axis=0)
        n_x = int(jnp.ceil((rx_max - rx_min) / min_spacing))
        n_y = int(jnp.ceil((ry_max - ry_min) / min_spacing))
        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min + min_spacing / 2, rx_max - min_spacing / 2, n_x),
            jnp.linspace(ry_min + min_spacing / 2, ry_max - min_spacing / 2, n_y))
        rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
        inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
        orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([centroid_x, centroid_y])
        return orig_pts[:, 0], orig_pts[:, 1]

    def filter_inside(cx, cy):
        n_v = boundary.shape[0]
        def edge_d(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_v]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (cx - x1) * (-ey / el) + (cy - y1) * (ex / el)
        inside = jnp.min(jax.vmap(edge_d)(jnp.arange(n_v)), axis=0) > 0
        return cx[inside], cy[inside]

    def select_turbines(ix, iy, n):
        x_min, y_min = jnp.min(boundary, axis=0)
        x_max, y_max = jnp.max(boundary, axis=0)
        if len(ix) >= n:
            idx = jnp.round(jnp.linspace(0, len(ix) - 1, n)).astype(int)
            return ix[idx], iy[idx]
        else:
            key = jax.random.PRNGKey(42)
            rx = jax.random.uniform(key, (n,), minval=float(x_min), maxval=float(x_max))
            key, _ = jax.random.split(key)
            ry = jax.random.uniform(key, (n,), minval=float(y_min), maxval=float(y_max))
            return rx, ry

    def run_pipeline(init_x, init_y):
        x1, y1 = topfarm_sgd_solve(objective, init_x, init_y,
                                    boundary, min_spacing, settings_s1)
        x2, y2 = topfarm_sgd_solve(objective, x1, y1,
                                    boundary, min_spacing, settings_s2)
        return x2, y2

    # --- Compute dominant wind direction ---
    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))

    # Start 1: Perpendicular to dominant wind
    angle1 = dominant + jnp.pi / 2
    cx1, cy1 = make_grid_init(angle1)
    fx1, fy1 = filter_inside(cx1, cy1)
    init_x1, init_y1 = select_turbines(fx1, fy1, n_target)

    # Start 2: 60-degree offset
    angle2 = dominant + jnp.pi / 3
    cx2, cy2 = make_grid_init(angle2)
    fx2, fy2 = filter_inside(cx2, cy2)
    init_x2, init_y2 = select_turbines(fx2, fy2, n_target)

    # Run both pipelines
    opt_x1, opt_y1 = run_pipeline(init_x1, init_y1)
    opt_x2, opt_y2 = run_pipeline(init_x2, init_y2)

    # Pick the one with better (more negative) objective
    aep1 = objective(opt_x1, opt_y1)
    aep2 = objective(opt_x2, opt_y2)

    use_first = aep1 < aep2
    opt_x = jnp.where(use_first, opt_x1, opt_x2)
    opt_y = jnp.where(use_first, opt_y1, opt_y2)

    return opt_x, opt_y
