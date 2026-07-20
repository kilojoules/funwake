"""2-start topfarm: run 2 different initializations, pick best.

All previous attempts use the same wind-aware grid, always converging
to the same local optimum (~5556). This tries 2 starts:
- Start 1: Standard wind-aware grid
- Start 2: Grid shifted by 0.5*min_spacing in the dominant wind direction

Each start does a compact 2-stage topfarm (explore + enforce).
Pick the one with higher AEP (if both feasible).
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # Stage 1: Explore with low constraints
    settings_explore = SGDSettings(
        learning_rate=150.0,
        max_iter=2500,
        additional_constant_lr_iterations=1200,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=2.0,
        boundary_weight=2.0,
        ks_rho=50.0,
        gamma_min_factor=0.01,
    )

    # Stage 2: Enforce constraints
    settings_enforce = SGDSettings(
        learning_rate=50.0,
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=300.0,
        boundary_weight=300.0,
        ks_rho=150.0,
        gamma_min_factor=0.01,
    )

    # --- Helper: create wind-aware grid init with offset ---
    def make_init(offset_frac):
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

        # Apply offset in rotated space
        x_offset = offset_frac * min_spacing
        y_offset = offset_frac * min_spacing * 0.5

        nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
        ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))
        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min + min_spacing / 2 + x_offset,
                         rx_max - min_spacing / 2 + x_offset, nx),
            jnp.linspace(ry_min + min_spacing / 2 + y_offset,
                         ry_max - min_spacing / 2 + y_offset, ny))
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

        x_min, y_min = jnp.min(boundary, axis=0)
        x_max, y_max = jnp.max(boundary, axis=0)
        if len(ix) >= n_target:
            idx = jnp.round(jnp.linspace(0, len(ix) - 1, n_target)).astype(int)
            return ix[idx], iy[idx]
        else:
            key = jax.random.PRNGKey(42)
            _x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
            key, _ = jax.random.split(key)
            _y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))
            return _x, _y

    # Run start 1: standard grid
    init_x1, init_y1 = make_init(0.0)
    x1e, y1e = topfarm_sgd_solve(objective, init_x1, init_y1,
                                  boundary, min_spacing, settings_explore)
    x1f, y1f = topfarm_sgd_solve(objective, x1e, y1e,
                                  boundary, min_spacing, settings_enforce)
    aep1 = objective(x1f, y1f)

    # Run start 2: shifted grid
    init_x2, init_y2 = make_init(0.4)
    x2e, y2e = topfarm_sgd_solve(objective, init_x2, init_y2,
                                  boundary, min_spacing, settings_explore)
    x2f, y2f = topfarm_sgd_solve(objective, x2e, y2e,
                                  boundary, min_spacing, settings_enforce)
    aep2 = objective(x2f, y2f)

    # Pick best (lower objective = higher AEP since objective is negative)
    use_start2 = aep2 < aep1
    opt_x = jnp.where(use_start2, x2f, x1f)
    opt_y = jnp.where(use_start2, y2f, y1f)

    return opt_x, opt_y
