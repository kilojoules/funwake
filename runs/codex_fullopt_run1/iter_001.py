"""Two-start TopFarm SGD with wind-aware rotated initialization.

HYPOTHESIS: A wind-perpendicular grid plus one perturbed restart gives the
TopFarm-style SGD solver enough diversity to escape the plain grid layout
without exceeding the scoring timeout.

AXIS: topfarm_sgd_solve with wind-aware initialization and one local restart.

LESSON: Pending score.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    x_min = float(jnp.min(boundary[:, 0]))
    x_max = float(jnp.max(boundary[:, 0]))
    y_min = float(jnp.min(boundary[:, 1]))
    y_max = float(jnp.max(boundary[:, 1]))
    n_verts = boundary.shape[0]

    def is_inside(xi, yi):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex = x2 - x1
            ey = y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (xi - x1) * (-ey / el) + (yi - y1) * (ex / el)

        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0.0

    def rotated_grid(angle, spacing_mult):
        cx = jnp.mean(boundary[:, 0])
        cy = jnp.mean(boundary[:, 1])
        ca = jnp.cos(angle)
        sa = jnp.sin(angle)
        translated = boundary - jnp.array([cx, cy])
        rot = jnp.array([[ca, -sa], [sa, ca]])
        rot_bnd = (rot @ translated.T).T
        rx_min, ry_min = jnp.min(rot_bnd, axis=0)
        rx_max, ry_max = jnp.max(rot_bnd, axis=0)

        spacing = min_spacing * spacing_mult
        nx = max(4, int(jnp.ceil((rx_max - rx_min) / spacing)) + 1)
        ny = max(4, int(jnp.ceil((ry_max - ry_min) / spacing)) + 1)
        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min + 0.45 * spacing, rx_max - 0.45 * spacing, nx),
            jnp.linspace(ry_min + 0.45 * spacing, ry_max - 0.45 * spacing, ny),
        )
        rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=1)
        inv_rot = jnp.array([[ca, sa], [-sa, ca]])
        pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
        inside = jax.vmap(is_inside)(pts[:, 0], pts[:, 1])
        px = pts[:, 0][inside]
        py = pts[:, 1][inside]
        if len(px) >= n_target:
            idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
            return px[idx], py[idx]

        key = jax.random.PRNGKey(17)
        ix = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
        key, _ = jax.random.split(key)
        iy = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)
        return ix, iy

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    init_x, init_y = rotated_grid(mean_wind + jnp.pi / 2.0, 1.0)

    settings = SGDSettings(
        learning_rate=150.0,
        max_iter=3500,
        additional_constant_lr_iterations=1500,
        tol=1e-7,
        beta1=0.12,
        beta2=0.22,
        gamma_min_factor=0.0025,
        ks_rho=120.0,
        spacing_weight=100.0,
        boundary_weight=100.0,
    )

    opt_x1, opt_y1 = topfarm_sgd_solve(
        objective, init_x, init_y, boundary, min_spacing, settings
    )
    obj1 = objective(opt_x1, opt_y1)

    key = jax.random.PRNGKey(777)
    noise = 0.4 * min_spacing
    dx = jax.random.normal(key, (n_target,)) * noise
    key, _ = jax.random.split(key)
    dy = jax.random.normal(key, (n_target,)) * noise
    init_x2 = jnp.clip(opt_x1 + dx, x_min, x_max)
    init_y2 = jnp.clip(opt_y1 + dy, y_min, y_max)

    opt_x2, opt_y2 = topfarm_sgd_solve(
        objective, init_x2, init_y2, boundary, min_spacing, settings
    )
    obj2 = objective(opt_x2, opt_y2)

    if obj1 < obj2:
        return opt_x1, opt_y1
    return opt_x2, opt_y2
