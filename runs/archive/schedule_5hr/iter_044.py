"""Hex grid + aggressive exploration (sw=0.5) + strong enforcement.

Testing if near-zero spacing penalty in exploration finds a higher-AEP
basin than sw=2. Hex grid from iter_039 (nearly matched best feasible).
Keep boundary penalty moderate (bw=10) during exploration to prevent
turbines leaving the boundary entirely.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # Stage 1: Feasibility push
    settings_s1 = SGDSettings(
        learning_rate=200.0,
        max_iter=1500,
        additional_constant_lr_iterations=750,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=200.0,
        boundary_weight=200.0,
        ks_rho=100.0,
        gamma_min_factor=0.01,
    )

    # Stage 2: Aggressive AEP exploration (near-zero spacing penalty)
    settings_s2 = SGDSettings(
        learning_rate=100.0,
        max_iter=3000,
        additional_constant_lr_iterations=1500,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=0.5,
        boundary_weight=10.0,
        ks_rho=50.0,
        gamma_min_factor=0.01,
    )

    # Stage 3: Strong enforcement to repair
    settings_s3 = SGDSettings(
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

    # --- Hexagonal grid initialization ---
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

    row_spacing = min_spacing * 0.866
    n_cols = int(jnp.ceil((rx_max - rx_min) / min_spacing)) + 1
    n_rows = int(jnp.ceil((ry_max - ry_min) / row_spacing)) + 1

    hex_x_list = []
    hex_y_list = []
    for row in range(n_rows):
        y_pos = ry_min + row_spacing * 0.5 + row * row_spacing
        offset = min_spacing * 0.5 if row % 2 else 0.0
        for col in range(n_cols):
            x_pos = rx_min + min_spacing * 0.5 + col * min_spacing + offset
            if x_pos <= rx_max and y_pos <= ry_max:
                hex_x_list.append(float(x_pos))
                hex_y_list.append(float(y_pos))

    rot_pts = jnp.array([hex_x_list, hex_y_list]).T
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
        init_x, init_y = ix[idx], iy[idx]
    else:
        key = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # 3-stage pipeline
    x1, y1 = topfarm_sgd_solve(objective, init_x, init_y,
                                boundary, min_spacing, settings_s1)
    x2, y2 = topfarm_sgd_solve(objective, x1, y1,
                                boundary, min_spacing, settings_s2)
    opt_x, opt_y = topfarm_sgd_solve(objective, x2, y2,
                                      boundary, min_spacing, settings_s3)

    return opt_x, opt_y
