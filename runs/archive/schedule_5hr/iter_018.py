"""Three-stage: lighter feasibility, longer AEP explore, moderate re-enforce.

Variant of iter_017 with more iterations in stage 2 (AEP) and slightly
less aggressive stages 1/3 to avoid over-constraining.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # Stage 1: Moderate feasibility push
    settings_s1 = SGDSettings(
        learning_rate=150.0,
        max_iter=1200,
        additional_constant_lr_iterations=600,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=100.0,
        boundary_weight=100.0,
        ks_rho=100.0,
        gamma_min_factor=0.01,
    )

    # Stage 2: Long AEP exploration with relaxed constraints
    settings_s2 = SGDSettings(
        learning_rate=120.0,
        max_iter=3000,
        additional_constant_lr_iterations=1500,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=1.0,
        boundary_weight=1.0,
        ks_rho=30.0,
        gamma_min_factor=0.01,
    )

    # Stage 3: Re-enforce constraints + final refinement
    settings_s3 = SGDSettings(
        learning_rate=60.0,
        max_iter=1800,
        additional_constant_lr_iterations=900,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        spacing_weight=250.0,
        boundary_weight=250.0,
        ks_rho=120.0,
        gamma_min_factor=0.01,
    )

    # Wind-aware grid initialization
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
        jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
        jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
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

    # Stage 1
    x1, y1 = topfarm_sgd_solve(objective, init_x, init_y,
                                boundary, min_spacing, settings_s1)
    # Stage 2
    x2, y2 = topfarm_sgd_solve(objective, x1, y1,
                                boundary, min_spacing, settings_s2)
    # Stage 3
    opt_x, opt_y = topfarm_sgd_solve(objective, x2, y2,
                                      boundary, min_spacing, settings_s3)

    return opt_x, opt_y
