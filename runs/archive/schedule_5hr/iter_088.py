"""Simplified 2-stage: let topfarm's internal alpha ramp do the work.

Insight: topfarm internally scales alpha = alpha0 * lr0/lr. With
gamma_min_factor=0.01, the effective penalty increases 100x during
the decay phase. This IS the explore→enforce schedule.

By using moderate sw=8, bw=25 and a long constant-LR phase (5000),
we get:
- 5000 iters at constant LR: effective sw=8 (exploration)
- 5000 iters with decaying LR: effective sw ramps to 800 (enforcement)

This single call replaces the complex 3-stage pipeline.
Plus a brief feasibility push first and polish after.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # --- Wind-aware grid initialization ---
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
    _nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
    _ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing / 2, rx_max - min_spacing / 2, _nx),
        jnp.linspace(ry_min + min_spacing / 2, ry_max - min_spacing / 2, _ny))
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

    # Stage 1: Quick feasibility push
    s1 = SGDSettings(
        learning_rate=200.0,
        max_iter=800, additional_constant_lr_iterations=400,
        tol=1e-6, beta1=0.1, beta2=0.2,
        spacing_weight=200.0, boundary_weight=200.0,
        ks_rho=100.0, gamma_min_factor=0.01,
    )

    # Stage 2: Combined explore+enforce via internal alpha ramp
    s2 = SGDSettings(
        learning_rate=200.0,
        max_iter=5000, additional_constant_lr_iterations=5000,
        tol=1e-6, beta1=0.1, beta2=0.2,
        spacing_weight=8.0, boundary_weight=25.0,
        ks_rho=80.0, gamma_min_factor=0.01,
    )

    x1, y1 = topfarm_sgd_solve(objective, init_x, init_y,
                                boundary, min_spacing, s1)
    opt_x, opt_y = topfarm_sgd_solve(objective, x1, y1,
                                      boundary, min_spacing, s2)

    return opt_x, opt_y
