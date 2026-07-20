"""Triple basin-hop with progressively higher LR hops.

iter_055's double hop improved to 5557.79 (+17.08). Try 3 hops with
escalating exploration aggressiveness (lr=200→300→400). Skip initial
feasibility stage to save iterations for hopping.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def make_explore(lr, iters, const_iters, sw=0.01, bw=2.0):
        return SGDSettings(
            learning_rate=lr, max_iter=iters,
            additional_constant_lr_iterations=const_iters,
            tol=1e-6, beta1=0.1, beta2=0.2,
            spacing_weight=sw, boundary_weight=bw,
            ks_rho=30.0, gamma_min_factor=0.01,
        )

    def make_enforce(iters, const_iters):
        return SGDSettings(
            learning_rate=50.0, max_iter=iters,
            additional_constant_lr_iterations=const_iters,
            tol=1e-6, beta1=0.1, beta2=0.2,
            spacing_weight=300.0, boundary_weight=300.0,
            ks_rho=150.0, gamma_min_factor=0.01,
        )

    # Initial explore + enforce
    s1 = make_explore(120.0, 1800, 900, sw=0.5, bw=5.0)
    s2 = make_enforce(1000, 500)
    # Hop 1
    s3 = make_explore(200.0, 1000, 500)
    s4 = make_enforce(800, 400)
    # Hop 2
    s5 = make_explore(300.0, 800, 400)
    s6 = make_enforce(800, 400)
    # Hop 3
    s7 = make_explore(400.0, 600, 300)
    s8 = make_enforce(1000, 500)

    # --- Wind-aware grid initialization ---
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

    # 8-stage pipeline with triple basin-hop
    x1, y1 = topfarm_sgd_solve(objective, init_x, init_y, boundary, min_spacing, s1)
    x2, y2 = topfarm_sgd_solve(objective, x1, y1, boundary, min_spacing, s2)
    x3, y3 = topfarm_sgd_solve(objective, x2, y2, boundary, min_spacing, s3)
    x4, y4 = topfarm_sgd_solve(objective, x3, y3, boundary, min_spacing, s4)
    x5, y5 = topfarm_sgd_solve(objective, x4, y4, boundary, min_spacing, s5)
    x6, y6 = topfarm_sgd_solve(objective, x5, y5, boundary, min_spacing, s6)
    x7, y7 = topfarm_sgd_solve(objective, x6, y6, boundary, min_spacing, s7)
    opt_x, opt_y = topfarm_sgd_solve(objective, x7, y7, boundary, min_spacing, s8)

    return opt_x, opt_y
