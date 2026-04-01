"""Baseline optimizer — grid initialization + single-start topfarm_sgd_solve."""

import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # Generate initial layout: grid inside the polygon
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    nx = int(jnp.ceil((x_max - x_min) / min_spacing))
    ny = int(jnp.ceil((y_max - y_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(x_min + min_spacing/2, x_max - min_spacing/2, nx),
        jnp.linspace(y_min + min_spacing/2, y_max - min_spacing/2, ny))
    candidates_x = gx.flatten()
    candidates_y = gy.flatten()

    # Inside-polygon check (convex): all signed edge distances positive
    n_verts = boundary.shape[0]
    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        edge_x, edge_y = x2 - x1, y2 - y1
        edge_len = jnp.sqrt(edge_x**2 + edge_y**2) + 1e-10
        nx, ny = -edge_y / edge_len, edge_x / edge_len
        return (candidates_x - x1) * nx + (candidates_y - y1) * ny
    all_dists = jax.vmap(edge_dist)(jnp.arange(n_verts))
    inside = jnp.min(all_dists, axis=0) > 0

    inside_x = candidates_x[inside]
    inside_y = candidates_y[inside]

    if len(inside_x) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(inside_x) - 1, n_target)).astype(int)
        init_x = inside_x[idx]
        init_y = inside_y[idx]
    else:
        key = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    settings = SGDSettings(learning_rate=50.0, max_iter=4000,
                           additional_constant_lr_iterations=2000, tol=1e-6)
    opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y,
                                      boundary, min_spacing, settings)
    return opt_x, opt_y
