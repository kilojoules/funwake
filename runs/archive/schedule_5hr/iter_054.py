"""Basin-hopping: converge to 5557 basin, shake out, re-converge.

All approaches plateau at ~5557 feasible. This first converges there
with standard 3-stage, then does a violent re-exploration (lr=250,
sw=0.01) to escape the local optimum, then re-enforces constraints.
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
    s1 = SGDSettings(
        learning_rate=200.0, max_iter=1000,
        additional_constant_lr_iterations=500,
        tol=1e-6, beta1=0.1, beta2=0.2,
        spacing_weight=200.0, boundary_weight=200.0,
        ks_rho=100.0, gamma_min_factor=0.01,
    )

    # Stage 2: AEP exploration
    s2 = SGDSettings(
        learning_rate=100.0, max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-6, beta1=0.1, beta2=0.2,
        spacing_weight=0.5, boundary_weight=5.0,
        ks_rho=50.0, gamma_min_factor=0.01,
    )

    # Stage 3: First enforcement
    s3 = SGDSettings(
        learning_rate=50.0, max_iter=1500,
        additional_constant_lr_iterations=750,
        tol=1e-6, beta1=0.1, beta2=0.2,
        spacing_weight=300.0, boundary_weight=300.0,
        ks_rho=150.0, gamma_min_factor=0.01,
    )

    # Stage 4: Basin-hop (violent re-exploration from converged solution)
    s4 = SGDSettings(
        learning_rate=250.0, max_iter=1500,
        additional_constant_lr_iterations=750,
        tol=1e-6, beta1=0.1, beta2=0.2,
        spacing_weight=0.01, boundary_weight=2.0,
        ks_rho=30.0, gamma_min_factor=0.01,
    )

    # Stage 5: Re-enforcement
    s5 = SGDSettings(
        learning_rate=50.0, max_iter=1200,
        additional_constant_lr_iterations=600,
        tol=1e-6, beta1=0.1, beta2=0.2,
        spacing_weight=300.0, boundary_weight=300.0,
        ks_rho=150.0, gamma_min_factor=0.01,
    )

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

    # 5-stage pipeline: converge → hop → re-converge
    x1, y1 = topfarm_sgd_solve(objective, init_x, init_y, boundary, min_spacing, s1)
    x2, y2 = topfarm_sgd_solve(objective, x1, y1, boundary, min_spacing, s2)
    x3, y3 = topfarm_sgd_solve(objective, x2, y2, boundary, min_spacing, s3)
    x4, y4 = topfarm_sgd_solve(objective, x3, y3, boundary, min_spacing, s4)
    opt_x, opt_y = topfarm_sgd_solve(objective, x4, y4, boundary, min_spacing, s5)

    return opt_x, opt_y
