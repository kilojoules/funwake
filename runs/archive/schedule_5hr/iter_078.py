"""BFGS optimizer: second-order method for wind farm layout.

All previous 68+ attempts used first-order ADAM/SGD. BFGS approximates
the inverse Hessian to make curvature-aware steps, converging faster
on smooth landscapes. With 50 turbines (100 variables), the 100x100
Hessian approximation is efficient.

Strategy: Progressive penalty BFGS → topfarm polish
1. BFGS with alpha=2 (explore AEP landscape, 300 iters)
2. BFGS with alpha=30 (moderate constraint push, 200 iters)
3. BFGS with alpha=200 (strong enforcement, 200 iters)
4. Topfarm polish for final feasibility

Key differences from all previous attempts:
- Second-order optimization (BFGS) instead of first-order (ADAM)
- jax.scipy.optimize.minimize for curvature-aware line search
- Separate boundary vs spacing penalty weights
"""
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


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

    if len(ix) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(ix) - 1, n_target)).astype(int)
        init_x, init_y = ix[idx], iy[idx]
    else:
        key = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    n = n_target

    # === BFGS with progressive penalty ===
    def make_loss(alpha_sp, alpha_bp):
        def loss(z):
            x, y = z[:n], z[n:]
            r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
            p = r.power()[:, :len(x)]
            aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
            sp = spacing_penalty(x, y, min_spacing)
            bp = boundary_penalty(x, y, boundary)
            return aep + alpha_sp * sp + alpha_bp * bp
        return loss

    z0 = jnp.concatenate([init_x, init_y])

    # Phase 1: Explore with low penalty
    res1 = jax_minimize(make_loss(2.0, 10.0), z0, method='BFGS',
                        options={'maxiter': 300})
    z1 = res1.x

    # Phase 2: Moderate enforcement
    res2 = jax_minimize(make_loss(30.0, 50.0), z1, method='BFGS',
                        options={'maxiter': 200})
    z2 = res2.x

    # Phase 3: Strong enforcement
    res3 = jax_minimize(make_loss(200.0, 200.0), z2, method='BFGS',
                        options={'maxiter': 200})
    z3 = res3.x

    explored_x, explored_y = z3[:n], z3[n:]

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
