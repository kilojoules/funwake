"""Differential-evolution tuned bump schedule inside a full optimizer.

HYPOTHESIS: The DE-tuned dual-bump schedule found in prior ablation logs can
transfer into a normal optimize() implementation and beat the hex/SGD basin.
AXIS: scipy_differential_evolution-tuned schedule parameters in the Adam skeleton.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    total_steps = 8000

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, : len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def con_penalty(x, y):
        return boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)

    grad_obj = jax.grad(objective, argnums=(0, 1))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    n_verts = boundary.shape[0]

    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    angle = dominant + jnp.pi / 2.0
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    cx = jnp.mean(boundary[:, 0])
    cy = jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)
    grid_spacing = min_spacing * 1.02
    nx = int(jnp.maximum(2, jnp.floor((rx_max - rx_min) / grid_spacing)))
    ny = int(jnp.maximum(2, jnp.floor((ry_max - ry_min) / grid_spacing)))
    gx, gy = jnp.meshgrid(
        rx_min + min_spacing * 0.55 + jnp.arange(nx) * grid_spacing,
        ry_min + min_spacing * 0.55 + jnp.arange(ny) * grid_spacing,
    )
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
    cand_x = orig_pts[:, 0]
    cand_y = orig_pts[:, 1]

    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)

    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0.0
    ix = cand_x[inside]
    iy = cand_y[inside]

    if len(ix) >= n_target:
        key = jax.random.PRNGKey(0)
        indices = jax.random.choice(key, len(ix), (n_target,), replace=False)
        x = ix[indices]
        y = iy[indices]
    else:
        key = jax.random.PRNGKey(0)
        key, kx, ky = jax.random.split(key, 3)
        x = jax.random.uniform(kx, (n_target,), minval=float(x_min), maxval=float(x_max))
        y = jax.random.uniform(ky, (n_target,), minval=float(y_min), maxval=float(y_max))

    gox, goy = grad_obj(x, y)
    lr0 = 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0

    def schedule(step):
        t = step / total_steps

        k = 4.788904698072376
        log_m = 2.5444236095801482
        warm = 0.031100666821230236
        amp1 = 0.33306582948262714
        amp2 = 0.3144610739896076
        c1 = 0.7118630839032932
        c2 = 0.4830074661356639
        w1 = 0.0357338045274541
        w2 = 0.13157290561309115
        alpha_c = 4.259958160161949
        alpha_d = 12.487427931615652
        beta1 = 0.11602857790365473
        beta2 = 0.8665970021316327

        lr_init = k * lr0
        lr_min = lr_init / (10.0**log_m)
        warm_lr = lr_init * t / jnp.maximum(warm, 1e-6)
        cosine_t = (t - warm) / jnp.maximum(1.0 - warm, 1e-6)
        cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (
            1.0 + jnp.cos(jnp.pi * cosine_t)
        )
        lr_base = jnp.where(t < warm, warm_lr, cosine_lr)
        bump1 = amp1 * lr_init * jnp.exp(-0.5 * ((t - c1) / w1) ** 2)
        bump2 = amp2 * lr_init * jnp.exp(-0.5 * ((t - c2) / w2) ** 2)
        lr = jnp.maximum(lr_base + bump1 + bump2, 1e-10)

        alpha_base = alpha_c * alpha0 * lr_init / lr
        late = jnp.maximum(t - 0.5, 0.0) / 0.5
        alpha = alpha_base + alpha_d * alpha0 * late**2
        return lr, alpha, beta1, beta2

    @jax.jit
    def run_loop(init_x, init_y):
        mx = jnp.zeros_like(init_x)
        my = jnp.zeros_like(init_y)
        vx = jnp.zeros_like(init_x)
        vy = jnp.zeros_like(init_y)

        def step(i, carry):
            x_cur, y_cur, mx_cur, my_cur, vx_cur, vy_cur = carry
            lr, alpha, beta1, beta2 = schedule(i.astype(float))
            gox, goy = grad_obj(x_cur, y_cur)
            gcx, gcy = grad_con(x_cur, y_cur)
            gx = gox + alpha * gcx
            gy = goy + alpha * gcy
            it = (i + 1).astype(float)

            mx_new = beta1 * mx_cur + (1.0 - beta1) * gx
            my_new = beta1 * my_cur + (1.0 - beta1) * gy
            vx_new = beta2 * vx_cur + (1.0 - beta2) * gx**2
            vy_new = beta2 * vy_cur + (1.0 - beta2) * gy**2

            mx_hat = mx_new / (1.0 - beta1**it)
            my_hat = my_new / (1.0 - beta1**it)
            vx_hat = vx_new / (1.0 - beta2**it)
            vy_hat = vy_new / (1.0 - beta2**it)
            x_next = x_cur - lr * mx_hat / (jnp.sqrt(vx_hat) + 1e-12)
            y_next = y_cur - lr * my_hat / (jnp.sqrt(vy_hat) + 1e-12)
            return x_next, y_next, mx_new, my_new, vx_new, vy_new

        final = jax.lax.fori_loop(0, total_steps, step, (init_x, init_y, mx, my, vx, vy))
        return final[0], final[1]

    opt_x, opt_y = run_loop(x, y)
    ok = (boundary_penalty(opt_x, opt_y, boundary) < 1e-3) & (
        spacing_penalty(opt_x, opt_y, min_spacing) < 1e-3
    )
    return jnp.where(ok, opt_x, x), jnp.where(ok, opt_y, y)
