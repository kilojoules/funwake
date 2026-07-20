"""Exact-grid AL-NAdam with topfarm warm-start polish.

HYPOTHESIS: The exact-grid AL-NAdam basin is strong, but its final projected
layout may stop short of a nearby TopFarm penalty optimum.  A short
topfarm_sgd_solve polish from the best custom layout can refine turbine
positions without paying for random multistarts.

AXIS: BO-tuned exact grid, staged AL-NAdam, then one warm-start topfarm polish.

LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.boundary import polygon_sdf
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    @jax.jit
    def aep_obj(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    wd_rad = jnp.deg2rad(wd)
    vx = jnp.sum(jnp.cos(wd_rad) * weights)
    vy = jnp.sum(jnp.sin(wd_rad) * weights)
    dominant_wd = jnp.arctan2(vy, vx)

    @jax.jit
    def generate_grid(params):
        sx, sy, theta_off, ox, oy, shear, aspect = params
        theta = dominant_wd + theta_off
        sy_actual = sy * aspect
        n_side = int(np.sqrt(n_target)) + 15
        ii, jj = jnp.meshgrid(
            jnp.arange(n_side) - n_side // 2, jnp.arange(n_side) - n_side // 2
        )
        ix = ii.ravel()
        iy = jj.ravel()
        hx = ix * sx * min_spacing + (iy % 2) * sx * min_spacing * 0.5
        hy = iy * sy_actual * min_spacing * jnp.sqrt(3.0) * 0.5
        hx = hx + shear * hy
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        sdf = polygon_sdf(rx, ry, boundary)
        idx = jnp.argsort(sdf)[:n_target]
        return rx[idx], ry[idx]

    def grid_score(params):
        gx, gy = generate_grid(params)
        return -aep_obj(gx, gy)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    bounds_low = jnp.array([1.02, 1.02, -jnp.pi / 4, x_min, y_min, -0.6, 0.8])
    bounds_high = jnp.array([5.0, 5.0, jnp.pi / 4, x_max, y_max, 0.6, 1.3])

    def scale_params(p):
        return bounds_low + p * (bounds_high - bounds_low)

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    x_raw = jax.random.uniform(subkey, (12, 7))
    y_raw = jnp.array([grid_score(scale_params(p)) for p in x_raw])

    @jax.jit
    def gp_predict(x_test, x_train, y_train, length=0.3):
        def kernel(a, b):
            d = jnp.sqrt(jnp.sum((a - b) ** 2) + 1e-8)
            return (
                1.0
                + jnp.sqrt(5.0) * d / length
                + 5.0 * d**2 / (3.0 * length**2)
            ) * jnp.exp(-jnp.sqrt(5.0) * d / length)

        k = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(x_train))(x_train)
        k = k + jnp.eye(len(x_train)) * 1e-5
        chol = jnp.linalg.cholesky(k)
        ks = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(x_train))(x_test)
        alpha = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, y_train))
        mu = ks @ alpha
        v = jnp.linalg.solve(chol, ks.T)
        var = jax.vmap(lambda a: kernel(a, a))(x_test) - jnp.sum(v**2, axis=0)
        return mu, jnp.sqrt(jnp.maximum(1e-9, var))

    for _ in range(18):
        key, subkey = jax.random.split(key)
        cand = jax.random.uniform(subkey, (800, 7))
        mu, sig = gp_predict(cand, x_raw, y_raw)
        incumbent = jnp.max(y_raw)
        z = (mu - incumbent) / sig
        cdf = 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))
        pdf = jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)
        ei = (mu - incumbent) * cdf + sig * pdf
        nxt = cand[jnp.argmax(ei)]
        val = grid_score(scale_params(nxt))
        x_raw = jnp.vstack([x_raw, nxt])
        y_raw = jnp.append(y_raw, val)

    top_params = x_raw[jnp.argsort(y_raw)[-10:][::-1]]

    @jax.jit
    def constraints_penalty(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        vb = jnp.maximum(0.0, sdf + 0.01)
        pen_b = jnp.sum(lam_b * vb + 0.5 * mu * vb**2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_target, n_target)), k=1)
        vs = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        pen_s = jnp.sum(mask * (lam_s * vs + 0.5 * mu * vs**2))
        return pen_b + pen_s

    @jax.jit
    def total_obj(x, y, lam_b, lam_s, mu):
        return aep_obj(x, y) + constraints_penalty(x, y, lam_b, lam_s, mu)

    grad_total = jax.jit(
        jax.vmap(jax.grad(total_obj, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None))
    )

    def run_al_nadam_batch(x, y, n_steps, lr, mu_init, mu_rate):
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx_acc = jnp.zeros_like(x)
        vy_acc = jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_target, n_target))
        mu = mu_init
        cx, cy = x, y
        n_inner = 5
        steps_per_inner = n_steps // n_inner

        for i in range(n_inner):
            def step(carry, t):
                px, py, ax, ay, bx, by = carry
                gx, gy = grad_total(px, py, lam_b, lam_s, mu)
                b1, b2 = 0.9, 0.999
                ax = b1 * ax + (1.0 - b1) * gx
                ay = b1 * ay + (1.0 - b1) * gy
                bx = b2 * bx + (1.0 - b2) * gx**2
                by = b2 * by + (1.0 - b2) * gy**2
                ahx = (b1 * ax + (1.0 - b1) * gx) / (1.0 - b1 ** (t + 1))
                ahy = (b1 * ay + (1.0 - b1) * gy) / (1.0 - b1 ** (t + 1))
                vhx = bx / (1.0 - b2 ** (t + 1))
                vhy = by / (1.0 - b2 ** (t + 1))
                px = px - lr * ahx / (jnp.sqrt(vhx) + 1e-8)
                py = py - lr * ahy / (jnp.sqrt(vhy) + 1e-8)
                return (px, py, ax, ay, bx, by), None

            (cx, cy, mx, my, vx_acc, vy_acc), _ = jax.lax.scan(
                step,
                (cx, cy, mx, my, vx_acc, vy_acc),
                jnp.arange(i * steps_per_inner, (i + 1) * steps_per_inner),
            )
            sdf = jax.vmap(lambda px, py: polygon_sdf(px, py, boundary))(cx, cy)
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx = cx[:, :, None] - cx[:, None, :]
            dy = cy[:, :, None] - cy[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= mu_rate
        return cx, cy

    layouts = [generate_grid(scale_params(p)) for p in top_params]
    cur_x = jnp.stack([layout[0] for layout in layouts])
    cur_y = jnp.stack([layout[1] for layout in layouts])

    cur_x, cur_y = run_al_nadam_batch(cur_x, cur_y, 250, 15.0, 1e2, 2.5)
    idx = jnp.argsort(jax.vmap(aep_obj)(cur_x, cur_y))[:5]
    cur_x, cur_y = cur_x[idx], cur_y[idx]

    cur_x, cur_y = run_al_nadam_batch(cur_x, cur_y, 500, 8.0, 1e3, 3.0)
    idx = jnp.argsort(jax.vmap(aep_obj)(cur_x, cur_y))[:2]
    cur_x, cur_y = cur_x[idx], cur_y[idx]

    cur_x, cur_y = run_al_nadam_batch(cur_x, cur_y, 1000, 4.0, 1e4, 2.5)
    best_idx = jnp.argmin(jax.vmap(aep_obj)(cur_x, cur_y))
    cur_x = cur_x[best_idx : best_idx + 1]
    cur_y = cur_y[best_idx : best_idx + 1]

    cur_x, cur_y = run_al_nadam_batch(cur_x, cur_y, 1000, 2.0, 1e5, 2.0)

    def project(x, y):
        for _ in range(20):
            sdf = polygon_sdf(x, y, boundary)
            grad_b = jax.vmap(
                jax.grad(
                    lambda px, py: polygon_sdf(
                        jnp.array([px]), jnp.array([py]), boundary
                    )[0],
                    argnums=(0, 1),
                )
            )(x, y)
            x = x - jnp.maximum(0.0, sdf + 0.02) * grad_b[0]
            y = y - jnp.maximum(0.0, sdf + 0.02) * grad_b[1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            force = jnp.maximum(0.0, min_spacing * 1.002 - dist)
            x = x + jnp.sum(force * (dx / dist), axis=1) * 0.15
            y = y + jnp.sum(force * (dy / dist), axis=1) * 0.15
        return x, y

    px, py = project(cur_x[0], cur_y[0])

    settings = SGDSettings(
        learning_rate=18.0,
        max_iter=900,
        additional_constant_lr_iterations=250,
        tol=1e-7,
        beta1=0.9,
        beta2=0.999,
        gamma_min_factor=0.02,
        ks_rho=100.0,
        spacing_weight=1.0,
        boundary_weight=1.0,
    )
    sx, sy = topfarm_sgd_solve(aep_obj, px, py, boundary, min_spacing, settings)
    sx, sy = project(sx, sy)
    return sx, sy
