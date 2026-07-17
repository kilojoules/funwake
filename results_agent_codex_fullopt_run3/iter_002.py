"""Exact-size wind-aligned grid with staged custom Adam refinement.

HYPOTHESIS: The last run's built-in SGD improved from a hex start but likely
left value in the lattice orientation/phase. A small surrogate search over
exact-size sheared grids, followed by batched custom Adam with augmented
constraint penalties, should reach a different and higher-AEP basin.
AXIS: custom_adam staged batched NAdam over wind-aligned exact lattice starts.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.boundary import polygon_sdf


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    @jax.jit
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    wd_rad = jnp.deg2rad(wd)
    wind_x = jnp.sum(jnp.cos(wd_rad) * weights)
    wind_y = jnp.sum(jnp.sin(wd_rad) * weights)
    dominant_wd = jnp.arctan2(wind_y, wind_x)

    @jax.jit
    def generate_grid(params):
        sx, sy, theta_off, ox, oy, shear, aspect = params
        theta = dominant_wd + theta_off
        sy_eff = sy * aspect
        n_side = int(np.sqrt(n_target)) + 15
        ii, jj = jnp.meshgrid(
            jnp.arange(n_side) - n_side // 2,
            jnp.arange(n_side) - n_side // 2,
        )
        ix = ii.ravel()
        iy = jj.ravel()
        hx = ix * sx * min_spacing + (iy % 2) * sx * min_spacing * 0.5
        hy = iy * sy_eff * min_spacing * jnp.sqrt(3.0) * 0.5
        hx = hx + shear * hy
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        sdf = polygon_sdf(rx, ry, boundary)
        idx = jnp.argsort(sdf)[:n_target]
        return rx[idx], ry[idx]

    def grid_score(params):
        gx, gy = generate_grid(params)
        return -objective(gx, gy)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    bounds_low = jnp.array([1.02, 1.02, -jnp.pi / 4.0, x_min, y_min, -0.6, 0.8])
    bounds_high = jnp.array([5.0, 5.0, jnp.pi / 4.0, x_max, y_max, 0.6, 1.3])

    def scale_params(raw):
        return bounds_low + raw * (bounds_high - bounds_low)

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    x_raw = jax.random.uniform(subkey, (12, 7))
    y_raw = jnp.array([grid_score(scale_params(p)) for p in x_raw])

    @jax.jit
    def gp_predict(x_test, x_train, y_train, length=0.3):
        def kernel(a, b):
            d = jnp.sqrt(jnp.sum((a - b) ** 2) + 1e-8)
            scaled = jnp.sqrt(5.0) * d / length
            return (1.0 + scaled + scaled**2 / 3.0) * jnp.exp(-scaled)

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
        x_raw = jnp.vstack([x_raw, nxt])
        y_raw = jnp.append(y_raw, grid_score(scale_params(nxt)))

    top_params = x_raw[jnp.argsort(y_raw)[-10:][::-1]]

    @jax.jit
    def constraint_penalty(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        boundary_violation = jnp.maximum(0.0, sdf + 0.01)
        pen_b = jnp.sum(lam_b * boundary_violation + 0.5 * mu * boundary_violation**2)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        upper = jnp.triu(jnp.ones((n_target, n_target)), k=1)
        spacing_violation = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        pen_s = jnp.sum(upper * (lam_s * spacing_violation + 0.5 * mu * spacing_violation**2))
        return pen_b + pen_s

    @jax.jit
    def total_objective(x, y, lam_b, lam_s, mu):
        return objective(x, y) + constraint_penalty(x, y, lam_b, lam_s, mu)

    grad_total = jax.jit(
        jax.vmap(
            jax.grad(total_objective, argnums=(0, 1)),
            in_axes=(0, 0, 0, 0, None),
        )
    )

    def run_batch_adam(x, y, n_steps, lr, mu_init, mu_rate):
        mom_x = jnp.zeros_like(x)
        mom_y = jnp.zeros_like(y)
        var_x = jnp.zeros_like(x)
        var_y = jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_target, n_target))
        cur_x = x
        cur_y = y
        mu = mu_init
        n_outer = 5
        steps_per_outer = n_steps // n_outer

        for outer in range(n_outer):
            def step(carry, t):
                px, py, mx, my, vx, vy = carry
                gx, gy = grad_total(px, py, lam_b, lam_s, mu)
                beta1 = 0.9
                beta2 = 0.999
                mx = beta1 * mx + (1.0 - beta1) * gx
                my = beta1 * my + (1.0 - beta1) * gy
                vx = beta2 * vx + (1.0 - beta2) * gx**2
                vy = beta2 * vy + (1.0 - beta2) * gy**2
                look_x = (beta1 * mx + (1.0 - beta1) * gx) / (1.0 - beta1 ** (t + 1))
                look_y = (beta1 * my + (1.0 - beta1) * gy) / (1.0 - beta1 ** (t + 1))
                vx_hat = vx / (1.0 - beta2 ** (t + 1))
                vy_hat = vy / (1.0 - beta2 ** (t + 1))
                px = px - lr * look_x / (jnp.sqrt(vx_hat) + 1e-8)
                py = py - lr * look_y / (jnp.sqrt(vy_hat) + 1e-8)
                return (px, py, mx, my, vx, vy), None

            start = outer * steps_per_outer
            stop = (outer + 1) * steps_per_outer
            (cur_x, cur_y, mom_x, mom_y, var_x, var_y), _ = jax.lax.scan(
                step,
                (cur_x, cur_y, mom_x, mom_y, var_x, var_y),
                jnp.arange(start, stop),
            )

            sdf = jax.vmap(lambda px, py: polygon_sdf(px, py, boundary))(cur_x, cur_y)
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx = cur_x[:, :, None] - cur_x[:, None, :]
            dy = cur_y[:, :, None] - cur_y[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= mu_rate

        return cur_x, cur_y

    layouts = [generate_grid(scale_params(p)) for p in top_params]
    cur_x = jnp.stack([layout[0] for layout in layouts])
    cur_y = jnp.stack([layout[1] for layout in layouts])

    cur_x, cur_y = run_batch_adam(cur_x, cur_y, 250, 15.0, 1e2, 2.5)
    idx = jnp.argsort(jax.vmap(objective)(cur_x, cur_y))[:5]
    cur_x = cur_x[idx]
    cur_y = cur_y[idx]

    cur_x, cur_y = run_batch_adam(cur_x, cur_y, 500, 8.0, 1e3, 3.0)
    idx = jnp.argsort(jax.vmap(objective)(cur_x, cur_y))[:2]
    cur_x = cur_x[idx]
    cur_y = cur_y[idx]

    cur_x, cur_y = run_batch_adam(cur_x, cur_y, 1000, 4.0, 1e4, 2.5)
    best_idx = jnp.argmin(jax.vmap(objective)(cur_x, cur_y))
    cur_x = cur_x[best_idx : best_idx + 1]
    cur_y = cur_y[best_idx : best_idx + 1]

    cur_x, cur_y = run_batch_adam(cur_x, cur_y, 1500, 2.0, 1e5, 2.0)

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

    return project(cur_x[0], cur_y[0])
