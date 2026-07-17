"""Exact-grid plus boundary-farthest AL-NAdam ensemble.

HYPOTHESIS: The 5583 GWh exact-grid basin is reliable, but it may be missing
edge-biased packings that reserve more interior wake clearance.  Adding a few
boundary-offset farthest-point starts gives the AL solver a different basin
without giving up the incumbent BO lattice starts.

AXIS: incumbent BO exact grid plus deterministic boundary-farthest starts,
followed by the same staged custom augmented-Lagrangian NAdam and projection.

LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.boundary import polygon_sdf


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

    @jax.jit
    def generate_boundary_fps(params):
        theta_off, step_scale, off_x, off_y, inset, radial_bias = params
        theta = dominant_wd + theta_off
        center = jnp.mean(boundary, axis=0) + jnp.array([off_x, off_y])
        n_side = int(np.sqrt(n_target)) + 18
        ii, jj = jnp.meshgrid(
            jnp.arange(n_side) - n_side // 2, jnp.arange(n_side) - n_side // 2
        )
        gx = ii.ravel() * step_scale * min_spacing
        gy = jj.ravel() * step_scale * min_spacing
        rx = gx * jnp.cos(theta) - gy * jnp.sin(theta) + center[0]
        ry = gx * jnp.sin(theta) + gy * jnp.cos(theta) + center[1]
        sdf = polygon_sdf(rx, ry, boundary)
        valid = sdf < -0.02
        target_sdf = -inset * min_spacing
        boundary_band = -jnp.abs(sdf - target_sdf) / min_spacing
        rad = jnp.sqrt((rx - center[0]) ** 2 + (ry - center[1]) ** 2)
        base_score = jnp.where(valid, boundary_band + radial_bias * rad / min_spacing, -1e9)

        n_cand = rx.shape[0]
        sel_x = jnp.zeros((n_target,), dtype=rx.dtype)
        sel_y = jnp.zeros((n_target,), dtype=ry.dtype)
        chosen = jnp.zeros((n_cand,), dtype=bool)

        def first_step(carry, _):
            sx, sy, used = carry
            idx = jnp.argmax(jnp.where(used, -1e9, base_score))
            sx = sx.at[0].set(rx[idx])
            sy = sy.at[0].set(ry[idx])
            used = used.at[idx].set(True)
            return (sx, sy, used), None

        (sel_x, sel_y, chosen), _ = first_step((sel_x, sel_y, chosen), None)

        def step(carry, t):
            sx, sy, used = carry
            dx = rx[:, None] - sx[None, :]
            dy = ry[:, None] - sy[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            active = jnp.arange(n_target) < t
            min_dist = jnp.min(jnp.where(active[None, :], dist, 1e9), axis=1)
            spacing_score = min_dist / min_spacing
            score = jnp.where(used, -1e9, spacing_score + 0.35 * base_score)
            idx = jnp.argmax(score)
            sx = sx.at[t].set(rx[idx])
            sy = sy.at[t].set(ry[idx])
            used = used.at[idx].set(True)
            return (sx, sy, used), None

        (sel_x, sel_y, _), _ = jax.lax.scan(
            step, (sel_x, sel_y, chosen), jnp.arange(1, n_target)
        )
        return sel_x, sel_y

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

    fps_params = [
        jnp.array([0.0, 1.03, 0.0, 0.0, 0.9, 0.08]),
        jnp.array([jnp.pi / 10, 1.08, 0.15 * min_spacing, -0.10 * min_spacing, 1.2, 0.04]),
        jnp.array([-jnp.pi / 8, 1.12, -0.20 * min_spacing, 0.10 * min_spacing, 1.6, 0.02]),
        jnp.array([jnp.pi / 5, 1.18, 0.10 * min_spacing, 0.20 * min_spacing, 2.0, 0.00]),
    ]

    layouts = [generate_grid(scale_params(p)) for p in top_params[:8]]
    layouts += [generate_boundary_fps(p) for p in fps_params]
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

    cur_x, cur_y = run_al_nadam_batch(cur_x, cur_y, 1500, 2.0, 1e5, 2.0)

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
