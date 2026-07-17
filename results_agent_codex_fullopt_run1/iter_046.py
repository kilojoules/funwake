"""Wake-harm surplus pruning after compact BO grid search.

HYPOTHESIS: The previous surplus optimizer found a stronger basin, but pruning
by raw individual production can keep upstream wake blockers and discard
downstream turbines that would recover once blockers are removed.

AXIS: optimize a modest surplus grid, prune by several wind-weighted wake-harm
scores, then polish the best exact-size children with augmented-Lagrangian
NAdam.

LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.boundary import polygon_sdf


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    n_extra = 8 if n_target <= 60 else 6
    n_total = n_target + n_extra

    @jax.jit
    def aep_surplus(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_total]
        return jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    @jax.jit
    def obj_surplus(x, y):
        return -aep_surplus(x, y)

    wd_rad = jnp.deg2rad(wd)
    energy = weights * ws**3
    mean_wind = jnp.arctan2(
        jnp.sum(energy * jnp.sin(wd_rad)), jnp.sum(energy * jnp.cos(wd_rad))
    )

    @jax.jit
    def grid_from_params(params):
        sx, sy, theta_off, ox, oy, shear, aspect = params
        theta = mean_wind + theta_off
        sy = sy * aspect
        side = int(np.sqrt(n_total)) + 15
        ii, jj = jnp.meshgrid(
            jnp.arange(side) - side // 2, jnp.arange(side) - side // 2
        )
        ii = ii.ravel()
        jj = jj.ravel()
        hx = ii * sx * min_spacing + (jj % 2) * 0.5 * sx * min_spacing
        hy = jj * sy * min_spacing * jnp.sqrt(3.0) * 0.5
        hx = hx + shear * hy
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        sdf = polygon_sdf(rx, ry, boundary)
        keep = jnp.argsort(sdf)[:n_total]
        return rx[keep], ry[keep]

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    lo = jnp.array([1.00, 1.00, -0.65, x_min, y_min, -0.55, 0.82])
    hi = jnp.array([4.70, 4.70, 0.65, x_max, y_max, 0.55, 1.23])

    def scale(p):
        return lo + p * (hi - lo)

    def score_grid(p):
        gx, gy = grid_from_params(scale(p))
        return aep_surplus(gx, gy)

    key = jax.random.PRNGKey(2201)
    key, sub = jax.random.split(key)
    x_raw = jax.random.uniform(sub, (10, 7))
    y_raw = jnp.array([score_grid(p) for p in x_raw])

    @jax.jit
    def gp_predict(x_test, x_train, y_train, length=0.32):
        def kernel(a, b):
            d2 = jnp.sum((a - b) ** 2)
            d = jnp.sqrt(d2 + 1e-8)
            s5 = jnp.sqrt(5.0)
            return (1.0 + s5 * d / length + 5.0 * d2 / (3.0 * length**2)) * jnp.exp(
                -s5 * d / length
            )

        k = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(x_train))(x_train)
        k = k + jnp.eye(len(x_train)) * 1e-5
        l_chol = jnp.linalg.cholesky(k)
        ks = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(x_train))(x_test)
        alpha = jnp.linalg.solve(l_chol.T, jnp.linalg.solve(l_chol, y_train))
        mu = ks @ alpha
        v = jnp.linalg.solve(l_chol, ks.T)
        var = jax.vmap(lambda a: kernel(a, a))(x_test) - jnp.sum(v**2, axis=0)
        return mu, jnp.sqrt(jnp.maximum(var, 1e-9))

    for _ in range(13):
        key, sub = jax.random.split(key)
        cand = jax.random.uniform(sub, (650, 7))
        mu, sig = gp_predict(cand, x_raw, y_raw)
        incumbent = jnp.max(y_raw)
        z = (mu - incumbent) / sig
        cdf = 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))
        pdf = jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)
        ei = (mu - incumbent) * cdf + sig * pdf
        nxt = cand[jnp.argmax(ei)]
        val = score_grid(nxt)
        x_raw = jnp.vstack([x_raw, nxt])
        y_raw = jnp.append(y_raw, val)

    seeds = [grid_from_params(scale(p)) for p in x_raw[jnp.argsort(y_raw)[-7:][::-1]]]
    sx0 = jnp.stack([s[0] for s in seeds])
    sy0 = jnp.stack([s[1] for s in seeds])

    @jax.jit
    def surplus_penalty(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        vb = jnp.maximum(0.0, sdf + 0.01)
        pb = jnp.sum(lam_b * vb + 0.5 * mu * vb**2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_total, n_total)), k=1)
        vs = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        return pb + jnp.sum(mask * (lam_s * vs + 0.5 * mu * vs**2))

    @jax.jit
    def surplus_total(x, y, lam_b, lam_s, mu):
        return obj_surplus(x, y) + surplus_penalty(x, y, lam_b, lam_s, mu)

    grad_surplus = jax.jit(
        jax.vmap(jax.grad(surplus_total, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None))
    )

    def nadam_surplus(x, y):
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx = jnp.zeros_like(x)
        vy = jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_total, n_total))
        mu = 90.0
        lr = 11.0
        cx, cy = x, y
        for i in range(4):
            def step(carry, t):
                px, py, ax, ay, bx, by = carry
                gx, gy = grad_surplus(px, py, lam_b, lam_s, mu)
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

            (cx, cy, mx, my, vx, vy), _ = jax.lax.scan(
                step, (cx, cy, mx, my, vx, vy), jnp.arange(i * 180, (i + 1) * 180)
            )
            sdf = jax.vmap(lambda px, py: polygon_sdf(px, py, boundary))(cx, cy)
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx = cx[:, :, None] - cx[:, None, :]
            dy = cy[:, :, None] - cy[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= 2.1
        return cx, cy

    sx, sy = nadam_surplus(sx0, sy0)

    def wake_harm_score(x, y, harm_weight):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        indiv = jnp.sum(r.power() * weights[:, None], axis=0)
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)

        def one_dir(theta, w):
            ux = jnp.cos(theta)
            uy = jnp.sin(theta)
            down = dx * ux + dy * uy
            lat = jnp.abs(-dx * uy + dy * ux)
            cone = jnp.exp(-(lat / (0.55 * min_spacing + 0.075 * down)) ** 2)
            reach = jnp.exp(-jnp.maximum(down, 0.0) / (5.5 * min_spacing))
            affected = (down > 0.15 * min_spacing) & (lat < 1.15 * min_spacing)
            return w * jnp.sum(jnp.where(affected, cone * reach, 0.0), axis=1)

        harm = jnp.sum(jax.vmap(one_dir)(wd_rad, energy), axis=0)
        crowd = jnp.sum(jnp.exp(-(dist / (1.35 * min_spacing)) ** 2), axis=1) - 1.0
        harm = harm / (jnp.mean(harm) + 1e-9)
        crowd = crowd / (jnp.mean(crowd) + 1e-9)
        return indiv / (jnp.mean(indiv) + 1e-9) - harm_weight * harm - 0.10 * crowd

    def prune_child(x, y, harm_weight):
        score = wake_harm_score(x, y, harm_weight)
        keep = jnp.argsort(score)[-n_target:]
        return x[keep], y[keep]

    children = []
    for i in range(sx.shape[0]):
        for hw in (0.0, 0.18, 0.34, 0.55):
            children.append(prune_child(sx[i], sy[i], hw))

    px = jnp.stack([c[0] for c in children])
    py = jnp.stack([c[1] for c in children])

    @jax.jit
    def aep_final(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    @jax.jit
    def obj_final(x, y):
        return -aep_final(x, y)

    pre = jax.vmap(aep_final)(px, py)
    best_pre = jnp.argsort(pre)[-4:][::-1]
    px = px[best_pre]
    py = py[best_pre]

    @jax.jit
    def final_total(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        vb = jnp.maximum(0.0, sdf + 0.01)
        pb = jnp.sum(lam_b * vb + 0.5 * mu * vb**2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_target, n_target)), k=1)
        vs = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        ps = jnp.sum(mask * (lam_s * vs + 0.5 * mu * vs**2))
        return obj_final(x, y) + pb + ps

    grad_final = jax.jit(
        jax.vmap(jax.grad(final_total, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None))
    )

    def nadam_final(x, y):
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx = jnp.zeros_like(x)
        vy = jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_target, n_target))
        mu = 900.0
        lr = 3.7
        cx, cy = x, y
        for i in range(3):
            def step(carry, t):
                px, py, ax, ay, bx, by = carry
                gx, gy = grad_final(px, py, lam_b, lam_s, mu)
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

            (cx, cy, mx, my, vx, vy), _ = jax.lax.scan(
                step, (cx, cy, mx, my, vx, vy), jnp.arange(i * 360, (i + 1) * 360)
            )
            sdf = jax.vmap(lambda px, py: polygon_sdf(px, py, boundary))(cx, cy)
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx = cx[:, :, None] - cx[:, None, :]
            dy = cy[:, :, None] - cy[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= 3.0
        return cx, cy

    fx, fy = nadam_final(px, py)
    best = jnp.argmax(jax.vmap(aep_final)(fx, fy))
    out_x = fx[best]
    out_y = fy[best]

    def project(x, y):
        for _ in range(28):
            sdf = polygon_sdf(x, y, boundary)
            gb = jax.vmap(
                jax.grad(
                    lambda px, py: polygon_sdf(
                        jnp.array([px]), jnp.array([py]), boundary
                    )[0],
                    argnums=(0, 1),
                )
            )(x, y)
            x = x - jnp.maximum(0.0, sdf + 0.02) * gb[0]
            y = y - jnp.maximum(0.0, sdf + 0.02) * gb[1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            push = jnp.maximum(0.0, min_spacing * 1.002 - dist)
            x = x + 0.15 * jnp.sum(push * dx / dist, axis=1)
            y = y + 0.15 * jnp.sum(push * dy / dist, axis=1)
        return x, y

    return project(out_x, out_y)
