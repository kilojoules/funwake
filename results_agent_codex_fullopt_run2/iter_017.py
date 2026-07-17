"""SHGO search over wind-aligned surplus grids, then prune and polish.

HYPOTHESIS: SHGO can explore the low-dimensional lattice manifold more
systematically than the previous GP/EI sampler and uncover a better surplus
layout basin before turbine pruning.
AXIS: scipy_shgo global search for surplus-grid parameters with augmented
Lagrangian NAdam polishing and pruning.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import shgo

from pixwake.optim.boundary import polygon_sdf


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    n_extra = 8
    n_total = n_target + n_extra

    @jax.jit
    def aep_obj_total(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()
        aep = jnp.sum(p * weights[:, None]) * 8760.0 / 1e6
        return -aep

    wd_rad = jnp.deg2rad(wd)
    v_x = jnp.sum(jnp.cos(wd_rad) * weights)
    v_y = jnp.sum(jnp.sin(wd_rad) * weights)
    dominant_wd = jnp.arctan2(v_y, v_x)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    @jax.jit
    def generate_grid_surplus(params):
        sx, sy, theta_off, ox, oy, shear = params
        theta = dominant_wd + theta_off
        n_side = int(np.sqrt(n_total)) + 15
        ii, jj = jnp.meshgrid(
            jnp.arange(n_side) - n_side // 2,
            jnp.arange(n_side) - n_side // 2,
        )
        ix = ii.flatten()
        iy = jj.flatten()
        hx = ix * sx * min_spacing + (iy % 2) * sx * min_spacing / 2.0
        hy = iy * sy * min_spacing * jnp.sqrt(3.0) / 2.0
        hx = hx + shear * hy
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        sdf_vals = polygon_sdf(rx, ry, boundary)
        idx = jnp.argsort(sdf_vals)[:n_total]
        return rx[idx], ry[idx]

    @jax.jit
    def grid_aep(params):
        gx, gy = generate_grid_surplus(params)
        return -aep_obj_total(gx, gy)

    bounds = [
        (1.02, 4.25),
        (1.02, 4.25),
        (-float(jnp.pi) / 2.0, float(jnp.pi) / 2.0),
        (float(x_min), float(x_max)),
        (float(y_min), float(y_max)),
        (-0.55, 0.55),
    ]

    def shgo_objective(params_np):
        params = jnp.asarray(params_np)
        return -float(grid_aep(params))

    candidates = []
    try:
        result = shgo(
            shgo_objective,
            bounds,
            n=22,
            iters=1,
            sampling_method="simplicial",
            options={"maxfev": 80},
        )
        candidates.append(np.asarray(result.x, dtype=float))
        try:
            local_x = np.asarray(result.xl, dtype=float)
            local_f = np.asarray(result.funl, dtype=float)
            order = np.argsort(local_f)
            for idx in order[:5]:
                candidates.append(local_x[idx])
        except Exception:
            pass
    except Exception:
        pass

    cx = float(jnp.mean(boundary[:, 0]))
    cy = float(jnp.mean(boundary[:, 1]))
    candidates.extend(
        [
            np.array([1.08, 1.08, 0.0, cx, cy, 0.0]),
            np.array([1.20, 1.05, 0.35, cx, cy, 0.15]),
            np.array([1.05, 1.25, -0.35, cx, cy, -0.15]),
            np.array([1.45, 1.10, 0.75, cx, cy, 0.25]),
        ]
    )

    scored = []
    for cand in candidates:
        clipped = np.array(
            [np.clip(cand[i], bounds[i][0], bounds[i][1]) for i in range(6)]
        )
        val = float(grid_aep(jnp.asarray(clipped)))
        scored.append((val, clipped))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_params = jnp.asarray(np.stack([item[1] for item in scored[:6]]))

    @jax.jit
    def constraints_penalty_total(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        v_b = jnp.maximum(0.0, sdf + 0.01)
        pen_b = jnp.sum(lam_b * v_b + 0.5 * mu * v_b**2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_total, n_total)), k=1)
        v_s = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        pen_s = jnp.sum(mask * (lam_s * v_s + 0.5 * mu * v_s**2))
        return pen_b + pen_s

    @jax.jit
    def total_obj(x, y, lam_b, lam_s, mu):
        return aep_obj_total(x, y) + constraints_penalty_total(x, y, lam_b, lam_s, mu)

    grad_total = jax.jit(
        jax.vmap(jax.grad(total_obj, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None))
    )

    def run_surplus_polish(x, y, lr):
        m_x = jnp.zeros_like(x)
        m_y = jnp.zeros_like(y)
        v_x2 = jnp.zeros_like(x)
        v_y2 = jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_total, n_total))
        mu = 120.0
        curr_x = x
        curr_y = y

        for i in range(4):
            def step(carry, t):
                cx0, cy0, mx, my, vx2, vy2 = carry
                gx, gy = grad_total(cx0, cy0, lam_b, lam_s, mu)
                b1 = 0.9
                b2 = 0.999
                mx = b1 * mx + (1.0 - b1) * gx
                my = b1 * my + (1.0 - b1) * gy
                vx2 = b2 * vx2 + (1.0 - b2) * gx**2
                vy2 = b2 * vy2 + (1.0 - b2) * gy**2
                mx_h = (b1 * mx + (1.0 - b1) * gx) / (1.0 - b1 ** (t + 1))
                my_h = (b1 * my + (1.0 - b1) * gy) / (1.0 - b1 ** (t + 1))
                vx_h = vx2 / (1.0 - b2 ** (t + 1))
                vy_h = vy2 / (1.0 - b2 ** (t + 1))
                nx = cx0 - lr * mx_h / (jnp.sqrt(vx_h) + 1e-8)
                ny = cy0 - lr * my_h / (jnp.sqrt(vy_h) + 1e-8)
                return (nx, ny, mx, my, vx2, vy2), None

            init_carry = (curr_x, curr_y, m_x, m_y, v_x2, v_y2)
            (curr_x, curr_y, m_x, m_y, v_x2, v_y2), _ = jax.lax.scan(
                step, init_carry, jnp.arange(i * 180, (i + 1) * 180)
            )
            sdf = jax.vmap(lambda px, py: polygon_sdf(px, py, boundary))(curr_x, curr_y)
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx = curr_x[:, :, None] - curr_x[:, None, :]
            dy = curr_y[:, :, None] - curr_y[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= 2.2
        return curr_x, curr_y

    layouts = [generate_grid_surplus(top_params[i]) for i in range(top_params.shape[0])]
    cur_x = jnp.stack([layout[0] for layout in layouts])
    cur_y = jnp.stack([layout[1] for layout in layouts])
    cur_x, cur_y = run_surplus_polish(cur_x, cur_y, 10.0)

    def prune(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()
        indiv_aep = jnp.sum(p * weights[:, None], axis=0)
        idx = jnp.argsort(indiv_aep)[-n_target:]
        return x[idx], y[idx]

    pruned = [prune(cur_x[i], cur_y[i]) for i in range(cur_x.shape[0])]
    px = jnp.stack([layout[0] for layout in pruned])
    py = jnp.stack([layout[1] for layout in pruned])

    @jax.jit
    def aep_obj_final(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        aep = jnp.sum(p * weights[:, None]) * 8760.0 / 1e6
        return -aep

    @jax.jit
    def total_obj_final(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        v_b = jnp.maximum(0.0, sdf + 0.01)
        pen_b = jnp.sum(lam_b * v_b + 0.5 * mu * v_b**2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_target, n_target)), k=1)
        v_s = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        pen_s = jnp.sum(mask * (lam_s * v_s + 0.5 * mu * v_s**2))
        return aep_obj_final(x, y) + pen_b + pen_s

    grad_final = jax.jit(
        jax.vmap(jax.grad(total_obj_final, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None))
    )

    def run_final_polish(x, y, lr):
        m_x = jnp.zeros_like(x)
        m_y = jnp.zeros_like(y)
        v_x2 = jnp.zeros_like(x)
        v_y2 = jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_target, n_target))
        mu = 1000.0
        curr_x = x
        curr_y = y

        for i in range(3):
            def step(carry, t):
                cx0, cy0, mx, my, vx2, vy2 = carry
                gx, gy = grad_final(cx0, cy0, lam_b, lam_s, mu)
                b1 = 0.9
                b2 = 0.999
                mx = b1 * mx + (1.0 - b1) * gx
                my = b1 * my + (1.0 - b1) * gy
                vx2 = b2 * vx2 + (1.0 - b2) * gx**2
                vy2 = b2 * vy2 + (1.0 - b2) * gy**2
                mx_h = (b1 * mx + (1.0 - b1) * gx) / (1.0 - b1 ** (t + 1))
                my_h = (b1 * my + (1.0 - b1) * gy) / (1.0 - b1 ** (t + 1))
                vx_h = vx2 / (1.0 - b2 ** (t + 1))
                vy_h = vy2 / (1.0 - b2 ** (t + 1))
                nx = cx0 - lr * mx_h / (jnp.sqrt(vx_h) + 1e-8)
                ny = cy0 - lr * my_h / (jnp.sqrt(vy_h) + 1e-8)
                return (nx, ny, mx, my, vx2, vy2), None

            init_carry = (curr_x, curr_y, m_x, m_y, v_x2, v_y2)
            (curr_x, curr_y, m_x, m_y, v_x2, v_y2), _ = jax.lax.scan(
                step, init_carry, jnp.arange(i * 360, (i + 1) * 360)
            )
            sdf = jax.vmap(lambda pxx, pyy: polygon_sdf(pxx, pyy, boundary))(
                curr_x, curr_y
            )
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx = curr_x[:, :, None] - curr_x[:, None, :]
            dy = curr_y[:, :, None] - curr_y[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= 3.0
        return curr_x, curr_y

    seed_scores = jax.vmap(aep_obj_final)(px, py)
    best_seed_idx = jnp.argsort(seed_scores)[:3]
    fx, fy = run_final_polish(px[best_seed_idx], py[best_seed_idx], 4.0)
    best_idx = jnp.argmin(jax.vmap(aep_obj_final)(fx, fy))
    final_x = fx[best_idx]
    final_y = fy[best_idx]

    def project(x, y):
        for _ in range(28):
            sdf = polygon_sdf(x, y, boundary)
            grad_b = jax.vmap(
                jax.grad(
                    lambda px0, py0: polygon_sdf(
                        jnp.array([px0]), jnp.array([py0]), boundary
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

    return project(final_x, final_y)
