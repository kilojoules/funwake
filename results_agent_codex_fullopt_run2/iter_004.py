"""Projected custom Adam optimizer.

HYPOTHESIS: A hand-rolled Adam loop with projection after every step can follow
objective gradients differently from TopFarm SGD while keeping layouts feasible.
AXIS: custom_adam with explicit boundary and spacing projection.
LESSON: Pending score.
"""

from functools import partial

import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    signed_area = 0.5 * jnp.sum(
        boundary[:, 0] * jnp.roll(boundary[:, 1], -1)
        - jnp.roll(boundary[:, 0], -1) * boundary[:, 1]
    )
    boundary = jnp.where(signed_area < 0.0, boundary[::-1], boundary)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    n_verts = boundary.shape[0]

    def aep_gwh(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, : len(x)]
        return jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def edge_clearance(px, py):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-12
            return (px - x1) * (-ey / el) + (py - y1) * (ex / el)

        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0)

    def min_distance(x, y):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(len(x)) * 1e12)
        return jnp.min(dist)

    def candidate_cloud(step_factor, margin_factor):
        step = min_spacing * step_factor
        margin = min_spacing * margin_factor
        nx = int(jnp.maximum(3, jnp.ceil((x_max - x_min) / step)))
        ny = int(jnp.maximum(3, jnp.ceil((y_max - y_min) / step)))
        gx, gy = jnp.meshgrid(
            jnp.linspace(x_min + margin, x_max - margin, nx),
            jnp.linspace(y_min + margin, y_max - margin, ny),
        )
        cand_x = gx.flatten()
        cand_y = gy.flatten()
        inside = edge_clearance(cand_x, cand_y) > margin * 0.20
        return cand_x[inside], cand_y[inside]

    def farthest_init(cand_x, cand_y, mode):
        if len(cand_x) < n_target:
            gx, gy = jnp.meshgrid(
                jnp.linspace(x_min + min_spacing, x_max - min_spacing, n_target),
                jnp.array([0.5 * (y_min + y_max)]),
            )
            return gx.flatten()[:n_target], gy.flatten()[:n_target]

        cx = jnp.mean(boundary[:, 0])
        cy = jnp.mean(boundary[:, 1])
        energy = weights * ws**3
        theta = wd[jnp.argmax(energy)] * jnp.pi / 180.0
        down_x = jnp.sin(theta)
        down_y = jnp.cos(theta)
        cross_x = down_y
        cross_y = -down_x
        down = (cand_x - cx) * down_x + (cand_y - cy) * down_y
        cross = (cand_x - cx) * cross_x + (cand_y - cy) * cross_y
        radial = (cand_x - cx) ** 2 + (cand_y - cy) ** 2

        first_idx = jnp.argmax(
            jnp.where(mode == 0, radial, jnp.where(mode == 1, -down, cross))
        )
        init_x = jnp.zeros(n_target)
        init_y = jnp.zeros(n_target)
        init_x = init_x.at[0].set(cand_x[first_idx])
        init_y = init_y.at[0].set(cand_y[first_idx])
        best_dist2 = (cand_x - init_x[0]) ** 2 + (cand_y - init_y[0]) ** 2

        for i in range(1, n_target):
            valid = best_dist2 >= (min_spacing * 1.01) ** 2
            score = jnp.where(valid, best_dist2, -1.0)
            next_idx = jnp.argmax(score)
            next_idx = jnp.where(jnp.max(score) > 0.0, next_idx, jnp.argmax(best_dist2))
            init_x = init_x.at[i].set(cand_x[next_idx])
            init_y = init_y.at[i].set(cand_y[next_idx])
            dist2_new = (cand_x - init_x[i]) ** 2 + (cand_y - init_y[i]) ** 2
            best_dist2 = jnp.minimum(best_dist2, dist2_new)

        return init_x, init_y

    def project_boundary_once(x, y, margin):
        def edge_body(i, xy):
            px, py = xy
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-12
            nx = -ey / el
            ny = ex / el
            d = (px - x1) * nx + (py - y1) * ny
            shift = jnp.maximum(0.0, margin - d)
            return px + shift * nx, py + shift * ny

        return jax.lax.fori_loop(0, n_verts, edge_body, (x, y))

    def project_spacing_once(x, y, target):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        idx = jnp.arange(len(x), dtype=x.dtype)
        jitter_x = jnp.sin((idx[:, None] + 1.0) * (idx[None, :] + 2.0))
        jitter_y = jnp.cos((idx[:, None] + 3.0) * (idx[None, :] + 1.0))
        dx = dx + jnp.eye(len(x)) + (1.0 - jnp.eye(len(x))) * jitter_x * 1e-6
        dy = dy + (1.0 - jnp.eye(len(x))) * jitter_y * 1e-6
        dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(len(x)) * 1e12)
        overlap = jnp.maximum(0.0, target - dist) * (1.0 - jnp.eye(len(x)))
        push_x = jnp.sum(0.5 * overlap * dx / (dist + 1e-9), axis=1)
        push_y = jnp.sum(0.5 * overlap * dy / (dist + 1e-9), axis=1)
        return x + push_x, y + push_y

    def project(x, y):
        boundary_margin = min_spacing * 0.015
        spacing_target = min_spacing * 1.004

        def body(_, xy):
            px, py = xy
            px, py = project_boundary_once(px, py, boundary_margin)
            px, py = project_spacing_once(px, py, spacing_target)
            px, py = project_boundary_once(px, py, boundary_margin)
            return px, py

        return jax.lax.fori_loop(0, 5, body, (x, y))

    def loss_with_aux(x, y):
        aep = aep_gwh(x, y)
        clearance = edge_clearance(x, y)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(len(x)) * 1e12)
        i_upper, j_upper = jnp.triu_indices(len(x), k=1)
        pair_dist = dist[i_upper, j_upper]

        boundary_violation = jnp.maximum(0.0, min_spacing * 0.02 - clearance)
        spacing_violation = jnp.maximum(0.0, min_spacing * 1.006 - pair_dist)
        penalty = (
            5e-4 * jnp.sum(boundary_violation**2)
            + 8e-4 * jnp.sum(spacing_violation**2)
        )
        loss = -aep + penalty
        return loss, (aep, jnp.min(clearance), jnp.min(pair_dist))

    value_and_grad = jax.value_and_grad(loss_with_aux, argnums=(0, 1), has_aux=True)

    @partial(jax.jit, static_argnames=("steps",))
    def adam_solve(init_x, init_y, lr0, steps):
        init_x, init_y = project(init_x, init_y)
        init_aep = aep_gwh(init_x, init_y)
        carry0 = (
            init_x,
            init_y,
            jnp.zeros_like(init_x),
            jnp.zeros_like(init_y),
            jnp.zeros_like(init_x),
            jnp.zeros_like(init_y),
            init_x,
            init_y,
            init_aep,
        )

        def body(carry, t):
            x, y, mx, my, vx, vy, best_x, best_y, best_aep = carry
            (_, (aep, clearance, pair_dist)), (gx, gy) = value_and_grad(x, y)
            finite = jnp.all(jnp.isfinite(gx)) & jnp.all(jnp.isfinite(gy))
            gx = jnp.where(finite, gx, jnp.zeros_like(gx))
            gy = jnp.where(finite, gy, jnp.zeros_like(gy))

            b1 = 0.88
            b2 = 0.98
            step = t + 1
            mx = b1 * mx + (1.0 - b1) * gx
            my = b1 * my + (1.0 - b1) * gy
            vx = b2 * vx + (1.0 - b2) * gx**2
            vy = b2 * vy + (1.0 - b2) * gy**2
            mx_hat = mx / (1.0 - b1**step)
            my_hat = my / (1.0 - b1**step)
            vx_hat = vx / (1.0 - b2**step)
            vy_hat = vy / (1.0 - b2**step)

            frac = t.astype(x.dtype) / jnp.maximum(1.0, (steps - 1))
            lr = lr0 * (0.10 + 0.90 * (1.0 - frac) ** 1.6)
            x_new = x - lr * mx_hat / (jnp.sqrt(vx_hat) + 1e-9)
            y_new = y - lr * my_hat / (jnp.sqrt(vy_hat) + 1e-9)
            x_new, y_new = project(x_new, y_new)

            ok = (
                (clearance >= -1e-3)
                & (pair_dist >= min_spacing * 0.99)
                & jnp.isfinite(aep)
            )
            improve = ok & (aep > best_aep)
            best_x = jnp.where(improve, x, best_x)
            best_y = jnp.where(improve, y, best_y)
            best_aep = jnp.where(improve, aep, best_aep)
            return (x_new, y_new, mx, my, vx, vy, best_x, best_y, best_aep), None

        final, _ = jax.lax.scan(body, carry0, jnp.arange(steps))
        x, y, _, _, _, _, best_x, best_y, best_aep = final
        final_aep = aep_gwh(x, y)
        x, y = project(x, y)
        final_ok = (
            (boundary_penalty(x, y, boundary) < 1e-3)
            & (spacing_penalty(x, y, min_spacing) < 1e-3)
            & jnp.isfinite(final_aep)
        )
        use_final = final_ok & (final_aep > best_aep)
        return jnp.where(use_final, x, best_x), jnp.where(use_final, y, best_y)

    dense_x, dense_y = candidate_cloud(0.44, 0.10)
    open_x, open_y = candidate_cloud(0.72, 0.30)
    starts = (
        farthest_init(dense_x, dense_y, 0),
        farthest_init(dense_x, dense_y, 1),
        farthest_init(open_x, open_y, 2),
    )

    best_x, best_y = project(starts[0][0], starts[0][1])
    best_aep = jnp.where(
        (boundary_penalty(best_x, best_y, boundary) < 1e-3)
        & (spacing_penalty(best_x, best_y, min_spacing) < 1e-3),
        aep_gwh(best_x, best_y),
        -jnp.inf,
    )

    for init_x, init_y, lr0, steps in (
        (starts[0][0], starts[0][1], 90.0, 1700),
        (starts[1][0], starts[1][1], 130.0, 1700),
        (starts[2][0], starts[2][1], 70.0, 1200),
    ):
        cand_x, cand_y = adam_solve(init_x, init_y, lr0, steps)
        cand_x, cand_y = project(cand_x, cand_y)
        cand_aep = aep_gwh(cand_x, cand_y)
        cand_ok = (
            (boundary_penalty(cand_x, cand_y, boundary) < 1e-3)
            & (spacing_penalty(cand_x, cand_y, min_spacing) < 1e-3)
            & (min_distance(cand_x, cand_y) >= min_spacing * 0.99)
        )
        if cand_ok & (cand_aep > best_aep):
            best_aep = cand_aep
            best_x = cand_x
            best_y = cand_y

    return best_x, best_y
