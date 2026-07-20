"""Penalty-only custom Adam with staggered starts.

HYPOTHESIS: Projection kept the previous custom runs in a lower-AEP basin; a
penalty-only jax.grad Adam loop can use active constraints more fluidly while
still selecting only feasible iterates.
AXIS: custom_adam penalty optimizer with staggered lattice-like initialization.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, : len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    n_verts = boundary.shape[0]

    def edge_clearance(px, py):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (px - x1) * (-ey / el) + (py - y1) * (ex / el)

        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0)

    def candidate_cloud(step_x, step_y, margin, stagger):
        nx = int(jnp.maximum(3, jnp.ceil((x_max - x_min) / step_x)))
        ny = int(jnp.maximum(3, jnp.ceil((y_max - y_min) / step_y)))
        gx, gy = jnp.meshgrid(
            jnp.linspace(x_min + margin, x_max - margin, nx),
            jnp.linspace(y_min + margin, y_max - margin, ny),
        )
        row = jnp.arange(ny)[:, None]
        gx = gx + jnp.where(row % 2 == 0, 0.0, stagger * step_x)
        cand_x = gx.flatten()
        cand_y = gy.flatten()
        inside = edge_clearance(cand_x, cand_y) > margin * 0.1
        return cand_x[inside], cand_y[inside]

    def farthest_init(cand_x, cand_y, mode):
        if len(cand_x) < n_target:
            key = jax.random.PRNGKey(300 + mode)
            key, kx = jax.random.split(key)
            key, ky = jax.random.split(key)
            return (
                jax.random.uniform(kx, (n_target,), minval=float(x_min), maxval=float(x_max)),
                jax.random.uniform(ky, (n_target,), minval=float(y_min), maxval=float(y_max)),
            )

        cx = jnp.mean(boundary[:, 0])
        cy = jnp.mean(boundary[:, 1])
        energy = weights * ws**3
        theta = wd[jnp.argmax(energy)] * jnp.pi / 180.0
        down_x = jnp.sin(theta)
        down_y = jnp.cos(theta)
        proj = (cand_x - cx) * down_x + (cand_y - cy) * down_y
        radial = (cand_x - cx) ** 2 + (cand_y - cy) ** 2
        first_idx = jnp.where(mode == 0, jnp.argmax(radial), jnp.argmin(proj))

        init_x = jnp.zeros(n_target)
        init_y = jnp.zeros(n_target)
        init_x = init_x.at[0].set(cand_x[first_idx])
        init_y = init_y.at[0].set(cand_y[first_idx])
        best_dist2 = (cand_x - init_x[0]) ** 2 + (cand_y - init_y[0]) ** 2

        for i in range(1, n_target):
            valid = best_dist2 >= (min_spacing * 1.0) ** 2
            score = jnp.where(valid, best_dist2, -1.0)
            next_idx = jnp.argmax(score)
            next_idx = jnp.where(jnp.max(score) > 0.0, next_idx, jnp.argmax(best_dist2))
            init_x = init_x.at[i].set(cand_x[next_idx])
            init_y = init_y.at[i].set(cand_y[next_idx])
            dist2_new = (cand_x - init_x[i]) ** 2 + (cand_y - init_y[i]) ** 2
            best_dist2 = jnp.minimum(best_dist2, dist2_new)

        return init_x, init_y

    def ordered_init(cand_x, cand_y):
        if len(cand_x) < n_target:
            return farthest_init(cand_x, cand_y, 0)
        order = jnp.lexsort((cand_x, cand_y))
        sx = cand_x[order]
        sy = cand_y[order]
        idx = jnp.round(jnp.linspace(0, len(sx) - 1, n_target)).astype(int)
        return sx[idx], sy[idx]

    def min_distance(x, y):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + jnp.eye(len(x)) * 1e12)
        return jnp.min(dist)

    def feasible(x, y):
        return (
            (boundary_penalty(x, y, boundary) < 1e-3)
            & (spacing_penalty(x, y, min_spacing) < 1e-3)
            & (min_distance(x, y) >= min_spacing * 0.99)
        )

    def compute_mid(lr0, gamma_min, max_iter):
        lo = 0.0
        hi = 0.1
        for _ in range(80):
            mid = (lo + hi) * 0.5
            lr = lr0
            for t in range(1, max_iter + 1):
                lr = lr / (1.0 + mid * t)
            if lr < gamma_min:
                hi = mid
            else:
                lo = mid
        return (lo + hi) * 0.5

    def make_solver(lr0, max_iter, const_iter, beta1, beta2, spacing_w, boundary_w):
        mid = compute_mid(lr0, 0.01, max_iter)
        total_iter = max_iter + const_iter

        def constraint_penalty(x, y):
            return boundary_w * boundary_penalty(
                x, y, boundary
            ) + spacing_w * spacing_penalty(x, y, min_spacing)

        obj_vg = jax.value_and_grad(objective, argnums=(0, 1))
        con_grad = jax.grad(constraint_penalty, argnums=(0, 1))

        @jax.jit
        def solve(init_x, init_y):
            init_obj, (init_gx, init_gy) = obj_vg(init_x, init_y)
            grad_mag = jnp.concatenate([jnp.abs(init_gx), jnp.abs(init_gy)])
            alpha0 = jnp.mean(grad_mag) / lr0
            init_ok = feasible(init_x, init_y)
            init_aep = -init_obj
            carry0 = (
                init_x,
                init_y,
                jnp.zeros_like(init_x),
                jnp.zeros_like(init_y),
                jnp.zeros_like(init_x),
                jnp.zeros_like(init_y),
                jnp.array(lr0, dtype=init_x.dtype),
                alpha0,
                init_x,
                init_y,
                jnp.where(init_ok, init_aep, -jnp.inf),
            )

            def body(carry, t):
                x, y, mx, my, vx, vy, lr, alpha, best_x, best_y, best_aep = carry
                obj, (gx, gy) = obj_vg(x, y)
                cgx, cgy = con_grad(x, y)
                aep = -obj
                ok = feasible(x, y) & jnp.isfinite(aep)
                improve = ok & (aep > best_aep)
                best_x = jnp.where(improve, x, best_x)
                best_y = jnp.where(improve, y, best_y)
                best_aep = jnp.where(improve, aep, best_aep)

                full_gx = gx + alpha * cgx
                full_gy = gy + alpha * cgy
                finite = jnp.all(jnp.isfinite(full_gx)) & jnp.all(jnp.isfinite(full_gy))
                full_gx = jnp.where(finite, full_gx, jnp.zeros_like(full_gx))
                full_gy = jnp.where(finite, full_gy, jnp.zeros_like(full_gy))

                step = t + 1
                mx = beta1 * mx + (1.0 - beta1) * full_gx
                my = beta1 * my + (1.0 - beta1) * full_gy
                vx = beta2 * vx + (1.0 - beta2) * full_gx**2
                vy = beta2 * vy + (1.0 - beta2) * full_gy**2
                mx_hat = mx / (1.0 - beta1**step)
                my_hat = my / (1.0 - beta1**step)
                vx_hat = vx / (1.0 - beta2**step)
                vy_hat = vy / (1.0 - beta2**step)
                x = x - lr * mx_hat / (jnp.sqrt(vx_hat) + 1e-12)
                y = y - lr * my_hat / (jnp.sqrt(vy_hat) + 1e-12)

                decaying = step > const_iter
                decay_it = jnp.where(decaying, step - const_iter, 0)
                new_lr = jnp.where(decaying, lr / (1.0 + mid * decay_it), lr)
                new_alpha = jnp.where(decaying, alpha0 * lr0 / new_lr, alpha)
                return (x, y, mx, my, vx, vy, new_lr, new_alpha, best_x, best_y, best_aep), None

            final, _ = jax.lax.scan(body, carry0, jnp.arange(total_iter))
            x, y, _, _, _, _, _, _, best_x, best_y, best_aep = final
            final_aep = -objective(x, y)
            use_final = feasible(x, y) & (final_aep > best_aep)
            return jnp.where(use_final, x, best_x), jnp.where(use_final, y, best_y)

        return solve

    coarse_x, coarse_y = candidate_cloud(min_spacing, min_spacing * 0.8660254, min_spacing * 0.45, 0.5)
    dense_x, dense_y = candidate_cloud(min_spacing * 0.48, min_spacing * 0.42, min_spacing * 0.12, 0.5)
    starts = (
        ordered_init(coarse_x, coarse_y),
        farthest_init(coarse_x, coarse_y, 1),
        farthest_init(dense_x, dense_y, 0),
    )

    solve_main = make_solver(200.0, 4000, 2000, 0.1, 0.2, 3.0, 2.5)
    solve_soft = make_solver(120.0, 2800, 1200, 0.12, 0.25, 2.2, 2.0)

    best_x, best_y = starts[0]
    best_aep = jnp.where(feasible(best_x, best_y), -objective(best_x, best_y), -jnp.inf)

    for init_x, init_y, solver in (
        (starts[0][0], starts[0][1], solve_main),
        (starts[1][0], starts[1][1], solve_main),
        (starts[2][0], starts[2][1], solve_soft),
    ):
        cand_x, cand_y = solver(init_x, init_y)
        cand_aep = -objective(cand_x, cand_y)
        if feasible(cand_x, cand_y) & (cand_aep > best_aep):
            best_aep = cand_aep
            best_x = cand_x
            best_y = cand_y

        init_aep = -objective(init_x, init_y)
        if feasible(init_x, init_y) & (init_aep > best_aep):
            best_aep = init_aep
            best_x = init_x
            best_y = init_y

    return best_x, best_y
