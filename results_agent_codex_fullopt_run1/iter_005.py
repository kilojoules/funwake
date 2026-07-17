"""Hand-rolled low-momentum Adam with tolerance-aware polish.

HYPOTHESIS: A custom Adam loop that keeps TopFarm's useful inverse-LR
constraint ramp, but controls the final spacing tolerance explicitly, can
match the strong SGD basin while avoiding the previous projection-heavy
custom attempt's conservative moves.

AXIS: custom_adam with wind-aware initialization, one perturbation restart,
and a short relaxed-spacing Adam polish; no packaged layout solver call.

LESSON: Pending score.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    n_verts = boundary.shape[0]
    x_min = float(jnp.min(boundary[:, 0]))
    x_max = float(jnp.max(boundary[:, 0]))
    y_min = float(jnp.min(boundary[:, 1]))
    y_max = float(jnp.max(boundary[:, 1]))
    center = jnp.mean(boundary, axis=0)

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def edge_distances(x, y):
        def one_edge(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex = x2 - x1
            ey = y2 - y1
            el = jnp.sqrt(ex * ex + ey * ey) + 1e-10
            nx = -ey / el
            ny = ex / el
            return (x - x1) * nx + (y - y1) * ny

        return jax.vmap(one_edge)(jnp.arange(n_verts))

    def inside_mask(x, y):
        return jnp.min(edge_distances(x, y), axis=0) > 0.0

    def rotated_grid(angle, spacing_mult, offset):
        ca = jnp.cos(angle)
        sa = jnp.sin(angle)
        rot = jnp.array([[ca, -sa], [sa, ca]])
        inv_rot = jnp.array([[ca, sa], [-sa, ca]])
        rb = (rot @ (boundary - center).T).T
        rx_min, ry_min = jnp.min(rb, axis=0)
        rx_max, ry_max = jnp.max(rb, axis=0)
        spacing = min_spacing * spacing_mult
        nx = max(4, int(jnp.ceil((rx_max - rx_min) / spacing)) + 2)
        ny = max(4, int(jnp.ceil((ry_max - ry_min) / spacing)) + 2)
        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min + offset * spacing, rx_max - offset * spacing, nx),
            jnp.linspace(ry_min + offset * spacing, ry_max - offset * spacing, ny),
        )
        pts_r = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
        pts = (inv_rot @ pts_r.T).T + center
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        if len(px) >= n_target:
            rank = (
                (px - center[0]) * jnp.cos(angle + 0.21)
                + (py - center[1]) * jnp.sin(angle + 0.21)
            )
            order = jnp.argsort(rank)
            px = px[order]
            py = py[order]
            idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
            return px[idx], py[idx]

        key = jax.random.PRNGKey(19)
        return (
            jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max),
            jax.random.uniform(jax.random.PRNGKey(23), (n_target,), minval=y_min, maxval=y_max),
        )

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )

    grad_obj = jax.grad(objective, argnums=(0, 1))

    def compute_mid(lr0, gamma_min, n_steps):
        lo = 0.0
        hi = 0.1
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lr = lr0
            for t in range(1, n_steps + 1):
                lr *= 1.0 / (1.0 + mid * t)
            if lr < gamma_min:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def make_runner(
        const_steps,
        decay_steps,
        gamma_min,
        beta1,
        beta2,
        b_weight,
        s_weight,
        lr0,
    ):
        total_steps = const_steps + decay_steps
        mid = compute_mid(lr0, gamma_min, decay_steps)

        def con_penalty(x, y, spacing_scale):
            return (
                b_weight * boundary_penalty(x, y, boundary)
                + s_weight * spacing_penalty(x, y, min_spacing * spacing_scale)
            )

        grad_con = jax.grad(con_penalty, argnums=(0, 1))

        @jax.jit
        def run(z0, spacing_scale):
            x0 = z0[:n_target]
            y0 = z0[n_target:]
            g0x, g0y = grad_obj(x0, y0)
            alpha0 = jnp.mean(jnp.abs(jnp.concatenate([g0x, g0y]))) / lr0
            mx0 = jnp.zeros_like(x0)
            my0 = jnp.zeros_like(y0)
            vx0 = jnp.zeros_like(x0)
            vy0 = jnp.zeros_like(y0)

            def body(carry, t):
                x, y, mx, my, vx, vy, lr, alpha = carry
                gox, goy = grad_obj(x, y)
                gcx, gcy = grad_con(x, y, spacing_scale)
                gx = gox + alpha * gcx
                gy = goy + alpha * gcy

                step = t + 1.0
                mx = beta1 * mx + (1.0 - beta1) * gx
                my = beta1 * my + (1.0 - beta1) * gy
                vx = beta2 * vx + (1.0 - beta2) * gx * gx
                vy = beta2 * vy + (1.0 - beta2) * gy * gy
                mx_hat = mx / (1.0 - beta1**step)
                my_hat = my / (1.0 - beta1**step)
                vx_hat = vx / (1.0 - beta2**step)
                vy_hat = vy / (1.0 - beta2**step)

                x = x - lr * mx_hat / (jnp.sqrt(vx_hat) + 1e-12)
                y = y - lr * my_hat / (jnp.sqrt(vy_hat) + 1e-12)

                decaying = step > const_steps
                decay_step = jnp.where(decaying, step - const_steps, 0.0)
                new_lr = jnp.where(decaying, lr / (1.0 + mid * decay_step), lr)
                new_alpha = jnp.where(decaying, alpha0 * lr0 / new_lr, alpha)
                return (x, y, mx, my, vx, vy, new_lr, new_alpha), None

            init = (x0, y0, mx0, my0, vx0, vy0, lr0, alpha0)
            (x, y, _, _, _, _, _, _), _ = jax.lax.scan(
                body, init, jnp.arange(float(total_steps))
            )
            return x, y

        return run

    main_adam = make_runner(
        const_steps=1500,
        decay_steps=3500,
        gamma_min=0.0025,
        beta1=0.12,
        beta2=0.22,
        b_weight=100.0,
        s_weight=100.0,
        lr0=150.0,
    )
    polish_adam = make_runner(
        const_steps=80,
        decay_steps=520,
        gamma_min=0.15,
        beta1=0.35,
        beta2=0.82,
        b_weight=130.0,
        s_weight=70.0,
        lr0=28.0,
    )

    def feasible(x, y):
        bnd = boundary_penalty(x, y, boundary)
        spc = spacing_penalty(x, y, min_spacing * 0.99)
        return float(bnd) < 1e-3 and float(spc) < 1e-3

    def consider(best_x, best_y, best_obj, cand_x, cand_y):
        if feasible(cand_x, cand_y):
            cand_obj = float(objective(cand_x, cand_y))
            if cand_obj < best_obj:
                return cand_x, cand_y, cand_obj
        return best_x, best_y, best_obj

    init_x, init_y = rotated_grid(mean_wind + 0.5 * jnp.pi, 1.0, 0.45)
    best_x, best_y = init_x, init_y
    best_obj = float(objective(best_x, best_y)) if feasible(best_x, best_y) else 1.0e30

    opt_x, opt_y = main_adam(jnp.concatenate([init_x, init_y]), 1.0)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, opt_x, opt_y)

    key = jax.random.PRNGKey(777)
    dx = jax.random.normal(key, (n_target,)) * (0.36 * min_spacing)
    dy = jax.random.normal(jax.random.PRNGKey(778), (n_target,)) * (0.36 * min_spacing)
    span_x = x_max - x_min
    span_y = y_max - y_min
    restart_x = jnp.clip(opt_x + dx, x_min - 0.05 * span_x, x_max + 0.05 * span_x)
    restart_y = jnp.clip(opt_y + dy, y_min - 0.05 * span_y, y_max + 0.05 * span_y)
    opt2_x, opt2_y = main_adam(jnp.concatenate([restart_x, restart_y]), 1.0)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, opt2_x, opt2_y)

    tight_x, tight_y = polish_adam(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, tight_x, tight_y)

    return best_x, best_y
