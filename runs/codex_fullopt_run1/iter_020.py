"""Weighted wind-sector kmeans initializer with incumbent recovery.

HYPOTHESIS: kmeans clustering of weighted wind sectors can create a seed that
mixes row orientations from the major wind regimes, giving the local penalty
solver a different basin than the previous population search over one row
family.

AXIS: init_kmeans over circular wind directions to build a mixed-orientation
candidate set, followed by the established local penalty polish and incumbent
recovery starts.

LESSON: Pending score.
"""
import numpy as np
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    n_verts = boundary.shape[0]
    x_min = float(jnp.min(boundary[:, 0]))
    x_max = float(jnp.max(boundary[:, 0]))
    y_min = float(jnp.min(boundary[:, 1]))
    y_max = float(jnp.max(boundary[:, 1]))
    span_x = x_max - x_min
    span_y = y_max - y_min
    center = jnp.mean(boundary, axis=0)

    def objective_xy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    objective_xy_jit = jax.jit(objective_xy)

    def objective_z(z):
        return objective_xy(z[:n_target], z[n_target:])

    def constraint_z(z, spacing_scale):
        x = z[:n_target]
        y = z[n_target:]
        return 100.0 * boundary_penalty(x, y, boundary) + 100.0 * spacing_penalty(
            x, y, min_spacing * spacing_scale
        )

    obj_vg = jax.jit(jax.value_and_grad(objective_z))
    con_vg = jax.jit(jax.value_and_grad(constraint_z))

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

    def staggered_candidates(angle, spacing_mult, phase, skew):
        ca = jnp.cos(angle)
        sa = jnp.sin(angle)
        rot = jnp.array([[ca, -sa], [sa, ca]])
        inv_rot = jnp.array([[ca, sa], [-sa, ca]])
        rb = (rot @ (boundary - center).T).T
        rx_min, ry_min = jnp.min(rb, axis=0)
        rx_max, ry_max = jnp.max(rb, axis=0)
        spacing = min_spacing * spacing_mult
        row_step = spacing * jnp.sqrt(3.0) * 0.5
        nx = max(4, int(jnp.ceil((rx_max - rx_min) / spacing)) + 5)
        ny = max(4, int(jnp.ceil((ry_max - ry_min) / row_step)) + 5)
        gx = rx_min - 1.5 * spacing + spacing * phase + spacing * jnp.arange(nx)
        gy = ry_min - 1.5 * row_step + row_step * (0.41 + phase) + row_step * jnp.arange(ny)
        xx, yy = jnp.meshgrid(gx, gy)
        row = jnp.arange(ny)[:, None]
        xx = xx + (row % 2) * 0.5 * spacing + skew * (yy - center[1])
        pts_r = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        pts = (inv_rot @ pts_r.T).T + center
        mask = inside_mask(pts[:, 0], pts[:, 1])
        return pts[:, 0][mask], pts[:, 1][mask]

    def staggered_seed(angle, spacing_mult, phase, skew):
        px, py = staggered_candidates(angle, spacing_mult, phase, skew)
        if len(px) >= n_target:
            rank = (
                (px - center[0]) * jnp.cos(angle + 0.31)
                + (py - center[1]) * jnp.sin(angle + 0.31)
                + 0.08 * min_spacing * jnp.sin(px / (2.7 * min_spacing))
            )
            order = jnp.argsort(rank)
            px = px[order]
            py = py[order]
            idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
            return px[idx], py[idx]

        gx, gy = jnp.meshgrid(
            jnp.linspace(x_min + 0.06 * span_x, x_max - 0.06 * span_x, n_target),
            jnp.linspace(y_min + 0.06 * span_y, y_max - 0.06 * span_y, n_target),
        )
        pts = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]
        idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
        return px[idx], py[idx]

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )
    base_angle = mean_wind + 0.5 * jnp.pi

    def feasible(x, y):
        bnd = boundary_penalty(x, y, boundary)
        spc = spacing_penalty(x, y, min_spacing * 0.99)
        return float(bnd) < 1e-3 and float(spc) < 1e-3

    def circular_kmeans(n_clusters):
        theta = np.deg2rad(np.asarray(wd, dtype=float))
        speed = np.asarray(ws, dtype=float)
        mass = np.asarray(weights, dtype=float) * np.maximum(speed, 0.0) ** 3
        mass = mass / (np.sum(mass) + 1e-30)

        chosen = [int(np.argmax(mass))]
        while len(chosen) < n_clusters:
            best_i = 0
            best_value = -1.0
            for i in range(theta.size):
                if i in chosen:
                    continue
                nearest = min(abs(np.angle(np.exp(1j * (theta[i] - theta[j])))) for j in chosen)
                value = mass[i] * (0.35 + nearest)
                if value > best_value:
                    best_i = i
                    best_value = value
            chosen.append(best_i)

        centers = theta[np.asarray(chosen, dtype=int)].copy()
        labels = np.zeros(theta.size, dtype=int)
        for _ in range(10):
            diff = np.angle(np.exp(1j * (theta[:, None] - centers[None, :])))
            labels = np.argmin(np.abs(diff), axis=1)
            for k in range(n_clusters):
                m = labels == k
                if np.any(m):
                    centers[k] = np.arctan2(
                        np.sum(mass[m] * np.sin(theta[m])),
                        np.sum(mass[m] * np.cos(theta[m])),
                    )

        cluster_mass = np.array([np.sum(mass[labels == k]) for k in range(n_clusters)])
        order = np.argsort(-cluster_mass)
        return centers[order], cluster_mass[order]

    def kmeans_mixed_seed():
        cluster_centers, cluster_mass = circular_kmeans(3)
        xs = []
        ys = []
        scores = []
        for k in range(3):
            angle = float(cluster_centers[k] + 0.5 * np.pi)
            phase = float(np.mod(0.17 + 0.43 * k + 0.9 * cluster_mass[k], 1.0))
            skew = float((k - 1) * 0.007 + 0.010 * np.sin(cluster_centers[k]))
            px, py = staggered_candidates(angle, 1.000 + 0.003 * (k == 1), phase, skew)
            if len(px) == 0:
                continue
            px_np = np.asarray(px, dtype=float)
            py_np = np.asarray(py, dtype=float)
            dx = px_np - float(center[0])
            dy = py_np - float(center[1])
            score = np.zeros(px_np.shape, dtype=float)
            for c, w in zip(cluster_centers, cluster_mass):
                down = dx * np.cos(c) + dy * np.sin(c)
                cross = -dx * np.sin(c) + dy * np.cos(c)
                score += w * (
                    0.55 * np.sin(cross / (1.35 * float(min_spacing)) + 0.7 * k)
                    + 0.25 * np.cos(down / (2.20 * float(min_spacing)) - 0.4 * k)
                )
            margin = np.asarray(jnp.min(edge_distances(px, py), axis=0), dtype=float)
            score += 0.00005 * margin + 0.015 * np.sin(px_np / (3.1 * float(min_spacing)))
            xs.append(px_np)
            ys.append(py_np)
            scores.append(score)

        if not xs:
            return staggered_seed(base_angle, 1.000, 0.24, 0.000)

        cand_x = np.concatenate(xs)
        cand_y = np.concatenate(ys)
        cand_score = np.concatenate(scores)
        order = np.argsort(-cand_score)
        keep_x = []
        keep_y = []
        min_d2 = float(min_spacing * 0.995) ** 2
        for idx in order:
            x = cand_x[idx]
            y = cand_y[idx]
            ok = True
            for sx, sy in zip(keep_x, keep_y):
                if (x - sx) * (x - sx) + (y - sy) * (y - sy) < min_d2:
                    ok = False
                    break
            if ok:
                keep_x.append(x)
                keep_y.append(y)
                if len(keep_x) == n_target:
                    break

        if len(keep_x) < n_target:
            return staggered_seed(float(cluster_centers[0] + 0.5 * np.pi), 1.000, 0.31, 0.006)
        return jnp.asarray(keep_x), jnp.asarray(keep_y)

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

    def make_runner(lr0, gamma_min, const_steps, decay_steps, momentum):
        total_steps = const_steps + decay_steps
        mid = compute_mid(lr0, gamma_min, decay_steps)

        @jax.jit
        def run(z0, spacing_scale):
            _, g0 = obj_vg(z0)
            alpha0 = jnp.mean(jnp.abs(g0)) / lr0
            v0 = jnp.zeros_like(z0)

            def body(carry, t):
                z, v, lr, alpha = carry
                look = z + momentum * v
                _, go = obj_vg(look)
                _, gc = con_vg(look, spacing_scale)
                g = go + alpha * gc
                rms = jnp.sqrt(jnp.mean(g * g)) + 1e-12
                v = momentum * v - lr * g / rms
                z = z + v
                z = jnp.concatenate(
                    [
                        jnp.clip(z[:n_target], x_min - 0.08 * span_x, x_max + 0.08 * span_x),
                        jnp.clip(z[n_target:], y_min - 0.08 * span_y, y_max + 0.08 * span_y),
                    ]
                )
                step = t + 1.0
                decaying = step > const_steps
                decay_step = jnp.where(decaying, step - const_steps, 0.0)
                new_lr = jnp.where(decaying, lr / (1.0 + mid * decay_step), lr)
                new_alpha = jnp.where(decaying, alpha0 * lr0 / new_lr, alpha)
                return (z, v, new_lr, new_alpha), None

            (z, _, _, _), _ = jax.lax.scan(
                body, (z0, v0, lr0, alpha0), jnp.arange(total_steps)
            )
            return z

        return run

    main_run = make_runner(95.0, 0.0035, 1400, 3600, 0.38)
    kmeans_run = make_runner(86.0, 0.0045, 700, 2500, 0.32)
    polish_run = make_runner(18.0, 0.10, 80, 520, 0.18)

    def consider(best_x, best_y, best_obj, cand_z):
        x = cand_z[:n_target]
        y = cand_z[n_target:]
        if feasible(x, y):
            obj = float(objective_xy_jit(x, y))
            if obj < best_obj:
                return x, y, obj
        return best_x, best_y, best_obj

    start0 = staggered_seed(base_angle, 1.000, 0.24, 0.000)
    start1 = staggered_seed(base_angle + 0.16, 1.006, 0.58, 0.015)
    start2 = kmeans_mixed_seed()

    best_x, best_y = start0
    best_obj = float(objective_xy_jit(best_x, best_y)) if feasible(best_x, best_y) else 1.0e30
    for init_x, init_y in (start0, start1, start2):
        if feasible(init_x, init_y):
            init_obj = float(objective_xy_jit(init_x, init_y))
            if init_obj < best_obj:
                best_x, best_y, best_obj = init_x, init_y, init_obj

    for init_x, init_y in (start0, start1):
        z = main_run(jnp.concatenate([init_x, init_y]), 1.0)
        best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)

    z = kmeans_run(jnp.concatenate([start2[0], start2[1]]), 1.0)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)

    z = polish_run(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)
    return best_x, best_y
