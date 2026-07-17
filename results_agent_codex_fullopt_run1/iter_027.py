"""Active-contact tangent polish after incumbent recovery.

HYPOTHESIS: The current best basin is spacing/boundary active. Instead of
global affine perturbations, slide selected turbines along local active
constraint tangents so feasibility is preserved while wake relationships change.

AXIS: incumbent Nesterov penalty recovery plus deterministic contact-tangent
single and paired moves around nearest spacing contacts and boundary edges.

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

    def turbine_power_xy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return jnp.sum(p * weights[:, None], axis=0)

    turbine_power_jit = jax.jit(turbine_power_xy)

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

    edge_dx = np.asarray(jnp.roll(boundary[:, 0], -1) - boundary[:, 0], dtype=float)
    edge_dy = np.asarray(jnp.roll(boundary[:, 1], -1) - boundary[:, 1], dtype=float)
    edge_len = np.sqrt(edge_dx * edge_dx + edge_dy * edge_dy) + 1e-12
    edge_tx = edge_dx / edge_len
    edge_ty = edge_dy / edge_len

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
    energy_mass = weights * (ws**3)
    mean_wind = jnp.arctan2(
        jnp.sum(energy_mass * jnp.sin(wd_rad)),
        jnp.sum(energy_mass * jnp.cos(wd_rad)),
    )
    base_angle = mean_wind + 0.5 * jnp.pi

    def feasible_loose(x, y):
        bnd = boundary_penalty(x, y, boundary)
        spc = spacing_penalty(x, y, min_spacing * 0.99)
        return float(bnd) < 1e-3 and float(spc) < 1e-3

    def feasible_exact(x, y):
        bnd = boundary_penalty(x, y, boundary)
        spc = spacing_penalty(x, y, min_spacing)
        return float(bnd) < 1e-5 and float(spc) < 1e-5

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
    polish_run = make_runner(18.0, 0.10, 80, 520, 0.18)
    micro_run = make_runner(7.5, 0.20, 18, 180, 0.08)

    def consider(best_x, best_y, best_obj, cand_z):
        x = cand_z[:n_target]
        y = cand_z[n_target:]
        if feasible_loose(x, y):
            obj = float(objective_xy_jit(x, y))
            if obj < best_obj:
                return x, y, obj
        return best_x, best_y, best_obj

    def try_candidate(best_x, best_y, best_obj, cand_x, cand_y):
        if feasible_exact(cand_x, cand_y):
            cand_obj = float(objective_xy_jit(cand_x, cand_y))
            if cand_obj < best_obj - 1.0e-5:
                return cand_x, cand_y, cand_obj, True
        return best_x, best_y, best_obj, False

    def add_dir(dirs, vec):
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            return
        v = vec / norm
        if all(abs(float(np.dot(v, u))) < 0.985 for u in dirs):
            dirs.append(v)

    def contact_tangent_refine(x0, y0, obj0):
        best_x = x0
        best_y = y0
        best_obj = obj0
        if not feasible_exact(best_x, best_y):
            return best_x, best_y, best_obj

        theta = np.asarray(wd_rad, dtype=float)
        mass = np.asarray(energy_mass, dtype=float)
        angles = [float(mean_wind)]
        for idx in np.argsort(-mass)[:3]:
            angles.append(float(theta[idx]))

        wind_dirs = []
        for angle in angles:
            add_dir(wind_dirs, np.array([np.cos(angle), np.sin(angle)], dtype=float))
            add_dir(wind_dirs, np.array([np.cos(angle + 0.5 * np.pi), np.sin(angle + 0.5 * np.pi)], dtype=float))
        wind_dirs = wind_dirs[:4]

        for step in (0.115, 0.070, 0.042, 0.025, 0.015):
            stride = float(step * min_spacing)
            prod = np.asarray(turbine_power_jit(best_x, best_y), dtype=float)
            _, grad = obj_vg(jnp.concatenate([best_x, best_y]))
            gx = np.asarray(grad[:n_target], dtype=float)
            gy = np.asarray(grad[n_target:], dtype=float)
            gnorm = np.sqrt(gx * gx + gy * gy)
            bx = np.asarray(best_x, dtype=float)
            by = np.asarray(best_y, dtype=float)
            dmat = np.sqrt(
                (bx[:, None] - bx[None, :]) ** 2
                + (by[:, None] - by[None, :]) ** 2
                + np.eye(n_target) * 1.0e12
            )
            edge_d = np.asarray(edge_distances(best_x, best_y), dtype=float)
            nearest_edge = np.argmin(edge_d, axis=0)
            nearest_bnd = np.min(edge_d, axis=0)

            order = []
            for idx in np.argsort(prod)[: min(9, n_target)]:
                order.append(int(idx))
            for idx in np.argsort(-gnorm)[: min(8, n_target)]:
                idx = int(idx)
                if idx not in order:
                    order.append(idx)

            improved = False
            for i in order:
                dirs = []
                if nearest_bnd[i] < 0.32 * float(min_spacing):
                    e = int(nearest_edge[i])
                    add_dir(dirs, np.array([edge_tx[e], edge_ty[e]], dtype=float))

                for j in np.argsort(dmat[i])[:5]:
                    j = int(j)
                    if dmat[i, j] > 1.20 * float(min_spacing):
                        continue
                    nvec = np.array([bx[i] - bx[j], by[i] - by[j]], dtype=float)
                    tangent = np.array([-nvec[1], nvec[0]], dtype=float)
                    add_dir(dirs, tangent)

                gvec = np.array([-gx[i], -gy[i]], dtype=float)
                if len(dirs) == 0:
                    add_dir(dirs, gvec)
                dirs.extend(wind_dirs[:2])

                local_x = best_x
                local_y = best_y
                local_obj = best_obj
                for v in dirs[:6]:
                    for sign in (-1.0, 1.0):
                        dx = sign * stride * float(v[0])
                        dy = sign * stride * float(v[1])
                        cand_x = best_x.at[i].add(dx)
                        cand_y = best_y.at[i].add(dy)
                        local_x, local_y, local_obj, ok = try_candidate(
                            local_x, local_y, local_obj, cand_x, cand_y
                        )
                        if ok:
                            continue

                        for j in np.argsort(dmat[i])[:3]:
                            j = int(j)
                            if dmat[i, j] > 1.16 * float(min_spacing):
                                continue
                            for share in (0.45, 0.85):
                                cand_x = best_x.at[i].add(dx).at[j].add(-share * dx)
                                cand_y = best_y.at[i].add(dy).at[j].add(-share * dy)
                                local_x, local_y, local_obj, _ = try_candidate(
                                    local_x, local_y, local_obj, cand_x, cand_y
                                )

                if local_obj < best_obj - 1.0e-5:
                    best_x = local_x
                    best_y = local_y
                    best_obj = local_obj
                    improved = True

            if improved:
                z = micro_run(jnp.concatenate([best_x, best_y]), 0.996)
                best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)
            elif step < 0.03:
                break

        return best_x, best_y, best_obj

    start0 = staggered_seed(base_angle, 1.000, 0.24, 0.000)
    start1 = staggered_seed(base_angle + 0.16, 1.006, 0.58, 0.015)

    best_x, best_y = start0
    best_obj = float(objective_xy_jit(best_x, best_y)) if feasible_loose(best_x, best_y) else 1.0e30
    for init_x, init_y in (start0, start1):
        if feasible_loose(init_x, init_y):
            init_obj = float(objective_xy_jit(init_x, init_y))
            if init_obj < best_obj:
                best_x, best_y, best_obj = init_x, init_y, init_obj
        z = main_run(jnp.concatenate([init_x, init_y]), 1.0)
        best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)

    z = polish_run(jnp.concatenate([best_x, best_y]), 0.992)
    best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)
    best_x, best_y, best_obj = contact_tangent_refine(best_x, best_y, best_obj)
    return best_x, best_y
