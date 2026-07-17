"""Affine recovery with active-contact tangent cleanup.

HYPOTHESIS: Attempt 26's affine perturbation reaches a strong basin, but
remaining gains may be blocked by active spacing/boundary constraints. A final
contact-tangent pass can slide constrained turbines without leaving feasibility.

AXIS: incumbent affine recovery followed by local tangent moves on nearest
spacing contacts and boundary-active turbines.

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
    edge_dx = np.asarray(jnp.roll(boundary[:, 0], -1) - boundary[:, 0], dtype=float)
    edge_dy = np.asarray(jnp.roll(boundary[:, 1], -1) - boundary[:, 1], dtype=float)
    edge_len = np.sqrt(edge_dx * edge_dx + edge_dy * edge_dy) + 1e-12
    edge_tx = edge_dx / edge_len
    edge_ty = edge_dy / edge_len

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
    micro_run = make_runner(8.5, 0.18, 24, 260, 0.10)

    def consider(best_x, best_y, best_obj, cand_z):
        x = cand_z[:n_target]
        y = cand_z[n_target:]
        if feasible_loose(x, y):
            obj = float(objective_xy_jit(x, y))
            if obj < best_obj:
                return x, y, obj
        return best_x, best_y, best_obj

    def affine_gradient_refine(x0, y0, obj0):
        best_x = x0
        best_y = y0
        best_obj = obj0
        if not feasible_exact(best_x, best_y):
            return best_x, best_y, best_obj

        theta = np.asarray(wd_rad, dtype=float)
        mass = np.asarray(energy_mass, dtype=float)
        top = np.argsort(-mass)[:3]
        mean = float(mean_wind)
        angles = [mean]
        for idx in top:
            angles.append(float(theta[idx]))

        base_u = np.array([np.cos(mean), np.sin(mean)], dtype=float)
        base_v = np.array([-np.sin(mean), np.cos(mean)], dtype=float)
        center_x = float(jnp.mean(best_x))
        center_y = float(jnp.mean(best_y))

        affine_params = [
            (0.008, 1.000, 1.000, 0.000, 0.000, 0.000),
            (-0.008, 1.000, 1.000, 0.000, 0.000, 0.000),
            (0.000, 0.998, 1.002, 0.000, 0.000, 0.000),
            (0.000, 1.002, 0.998, 0.000, 0.000, 0.000),
            (0.000, 1.000, 1.000, 0.010, 0.000, 0.000),
            (0.000, 1.000, 1.000, -0.010, 0.000, 0.000),
            (0.004, 0.999, 1.001, 0.006, 0.000, 0.000),
            (-0.004, 0.999, 1.001, -0.006, 0.000, 0.000),
            (0.000, 1.000, 1.000, 0.000, 0.018, 0.000),
            (0.000, 1.000, 1.000, 0.000, -0.018, 0.000),
            (0.000, 1.000, 1.000, 0.000, 0.000, 0.018),
            (0.000, 1.000, 1.000, 0.000, 0.000, -0.018),
        ]

        for rot, scale_u, scale_v, shear, shift_u, shift_v in affine_params:
            ca = np.cos(rot)
            sa = np.sin(rot)
            u = ca * base_u + sa * base_v
            v = -sa * base_u + ca * base_v
            rx = np.asarray(best_x, dtype=float) - center_x
            ry = np.asarray(best_y, dtype=float) - center_y
            qu = rx * u[0] + ry * u[1]
            qv = rx * v[0] + ry * v[1]
            qu2 = scale_u * qu
            qv2 = scale_v * qv + shear * qu
            cand_x = jnp.asarray(center_x + qu2 * u[0] + qv2 * v[0] + shift_u * min_spacing * u[0] + shift_v * min_spacing * v[0])
            cand_y = jnp.asarray(center_y + qu2 * u[1] + qv2 * v[1] + shift_u * min_spacing * u[1] + shift_v * min_spacing * v[1])
            if feasible_exact(cand_x, cand_y):
                cand_obj = float(objective_xy_jit(cand_x, cand_y))
                if cand_obj < best_obj - 1.0e-5:
                    best_x = cand_x
                    best_y = cand_y
                    best_obj = cand_obj

            z = micro_run(jnp.concatenate([cand_x, cand_y]), 0.996)
            best_x, best_y, best_obj = consider(best_x, best_y, best_obj, z)

        dirs = []
        for angle in angles:
            dirs.append((np.cos(angle), np.sin(angle)))
            dirs.append((np.cos(angle + 0.5 * np.pi), np.sin(angle + 0.5 * np.pi)))

        unique_dirs = []
        for dx, dy in dirs:
            v = np.array([dx, dy], dtype=float)
            v /= np.linalg.norm(v) + 1e-12
            if all(abs(float(np.dot(v, u))) < 0.965 for u in unique_dirs):
                unique_dirs.append(v)
            if len(unique_dirs) >= 5:
                break

        for step in (0.080, 0.050, 0.030, 0.018):
            prod = np.asarray(turbine_power_jit(best_x, best_y), dtype=float)
            _, grad = obj_vg(jnp.concatenate([best_x, best_y]))
            gx = np.asarray(grad[:n_target], dtype=float)
            gy = np.asarray(grad[n_target:], dtype=float)
            gnorm = np.sqrt(gx * gx + gy * gy)
            order = []
            for idx in np.argsort(prod)[: min(12, n_target)]:
                order.append(int(idx))
            for idx in np.argsort(-gnorm)[: min(8, n_target)]:
                idx = int(idx)
                if idx not in order:
                    order.append(idx)
            stride = float(step * min_spacing)
            improved = False

            for i in order:
                i = int(i)
                local_x = best_x
                local_y = best_y
                local_obj = best_obj
                gvec = np.array([-gx[i], -gy[i]], dtype=float)
                if np.linalg.norm(gvec) > 1.0e-12:
                    gvec = gvec / np.linalg.norm(gvec)
                    trial_dirs = [gvec, np.array([-gvec[1], gvec[0]])]
                else:
                    trial_dirs = []
                trial_dirs.extend(unique_dirs[:3])
                for v in trial_dirs:
                    v = np.asarray(v, dtype=float)
                    v = v / (np.linalg.norm(v) + 1.0e-12)
                    for sign in (-1.0, 1.0):
                        cand_x = best_x.at[i].add(sign * stride * float(v[0]))
                        cand_y = best_y.at[i].add(sign * stride * float(v[1]))
                        if feasible_exact(cand_x, cand_y):
                            cand_obj = float(objective_xy_jit(cand_x, cand_y))
                            if cand_obj < local_obj - 1.0e-5:
                                local_x = cand_x
                                local_y = cand_y
                                local_obj = cand_obj

                if local_obj < best_obj - 1.0e-5:
                    best_x = local_x
                    best_y = local_y
                    best_obj = local_obj
                    improved = True

            if not improved and step < 0.04:
                break
        return best_x, best_y, best_obj

    def add_dir(dirs, vec, max_dot=0.985):
        norm = np.linalg.norm(vec)
        if norm <= 1.0e-12:
            return
        v = vec / norm
        if all(abs(float(np.dot(v, u))) < max_dot for u in dirs):
            dirs.append(v)

    def contact_tangent_refine(x0, y0, obj0):
        best_x = x0
        best_y = y0
        best_obj = obj0
        if not feasible_exact(best_x, best_y):
            return best_x, best_y, best_obj

        theta = np.asarray(wd_rad, dtype=float)
        mass = np.asarray(energy_mass, dtype=float)
        wind_dirs = []
        add_dir(wind_dirs, np.array([np.cos(float(mean_wind)), np.sin(float(mean_wind))]))
        add_dir(wind_dirs, np.array([-np.sin(float(mean_wind)), np.cos(float(mean_wind))]))
        for idx in np.argsort(-mass)[:2]:
            angle = float(theta[idx])
            add_dir(wind_dirs, np.array([np.cos(angle), np.sin(angle)]))
            add_dir(wind_dirs, np.array([-np.sin(angle), np.cos(angle)]))

        for step in (0.040, 0.025, 0.015):
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
            for idx in np.argsort(prod)[: min(8, n_target)]:
                order.append(int(idx))
            for idx in np.argsort(-gnorm)[: min(6, n_target)]:
                idx = int(idx)
                if idx not in order:
                    order.append(idx)

            improved = False
            for i in order:
                dirs = []
                if nearest_bnd[i] < 0.28 * float(min_spacing):
                    e = int(nearest_edge[i])
                    add_dir(dirs, np.array([edge_tx[e], edge_ty[e]], dtype=float))
                for j in np.argsort(dmat[i])[:4]:
                    j = int(j)
                    if dmat[i, j] > 1.12 * float(min_spacing):
                        continue
                    sep = np.array([bx[i] - bx[j], by[i] - by[j]], dtype=float)
                    add_dir(dirs, np.array([-sep[1], sep[0]], dtype=float))
                add_dir(dirs, np.array([-gx[i], -gy[i]], dtype=float), max_dot=0.995)
                dirs.extend(wind_dirs[:2])

                local_x = best_x
                local_y = best_y
                local_obj = best_obj
                for v in dirs[:5]:
                    v = np.asarray(v, dtype=float)
                    v = v / (np.linalg.norm(v) + 1.0e-12)
                    for sign in (-1.0, 1.0):
                        dx = sign * stride * float(v[0])
                        dy = sign * stride * float(v[1])
                        cand_x = best_x.at[i].add(dx)
                        cand_y = best_y.at[i].add(dy)
                        if feasible_exact(cand_x, cand_y):
                            cand_obj = float(objective_xy_jit(cand_x, cand_y))
                            if cand_obj < local_obj - 1.0e-5:
                                local_x = cand_x
                                local_y = cand_y
                                local_obj = cand_obj

                        for j in np.argsort(dmat[i])[:2]:
                            j = int(j)
                            if dmat[i, j] > 1.10 * float(min_spacing):
                                continue
                            cand_x = best_x.at[i].add(dx).at[j].add(-0.55 * dx)
                            cand_y = best_y.at[i].add(dy).at[j].add(-0.55 * dy)
                            if feasible_exact(cand_x, cand_y):
                                cand_obj = float(objective_xy_jit(cand_x, cand_y))
                                if cand_obj < local_obj - 1.0e-5:
                                    local_x = cand_x
                                    local_y = cand_y
                                    local_obj = cand_obj

                if local_obj < best_obj - 1.0e-5:
                    best_x = local_x
                    best_y = local_y
                    best_obj = local_obj
                    improved = True

            if not improved and step < 0.03:
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
    best_x, best_y, best_obj = affine_gradient_refine(best_x, best_y, best_obj)
    best_x, best_y, best_obj = contact_tangent_refine(best_x, best_y, best_obj)
    return best_x, best_y
