"""Simulated annealing over lattice basins and projected layout moves.

HYPOTHESIS: The previous best layout was likely trapped in a polished
wind-grid basin. A true simulated_annealing phase can accept temporary AEP
losses while moving individual turbines and wind-frame bands, then keep only
strictly feasible layouts after projection.

AXIS: simulated_annealing using scipy dual_annealing on compact lattice
parameters, followed by temperature-controlled projected coordinate and band
moves.

LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import dual_annealing


def _edge_clearance(x, y, boundary):
    n_verts = boundary.shape[0]

    def edge(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex = x2 - x1
        ey = y2 - y1
        el = jnp.sqrt(ex * ex + ey * ey) + 1e-12
        return (x - x1) * (-ey / el) + (y - y1) * (ex / el)

    return jax.vmap(edge)(jnp.arange(n_verts))


def _min_distance(x, y):
    n = x.shape[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n) * 1e18)
    return jnp.min(dist)


def _feasible(x, y, boundary, min_spacing):
    return (
        (jnp.min(_edge_clearance(x, y, boundary)) >= -1e-5)
        & (_min_distance(x, y) >= min_spacing * 0.9995)
    )


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    @jax.jit
    def neg_aep_xy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    score_batch = jax.jit(jax.vmap(neg_aep_xy))
    feasible_batch = jax.jit(
        jax.vmap(lambda px, py: _feasible(px, py, boundary, min_spacing))
    )

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    center = jnp.mean(boundary, axis=0)
    span_x = x_max - x_min
    span_y = y_max - y_min
    diag = jnp.sqrt(span_x * span_x + span_y * span_y)

    wd_rad = jnp.deg2rad(wd)
    energy = weights * ws**3
    dominant = jnp.arctan2(
        jnp.sum(jnp.sin(wd_rad) * energy),
        jnp.sum(jnp.cos(wd_rad) * energy),
    )
    u_wind = jnp.cos(dominant)
    v_wind = jnp.sin(dominant)
    u_cross = -v_wind
    v_cross = u_wind

    low = jnp.array([1.02, 1.03, -jnp.pi / 3.0, 0.02, 0.02, -0.24, 0.86])
    high = jnp.array([4.15, 4.10, jnp.pi / 3.0, 0.98, 0.98, 0.24, 1.30])

    def scale(raw):
        return low + jnp.asarray(raw, dtype=jnp.float64) * (high - low)

    def lattice(raw):
        sx, sy, theta_off, ox_raw, oy_raw, shear, aspect = scale(raw)
        theta = dominant + theta_off
        row_step = sy * aspect * min_spacing * 0.8660254037844386
        n_side = int(np.ceil(float(diag / (min_spacing * 0.70)))) + 17
        ii, jj = jnp.meshgrid(
            jnp.arange(n_side) - n_side // 2,
            jnp.arange(n_side) - n_side // 2,
        )
        ix = ii.ravel()
        iy = jj.ravel()
        hx = (ix + 0.5 * (iy % 2)) * sx * min_spacing
        hy = iy * row_step
        hx = hx + shear * hy
        ox = x_min + ox_raw * span_x
        oy = y_min + oy_raw * span_y
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        clearance = jnp.min(_edge_clearance(rx, ry, boundary), axis=0)
        edge_pull = clearance / diag
        spread = ((rx - center[0]) ** 2 + (ry - center[1]) ** 2) / (diag * diag)
        score = jnp.where(
            clearance > min_spacing * 0.018, edge_pull + 0.024 * spread, -1e12
        )
        idx = jnp.argsort(score)[-n_target:]
        return rx[idx], ry[idx]

    def project(x, y, n_steps=12, boundary_margin=0.0, spacing_margin=1.0004):
        n_edges = boundary.shape[0]
        for _ in range(n_steps):
            for e in range(n_edges):
                x1, y1 = boundary[e]
                x2, y2 = boundary[(e + 1) % n_edges]
                ex = x2 - x1
                ey = y2 - y1
                el = jnp.sqrt(ex * ex + ey * ey) + 1e-12
                nx = -ey / el
                ny = ex / el
                clearance = (x - x1) * nx + (y - y1) * ny
                push = jnp.maximum(0.0, boundary_margin - clearance)
                x = x + push * nx
                y = y + push * ny

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n_target) * 1e18)
            overlap = jnp.maximum(0.0, min_spacing * spacing_margin - dist)
            x = x + jnp.sum(overlap * dx / dist, axis=1) * 0.52
            y = y + jnp.sum(overlap * dy / dist, axis=1) * 0.52
        return x, y

    project_batch = jax.jit(
        jax.vmap(
            lambda px, py: project(
                px, py, n_steps=12, boundary_margin=0.02, spacing_margin=1.0006
            )
        )
    )

    def lattice_score(raw):
        raw = jnp.clip(jnp.asarray(raw, dtype=jnp.float64), 0.0, 1.0)
        x, y = lattice(raw)
        x, y = project(x, y, n_steps=10, boundary_margin=0.02, spacing_margin=1.0005)
        val = neg_aep_xy(x, y)
        return val if _feasible(x, y, boundary, min_spacing) else val + 1e6

    seed_raw = np.array(
        [
            [0.02, 0.03, 0.50, 0.50, 0.50, 0.50, 0.20],
            [0.04, 0.06, 0.32, 0.43, 0.46, 0.43, 0.24],
            [0.05, 0.02, 0.69, 0.56, 0.52, 0.57, 0.23],
            [0.14, 0.10, 0.18, 0.45, 0.58, 0.50, 0.30],
            [0.10, 0.16, 0.84, 0.53, 0.42, 0.58, 0.36],
            [0.19, 0.05, 0.61, 0.48, 0.50, 0.63, 0.18],
            [0.06, 0.20, 0.39, 0.58, 0.48, 0.37, 0.35],
        ],
        dtype=np.float64,
    )
    scored_raw = [(float(lattice_score(row)), row.copy()) for row in seed_raw]

    bounds = [(0.0, 1.0)] * 7
    n_outer = 2 if n_target <= 55 else 1
    for i, (_, x0) in enumerate(sorted(scored_raw, key=lambda item: item[0])[:n_outer]):
        result = dual_annealing(
            lambda z: float(lattice_score(z)),
            bounds=bounds,
            x0=x0,
            maxiter=20 if n_target <= 55 else 8,
            maxfun=390 if n_target <= 55 else 150,
            initial_temp=2600.0,
            restart_temp_ratio=2e-4,
            visit=2.45,
            accept=-6.0,
            no_local_search=True,
            seed=811 + i,
        )
        scored_raw.append((float(result.fun), np.clip(result.x, 0.0, 1.0)))

    start_layouts = []
    for _, raw in sorted(scored_raw, key=lambda item: item[0])[:4]:
        sx, sy = lattice(raw)
        start_layouts.append(
            project(sx, sy, n_steps=18, boundary_margin=0.02, spacing_margin=1.0008)
        )

    def eval_layouts(xs, ys):
        scores = score_batch(xs, ys)
        feas = feasible_batch(xs, ys)
        return jnp.where(feas, scores, scores + 1e9)

    def best_of(xs, ys):
        vals = eval_layouts(xs, ys)
        idx = jnp.argmin(vals)
        return xs[idx], ys[idx], vals[idx]

    def anneal_layout(seed_x, seed_y, rng_seed):
        rng = np.random.default_rng(rng_seed)
        cur_x, cur_y = project(
            seed_x, seed_y, n_steps=20, boundary_margin=0.02, spacing_margin=1.0008
        )
        cur_val = float(
            jnp.where(
                _feasible(cur_x, cur_y, boundary, min_spacing),
                neg_aep_xy(cur_x, cur_y),
                1e9,
            )
        )
        best_x = cur_x
        best_y = cur_y
        best_val = cur_val

        n_steps = 58 if n_target <= 55 else 24
        batch_size = 32 if n_target <= 55 else 14
        base_scale = float(min_spacing)
        for step in range(n_steps):
            frac = step / max(1, n_steps - 1)
            temp = 3.2 * (1.0 - frac) ** 1.7 + 0.08
            move_scale = base_scale * (0.38 * (1.0 - frac) + 0.045)

            raw_x = []
            raw_y = []
            u_coord = np.asarray(cur_x * u_wind + cur_y * v_wind)
            v_coord = np.asarray(cur_x * u_cross + cur_y * v_cross)
            u_edges = np.quantile(u_coord, [0.0, 0.25, 0.50, 0.75, 1.0])
            v_edges = np.quantile(v_coord, [0.0, 0.25, 0.50, 0.75, 1.0])

            for _ in range(batch_size):
                kind = rng.integers(0, 5)
                x = cur_x
                y = cur_y
                if kind == 0:
                    idx = int(rng.integers(0, n_target))
                    du, dv = rng.normal(0.0, [move_scale, move_scale * 0.65])
                    mask = (jnp.arange(n_target) == idx).astype(cur_x.dtype)
                    x = x + mask * (du * u_wind + dv * u_cross)
                    y = y + mask * (du * v_wind + dv * v_cross)
                elif kind == 1:
                    axis = int(rng.integers(0, 2))
                    band = int(rng.integers(0, 4))
                    if axis == 0:
                        lo, hi = u_edges[band], u_edges[band + 1]
                        mask_np = (u_coord >= lo) & (u_coord <= hi)
                    else:
                        lo, hi = v_edges[band], v_edges[band + 1]
                        mask_np = (v_coord >= lo) & (v_coord <= hi)
                    mask = jnp.asarray(mask_np.astype(np.float64))
                    du, dv = rng.normal(0.0, [move_scale * 0.70, move_scale * 0.48])
                    x = x + mask * (du * u_wind + dv * u_cross)
                    y = y + mask * (du * v_wind + dv * v_cross)
                elif kind == 2:
                    tx, ty = rng.normal(0.0, move_scale * 0.28, size=2)
                    x = x + tx
                    y = y + ty
                elif kind == 3:
                    cx = jnp.mean(x)
                    cy = jnp.mean(y)
                    rot = rng.normal(0.0, 0.008 * (1.0 - frac) + 0.0015)
                    px = x - cx
                    py = y - cy
                    x = px * jnp.cos(rot) - py * jnp.sin(rot) + cx
                    y = px * jnp.sin(rot) + py * jnp.cos(rot) + cy
                else:
                    cx = jnp.mean(x)
                    cy = jnp.mean(y)
                    scale = 1.0 + rng.normal(0.0, 0.008 * (1.0 - frac) + 0.0015)
                    shear = rng.normal(0.0, 0.010 * (1.0 - frac) + 0.002)
                    px = x - cx
                    py = y - cy
                    x = px * scale + shear * py + cx
                    y = py / scale + cy
                raw_x.append(x)
                raw_y.append(y)

            cand_x = jnp.concatenate([cur_x[None, :], jnp.stack(raw_x)], axis=0)
            cand_y = jnp.concatenate([cur_y[None, :], jnp.stack(raw_y)], axis=0)
            cand_x, cand_y = project_batch(cand_x, cand_y)
            vals = np.asarray(eval_layouts(cand_x, cand_y))
            delta = vals - cur_val
            accept = (delta < 0.0) | (
                rng.random(vals.shape[0]) < np.exp(-np.maximum(delta, 0.0) / temp)
            )
            accept[0] = True
            accepted_idx = np.flatnonzero(accept)
            chosen = accepted_idx[np.argmin(vals[accepted_idx])]
            cur_x = cand_x[chosen]
            cur_y = cand_y[chosen]
            cur_val = float(vals[chosen])
            if cur_val < best_val:
                best_x, best_y, best_val = cur_x, cur_y, cur_val

        return best_x, best_y, best_val

    def greedy_finish(seed_x, seed_y):
        cur_x, cur_y = seed_x, seed_y
        cur_val = float(eval_layouts(cur_x[None, :], cur_y[None, :])[0])
        dirs = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [0.65, 0.45],
                [-0.65, -0.45],
            ],
            dtype=np.float64,
        )
        for step_len in [95.0, 60.0, 38.0, 24.0, 15.0]:
            raw_x = [cur_x]
            raw_y = [cur_y]
            for i in range(n_target):
                mask = (jnp.arange(n_target) == i).astype(cur_x.dtype)
                for du, dv in dirs:
                    raw_x.append(cur_x + mask * step_len * (du * u_wind + dv * u_cross))
                    raw_y.append(cur_y + mask * step_len * (du * v_wind + dv * v_cross))

            u_coord = cur_x * u_wind + cur_y * v_wind
            v_coord = cur_x * u_cross + cur_y * v_cross
            for axis_coord, axis_id in ((u_coord, 0), (v_coord, 1)):
                edges = jnp.quantile(axis_coord, jnp.array([0.0, 0.33, 0.66, 1.0]))
                for band in range(3):
                    mask = ((axis_coord >= edges[band]) & (axis_coord <= edges[band + 1])).astype(
                        cur_x.dtype
                    )
                    for sign in (-1.0, 1.0):
                        du = sign * step_len * (0.55 if axis_id == 0 else 0.20)
                        dv = sign * step_len * (0.25 if axis_id == 0 else 0.55)
                        raw_x.append(cur_x + mask * (du * u_wind + dv * u_cross))
                        raw_y.append(cur_y + mask * (du * v_wind + dv * v_cross))

            cand_x = jnp.stack(raw_x)
            cand_y = jnp.stack(raw_y)
            cand_x, cand_y = project_batch(cand_x, cand_y)
            vals = eval_layouts(cand_x, cand_y)
            idx = jnp.argmin(vals)
            if float(vals[idx]) < cur_val:
                cur_x = cand_x[idx]
                cur_y = cand_y[idx]
                cur_val = float(vals[idx])
        return cur_x, cur_y, cur_val

    annealed = []
    for i, (sx, sy) in enumerate(start_layouts[:3]):
        annealed.append(anneal_layout(sx, sy, 1401 + i))

    cand_x = jnp.stack([item[0] for item in annealed] + [item[0] for item in start_layouts])
    cand_y = jnp.stack([item[1] for item in annealed] + [item[1] for item in start_layouts])
    best_x, best_y, _ = best_of(cand_x, cand_y)

    finish_x, finish_y, _ = greedy_finish(best_x, best_y)
    cand_x = jnp.stack([best_x, finish_x])
    cand_y = jnp.stack([best_y, finish_y])
    best_x, best_y, _ = best_of(cand_x, cand_y)
    return jnp.asarray(best_x), jnp.asarray(best_y)
