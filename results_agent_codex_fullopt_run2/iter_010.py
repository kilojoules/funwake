"""SciPy L-BFGS-B penalty polish from lattice starts.

HYPOTHESIS: L-BFGS-B can cheaply polish the best lattice basins because the
wake objective is smooth enough locally, and a staged soft penalty can let it
skim active spacing/boundary constraints without the overhead of full SLSQP.
AXIS: scipy_lbfgs penalty method with feasible-candidate tracking.
LESSON: Pending score.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    n_verts = boundary.shape[0]
    start_time = time.time()

    def objective_xy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, : len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

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
        inside = edge_clearance(cand_x, cand_y) > margin * 0.08
        return cand_x[inside], cand_y[inside]

    def farthest_init(cand_x, cand_y, mode):
        if len(cand_x) < n_target:
            key = jax.random.PRNGKey(410 + mode)
            kx, ky = jax.random.split(key)
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

    def packed(vec):
        x = vec[:n_target]
        y = vec[n_target:]
        return x, y

    def merit_with_aux(vec, penalty_scale):
        x, y = packed(vec)
        neg_aep = objective_xy(x, y)
        bpen = boundary_penalty(x, y, boundary)
        spen = spacing_penalty(x, y, min_spacing)
        md = min_distance(x, y)
        total = neg_aep + penalty_scale * (4.0 * bpen + spen)
        return total, (-neg_aep, bpen, spen, md)

    merit_vg = jax.jit(jax.value_and_grad(merit_with_aux, argnums=0, has_aux=True))

    coarse_x, coarse_y = candidate_cloud(
        min_spacing, min_spacing * 0.8660254, min_spacing * 0.45, 0.5
    )
    dense_x, dense_y = candidate_cloud(
        min_spacing * 0.52, min_spacing * 0.45, min_spacing * 0.14, 0.5
    )
    starts = (
        ordered_init(coarse_x, coarse_y),
        farthest_init(coarse_x, coarse_y, 1),
        farthest_init(dense_x, dense_y, 0),
    )

    best_x, best_y = starts[0]
    best_aep = jnp.where(feasible(best_x, best_y), -objective_xy(best_x, best_y), -jnp.inf)

    bounds = [(float(x_min), float(x_max))] * n_target + [
        (float(y_min), float(y_max))
    ] * n_target

    for init_x, init_y in starts:
        if time.time() - start_time > 43.0:
            break

        init_aep = -objective_xy(init_x, init_y)
        if feasible(init_x, init_y) & (init_aep > best_aep):
            best_aep = init_aep
            best_x = init_x
            best_y = init_y

        z0 = np.concatenate([np.asarray(init_x), np.asarray(init_y)])
        local_best = {"aep": float(best_aep), "z": z0.copy()}

        def scipy_fun(z, penalty_scale):
            (value, aux), grad = merit_vg(jnp.asarray(z), penalty_scale)
            aep, bpen, spen, md = aux
            aep_f = float(aep)
            ok = (
                np.isfinite(aep_f)
                and float(bpen) < 1e-3
                and float(spen) < 1e-3
                and float(md) >= float(min_spacing) * 0.99
            )
            if ok and aep_f > local_best["aep"]:
                local_best["aep"] = aep_f
                local_best["z"] = np.asarray(z).copy()

            value_f = float(value)
            grad_np = np.asarray(grad, dtype=float)
            if not np.isfinite(value_f) or not np.all(np.isfinite(grad_np)):
                return 1e30, np.zeros_like(z)
            return value_f, grad_np

        z = z0
        for penalty_scale, maxiter in ((0.02, 35), (0.18, 45)):
            if time.time() - start_time > 48.0:
                break
            res = minimize(
                scipy_fun,
                z,
                args=(penalty_scale,),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": maxiter,
                    "maxls": 12,
                    "ftol": 1e-7,
                    "gtol": 1e-5,
                    "disp": False,
                },
            )
            z = res.x

        cand = jnp.asarray(local_best["z"])
        cand_x, cand_y = packed(cand)
        cand_aep = -objective_xy(cand_x, cand_y)
        if feasible(cand_x, cand_y) & (cand_aep > best_aep):
            best_aep = cand_aep
            best_x = cand_x
            best_y = cand_y

    return best_x, best_y
