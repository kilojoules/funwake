"""Basin hopping over constrained local SLSQP polishes from a strong basin.

HYPOTHESIS: A good rotated-lattice Adam solution is close to the incumbent, but
small stochastic basin hops followed by exact SLSQP constraints can cross active
spacing/boundary faces and find a better nearby local minimum than restarts.
AXIS: scipy_basin_hopping with constrained SLSQP local minimizers after one
custom_adam warm start.
LESSON: Pending score.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import basinhopping

from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    start_time = time.time()
    total_steps = 8000 if n_target <= 60 else 4800

    bnd = np.asarray(boundary, dtype=float)
    signed_area = 0.5 * np.sum(
        bnd[:, 0] * np.roll(bnd[:, 1], -1) - np.roll(bnd[:, 0], -1) * bnd[:, 1]
    )
    if signed_area < 0.0:
        bnd = bnd[::-1].copy()
        boundary = boundary[::-1]

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, : len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def constraint_value(x, y):
        return boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)

    objective_vg = jax.value_and_grad(objective, argnums=(0, 1))
    constraint_vg = jax.value_and_grad(constraint_value, argnums=(0, 1))

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    n_verts = boundary.shape[0]

    wd_rad = jnp.deg2rad(wd)
    vec_x = jnp.sum(weights * jnp.sin(wd_rad))
    vec_y = jnp.sum(weights * jnp.cos(wd_rad))
    angle = jnp.arctan2(vec_x, vec_y) + jnp.pi / 2.0
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    cx = jnp.mean(boundary[:, 0])
    cy = jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)
    grid_spacing = min_spacing * 1.02
    nx = int(jnp.maximum(2, jnp.floor((rx_max - rx_min) / grid_spacing)))
    ny = int(jnp.maximum(2, jnp.floor((ry_max - ry_min) / grid_spacing)))
    gx, gy = jnp.meshgrid(
        rx_min + min_spacing * 0.55 + jnp.arange(nx) * grid_spacing,
        ry_min + min_spacing * 0.55 + jnp.arange(ny) * grid_spacing,
    )
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
    cand_x = orig_pts[:, 0]
    cand_y = orig_pts[:, 1]

    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)

    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0.0
    ix = cand_x[inside]
    iy = cand_y[inside]

    def fallback_start(seed):
        key = jax.random.PRNGKey(seed)
        key, kx, ky = jax.random.split(key, 3)
        return (
            jax.random.uniform(kx, (n_target,), minval=float(x_min), maxval=float(x_max)),
            jax.random.uniform(ky, (n_target,), minval=float(y_min), maxval=float(y_max)),
        )

    def sampled_start(seed):
        if len(ix) >= n_target:
            key = jax.random.PRNGKey(seed)
            indices = jax.random.choice(key, len(ix), (n_target,), replace=False)
            return ix[indices], iy[indices]
        return fallback_start(seed)

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

    x0, y0 = sampled_start(0)
    _, (gox0, goy0) = objective_vg(x0, y0)
    lr0 = 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox0, goy0]))) / lr0

    def schedule(step):
        t = step / total_steps
        k = 4.788904698072376
        log_m = 2.5444236095801482
        warm = 0.031100666821230236
        amp1 = 0.33306582948262714
        amp2 = 0.3144610739896076
        c1 = 0.7118630839032932
        c2 = 0.4830074661356639
        w1 = 0.0357338045274541
        w2 = 0.13157290561309115
        alpha_c = 4.259958160161949
        alpha_d = 12.487427931615652
        b1 = 0.11602857790365473
        b2 = 0.8665970021316327

        lr_init = k * lr0
        lr_min = lr_init / (10.0**log_m)
        warm_lr = lr_init * t / jnp.maximum(warm, 1e-6)
        cosine_t = (t - warm) / jnp.maximum(1.0 - warm, 1e-6)
        cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (
            1.0 + jnp.cos(jnp.pi * cosine_t)
        )
        lr_base = jnp.where(t < warm, warm_lr, cosine_lr)
        bump1 = amp1 * lr_init * jnp.exp(-0.5 * ((t - c1) / w1) ** 2)
        bump2 = amp2 * lr_init * jnp.exp(-0.5 * ((t - c2) / w2) ** 2)
        lr = jnp.maximum(lr_base + bump1 + bump2, 1e-10)

        alpha_base = alpha_c * alpha0 * lr_init / lr
        late = jnp.maximum(t - 0.5, 0.0) / 0.5
        alpha = alpha_base + alpha_d * alpha0 * late**2
        return lr, alpha, b1, b2

    @jax.jit
    def run_loop(init_x, init_y):
        mx0 = jnp.zeros_like(init_x)
        my0 = jnp.zeros_like(init_y)
        vx0 = jnp.zeros_like(init_x)
        vy0 = jnp.zeros_like(init_y)

        def one_step(i, carry):
            x_cur, y_cur, mx_cur, my_cur, vx_cur, vy_cur = carry
            lr, alpha, b1, b2 = schedule(i.astype(float))
            _, (gox, goy) = objective_vg(x_cur, y_cur)
            _, (gcx, gcy) = constraint_vg(x_cur, y_cur)
            gx_all = gox + alpha * gcx
            gy_all = goy + alpha * gcy
            it = (i + 1).astype(float)

            mx_new = b1 * mx_cur + (1.0 - b1) * gx_all
            my_new = b1 * my_cur + (1.0 - b1) * gy_all
            vx_new = b2 * vx_cur + (1.0 - b2) * gx_all**2
            vy_new = b2 * vy_cur + (1.0 - b2) * gy_all**2

            mx_hat = mx_new / (1.0 - b1**it)
            my_hat = my_new / (1.0 - b1**it)
            vx_hat = vx_new / (1.0 - b2**it)
            vy_hat = vy_new / (1.0 - b2**it)
            x_next = x_cur - lr * mx_hat / (jnp.sqrt(vx_hat) + 1e-12)
            y_next = y_cur - lr * my_hat / (jnp.sqrt(vy_hat) + 1e-12)
            return x_next, y_next, mx_new, my_new, vx_new, vy_new

        final = jax.lax.fori_loop(
            0, total_steps, one_step, (init_x, init_y, mx0, my0, vx0, vy0)
        )
        return final[0], final[1]

    best_x, best_y = x0, y0
    best_aep = jnp.where(feasible(best_x, best_y), -objective(best_x, best_y), -jnp.inf)

    ox, oy = run_loop(x0, y0)
    opt_aep = -objective(ox, oy)
    if feasible(ox, oy) & (opt_aep > best_aep):
        best_aep = opt_aep
        best_x = ox
        best_y = oy

    if n_target > 60 or time.time() - start_time > 95.0:
        return best_x, best_y

    def objective_vec(z):
        x = z[:n_target]
        y = z[n_target:]
        return objective(x, y) / 1000.0

    value_grad = jax.jit(jax.value_and_grad(objective_vec))

    pair_i, pair_j = np.triu_indices(n_target, k=1)
    spacing_sq = float(min_spacing) ** 2
    bnd_np = np.asarray(bnd, dtype=float)
    n_verts_np = bnd_np.shape[0]

    rows = []
    lows = []
    for i in range(n_verts_np):
        x1, y1 = bnd_np[i]
        x2, y2 = bnd_np[(i + 1) % n_verts_np]
        ex = x2 - x1
        ey = y2 - y1
        el = np.hypot(ex, ey) + 1e-12
        nx_e = -ey / el
        ny_e = ex / el
        for t_idx in range(n_target):
            row = np.zeros(2 * n_target)
            row[t_idx] = nx_e / float(min_spacing)
            row[n_target + t_idx] = ny_e / float(min_spacing)
            rows.append(row)
            lows.append((nx_e * x1 + ny_e * y1) / float(min_spacing))
    boundary_constraint = LinearConstraint(
        np.vstack(rows), np.asarray(lows), np.full(len(lows), np.inf)
    )

    def spacing_fun(z):
        x = z[:n_target]
        y = z[n_target:]
        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        return (dx * dx + dy * dy) / spacing_sq - 1.0

    def spacing_jac(z):
        x = z[:n_target]
        y = z[n_target:]
        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        jac = np.zeros((len(pair_i), 2 * n_target))
        vals_x = 2.0 * dx / spacing_sq
        vals_y = 2.0 * dy / spacing_sq
        rows_idx = np.arange(len(pair_i))
        jac[rows_idx, pair_i] = vals_x
        jac[rows_idx, pair_j] = -vals_x
        jac[rows_idx, n_target + pair_i] = vals_y
        jac[rows_idx, n_target + pair_j] = -vals_y
        return jac

    spacing_constraint = NonlinearConstraint(
        spacing_fun, np.zeros(len(pair_i)), np.full(len(pair_i), np.inf), jac=spacing_jac
    )
    bounds = Bounds(
        np.concatenate([np.full(n_target, float(x_min)), np.full(n_target, float(y_min))]),
        np.concatenate([np.full(n_target, float(x_max)), np.full(n_target, float(y_max))]),
    )

    def feasible_vec(z):
        x = jnp.asarray(z[:n_target])
        y = jnp.asarray(z[n_target:])
        return bool(feasible(x, y))

    def aep_vec(z):
        x = jnp.asarray(z[:n_target])
        y = jnp.asarray(z[n_target:])
        return float(-objective(x, y))

    def scipy_fun(z):
        value, grad = value_grad(jnp.asarray(z))
        value_f = float(value)
        grad_np = np.asarray(grad, dtype=float)
        if not np.isfinite(value_f) or not np.all(np.isfinite(grad_np)):
            return 1e20, np.zeros_like(z)
        return value_f, grad_np

    best_z = np.concatenate([np.asarray(best_x), np.asarray(best_y)])
    best_aep_np = float(best_aep)
    local_best = {"z": best_z.copy(), "aep": best_aep_np}

    def track_candidate(z):
        if feasible_vec(z):
            val = aep_vec(z)
            if val > local_best["aep"]:
                local_best["aep"] = val
                local_best["z"] = np.asarray(z).copy()

    class StructuredStep:
        def __init__(self, step_size):
            self.stepsize = step_size
            self.rng = np.random.default_rng(15015)
            wd_np = np.asarray(wd, dtype=float)
            ws_np = np.asarray(ws, dtype=float)
            wt_np = np.asarray(weights, dtype=float)
            theta = np.deg2rad(wd_np[int(np.argmax(wt_np * ws_np**3))])
            self.down = np.array([np.sin(theta), np.cos(theta)])
            self.cross = np.array([self.down[1], -self.down[0]])

        def __call__(self, z):
            x = z[:n_target].copy()
            y = z[n_target:].copy()
            count = max(3, n_target // 5)
            idx = self.rng.choice(n_target, size=count, replace=False)
            along = self.rng.normal(0.0, 0.35 * self.stepsize, size=count)
            cross = self.rng.normal(0.0, self.stepsize, size=count)
            x[idx] += along * self.down[0] + cross * self.cross[0]
            y[idx] += along * self.down[1] + cross * self.cross[1]
            z[:n_target] = np.clip(x, float(x_min), float(x_max))
            z[n_target:] = np.clip(y, float(y_min), float(y_max))
            return z

    def callback(z, f, accept):
        track_candidate(z)
        return time.time() - start_time > 165.0

    minimizer_kwargs = {
        "method": "SLSQP",
        "jac": True,
        "bounds": bounds,
        "constraints": (boundary_constraint, spacing_constraint),
        "options": {
            "maxiter": 28,
            "ftol": 1e-8,
            "disp": False,
        },
    }

    try:
        result = basinhopping(
            scipy_fun,
            best_z,
            niter=2,
            T=0.02,
            take_step=StructuredStep(float(min_spacing) * 0.22),
            minimizer_kwargs=minimizer_kwargs,
            callback=callback,
            seed=15015,
            disp=False,
        )
        track_candidate(result.x)
    except Exception:
        pass

    out_z = local_best["z"]
    return jnp.asarray(out_z[:n_target]), jnp.asarray(out_z[n_target:])
