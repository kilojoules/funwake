"""Random-topfarm basin with batched feasible turbine/pair line search.

HYPOTHESIS: The 5583 GWh random-start basin may still contain small feasible
single-turbine or active-pair translations that TopFarm's smooth penalties do
not accept.  A batched exact-feasibility line search can harvest only measured
AEP gains while keeping the robust random-basin fallback.

AXIS: three sequential random topfarm starts, then feasible weak-turbine and
near-contact-pair translation search around the best returned layout.

LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
from pixwake.optim.boundary import polygon_sdf
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    @jax.jit
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    @jax.jit
    def turbine_energy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return jnp.sum(p * weights[:, None], axis=0)

    settings = SGDSettings(
        learning_rate=100.0,
        max_iter=2000,
        additional_constant_lr_iterations=1000,
        tol=1e-6,
        beta1=0.9,
        beta2=0.999,
        gamma_min_factor=0.01,
        ks_rho=100.0,
        spacing_weight=1.0,
        boundary_weight=1.0,
    )

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    best_aep = -jnp.inf
    best_x = jnp.zeros((n_target,))
    best_y = jnp.zeros((n_target,))
    key = jax.random.PRNGKey(0)

    for _ in range(3):
        key, subkey = jax.random.split(key)
        init_x = jax.random.uniform(
            subkey, (n_target,), minval=float(x_min), maxval=float(x_max)
        )
        key, subkey = jax.random.split(key)
        init_y = jax.random.uniform(
            subkey, (n_target,), minval=float(y_min), maxval=float(y_max)
        )
        opt_x, opt_y = topfarm_sgd_solve(
            objective, init_x, init_y, boundary, min_spacing, settings
        )
        aep = -objective(opt_x, opt_y)
        if aep > best_aep:
            best_aep = aep
            best_x = opt_x
            best_y = opt_y

    wd_rad = jnp.deg2rad(wd)
    vx = jnp.sum(jnp.cos(wd_rad) * weights)
    vy = jnp.sum(jnp.sin(wd_rad) * weights)
    norm = jnp.sqrt(vx**2 + vy**2) + 1e-12
    wind_u = jnp.array([vx / norm, vy / norm])
    wind_v = jnp.array([-wind_u[1], wind_u[0]])
    dirs = jnp.stack(
        [
            wind_u,
            -wind_u,
            wind_v,
            -wind_v,
            (wind_u + wind_v) / jnp.sqrt(2.0),
            (wind_u - wind_v) / jnp.sqrt(2.0),
            (-wind_u + wind_v) / jnp.sqrt(2.0),
            (-wind_u - wind_v) / jnp.sqrt(2.0),
        ]
    )

    mask = jnp.triu(jnp.ones((n_target, n_target), dtype=bool), k=1)

    @jax.jit
    def feasible_batch(xs, ys):
        sdf = jax.vmap(lambda x, y: polygon_sdf(x, y, boundary))(xs, ys)
        boundary_ok = jnp.max(sdf, axis=1) <= 1e-3
        dx = xs[:, :, None] - xs[:, None, :]
        dy = ys[:, :, None] - ys[:, None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        min_dist = jnp.min(jnp.where(mask[None, :, :], dist, 1e12), axis=(1, 2))
        spacing_ok = min_dist >= min_spacing * 0.999
        return boundary_ok & spacing_ok

    batch_objective = jax.jit(jax.vmap(objective))

    @jax.jit
    def single_candidates(x, y, idxs, step):
        n_idx = idxs.shape[0]
        n_dirs = dirs.shape[0]
        move_idx = jnp.repeat(idxs, n_dirs)
        move_dir = jnp.tile(dirs, (n_idx, 1))
        xs = jnp.tile(x[None, :], (move_idx.shape[0], 1))
        ys = jnp.tile(y[None, :], (move_idx.shape[0], 1))
        rows = jnp.arange(move_idx.shape[0])
        xs = xs.at[rows, move_idx].add(step * move_dir[:, 0])
        ys = ys.at[rows, move_idx].add(step * move_dir[:, 1])
        return xs, ys

    @jax.jit
    def pair_candidates(x, y, pairs, step):
        n_pairs = pairs.shape[0]
        n_dirs = dirs.shape[0]
        pair_idx = jnp.repeat(pairs, n_dirs, axis=0)
        move_dir = jnp.tile(dirs, (n_pairs, 1))
        xs = jnp.tile(x[None, :], (pair_idx.shape[0], 1))
        ys = jnp.tile(y[None, :], (pair_idx.shape[0], 1))
        rows = jnp.arange(pair_idx.shape[0])
        xs = xs.at[rows, pair_idx[:, 0]].add(step * move_dir[:, 0])
        ys = ys.at[rows, pair_idx[:, 0]].add(step * move_dir[:, 1])
        xs = xs.at[rows, pair_idx[:, 1]].add(step * move_dir[:, 0])
        ys = ys.at[rows, pair_idx[:, 1]].add(step * move_dir[:, 1])
        return xs, ys

    def accept_best(x, y, cur_obj, cand_x, cand_y):
        ok = feasible_batch(cand_x, cand_y)
        vals = batch_objective(cand_x, cand_y)
        vals = jnp.where(ok, vals, jnp.inf)
        idx = jnp.argmin(vals)
        exact_val = objective(cand_x[idx], cand_y[idx])
        if bool(ok[idx] & (exact_val < cur_obj - 1e-5)):
            return cand_x[idx], cand_y[idx], exact_val
        return x, y, cur_obj

    x = best_x
    y = best_y
    cur_obj = objective(x, y)

    for step_frac in (0.16, 0.08, 0.04, 0.02):
        step = min_spacing * step_frac
        energy = turbine_energy(x, y)
        low = jnp.argsort(energy)[:14]
        cand_x, cand_y = single_candidates(x, y, low, step)
        x, y, cur_obj = accept_best(x, y, cur_obj, cand_x, cand_y)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        near_score = jnp.where(mask, dist, 1e12)
        flat = jnp.argsort(near_score.ravel())[:10]
        pairs = jnp.stack([flat // n_target, flat % n_target], axis=1)
        cand_x, cand_y = pair_candidates(x, y, pairs, step)
        x, y, cur_obj = accept_best(x, y, cur_obj, cand_x, cand_y)

    return x, y
