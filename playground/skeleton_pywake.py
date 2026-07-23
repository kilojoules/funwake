"""Fixed optimization skeleton backed by py_wake (open-source test path).

Port of playground/skeleton.py with the same contract and semantics:
  - wind-direction-aware grid initialization inside the polygon (identical
    jax.random seeding, so initial layouts match the reference skeleton),
  - lr0 = 50.0, alpha0 = mean|initial AEP gradient| / lr0,
  - per-step schedule_fn(step, total_steps, lr0, alpha0)
        -> (lr, alpha, beta1, beta2),
  - Adam update on grad = grad_aep + alpha * grad_constraint,
  - optional early stopping: once lr <= es_threshold * peak lr, the AEP
    gradient is dropped and the step follows the constraint gradient only.

The AEP objective and its gradient come from the py_wake adapter
(pywake_adapter.WakeSimulation.make_neg_aep); constraint penalties and
their analytic gradients come from penalties_np.  The Adam loop itself is
plain numpy (the reference skeleton's JAX loop is replaced 1:1).
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from penalties_np import boundary_penalty_grad, spacing_penalty_grad


def _grid_init(boundary, min_spacing, wd, ws, weights, n_target, seed):
    """Wind-direction-aware grid initialization (mirrors skeleton.py)."""
    boundary = jnp.asarray(boundary, dtype=float)
    wd = jnp.asarray(wd, dtype=float)
    weights = jnp.asarray(weights, dtype=float)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    # Dominant wind direction
    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
    angle = dominant + jnp.pi / 2  # perpendicular to wind

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)
    nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
    ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
        jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
    cand_x, cand_y = orig_pts[:, 0], orig_pts[:, 1]

    # Filter inside boundary
    n_verts = boundary.shape[0]

    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)

    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
    ix, iy = cand_x[inside], cand_y[inside]

    if len(ix) >= n_target:
        key = jax.random.PRNGKey(seed)
        indices = jax.random.choice(key, len(ix), (n_target,), replace=False)
        x, y = ix[indices], iy[indices]
    else:
        key = jax.random.PRNGKey(seed)
        x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def run_with_schedule(schedule_fn, sim, n_target, boundary, min_spacing,
                      wd, ws, weights, total_steps=8000, seed=0,
                      early_stopping=False, es_threshold=0.1):
    """Run the fixed Adam skeleton with a given schedule_fn.

    sim must be a pywake_adapter.WakeSimulation.  Returns (opt_x, opt_y).
    """
    boundary_np = np.asarray(boundary, dtype=float)
    wd_np = np.asarray(wd, dtype=float)
    ws_np = np.asarray(ws, dtype=float)
    weights_np = np.asarray(weights, dtype=float)
    min_spacing = float(min_spacing)

    # ── Objective + constraint gradients ───────────────────────────
    _, grad_obj = sim.make_neg_aep(wd_np, ws_np, weights_np)

    def grad_con(x, y):
        bx, by = boundary_penalty_grad(x, y, boundary_np)
        sx, sy = spacing_penalty_grad(x, y, min_spacing)
        return bx + sx, by + sy

    # ── Wind-aware grid initialization ─────────────────────────────
    x, y = _grid_init(boundary, min_spacing, wd, ws, weights, n_target, seed)

    # ── Compute lr0 and alpha0 from problem scale ──────────────────
    gox, goy = grad_obj(x, y)
    lr0 = 50.0
    alpha0 = float(np.mean(np.abs(np.concatenate([gox, goy]))) / lr0)

    # Precompute the schedule (it depends only on step, lr0, alpha0).
    lrs = np.empty(total_steps)
    alphas = np.empty(total_steps)
    b1s = np.empty(total_steps)
    b2s = np.empty(total_steps)
    for i in range(total_steps):
        lr_i, alpha_i, b1_i, b2_i = schedule_fn(i, total_steps, lr0, alpha0)
        lrs[i] = float(lr_i)
        alphas[i] = float(alpha_i)
        b1s[i] = float(b1_i)
        b2s[i] = float(b2_i)
    max_lr = float(np.max(lrs))  # reference for the early-stopping trigger

    # ── Adam loop with the schedule ────────────────────────────────
    mx = np.zeros_like(x)
    my = np.zeros_like(y)
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)
    eps = 1e-12

    for i in range(total_steps):
        lr, alpha, b1, b2 = lrs[i], alphas[i], b1s[i], b2s[i]

        # early stopping: once lr <= es_threshold x peak, drop the AEP term
        es_on = early_stopping and lr <= es_threshold * max_lr
        if es_on:
            gox = np.zeros_like(x)
            goy = np.zeros_like(y)
        else:
            gox, goy = grad_obj(x, y)
        gcx, gcy = grad_con(x, y)
        jx = gox + alpha * gcx
        jy = goy + alpha * gcy

        # Adam update
        it = float(i + 1)
        mx = b1 * mx + (1 - b1) * jx
        my = b1 * my + (1 - b1) * jy
        vx = b2 * vx + (1 - b2) * jx**2
        vy = b2 * vy + (1 - b2) * jy**2

        mx_hat = mx / (1 - b1**it)
        my_hat = my / (1 - b1**it)
        vx_hat = vx / (1 - b2**it)
        vy_hat = vy / (1 - b2**it)

        x = x - lr * mx_hat / (np.sqrt(vx_hat) + eps)
        y = y - lr * my_hat / (np.sqrt(vy_hat) + eps)

    return x, y
