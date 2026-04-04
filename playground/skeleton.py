"""Fixed optimization skeleton. The LLM evolves ONLY the schedule_fn.

The skeleton handles: initialization, gradient computation, Adam update,
constraint penalties, convergence. The LLM controls the learning rate,
penalty weight, and Adam momentum schedules via schedule_fn.

schedule_fn(step, total_steps, lr0, alpha0) -> (lr, alpha, beta1, beta2)
  - step: current iteration (0 to total_steps-1)
  - total_steps: total number of iterations
  - lr0: initial learning rate (computed from problem scale)
  - alpha0: initial penalty weight (computed from gradient magnitude)
  Returns:
  - lr: learning rate for this step
  - alpha: constraint penalty multiplier for this step
  - beta1: Adam first moment decay (0 to 1)
  - beta2: Adam second moment decay (0 to 1)
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def run_with_schedule(schedule_fn, sim, n_target, boundary, min_spacing,
                      wd, ws, weights, total_steps=8000, seed=0):
    """Run the fixed Adam skeleton with a given schedule_fn.

    Returns (opt_x, opt_y).
    """

    # ── Objective + constraint gradients ───────────────────────────
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def con_penalty(x, y):
        return (boundary_penalty(x, y, boundary)
                + spacing_penalty(x, y, min_spacing))

    grad_obj = jax.grad(aep_objective, argnums=(0, 1))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    # ── Wind-aware grid initialization ─────────────────────────────
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

    # ── Compute lr0 and alpha0 from problem scale ──────────────────
    gox, goy = grad_obj(x, y)
    lr0 = 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0

    # ── JIT-compiled Adam loop with LLM's schedule ─────────────────
    @jax.jit
    def run_loop(x, y):
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx = jnp.zeros_like(x)
        vy = jnp.zeros_like(y)
        eps = 1e-12

        def step(i, carry):
            x, y, mx, my, vx, vy = carry

            # LLM's schedule controls lr, alpha, beta1, beta2
            lr, alpha, b1, b2 = schedule_fn(i, total_steps, lr0, alpha0)

            # Gradients
            gox, goy = grad_obj(x, y)
            gcx, gcy = grad_con(x, y)
            jx = gox + alpha * gcx
            jy = goy + alpha * gcy

            # Adam update
            it = (i + 1).astype(float)
            mx_new = b1 * mx + (1 - b1) * jx
            my_new = b1 * my + (1 - b1) * jy
            vx_new = b2 * vx + (1 - b2) * jx**2
            vy_new = b2 * vy + (1 - b2) * jy**2

            mx_hat = mx_new / (1 - b1**it)
            my_hat = my_new / (1 - b1**it)
            vx_hat = vx_new / (1 - b2**it)
            vy_hat = vy_new / (1 - b2**it)

            x_new = x - lr * mx_hat / (jnp.sqrt(vx_hat) + eps)
            y_new = y - lr * my_hat / (jnp.sqrt(vy_hat) + eps)

            return (x_new, y_new, mx_new, my_new, vx_new, vy_new)

        init = (x, y, mx, my, vx, vy)
        final = jax.lax.fori_loop(0, total_steps, step, init)
        return final[0], final[1]

    return run_loop(x, y)
