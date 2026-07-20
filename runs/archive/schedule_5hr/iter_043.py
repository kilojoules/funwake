"""3-phase vanilla SGD with lr restart in enforcement phase.

Phases 1-2: original v4 settings (lr_decay=0.999, feasibility→exploration).
Phase 3: lr restarts at 50 and decays, with moderate enforcement (alpha=150).
This avoids the overcorrection from iter_042 (alpha=400 was too aggressive)
while still having enough lr to move turbines for feasibility repair.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    # --- Wind-aware grid initialization ---
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
    angle = dominant + jnp.pi / 2

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
        jnp.linspace(rx_min + min_spacing / 2, rx_max - min_spacing / 2, nx),
        jnp.linspace(ry_min + min_spacing / 2, ry_max - min_spacing / 2, ny))
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
    cand_x, cand_y = orig_pts[:, 0], orig_pts[:, 1]

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
        idx = jnp.round(jnp.linspace(0, len(ix) - 1, n_target)).astype(int)
        init_x, init_y = ix[idx], iy[idx]
    else:
        key = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # --- Penalized objective ---
    def penalized_objective(x, y, alpha_sp, alpha_bp):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing)
        bp = boundary_penalty(x, y, boundary)
        return aep + alpha_sp * sp + alpha_bp * bp

    grad_fn = jax.grad(penalized_objective, argnums=(0, 1))

    # --- Hyperparameters ---
    total_iters = 14000
    max_step = 2.0 * min_spacing

    # Phase boundaries
    phase1_end = 4000   # feasibility
    phase2_end = 10000  # AEP exploration
    # phase 3: 10000-14000 enforcement with lr restart

    # Phase 1-2: v4 best settings
    lr_phases12 = 250.0
    lr_decay_phases12 = 0.999
    alpha_p1 = 250.0
    alpha_p2 = 3.0

    # Phase 3: lr restart for enforcement
    lr_phase3 = 50.0
    lr_decay_phase3 = 0.999
    alpha_p3 = 150.0

    def sgd_step(i, state):
        x, y = state

        in_p1 = i < phase1_end
        in_p2 = (i >= phase1_end) & (i < phase2_end)
        in_p3 = i >= phase2_end

        # Piecewise lr: phases 1-2 share one decay, phase 3 restarts
        lr_12 = lr_phases12 * (lr_decay_phases12 ** i)
        lr_3 = lr_phase3 * (lr_decay_phase3 ** (i - phase2_end))
        lr = jnp.where(in_p3, lr_3, lr_12)

        alpha = jnp.where(in_p1, alpha_p1,
                          jnp.where(in_p2, alpha_p2, alpha_p3))

        gx, gy = grad_fn(x, y, alpha, alpha)

        # Per-turbine step clipping
        step_x = lr * gx
        step_y = lr * gy
        step_dist = jnp.sqrt(step_x**2 + step_y**2 + 1e-12)
        scale = jnp.minimum(1.0, max_step / step_dist)
        step_x = step_x * scale
        step_y = step_y * scale

        x = x - step_x
        y = y - step_y
        return (x, y)

    final_x, final_y = jax.lax.fori_loop(
        0, total_iters, sgd_step, (init_x, init_y))

    return final_x, final_y
