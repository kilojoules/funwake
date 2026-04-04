"""Iter 010: Custom vanilla SGD + 2-stage penalty annealing (v4-style).

Closely follows v4 best approach (+59 GWh):
- Vanilla SGD (no momentum) — more stable with penalty switching
- 12000 iterations via fori_loop
- Stage 1 (4000): aggressive feasibility (alpha=250, rho=150)
- Stage 2 (8000): AEP refinement (alpha=3, rho=50)
- Wind-aware grid initialization
- LR=250, decay=0.999
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    x_min_orig, y_min_orig = jnp.min(boundary, axis=0)
    x_max_orig, y_max_orig = jnp.max(boundary, axis=0)

    # ── Wind-aware grid initialization ─────────────────────────────
    wd_rad = jnp.deg2rad(wd)
    sin_sum = jnp.sum(weights * jnp.sin(wd_rad))
    cos_sum = jnp.sum(weights * jnp.cos(wd_rad))
    dominant_wd_rad = jnp.arctan2(sin_sum, cos_sum)
    grid_angle = dominant_wd_rad + jnp.pi / 2.0

    cos_t = jnp.cos(grid_angle)
    sin_t = jnp.sin(grid_angle)

    cx = jnp.mean(boundary[:, 0])
    cy = jnp.mean(boundary[:, 1])

    translated = boundary - jnp.array([cx, cy])
    rot_mat = jnp.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rot_bnd = (rot_mat @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)

    nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
    ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))

    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing / 2, rx_max - min_spacing / 2, nx),
        jnp.linspace(ry_min + min_spacing / 2, ry_max - min_spacing / 2, ny))

    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_t, sin_t], [-sin_t, cos_t]])
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
    inside_x, inside_y = cand_x[inside], cand_y[inside]

    if len(inside_x) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(inside_x) - 1, n_target)).astype(int)
        init_x, init_y = inside_x[idx], inside_y[idx]
    else:
        key = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min_orig), maxval=float(x_max_orig))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min_orig), maxval=float(y_max_orig))

    # ── Penalized objective ────────────────────────────────────────
    def objective_penalized(x, y, alpha_s, alpha_b, ks_rho):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        aep = -jnp.sum(p * weights[:, None]) * 8760 / 1e6
        sp = spacing_penalty(x, y, min_spacing, ks_rho)
        bp = boundary_penalty(x, y, boundary, ks_rho)
        return aep + alpha_s * sp + alpha_b * bp

    grad_fn = jax.grad(objective_penalized, argnums=(0, 1))

    # ── Vanilla SGD with 2-stage penalty annealing ─────────────────
    total_iters = 12000
    stage1_iters = 4000

    alpha_s1, alpha_b1, rho1 = 250.0, 250.0, 150.0
    alpha_s2, alpha_b2, rho2 = 3.0, 3.0, 50.0

    initial_lr = 250.0
    lr_decay = 0.999

    @jax.jit
    def sgd_step(i, state):
        x, y = state

        lr = initial_lr * (lr_decay ** i)

        a_s = jnp.where(i < stage1_iters, alpha_s1, alpha_s2)
        a_b = jnp.where(i < stage1_iters, alpha_b1, alpha_b2)
        rho = jnp.where(i < stage1_iters, rho1, rho2)

        gx, gy = grad_fn(x, y, a_s, a_b, rho)

        x = x - lr * gx
        y = y - lr * gy

        return x, y

    state0 = (init_x, init_y)
    final_x, final_y = jax.lax.fori_loop(0, total_iters, sgd_step, state0)

    return final_x, final_y
