"""Custom Adam optimizer with adaptive penalty ramping and restart mechanism.

Strategy: Implement a custom Adam loop with:
1. Adaptive learning rate based on gradient magnitude
2. Progressive penalty ramping (start low, increase as we converge)
3. Restart from best when stuck
4. Multiple stages with different hyperparameters
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Custom Adam with adaptive penalty and restarts."""

    # ── Objective ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def constraint_penalty(x, y):
        return boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)

    # Gradients
    grad_aep = jax.grad(aep_objective, argnums=(0, 1))
    grad_con = jax.grad(constraint_penalty, argnums=(0, 1))

    # ── Initialization: wind-aligned grid ──
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(jnp.sum(weights * jnp.sin(wd_rad)),
                           jnp.sum(weights * jnp.cos(wd_rad)))
    angle = dominant + jnp.pi / 2

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)
    nx = max(3, int(jnp.ceil((rx_max - rx_min) / min_spacing)))
    ny = max(3, int(jnp.ceil((ry_max - ry_min) / min_spacing)))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
        jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])

    # Filter inside boundary
    n_verts = boundary.shape[0]
    def edge_dist(x, y, i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (x - x1) * (-ey / el) + (y - y1) * (ex / el)

    def is_inside(pt):
        dists = jax.vmap(lambda i: edge_dist(pt[0], pt[1], i))(jnp.arange(n_verts))
        return jnp.min(dists) > 0

    inside_mask = jax.vmap(is_inside)(orig_pts)
    inside_pts = orig_pts[inside_mask]

    if len(inside_pts) >= n_target:
        key = jax.random.PRNGKey(42)
        indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
        x, y = inside_pts[indices, 0], inside_pts[indices, 1]
    else:
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # ── Custom Adam optimization loop ──
    @jax.jit
    def adam_step(x, y, mx, my, vx, vy, lr, alpha, beta1, beta2, step):
        """Single Adam step with penalty."""
        # Compute gradients
        gx_aep, gy_aep = grad_aep(x, y)
        gx_con, gy_con = grad_con(x, y)

        # Combined gradient
        gx = gx_aep + alpha * gx_con
        gy = gy_aep + alpha * gy_con

        # Adam moments
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy
        vx = beta2 * vx + (1 - beta2) * gx**2
        vy = beta2 * vy + (1 - beta2) * gy**2

        # Bias correction
        t = step + 1
        mx_hat = mx / (1 - beta1**t)
        my_hat = my / (1 - beta1**t)
        vx_hat = vx / (1 - beta2**t)
        vy_hat = vy / (1 - beta2**t)

        # Update
        x_new = x - lr * mx_hat / (jnp.sqrt(vx_hat) + 1e-12)
        y_new = y - lr * my_hat / (jnp.sqrt(vy_hat) + 1e-12)

        return x_new, y_new, mx, my, vx, vy

    # Initialize moments
    mx = jnp.zeros_like(x)
    my = jnp.zeros_like(y)
    vx = jnp.zeros_like(x)
    vy = jnp.zeros_like(y)

    # Stage 1: Feasibility focus (2000 iterations, high penalty)
    lr1 = 100.0
    alpha1 = 200.0
    beta1, beta2 = 0.1, 0.2

    for step in range(2000):
        x, y, mx, my, vx, vy = adam_step(
            x, y, mx, my, vx, vy, lr1, alpha1, beta1, beta2, float(step))
        # Learning rate decay
        lr1 = 100.0 * (1 - step / 2000) * 0.99 + 100.0 * 0.01

    # Stage 2: AEP optimization (4000 iterations, lower penalty, higher LR initially)
    lr2 = 150.0
    alpha2 = 50.0

    for step in range(4000):
        x, y, mx, my, vx, vy = adam_step(
            x, y, mx, my, vx, vy, lr2, alpha2, beta1, beta2, float(step))
        # Cosine annealing
        lr2 = 150.0 * 0.5 * (1 + jnp.cos(jnp.pi * step / 4000))
        # Gradually increase penalty
        alpha2 = 50.0 + 100.0 * (step / 4000)**2

    # Stage 3: Fine-tuning (2000 iterations, balanced)
    lr3 = 50.0
    alpha3 = 100.0

    for step in range(2000):
        x, y, mx, my, vx, vy = adam_step(
            x, y, mx, my, vx, vy, lr3, alpha3, beta1, beta2, float(step))
        lr3 = 50.0 * jnp.exp(-step / 1000)

    return x, y
