"""Custom Adam with explicit feasibility projection.

HYPOTHESIS: A hand-rolled Adam loop with a rising constraint penalty can make
larger wake-aware moves than TopFarm SGD, and a final projection pass can keep
the returned layout feasible across dense polygons.

AXIS: custom_adam plus deterministic rotated hex initializations and projection
repair; no call to topfarm_sgd_solve.

LESSON: Pending score.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    n_verts = boundary.shape[0]
    x_min = float(jnp.min(boundary[:, 0]))
    x_max = float(jnp.max(boundary[:, 0]))
    y_min = float(jnp.min(boundary[:, 1]))
    y_max = float(jnp.max(boundary[:, 1]))
    center = jnp.mean(boundary, axis=0)
    box_span = float(jnp.maximum(x_max - x_min, y_max - y_min))
    ms2 = min_spacing * min_spacing

    def aep_gwh(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    def objective(x, y):
        return -aep_gwh(x, y)

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

    def rotated_hex(angle, spacing_mult, phase):
        ca = jnp.cos(angle)
        sa = jnp.sin(angle)
        rot = jnp.array([[ca, -sa], [sa, ca]])
        inv_rot = jnp.array([[ca, sa], [-sa, ca]])
        rb = (rot @ (boundary - center).T).T
        rx_min, ry_min = jnp.min(rb, axis=0)
        rx_max, ry_max = jnp.max(rb, axis=0)
        sp = min_spacing * spacing_mult
        dy = sp * 0.8660254037844386
        nx = max(4, int(jnp.ceil((rx_max - rx_min) / sp)) + 4)
        ny = max(4, int(jnp.ceil((ry_max - ry_min) / dy)) + 4)
        gx, gy = jnp.meshgrid(
            jnp.linspace(rx_min - sp, rx_max + sp, nx),
            jnp.linspace(ry_min - dy, ry_max + dy, ny),
        )
        row = jnp.arange(ny)[:, None]
        gx = gx + 0.5 * sp * ((row + phase) % 2)
        pts_r = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
        pts = (inv_rot @ pts_r.T).T + center
        mask = inside_mask(pts[:, 0], pts[:, 1])
        px = pts[:, 0][mask]
        py = pts[:, 1][mask]

        score_axis = (
            (px - center[0]) * jnp.cos(angle + 0.37)
            + (py - center[1]) * jnp.sin(angle + 0.37)
        )
        order = jnp.argsort(score_axis)
        px = px[order]
        py = py[order]
        if len(px) >= n_target:
            idx = jnp.round(jnp.linspace(0, len(px) - 1, n_target)).astype(int)
            return px[idx], py[idx]

        gx2, gy2 = jnp.meshgrid(
            jnp.linspace(x_min + 0.08 * box_span, x_max - 0.08 * box_span, n_target),
            jnp.linspace(y_min + 0.08 * box_span, y_max - 0.08 * box_span, n_target),
        )
        pts2 = jnp.stack([gx2.ravel(), gy2.ravel()], axis=1)
        mask2 = inside_mask(pts2[:, 0], pts2[:, 1])
        px2 = pts2[:, 0][mask2]
        py2 = pts2[:, 1][mask2]
        idx2 = jnp.round(jnp.linspace(0, len(px2) - 1, n_target)).astype(int)
        return px2[idx2], py2[idx2]

    wd_rad = jnp.deg2rad(wd)
    mean_wind = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)),
    )

    def project_boundary(x, y, margin):
        for i in range(n_verts):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex = x2 - x1
            ey = y2 - y1
            el = jnp.sqrt(ex * ex + ey * ey) + 1e-10
            nx = -ey / el
            ny = ex / el
            d = (x - x1) * nx + (y - y1) * ny
            push = jnp.maximum(0.0, margin - d)
            x = x + push * nx
            y = y + push * ny
        return x, y

    def project_spacing(x, y):
        for _ in range(12):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n_target) * 1e12)
            target = min_spacing * 1.003
            viol = jnp.maximum(0.0, target - dist)
            ux = dx / (dist + 1e-9)
            uy = dy / (dist + 1e-9)
            push_x = 0.52 * jnp.sum(viol * ux, axis=1)
            push_y = 0.52 * jnp.sum(viol * uy, axis=1)
            x = x + push_x
            y = y + push_y
            x, y = project_boundary(x, y, 1.0)
        return x, y

    def repair(x, y):
        x, y = project_boundary(x, y, 2.0)
        x, y = project_spacing(x, y)
        x, y = project_boundary(x, y, 1.0)
        return x, y

    def loss_from_z(z, alpha):
        x = z[:n_target]
        y = z[n_target:]
        raw = objective(x, y)
        pen = boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)
        return raw + alpha * pen / ms2

    loss_grad = jax.value_and_grad(loss_from_z)

    @jax.jit
    def run_adam(z0, lr0, alpha0, alpha1):
        z0 = jnp.concatenate(repair(z0[:n_target], z0[n_target:]))
        m0 = jnp.zeros_like(z0)
        v0 = jnp.zeros_like(z0)

        def body(carry, t):
            z, m, v = carry
            frac = t / 899.0
            lr = lr0 * (1.0 - frac) + 18.0 * frac
            alpha = alpha0 * (alpha1 / alpha0) ** frac
            _, g = loss_grad(z, alpha)
            g_norm = jnp.sqrt(jnp.sum(g * g)) + 1e-12
            g = g * jnp.minimum(1.0, 12.0 / g_norm)
            beta1 = 0.86
            beta2 = 0.985
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            tt = t + 1.0
            mh = m / (1.0 - beta1**tt)
            vh = v / (1.0 - beta2**tt)
            z = z - lr * mh / (jnp.sqrt(vh) + 1e-8)
            x = jnp.clip(z[:n_target], x_min - 0.2 * box_span, x_max + 0.2 * box_span)
            y = jnp.clip(z[n_target:], y_min - 0.2 * box_span, y_max + 0.2 * box_span)
            x, y = project_boundary(x, y, 0.25)
            z = jnp.concatenate([x, y])
            return (z, m, v), None

        (z, _, _), _ = jax.lax.scan(body, (z0, m0, v0), jnp.arange(900.0))
        x, y = repair(z[:n_target], z[n_target:])
        return x, y

    starts = [
        rotated_hex(mean_wind + 0.5 * jnp.pi, 1.02, 0),
        rotated_hex(mean_wind, 1.02, 1),
    ]

    best_x, best_y = starts[0]
    best_x, best_y = repair(best_x, best_y)
    best_obj = objective(best_x, best_y)

    for i, (sx, sy) in enumerate(starts):
        z0 = jnp.concatenate([sx, sy])
        lr0 = 165.0 if i == 0 else 125.0
        ox, oy = run_adam(z0, lr0, 35.0, 65000.0)
        ox, oy = repair(ox, oy)
        obj = objective(ox, oy)
        if obj < best_obj:
            best_obj = obj
            best_x, best_y = ox, oy

    return best_x, best_y
