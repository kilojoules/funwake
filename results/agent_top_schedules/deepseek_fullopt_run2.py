"""
HYPOTHESIS: Wind-aware staggered initialization improves AEP by minimizing wakes from the start.
AXIS: wind_aware_staggered_init
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    # ── Objective ──────────────────────────────────────────────────
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Wind-aware staggered initialization ────────────────────────
    wind_energy = (ws ** 3) * weights
    weighted_wd = wd * wind_energy
    total_energy = jnp.sum(wind_energy)
    dominant_dir = jnp.sum(weighted_wd) / total_energy

    dominant_rad = dominant_dir * jnp.pi / 180.0
    cross_wind_rad = dominant_rad + jnp.pi / 2.0

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    corners = jnp.array([[x_min, y_min], [x_max, y_min], 
                         [x_max, y_max], [x_min, y_max]])
    cx = jnp.cos(cross_wind_rad)
    cy = jnp.sin(cross_wind_rad)
    cross_proj = corners[:, 0] * cx + corners[:, 1] * cy
    along_proj = corners[:, 0] * (-cy) + corners[:, 1] * cx

    cross_min, cross_max = jnp.min(cross_proj), jnp.max(cross_proj)
    along_min, along_max = jnp.min(along_proj), jnp.max(along_proj)

    cross_range = cross_max - cross_min
    along_range = along_max - along_min

    cross_spacing = min_spacing * 3.0
    along_spacing = min_spacing * 5.0

    n_cross = max(3, int(jnp.ceil(cross_range / cross_spacing)))
    n_along = max(2, int(jnp.ceil(along_range / along_spacing)))

    # Safety margin
    margin = min_spacing * 0.5
    cross_min_safe = cross_min + margin
    cross_max_safe = cross_max - margin
    along_min_safe = along_min + margin
    along_max_safe = along_max - margin
    if cross_max_safe <= cross_min_safe:
        cross_min_safe = cross_min
        cross_max_safe = cross_max - 1.0
    if along_max_safe <= along_min_safe:
        along_min_safe = along_min
        along_max_safe = along_max - 1.0

    t = jnp.linspace(cross_min_safe, cross_max_safe, max(3, n_cross * 2))
    s = jnp.linspace(along_min_safe, along_max_safe, max(2, n_along * 2))

    all_cross, all_along = [], []
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    for j in range(len(s)):
        stagger = 0.0 if j % 2 == 0 else dt * 0.5
        for i in range(len(t)):
            c = t[i] + stagger
            if c > cross_max_safe:
                continue
            all_cross.append(c)
            all_along.append(s[j])

    cand_cross = jnp.array(all_cross) if all_cross else jnp.array([cross_min_safe])
    cand_along = jnp.array(all_along) if all_along else jnp.array([along_min_safe])

    cand_x = cand_cross * cx - cand_along * cy
    cand_y = cand_cross * cy + cand_along * cx

    # Filter to points inside boundary with margin
    n_verts = boundary.shape[0]
    def point_inside(px, py):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (px - x1) * (-ey / el) + (py - y1) * (ex / el) - min_spacing * 0.2
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0

    inside_mask = jax.vmap(point_inside)(cand_x, cand_y)
    inside_x, inside_y = cand_x[inside_mask], cand_y[inside_mask]

    n_inside = len(inside_x)
    if n_inside >= n_target:
        idx = jnp.round(jnp.linspace(0, n_inside - 1, n_target)).astype(int)
        init_x, init_y = inside_x[idx], inside_y[idx]
    else:
        nx = int(jnp.ceil((x_max - x_min) / min_spacing))
        ny = int(jnp.ceil((y_max - y_min) / min_spacing))
        gx, gy = jnp.meshgrid(
            jnp.linspace(x_min + min_spacing, x_max - min_spacing, max(2, nx)),
            jnp.linspace(y_min + min_spacing, y_max - min_spacing, max(2, ny)))
        flat_x, flat_y = gx.flatten(), gy.flatten()
        in_mask = jax.vmap(lambda px, py: point_inside(px, py))(flat_x, flat_y)
        fx, fy = flat_x[in_mask], flat_y[in_mask]
        nf = len(fx)
        if nf >= n_target:
            idx = jnp.round(jnp.linspace(0, nf - 1, n_target)).astype(int)
            init_x, init_y = fx[idx], fy[idx]
        else:
            key = jax.random.PRNGKey(0)
            rx = jax.random.uniform(key, (n_target,), minval=float(x_min + min_spacing), 
                                   maxval=float(x_max - min_spacing))
            key, _ = jax.random.split(key)
            ry = jax.random.uniform(key, (n_target,), minval=float(y_min + min_spacing),
                                   maxval=float(y_max - min_spacing))
            ri_mask = jax.vmap(lambda px, py: point_inside(px, py))(rx, ry)
            rx_in, ry_in = rx[ri_mask], ry[ri_mask]
            all_x = jnp.concatenate([fx, rx_in])
            all_y = jnp.concatenate([fy, ry_in])
            n_all = len(all_x)
            if n_all >= n_target:
                idx = jnp.round(jnp.linspace(0, n_all - 1, n_target)).astype(int)
                init_x, init_y = all_x[idx], all_y[idx]
            else:
                idx = jnp.round(jnp.linspace(0, nf - 1, n_target)).astype(int)
                init_x, init_y = fx[idx % nf], fy[idx % nf]

    # ── Optimizer settings ──────────────────────────────────────────
    # Higher boundary_weight and lower tol to ensure constraint satisfaction
    settings = SGDSettings(
        learning_rate=100.0,
        max_iter=5000,
        additional_constant_lr_iterations=2000,
        tol=1e-8,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.01,
        ks_rho=200.0,
        spacing_weight=2.0,
        boundary_weight=5.0,
    )

    opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y,
                                      boundary, min_spacing, settings)
    return opt_x, opt_y