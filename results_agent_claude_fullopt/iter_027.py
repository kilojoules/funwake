"""Adaptive dual-phase optimizer: coarse global search + fine local refinement.

Strategy:
- Phase 1: Population-based exploration (3 parallel starts, pick best)
  - Wind-aware hex grid
  - Rotated hex grid (+45deg)
  - Random with wind clustering
- Phase 2: Intensive refinement of best solution
- Adaptive penalty ramping
- Budget: ~40s JIT + 40s phase 1 + 100s phase 2
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Adaptive dual-phase optimizer."""

    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))
    n_verts = boundary.shape[0]

    def is_inside_boundary(xi, yi):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (xi - x1) * (-ey / el) + (yi - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > 0

    def create_hex_grid(angle_offset):
        """Create hexagonal grid with given rotation."""
        wd_rad = jnp.deg2rad(wd)
        dominant = jnp.arctan2(
            jnp.sum(weights * jnp.sin(wd_rad)),
            jnp.sum(weights * jnp.cos(wd_rad)))
        angle = dominant + jnp.pi / 2 + angle_offset

        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
        translated = boundary - jnp.array([cx, cy])
        rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_bnd = (rot @ translated.T).T

        rx_min, ry_min = jnp.min(rot_bnd, axis=0)
        rx_max, ry_max = jnp.max(rot_bnd, axis=0)

        h_spacing = min_spacing * 1.08
        v_spacing = h_spacing * jnp.sqrt(3) / 2

        nx = max(3, int(jnp.ceil((rx_max - rx_min) / h_spacing)))
        ny = max(3, int(jnp.ceil((ry_max - ry_min) / v_spacing)))

        candidates = []
        for i in range(ny):
            for j in range(nx):
                offset = (h_spacing / 2) if i % 2 == 1 else 0
                rx = float(rx_min) + j * float(h_spacing) + offset
                ry = float(ry_min) + i * float(v_spacing)
                candidates.append((rx, ry))

        candidates = jnp.array(candidates)
        inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
        orig_pts = (inv_rot @ candidates.T).T + jnp.array([cx, cy])

        inside_mask = jnp.array([is_inside_boundary(pt[0], pt[1]) for pt in orig_pts])
        inside_pts = orig_pts[inside_mask]

        if len(inside_pts) >= n_target:
            idx = jnp.round(jnp.linspace(0, len(inside_pts) - 1, n_target)).astype(int)
            return inside_pts[idx, 0], inside_pts[idx, 1]
        else:
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
            key, _ = jax.random.split(key)
            y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)
            return x, y

    # ── Phase 1: Quick exploration of 3 diverse starts ─────────────
    # Fast SGD settings for exploration
    explore_settings = SGDSettings(
        learning_rate=100.0,
        max_iter=800,
        additional_constant_lr_iterations=400,
        tol=1e-6,
        beta1=0.15,
        beta2=0.25,
        gamma_min_factor=0.02,
        ks_rho=100.0,
        spacing_weight=1.2,
        boundary_weight=1.2,
    )

    # Start 1: Standard wind-aware hex
    init_x1, init_y1 = create_hex_grid(0.0)
    opt_x1, opt_y1 = topfarm_sgd_solve(
        objective, init_x1, init_y1, boundary, min_spacing, explore_settings
    )
    aep1 = -objective(opt_x1, opt_y1)

    # Start 2: Rotated hex (+30deg)
    init_x2, init_y2 = create_hex_grid(jnp.pi / 6)
    opt_x2, opt_y2 = topfarm_sgd_solve(
        objective, init_x2, init_y2, boundary, min_spacing, explore_settings
    )
    aep2 = -objective(opt_x2, opt_y2)

    # Start 3: Wind-clustered random
    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))

    key = jax.random.PRNGKey(789)
    # Cluster positions perpendicular to dominant wind
    angle_perp = dominant + jnp.pi / 2
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])

    # Random positions along perpendicular axis
    spread = min(x_max - x_min, y_max - y_min) * 0.8
    offsets = jax.random.uniform(key, (n_target,), minval=-spread/2, maxval=spread/2)
    key, _ = jax.random.split(key)
    cross_offsets = jax.random.uniform(key, (n_target,), minval=-spread/4, maxval=spread/4)

    init_x3 = cx + offsets * jnp.cos(angle_perp) + cross_offsets * jnp.sin(angle_perp)
    init_y3 = cy + offsets * jnp.sin(angle_perp) - cross_offsets * jnp.cos(angle_perp)

    # Clamp to boundary box
    init_x3 = jnp.clip(init_x3, x_min + min_spacing, x_max - min_spacing)
    init_y3 = jnp.clip(init_y3, y_min + min_spacing, y_max - min_spacing)

    opt_x3, opt_y3 = topfarm_sgd_solve(
        objective, init_x3, init_y3, boundary, min_spacing, explore_settings
    )
    aep3 = -objective(opt_x3, opt_y3)

    # Select best
    best_aep = jnp.max(jnp.array([aep1, aep2, aep3]))
    best_idx = jnp.argmax(jnp.array([aep1, aep2, aep3]))

    best_x = jnp.where(best_idx == 0, opt_x1,
                       jnp.where(best_idx == 1, opt_x2, opt_x3))
    best_y = jnp.where(best_idx == 0, opt_y1,
                       jnp.where(best_idx == 1, opt_y2, opt_y3))

    # ── Phase 2: Intensive refinement of best solution ─────────────
    refine_settings = SGDSettings(
        learning_rate=80.0,
        max_iter=4000,
        additional_constant_lr_iterations=2000,
        tol=1e-8,
        beta1=0.25,
        beta2=0.35,
        gamma_min_factor=0.001,
        ks_rho=180.0,
        spacing_weight=2.0,
        boundary_weight=2.0,
    )

    final_x, final_y = topfarm_sgd_solve(
        objective, best_x, best_y, boundary, min_spacing, refine_settings
    )

    return final_x, final_y
