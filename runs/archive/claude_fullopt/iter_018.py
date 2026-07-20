"""Wake-aware greedy placement followed by SGD refinement.

Strategy:
1. Greedy sequential placement: add turbines one at a time
2. Each new turbine placed where it maximizes incremental AEP
3. Consider wake effects from already-placed turbines
4. After all turbines placed, refine entire layout with SGD
5. Multiple greedy runs with different starting positions
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Wake-aware greedy placement + SGD refinement."""

    # ── Objective ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Create candidate grid ──
    x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
    x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

    # Dense candidate grid
    spacing = min_spacing * 0.9
    nx = max(5, int(jnp.ceil((x_max - x_min) / spacing)))
    ny = max(5, int(jnp.ceil((y_max - y_min) / spacing)))

    gx, gy = jnp.meshgrid(
        jnp.linspace(x_min + spacing/2, x_max - spacing/2, nx),
        jnp.linspace(y_min + spacing/2, y_max - spacing/2, ny))
    cand_x_all = gx.flatten()
    cand_y_all = gy.flatten()

    # Filter inside boundary
    n_verts = boundary.shape[0]
    def is_inside(xi, yi):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (xi - x1) * (-ey / el) + (yi - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > min_spacing * 0.4

    inside_mask = jax.vmap(is_inside)(cand_x_all, cand_y_all)
    cand_x = cand_x_all[inside_mask]
    cand_y = cand_y_all[inside_mask]

    # ── Greedy placement function ──
    def greedy_placement(start_idx):
        """Place turbines greedily starting from a specific candidate."""
        placed_x = jnp.array([cand_x[start_idx]])
        placed_y = jnp.array([cand_y[start_idx]])

        for _ in range(n_target - 1):
            # Find candidates far enough from placed turbines
            def min_dist_to_placed(xi, yi):
                dx = placed_x - xi
                dy = placed_y - yi
                dists = jnp.sqrt(dx**2 + dy**2)
                return jnp.min(dists)

            dists_to_placed = jax.vmap(min_dist_to_placed)(cand_x, cand_y)
            valid_mask = dists_to_placed >= min_spacing * 0.95

            if jnp.sum(valid_mask) == 0:
                # No valid candidates, fall back to random
                key = jax.random.PRNGKey(len(placed_x))
                new_x = jax.random.uniform(key, (), minval=x_min, maxval=x_max)
                key, _ = jax.random.split(key)
                new_y = jax.random.uniform(key, (), minval=y_min, maxval=y_max)
                placed_x = jnp.append(placed_x, new_x)
                placed_y = jnp.append(placed_y, new_y)
                continue

            valid_x = cand_x[valid_mask]
            valid_y = cand_y[valid_mask]

            # Evaluate each valid candidate
            best_aep = float('inf')
            best_x, best_y = valid_x[0], valid_y[0]

            # Sample subset of candidates for speed
            n_valid = len(valid_x)
            sample_size = min(20, n_valid)
            key = jax.random.PRNGKey(len(placed_x))
            sample_indices = jax.random.choice(key, n_valid, (sample_size,), replace=False)

            for idx in sample_indices:
                test_x = jnp.append(placed_x, valid_x[idx])
                test_y = jnp.append(placed_y, valid_y[idx])
                aep = aep_objective(test_x, test_y)

                if aep < best_aep:
                    best_aep = aep
                    best_x = valid_x[idx]
                    best_y = valid_y[idx]

            placed_x = jnp.append(placed_x, best_x)
            placed_y = jnp.append(placed_y, best_y)

        return placed_x, placed_y

    # ── Run greedy from multiple starting points ──
    best_x, best_y = None, None
    best_aep = float('inf')

    n_starts = min(5, len(cand_x))
    key = jax.random.PRNGKey(42)
    start_indices = jax.random.choice(key, len(cand_x), (n_starts,), replace=False)

    for start_idx in start_indices:
        try:
            gx, gy = greedy_placement(int(start_idx))
            if len(gx) == n_target:
                aep = aep_objective(gx, gy)
                if aep < best_aep:
                    best_aep = aep
                    best_x, best_y = gx, gy
        except Exception:
            continue

    # Fallback if greedy failed
    if best_x is None:
        dx = min_spacing * 1.1
        dy = min_spacing * 1.1 * jnp.sqrt(3) / 2
        nx = max(3, int(jnp.ceil((x_max - x_min) / dx)) + 1)
        ny = max(3, int(jnp.ceil((y_max - y_min) / dy)) + 1)

        pts = []
        for i in range(ny):
            for j in range(nx):
                offset = dx / 2 if i % 2 == 1 else 0
                xi = x_min + j * dx + offset
                yi = y_min + i * dy
                pts.append([xi, yi])

        pts = jnp.array(pts)
        inside_mask = jax.vmap(lambda pt: is_inside(pt[0], pt[1]))(pts)
        inside_pts = pts[inside_mask]

        if len(inside_pts) >= n_target:
            indices = jnp.round(jnp.linspace(0, len(inside_pts) - 1, n_target)).astype(int)
            best_x = inside_pts[indices, 0]
            best_y = inside_pts[indices, 1]
        else:
            key = jax.random.PRNGKey(42)
            best_x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
            key, _ = jax.random.split(key)
            best_y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

    # ── Refine with SGD ──
    settings = SGDSettings(
        learning_rate=120.0,
        max_iter=3500,
        additional_constant_lr_iterations=2000,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.005,
        ks_rho=100.0,
        spacing_weight=80.0,
        boundary_weight=80.0,
    )

    opt_x, opt_y = topfarm_sgd_solve(
        aep_objective, best_x, best_y,
        boundary, min_spacing, settings)

    return opt_x, opt_y
