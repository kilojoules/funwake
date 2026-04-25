"""Ensemble optimizer: run multiple methods and select best.

Strategy:
1. Run 4 different optimization strategies in sequence
2. Each with limited iterations for speed
3. Track best result across all methods
4. Final polish with SGD on the overall best
5. Methods: L-BFGS-B, trust-constr, multi-start SGD, coarse-to-fine

This is a meta-optimizer that hedges bets across different approaches.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Ensemble meta-optimizer."""

    # ── Objective ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Shared initialization ──
    def get_hex_init(seed=0):
        x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
        x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

        dx = min_spacing * 1.1
        dy = min_spacing * 1.1 * np.sqrt(3) / 2
        nx = max(3, int(np.ceil((x_max - x_min) / dx)) + 1)
        ny = max(3, int(np.ceil((y_max - y_min) / dy)) + 1)

        pts = []
        for i in range(ny):
            for j in range(nx):
                offset = dx / 2 if i % 2 == 1 else 0
                xi = x_min + j * dx + offset
                yi = y_min + i * dy
                pts.append([xi, yi])

        pts = jnp.array(pts)

        n_verts = boundary.shape[0]
        def is_inside(pt):
            def edge_dist(i):
                x1, y1 = boundary[i]
                x2, y2 = boundary[(i + 1) % n_verts]
                ex, ey = x2 - x1, y2 - y1
                el = jnp.sqrt(ex**2 + ey**2) + 1e-10
                return (pt[0] - x1) * (-ey / el) + (pt[1] - y1) * (ex / el)
            return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > min_spacing * 0.35

        inside_mask = jax.vmap(is_inside)(pts)
        inside_pts = pts[inside_mask]

        if len(inside_pts) >= n_target:
            key = jax.random.PRNGKey(seed)
            indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
            return inside_pts[indices, 0], inside_pts[indices, 1]
        else:
            key = jax.random.PRNGKey(seed)
            x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
            key, _ = jax.random.split(key)
            y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)
            return x, y

    # Track best across all methods
    best_x, best_y = None, None
    best_aep = float('inf')

    # ── Method 1: L-BFGS-B with penalties ──
    try:
        def penalized_flat(xy_flat, alpha=100.0):
            x = jnp.array(xy_flat[:n_target])
            y = jnp.array(xy_flat[n_target:])
            aep = aep_objective(x, y)
            penalty = boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)
            return float(aep + alpha * penalty)

        grad_aep = jax.grad(aep_objective, argnums=(0, 1))
        grad_pen = jax.grad(lambda x, y: boundary_penalty(x, y, boundary) +
                            spacing_penalty(x, y, min_spacing), argnums=(0, 1))

        def penalized_jac_flat(xy_flat, alpha=100.0):
            x = jnp.array(xy_flat[:n_target])
            y = jnp.array(xy_flat[n_target:])
            gx, gy = grad_aep(x, y)
            px, py = grad_pen(x, y)
            return np.concatenate([np.array(gx + alpha * px), np.array(gy + alpha * py)])

        init_x, init_y = get_hex_init(0)
        init = np.concatenate([np.array(init_x), np.array(init_y)])

        x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
        x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))
        bounds = [(x_min, x_max)] * n_target + [(y_min, y_max)] * n_target

        result = minimize(
            lambda xy: penalized_flat(xy, alpha=100.0),
            init,
            method='L-BFGS-B',
            jac=lambda xy: penalized_jac_flat(xy, alpha=100.0),
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        x1 = jnp.array(result.x[:n_target])
        y1 = jnp.array(result.x[n_target:])
        aep1 = aep_objective(x1, y1)

        if aep1 < best_aep:
            best_aep = aep1
            best_x, best_y = x1, y1
    except Exception:
        pass

    # ── Method 2: Quick SGD with aggressive settings ──
    try:
        init_x, init_y = get_hex_init(1)

        settings = SGDSettings(
            learning_rate=180.0,
            max_iter=1500,
            additional_constant_lr_iterations=1000,
            tol=1e-6,
            beta1=0.15,
            beta2=0.25,
            gamma_min_factor=0.01,
            ks_rho=120.0,
            spacing_weight=90.0,
            boundary_weight=90.0,
        )

        x2, y2 = topfarm_sgd_solve(aep_objective, init_x, init_y,
                                    boundary, min_spacing, settings)
        aep2 = aep_objective(x2, y2)

        if aep2 < best_aep:
            best_aep = aep2
            best_x, best_y = x2, y2
    except Exception:
        pass

    # ── Method 3: Coarse-to-fine (2 stages, fast) ──
    try:
        init_x, init_y = get_hex_init(2)

        # Coarse
        settings_coarse = SGDSettings(
            learning_rate=120.0,
            max_iter=1000,
            additional_constant_lr_iterations=500,
            tol=1e-6,
            beta1=0.1,
            beta2=0.2,
            gamma_min_factor=0.01,
            ks_rho=80.0,
            spacing_weight=40.0,
            boundary_weight=60.0,
        )

        x3a, y3a = topfarm_sgd_solve(aep_objective, init_x, init_y,
                                      boundary, min_spacing * 0.9, settings_coarse)

        # Fine
        settings_fine = SGDSettings(
            learning_rate=100.0,
            max_iter=1500,
            additional_constant_lr_iterations=1000,
            tol=1e-6,
            beta1=0.1,
            beta2=0.2,
            gamma_min_factor=0.005,
            ks_rho=120.0,
            spacing_weight=100.0,
            boundary_weight=100.0,
        )

        x3, y3 = topfarm_sgd_solve(aep_objective, x3a, y3a,
                                    boundary, min_spacing, settings_fine)
        aep3 = aep_objective(x3, y3)

        if aep3 < best_aep:
            best_aep = aep3
            best_x, best_y = x3, y3
    except Exception:
        pass

    # ── Method 4: Multi-start mini-SGD (3 starts, short runs) ──
    try:
        for seed in [10, 20, 30]:
            init_x, init_y = get_hex_init(seed)

            settings = SGDSettings(
                learning_rate=150.0,
                max_iter=1000,
                additional_constant_lr_iterations=800,
                tol=1e-6,
                beta1=0.12,
                beta2=0.22,
                gamma_min_factor=0.01,
                ks_rho=100.0,
                spacing_weight=80.0,
                boundary_weight=80.0,
            )

            x4, y4 = topfarm_sgd_solve(aep_objective, init_x, init_y,
                                        boundary, min_spacing, settings)
            aep4 = aep_objective(x4, y4)

            if aep4 < best_aep:
                best_aep = aep4
                best_x, best_y = x4, y4
    except Exception:
        pass

    # ── Final polish on best result ──
    if best_x is not None:
        try:
            settings_polish = SGDSettings(
                learning_rate=80.0,
                max_iter=2000,
                additional_constant_lr_iterations=1000,
                tol=1e-7,
                beta1=0.1,
                beta2=0.2,
                gamma_min_factor=0.002,
                ks_rho=150.0,
                spacing_weight=120.0,
                boundary_weight=120.0,
            )

            opt_x, opt_y = topfarm_sgd_solve(aep_objective, best_x, best_y,
                                              boundary, min_spacing, settings_polish)
            return opt_x, opt_y
        except Exception:
            return best_x, best_y
    else:
        # Complete fallback
        init_x, init_y = get_hex_init(0)
        return init_x, init_y
