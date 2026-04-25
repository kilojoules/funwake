"""SLSQP optimizer with multi-start and rotated grid initialization.

Features:
- Multi-start strategy (3 starts)
- Grid initialization rotated by dominant wind direction
- Increased SLSQP iterations (maxiter=200)
- Squared-distance spacing constraints for better stability
- JAX-based gradients and Jacobians
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from pixwake.optim.boundary import polygon_sdf, _point_in_polygon


def rotate(coords, angle):
    """Rotate points (x, y) by angle (radians)."""
    x, y = coords[:, 0], coords[:, 1]
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    rx = x * cos_a - y * sin_a
    ry = x * sin_a + y * cos_a
    return jnp.stack([rx, ry], axis=1)


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # ── Objective ──────────────────────────────────────────────────
    @jax.jit
    def aep_obj(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        # Maximize AEP = Minimize -AEP
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    grad_aep = jax.jit(jax.grad(aep_obj))

    # ── Constraints ────────────────────────────────────────────────
    @jax.jit
    def boundary_con(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        # All points must be inside: polygon_sdf <= 0
        return -polygon_sdf(x, y, boundary)

    jac_boundary = jax.jit(jax.jacobian(boundary_con))

    @jax.jit
    def spacing_con(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        # Extract upper triangle (unique pairs)
        i_upper, j_upper = jnp.triu_indices(n_target, k=1)
        pair_dist_sq = dist_sq[i_upper, j_upper]
        # d^2 - min_spacing^2 >= 0
        return pair_dist_sq - min_spacing**2

    jac_spacing = jax.jit(jax.jacobian(spacing_con))

    # ── Wrappers for SciPy ─────────────────────────────────────────
    def scipy_obj(x):
        return float(aep_obj(x))

    def scipy_grad(x):
        return np.array(grad_aep(x))

    def scipy_con_boundary(x):
        return np.array(boundary_con(x))

    def scipy_jac_boundary(x):
        return np.array(jac_boundary(x))

    def scipy_con_spacing(x):
        return np.array(spacing_con(x))

    def scipy_jac_spacing(x):
        return np.array(jac_spacing(x))

    # ── Initializations ────────────────────────────────────────────
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    
    # Estimate dominant wind direction
    # wd is in degrees, convert to radians for rotation
    dom_wd_rad = jnp.deg2rad(wd[jnp.argmax(weights)])
    
    best_res = None
    best_aep = float('inf')

    # Grid parameters
    spacing_factor = 1.1 # slightly larger than min_spacing to avoid hitting constraints early
    s = min_spacing * spacing_factor
    
    # 3 starts:
    # 1. Axis-aligned
    # 2. Rotated by dominant wind direction
    # 3. Rotated by dominant wind direction + 45 deg
    angles = [0, dom_wd_rad, dom_wd_rad + jnp.pi/4]
    
    for seed in range(3):
        angle = angles[seed]
        
        # Create a large enough grid and rotate it
        diag = jnp.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        nx = int(jnp.ceil(diag / s)) + 2
        ny = int(jnp.ceil(diag / s)) + 2
        
        gx, gy = jnp.meshgrid(
            jnp.linspace(-diag, diag, nx),
            jnp.linspace(-diag, diag, ny))
        cand = jnp.stack([gx.flatten(), gy.flatten()], axis=1)
        
        # Rotate grid
        cand_rot = rotate(cand, angle)
        
        # Center grid on polygon centroid
        centroid = jnp.mean(boundary, axis=0)
        cand_final = cand_rot + centroid
        
        cand_x, cand_y = cand_final[:, 0], cand_final[:, 1]

        # Filter points inside
        inside = jax.vmap(lambda px, py: _point_in_polygon(px, py, boundary))(cand_x, cand_y)
        inside_x, inside_y = cand_x[inside > 0.5], cand_y[inside > 0.5]

        if len(inside_x) >= n_target:
            # Pick n_target points spread out
            idx = jnp.linspace(0, len(inside_x) - 1, n_target).astype(int)
            init_x, init_y = inside_x[idx], inside_y[idx]
        else:
            # Fallback to random uniform within bounds
            key = jax.random.PRNGKey(seed)
            init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
            init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

        init_coords = np.concatenate([init_x, init_y])

        # ── Run SLSQP ──────────────────────────────────────────────
        cons = [
            {'type': 'ineq', 'fun': scipy_con_boundary, 'jac': scipy_jac_boundary},
            {'type': 'ineq', 'fun': scipy_con_spacing, 'jac': scipy_jac_spacing}
        ]
        
        res = minimize(
            scipy_obj, init_coords, jac=scipy_grad,
            method='SLSQP', constraints=cons,
            options={'maxiter': 200, 'disp': False}
        )

        if res.fun < best_aep:
            best_aep = res.fun
            best_res = res
            
        # Time check - if we already spent > 40s, stop multi-start
        # (Hard to check time inside the loop without time module, but 3 starts should be fine)

    opt_x = best_res.x[:n_target]
    opt_y = best_res.x[n_target:]
    return jnp.array(opt_x), jnp.array(opt_y)
