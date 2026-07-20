"""Pruning-Augmented AL-NAdam with BO-tuned Surplus Grid.

HYPOTHESIS: Optimizing a surplus of turbines (n_target + 8) and then pruning 
down to n_target allows the optimizer to discover more efficient subsets 
of the grid that better align with the boundary and wind rose, 
effectively bypassing some local optima.

AXIS: Initialization (Surplus BO Grid) + Search (AL-NAdam + Pruning)

FAMILY: BayesianOptimization, constraint_augmented_lagrangian, nesterov_momentum
"""

import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.boundary import polygon_sdf

def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    n_extra = 8
    n_total = n_target + n_extra

    # --- Objective (surplus) ---
    @jax.jit
    def aep_obj_total(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power() # (n_findex, n_total)
        aep = jnp.sum(p * weights[:, None]) * 8760 / 1e6
        return -aep

    # --- Wind-Aware Initialization ---
    wd_rad = jnp.deg2rad(wd)
    v_x = jnp.sum(jnp.cos(wd_rad) * weights)
    v_y = jnp.sum(jnp.sin(wd_rad) * weights)
    dominant_wd = jnp.arctan2(v_y, v_x)

    @jax.jit
    def generate_grid_surplus(params):
        # params: [sx, sy, theta_off, ox, oy, shear, aspect]
        sx, sy, theta_off, ox, oy, shear, aspect = params
        theta = dominant_wd + theta_off
        sy_actual = sy * aspect
        n_side = int(np.sqrt(n_total)) + 15
        i, j = jnp.meshgrid(jnp.arange(n_side) - n_side//2, jnp.arange(n_side) - n_side//2)
        ix, iy = i.flatten(), j.flatten()
        hx = ix * sx * min_spacing + (iy % 2) * sx * min_spacing / 2.0
        hy = iy * sy_actual * min_spacing * jnp.sqrt(3) / 2.0
        hx = hx + shear * hy
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        sdf_vals = polygon_sdf(rx, ry, boundary)
        idx = jnp.argsort(sdf_vals)[:n_total]
        return rx[idx], ry[idx]

    # --- BO to find best surplus starting points ---
    def get_aep_grid(params):
        gx, gy = generate_grid_surplus(params)
        return -aep_obj_total(gx, gy)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    bounds_low = jnp.array([1.02, 1.02, -jnp.pi/4, x_min, y_min, -0.5, 0.85])
    bounds_high = jnp.array([4.5, 4.5, jnp.pi/4, x_max, y_max, 0.5, 1.2])
    def scale_params(p): return bounds_low + p * (bounds_high - bounds_low)

    key = jax.random.PRNGKey(777)
    n_init = 10
    key, subkey = jax.random.split(key)
    X_raw = jax.random.uniform(subkey, (n_init, 7))
    Y = jnp.array([get_aep_grid(scale_params(p)) for p in X_raw])

    @jax.jit
    def gp_predict(X_test, X_train, Y_train, l=0.3):
        def kernel(a, b):
            d = jnp.sqrt(jnp.sum((a - b)**2) + 1e-8)
            return (1.0 + jnp.sqrt(5.0) * d / l + 5.0 * d**2 / (3.0 * l**2)) * jnp.exp(-jnp.sqrt(5.0) * d / l)
        K = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(X_train))(X_train)
        K += jnp.eye(len(X_train)) * 1e-5
        L = jnp.linalg.cholesky(K)
        K_s = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(X_train))(X_test)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Y_train))
        mu = K_s @ alpha
        v = jnp.linalg.solve(L, K_s.T)
        var = jax.vmap(lambda a: kernel(a, a))(X_test) - jnp.sum(v**2, axis=0)
        return mu, jnp.sqrt(jnp.maximum(1e-9, var))

    for _ in range(15):
        key, subkey = jax.random.split(key)
        X_cand = jax.random.uniform(subkey, (600, 7))
        mu, std = gp_predict(X_cand, X_raw, Y)
        f_best = jnp.max(Y)
        z = (mu - f_best) / std
        ei = (mu - f_best) * (0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))) + std * (jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi))
        x_next = X_cand[jnp.argmax(ei)]
        y_next = get_aep_grid(scale_params(x_next))
        X_raw = jnp.vstack([X_raw, x_next])
        Y = jnp.append(Y, y_next)

    top_indices = jnp.argsort(Y)[-8:][::-1]
    top_params = X_raw[top_indices]

    # --- Augmented Lagrangian for n_total ---
    @jax.jit
    def constraints_penalty(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        v_b = jnp.maximum(0.0, sdf + 0.01)
        pen_b = jnp.sum(lam_b * v_b + 0.5 * mu * v_b**2)
        dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_total, n_total)), k=1)
        v_s = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        pen_s = jnp.sum(mask * (lam_s * v_s + 0.5 * mu * v_s**2))
        return pen_b + pen_s

    @jax.jit
    def total_obj(x, y, lam_b, lam_s, mu):
        return aep_obj_total(x, y) + constraints_penalty(x, y, lam_b, lam_s, mu)

    grad_total = jax.jit(jax.vmap(jax.grad(total_obj, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None)))

    def run_al_nadam_surplus(x, y, n_steps, lr):
        m_x, m_y = jnp.zeros_like(x), jnp.zeros_like(y)
        v_x, v_y = jnp.zeros_like(x), jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_total, n_total))
        mu = 100.0
        
        curr_x, curr_y = x, y
        for i in range(4):
            def step(carry, t):
                cx, cy, mx, my, vx, vy = carry
                gx, gy = grad_total(cx, cy, lam_b, lam_s, mu)
                b1, b2 = 0.9, 0.999
                mx = b1 * mx + (1 - b1) * gx
                my = b1 * my + (1 - b1) * gy
                vx = b2 * vx + (1 - b2) * gx**2
                vy = b2 * vy + (1 - b2) * gy**2
                mx_h = (b1 * mx + (1 - b1) * gx) / (1 - b1**(t+1))
                my_h = (b1 * my + (1 - b1) * gy) / (1 - b1**(t+1))
                vx_h = vx / (1 - b2**(t+1))
                vy_h = vy / (1 - b2**(t+1))
                nx = cx - lr * mx_h / (jnp.sqrt(vx_h) + 1e-8)
                ny = cy - lr * my_h / (jnp.sqrt(vy_h) + 1e-8)
                return (nx, ny, mx, my, vx, vy), None

            init_carry = (curr_x, curr_y, m_x, m_y, v_x, v_y)
            (curr_x, curr_y, m_x, m_y, v_x, v_y), _ = jax.lax.scan(step, init_carry, jnp.arange(i*200, (i+1)*200))
            
            # Update AL (simplified: individual updates)
            sdf = jax.vmap(lambda px, py: polygon_sdf(px, py, boundary))(curr_x, curr_y)
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx, dy = curr_x[:, :, None] - curr_x[:, None, :], curr_y[:, :, None] - curr_y[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= 2.0
        return curr_x, curr_y

    # --- Refine and Select ---
    layouts = [generate_grid_surplus(scale_params(p)) for p in top_params]
    cur_x = jnp.stack([l[0] for l in layouts])
    cur_y = jnp.stack([l[1] for l in layouts])

    cur_x, cur_y = run_al_nadam_surplus(cur_x, cur_y, 800, 12.0)
    
    # Prune each layout to n_target
    def prune(x, y):
        # Individual contribution
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()
        indiv_aep = jnp.sum(p * weights[:, None], axis=0)
        idx = jnp.argsort(indiv_aep)[-n_target:]
        return x[idx], y[idx]

    pruned = [prune(cur_x[i], cur_y[i]) for i in range(len(cur_x))]
    px = jnp.stack([l[0] for l in pruned])
    py = jnp.stack([l[1] for l in pruned])

    # Final refinement on pruned layouts
    @jax.jit
    def aep_obj_final(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        aep = jnp.sum(p * weights[:, None]) * 8760 / 1e6
        return -aep

    @jax.jit
    def total_obj_final(x, y, lam_b, lam_s, mu):
        sdf = polygon_sdf(x, y, boundary)
        v_b = jnp.maximum(0.0, sdf + 0.01)
        pen_b = jnp.sum(lam_b * v_b + 0.5 * mu * v_b**2)
        dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        mask = jnp.triu(jnp.ones((n_target, n_target)), k=1)
        v_s = jnp.maximum(0.0, min_spacing * 1.001 - dist)
        pen_s = jnp.sum(mask * (lam_s * v_s + 0.5 * mu * v_s**2))
        return aep_obj_final(x, y) + pen_b + pen_s

    grad_final = jax.jit(jax.vmap(jax.grad(total_obj_final, argnums=(0, 1)), in_axes=(0, 0, 0, 0, None)))

    def run_al_nadam_final(x, y, n_steps, lr):
        m_x, m_y = jnp.zeros_like(x), jnp.zeros_like(y)
        v_x, v_y = jnp.zeros_like(x), jnp.zeros_like(y)
        lam_b = jnp.zeros_like(x)
        lam_s = jnp.zeros((x.shape[0], n_target, n_target))
        mu = 1000.0
        curr_x, curr_y = x, y
        for i in range(3):
            def step(carry, t):
                cx, cy, mx, my, vx, vy = carry
                gx, gy = grad_final(cx, cy, lam_b, lam_s, mu)
                b1, b2 = 0.9, 0.999
                mx = b1 * mx + (1 - b1) * gx
                my = b1 * my + (1 - b1) * gy
                vx = b2 * vx + (1 - b2) * gx**2
                vy = b2 * vy + (1 - b2) * gy**2
                mx_h = (b1 * mx + (1 - b1) * gx) / (1 - b1**(t+1))
                my_h = (b1 * my + (1 - b1) * gy) / (1 - b1**(t+1))
                vx_h = vx / (1 - b2**(t+1))
                vy_h = vy / (1 - b2**(t+1))
                nx = cx - lr * mx_h / (jnp.sqrt(vx_h) + 1e-8)
                ny = cy - lr * my_h / (jnp.sqrt(vy_h) + 1e-8)
                return (nx, ny, mx, my, vx, vy), None
            init_carry = (curr_x, curr_y, m_x, m_y, v_x, v_y)
            (curr_x, curr_y, m_x, m_y, v_x, v_y), _ = jax.lax.scan(step, init_carry, jnp.arange(i*400, (i+1)*400))
            sdf = jax.vmap(lambda pxx, pyy: polygon_sdf(pxx, pyy, boundary))(curr_x, curr_y)
            lam_b = lam_b + mu * jnp.maximum(0.0, sdf + 0.01)
            dx, dy = curr_x[:, :, None] - curr_x[:, None, :], curr_y[:, :, None] - curr_y[:, None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            lam_s = lam_s + mu * jnp.maximum(0.0, min_spacing * 1.001 - dist)
            mu *= 3.0
        return curr_x, curr_y

    # Filter to top 2 before final
    e = jax.vmap(aep_obj_final)(px, py)
    idx = jnp.argsort(e)[:2]
    px, py = px[idx], py[idx]
    
    fx, fy = run_al_nadam_final(px, py, 1200, 4.0)
    best_idx = jnp.argmin(jax.vmap(aep_obj_final)(fx, fy))
    final_x, final_y = fx[best_idx], fy[best_idx]

    # Strict final projection
    def project(x, y):
        for _ in range(25):
            sdf = polygon_sdf(x, y, boundary)
            grad_b = jax.vmap(jax.grad(lambda px, py: polygon_sdf(jnp.array([px]), jnp.array([py]), boundary)[0], argnums=(0, 1)))(x, y)
            x = x - jnp.maximum(0.0, sdf + 0.02) * grad_b[0]
            y = y - jnp.maximum(0.0, sdf + 0.02) * grad_b[1]
            dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            force = jnp.maximum(0.0, min_spacing * 1.002 - dist)
            x += jnp.sum(force * (dx/dist), axis=1) * 0.15
            y += jnp.sum(force * (dy/dist), axis=1) * 0.15
        return x, y

    rx, ry = project(final_x, final_y)
    return rx, ry
