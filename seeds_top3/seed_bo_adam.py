"""Successive Halving Adam with Aggressive Refinement and BO-tuned Hex Initialization.

HYPOTHESIS: Pushing the Adam refinement steps even further, while maintaining 
a robust BO-tuned initialization, will lead to the highest possible AEP 
within the 60s timeout.

AXIS: Initialization (BO-tuned Sheared Hex Grid) + Search (Successive Halving Adam)

FAMILY: bayesian_optimization
"""

import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.boundary import polygon_sdf

def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # --- Objective ---
    @jax.jit
    def aep_obj(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        aep = jnp.sum(p * weights[:, None]) * 8760 / 1e6
        return -aep

    # --- Hex Grid Generator with Shear ---
    @jax.jit
    def generate_grid(params):
        # params: [sx, sy, theta, ox, oy, shear]
        sx, sy, theta, ox, oy, shear = params
        n_side = int(np.sqrt(n_target)) + 15
        i, j = jnp.meshgrid(jnp.arange(n_side) - n_side//2, jnp.arange(n_side) - n_side//2)
        ix, iy = i.flatten(), j.flatten()
        
        # Base hex pattern
        hx = ix * sx * min_spacing + (iy % 2) * sx * min_spacing / 2.0
        hy = iy * sy * min_spacing * jnp.sqrt(3) / 2.0
        
        # Apply shear
        hx = hx + shear * hy
        
        # Rotate and offset
        rx = hx * jnp.cos(theta) - hy * jnp.sin(theta) + ox
        ry = hx * jnp.sin(theta) + hy * jnp.cos(theta) + oy
        
        sdf_vals = polygon_sdf(rx, ry, boundary)
        idx = jnp.argsort(sdf_vals)[:n_target]
        return rx[idx], ry[idx]

    # --- Bayesian Optimization ---
    def get_aep_grid(params):
        gx, gy = generate_grid(params)
        return -aep_obj(gx, gy)

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    
    bounds_low = jnp.array([1.1, 1.1, 0.0, x_min, y_min, -0.4])
    bounds_high = jnp.array([4.5, 4.5, jnp.pi/3, x_max, y_max, 0.4])
    def scale_params(p): return bounds_low + p * (bounds_high - bounds_low)

    key = jax.random.PRNGKey(42)
    n_init = 16
    key, subkey = jax.random.split(key)
    X_raw = jax.random.uniform(subkey, (n_init, 6))
    Y = jnp.array([get_aep_grid(scale_params(p)) for p in X_raw])

    @jax.jit
    def gp_predict_with_std(X_test, X_train, Y_train, l=0.3):
        def kernel(a, b): return jnp.exp(-jnp.sum((a - b)**2) / (2 * l**2))
        K = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(X_train))(X_train)
        K += jnp.eye(len(X_train)) * 1e-5
        L = jnp.linalg.cholesky(K)
        K_s = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(X_train))(X_test)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Y_train))
        mu = K_s @ alpha
        v = jnp.linalg.solve(L, K_s.T)
        var = jax.vmap(lambda a: kernel(a, a))(X_test) - jnp.sum(v**2, axis=0)
        return mu, jnp.sqrt(jnp.maximum(0.0, var))

    n_bo = 14
    kappa = 2.0
    for _ in range(n_bo):
        key, subkey = jax.random.split(key)
        X_cand = jax.random.uniform(subkey, (800, 6))
        mu_cand, std_cand = gp_predict_with_std(X_cand, X_raw, Y)
        ucb_cand = mu_cand + kappa * std_cand
        x_next = X_cand[jnp.argmax(ucb_cand)]
        y_next = get_aep_grid(scale_params(x_next))
        X_raw = jnp.vstack([X_raw, x_next])
        Y = jnp.append(Y, y_next)

    top_indices = jnp.argsort(Y)[-8:][::-1]
    top_params = X_raw[top_indices]
    
    # --- Precision Projection ---
    @jax.jit
    def project_boundary(x, y):
        def single_sdf(px, py):
            return polygon_sdf(jnp.array([px]), jnp.array([py]), boundary)[0]
        
        def b_step(carry, _):
            x, y = carry
            sdf_val, sdf_grad_tuple = jax.vmap(jax.value_and_grad(single_sdf, argnums=(0, 1)))(x, y)
            gx, gy = sdf_grad_tuple
            nx = x - jnp.maximum(0.0, sdf_val + 0.1) * gx
            ny = y - jnp.maximum(0.0, sdf_val + 0.1) * gy
            return (nx, ny), None
        
        (nx, ny), _ = jax.lax.scan(b_step, (x, y), jnp.arange(4))
        return nx, ny

    @jax.jit
    def project_spacing(x, y):
        def repulsion_step(carry, _):
            x, y = carry
            dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
            force = jnp.maximum(0.0, min_spacing * 1.006 - dist)
            move_x = jnp.sum(force * (dx/dist), axis=1) * 0.12
            move_y = jnp.sum(force * (dy/dist), axis=1) * 0.12
            return (x + move_x, y + move_y), None
        (new_x, new_y), _ = jax.lax.scan(repulsion_step, (x, y), jnp.arange(50))
        return new_x, new_y

    @jax.jit
    def project_all(x, y):
        for _ in range(5):
            x, y = project_boundary(x, y)
            x, y = project_spacing(x, y)
        return project_boundary(x, y)

    # --- Successive Halving Adam ---
    grad_aep_batch = jax.jit(jax.vmap(jax.grad(aep_obj, argnums=(0, 1))))
    project_batch = jax.jit(jax.vmap(project_all))

    def run_adam_batch(x, y, n_steps, lr, decay):
        def step(carry, _):
            x, y, m_x, m_y, v_x, v_y, lr, t = carry
            gx, gy = grad_aep_batch(x, y)
            b1, b2 = 0.9, 0.999
            m_x = b1 * m_x + (1 - b1) * gx
            m_y = b1 * m_y + (1 - b1) * gy
            v_x = b2 * v_x + (1 - b2) * gx**2
            v_y = b2 * v_y + (1 - b2) * gy**2
            m_x_h = m_x / (1 - b1**(t+1))
            m_y_h = m_y / (1 - b1**(t+1))
            v_x_h = v_x / (1 - b2**(t+1))
            v_y_h = v_y / (1 - b2**(t+1))
            nx = x - lr * m_x_h / (jnp.sqrt(v_x_h) + 1e-8)
            ny = y - lr * m_y_h / (jnp.sqrt(v_y_h) + 1e-8)
            nx, ny = project_batch(nx, ny)
            return (nx, ny, m_x, m_y, v_x, v_y, lr * decay, t + 1), None

        m_x, m_y = jnp.zeros_like(x), jnp.zeros_like(y)
        v_x, v_y = jnp.zeros_like(x), jnp.zeros_like(y)
        init_carry = (x, y, m_x, m_y, v_x, v_y, lr, 0)
        (fx, fy, _, _, _, _, _, _), _ = jax.lax.scan(step, init_carry, jnp.arange(n_steps))
        return fx, fy

    layouts = [generate_grid(scale_params(p)) for p in top_params]
    cur_x = jnp.stack([l[0] for l in layouts])
    cur_y = jnp.stack([l[1] for l in layouts])

    # 8 -> 4
    cur_x, cur_y = run_adam_batch(cur_x, cur_y, 400, 12.0, 0.998)
    e = jax.vmap(aep_obj)(cur_x, cur_y)
    idx = jnp.argsort(e)[:4]
    cur_x, cur_y = cur_x[idx], cur_y[idx]

    # 4 -> 2
    cur_x, cur_y = run_adam_batch(cur_x, cur_y, 1000, 8.0, 0.999)
    e = jax.vmap(aep_obj)(cur_x, cur_y)
    idx = jnp.argsort(e)[:2]
    cur_x, cur_y = cur_x[idx], cur_y[idx]

    # 2 -> 1
    cur_x, cur_y = run_adam_batch(cur_x, cur_y, 2000, 4.0, 0.9995)
    e = jax.vmap(aep_obj)(cur_x, cur_y)
    idx = jnp.argmin(e)
    
    # Final refinement
    best_x, best_y = cur_x[idx:idx+1], cur_y[idx:idx+1]
    best_x, best_y = run_adam_batch(best_x, best_y, 2000, 1.2, 0.9998)
    
    return best_x[0], best_y[0]
