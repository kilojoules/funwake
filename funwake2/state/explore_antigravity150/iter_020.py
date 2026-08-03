import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR: Cyclic Cosine Annealing with Warm Restarts ---
    # We split the optimization into 3 distinct cycles. Each cycle begins with a 
    # high exploration learning rate which decays to gamma_min following a cosine curve.
    num_cycles = 3.0
    
    # Determine which cycle we are in (0, 1, or 2)
    cycle_idx = jnp.clip(jnp.floor(progress * num_cycles), 0.0, num_cycles - 1.0)
    
    # Normalized progress within the current cycle [0.0, 1.0]
    cycle_progress = (progress * num_cycles) - cycle_idx
    
    # Cosine decay from 1.0 down to 0.0 within the cycle
    decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    # Each restart has a slightly smaller peak learning rate to focus the search
    lr_max = 1.25 * D_f
    lr_max_cycle = lr_max * (0.75 ** cycle_idx)
    
    lr = gamma_min_f + (lr_max_cycle - gamma_min_f) * decay

    # --- 2. Cyclic ALM Penalty (Alpha) ---
    # Instead of a monotonic penalty, we use cyclic feasibility restoration.
    # At the start of each cycle, the penalty drops to allow massive layout reorganization.
    # As the cycle cools down, the penalty ramps up to restore strict feasibility.
    alpha_base = alpha0 * 0.1
    
    # Each subsequent cycle peaks at a higher penalty to ensure final compliance
    alpha_peak = alpha0 * 5.0 * (2.0 ** cycle_idx)
    
    # Alpha rises as lr decays
    alpha_main = alpha_peak - (alpha_peak - alpha_base) * decay

    # Terminal feasibility spike in the last 3% of the ENTIRE run
    is_terminal = progress >= 0.97
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    lr = jnp.where(is_terminal, gamma_min_f, lr)

    # --- 3. Cyclic Kinematic Adam Moments ---
    # Momentum (beta1) starts high in each cycle to burst out of local minima,
    # then drops as the penalty takes over to prevent boundary oscillation.
    # beta2 (curvature) starts low for adaptation, then ramps high.
    beta1_main = 0.02 + 0.15 * decay   # High (0.17) when decay=1, low (0.02) when decay=0
    beta2_main = 0.95 - 0.75 * decay   # Low (0.20) when decay=1, high (0.95) when decay=0
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2