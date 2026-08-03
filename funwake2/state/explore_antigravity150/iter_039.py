import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR Multi-Cycle Cosine Learning Rate ---
    # Three cycles of decreasing length. Warm restarts jolt the layout out of 
    # local optima, while cosine annealing allows it to settle into a new basin.
    c1_end = 0.45
    c2_end = 0.75
    c3_end = 0.90
    
    in_c1 = progress < c1_end
    in_c2 = (progress >= c1_end) & (progress < c2_end)
    in_c3 = (progress >= c2_end) & (progress < c3_end)
    is_terminal = progress >= c3_end
    
    p_c1 = progress / c1_end
    p_c2 = (progress - c1_end) / (c2_end - c1_end)
    p_c3 = (progress - c2_end) / (c3_end - c2_end)
    
    cycle_p = jnp.where(in_c1, p_c1,
              jnp.where(in_c2, p_c2,
              jnp.where(in_c3, p_c3, 1.0)))
              
    lr_peak = jnp.where(in_c1, 1.50 * D_f,
              jnp.where(in_c2, 0.75 * D_f,
              jnp.where(in_c3, 0.30 * D_f, gamma_min_f)))
              
    # Cosine annealing within cycle
    lr_main = gamma_min_f + 0.5 * (lr_peak - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_p))
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Cyclic & Ratcheting Alpha (Penalty) ---
    # Alpha drops at the start of each cycle to permit exploration, then 
    # ramps up to enforce feasibility. Both the floor and ceiling of the 
    # penalty ratchet upwards with each successive cycle to guarantee 
    # we eventually trap the layout in a feasible state.
    
    alpha_base = jnp.where(in_c1, alpha0 * 0.1,
                 jnp.where(in_c2, alpha0 * 1.0,
                 jnp.where(in_c3, alpha0 * 5.0, alpha0)))
                 
    alpha_peak = jnp.where(in_c1, alpha0 * 3.0,
                 jnp.where(in_c2, alpha0 * 12.0,
                 jnp.where(in_c3, alpha0 * 40.0, alpha0)))
                 
    alpha_cycle = alpha_base + 0.5 * (alpha_peak - alpha_base) * (1.0 - jnp.cos(jnp.pi * cycle_p))
    
    # Terminal feasibility spike ensures absolute compliance at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)

    # --- 3. Cyclic Adam Moments ---
    # Synchronize moments with the cycles:
    # High momentum (low beta2) at restart to ride the explosive LR.
    # Drop momentum (high beta2) as cycle ends to absorb constraint curvature.
    b1_start, b2_start = 0.12, 0.15
    b1_end, b2_end = 0.02, 0.85
    
    t_beta = 0.5 * (1.0 - jnp.cos(jnp.pi * cycle_p))
    
    beta1_main = b1_start + (b1_end - b1_start) * t_beta
    beta2_main = b2_start + (b2_end - b2_start) * t_beta
    
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2