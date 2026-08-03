import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Multi-Cycle SGDR (Warm Restarts) ---
    # Instead of a single WSD phase, we use 3 cycles of cosine annealing.
    # Each cycle starts with a high LR (exploration) and decays.
    # The restarts bounce the layout out of sub-optimal local minima.
    c1_end = 0.40
    c2_end = 0.75
    c3_end = 0.90
    
    in_c1 = progress < c1_end
    in_c2 = (progress >= c1_end) & (progress < c2_end)
    in_c3 = (progress >= c2_end) & (progress < c3_end)
    is_terminal = progress >= c3_end
    
    # Normalize progress within each cycle to [0, 1]
    cycle_prog = jnp.where(in_c1, progress / c1_end,
                 jnp.where(in_c2, (progress - c1_end) / (c2_end - c1_end),
                 jnp.where(in_c3, (progress - c2_end) / (c3_end - c2_end), 1.0)))
    
    # Decaying peak learning rates for successive cycles
    lr_max_c1 = 1.50 * D_f
    lr_max_c2 = 0.75 * D_f
    lr_max_c3 = 0.25 * D_f
    
    lr_max = jnp.where(in_c1, lr_max_c1,
             jnp.where(in_c2, lr_max_c2, lr_max_c3))
             
    # Cosine annealing within each cycle
    lr_main = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_prog))
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Cyclic Alpha (Coupled with SGDR) ---
    # At each restart, alpha drops to allow layout rearrangement.
    # It then ramps up logistically to a cycle-specific plateau to enforce feasibility locally
    # before the next cycle begins.
    
    alpha_base_c1 = alpha0 * 0.1
    alpha_base_c2 = alpha0 * 0.5
    alpha_base_c3 = alpha0 * 2.0
    
    alpha_plat_c1 = alpha0 * 5.0
    alpha_plat_c2 = alpha0 * 15.0
    alpha_plat_c3 = alpha0 * 30.0
    
    alpha_base = jnp.where(in_c1, alpha_base_c1,
                 jnp.where(in_c2, alpha_base_c2, alpha_base_c3))
                 
    alpha_plat = jnp.where(in_c1, alpha_plat_c1,
                 jnp.where(in_c2, alpha_plat_c2, alpha_plat_c3))
                 
    # Logistic ramp centered mid-cycle to smoothly phase in the penalty
    k = 15.0
    logistic_ramp = 1.0 / (1.0 + jnp.exp(-k * (cycle_prog - 0.5)))
    
    alpha_main = alpha_base + (alpha_plat - alpha_base) * logistic_ramp
    
    # Terminal feasibility spike (Filter method) ensures strict compliance at the very end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Cyclic Adam Moments ---
    # Low beta2 (high momentum) at cycle starts for exploration.
    # High beta2 (damping) at cycle ends as penalty kicks in to absorb constraint curvature.
    b1_start = 0.15
    b1_end = 0.02
    b2_start = 0.20
    b2_end = 0.90
    
    beta1_main = b1_start + (b1_end - b1_start) * logistic_ramp
    beta2_main = b2_start + (b2_end - b2_start) * logistic_ramp
    
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2