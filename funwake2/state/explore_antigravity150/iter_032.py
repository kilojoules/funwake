import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Multi-cycle SGDR Learning Rate ---
    # 3 cycles with increasing lengths (T_mult = 2). 
    # Cycle fractions: 1/7, 2/7, 4/7 of total steps.
    p_c1 = 1.0 / 7.0
    p_c2 = 3.0 / 7.0
    
    cycle_idx = jnp.where(progress < p_c1, 0.0,
                  jnp.where(progress < p_c2, 1.0, 2.0))
                  
    cycle_start = jnp.where(progress < p_c1, 0.0,
                    jnp.where(progress < p_c2, p_c1, p_c2))
                    
    cycle_len = jnp.where(progress < p_c1, p_c1,
                  jnp.where(progress < p_c2, 2.0 / 7.0, 4.0 / 7.0))
                  
    cycle_progress = (progress - cycle_start) / cycle_len
    # Bound cycle_progress to avoid numerical issues exactly at boundaries
    cycle_progress = jnp.clip(cycle_progress, 0.0, 1.0)
    
    # Warm restarts: massive initial exploration, cooling down in later cycles
    lr_max_0 = 1.5 * D_f
    lr_max_1 = 0.8 * D_f
    lr_max_2 = 0.3 * D_f
    
    lr_max = jnp.where(cycle_idx == 0.0, lr_max_0,
               jnp.where(cycle_idx == 1.0, lr_max_1, lr_max_2))
               
    # Cosine annealing within each cycle
    lr_main = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_progress))

    # --- 2. Cyclic Alpha with Mid-Run Feasibility Bursts ---
    # Decoupled penalty: drops at the start of each cycle to permit layout exploration,
    # then spikes at the end to restore feasibility (mid-run feasibility-restoration).
    alpha_base_0 = alpha0 * 0.05
    alpha_base_1 = alpha0 * 0.20
    alpha_base_2 = alpha0 * 1.00
    
    alpha_burst_0 = alpha0 * 2.0
    alpha_burst_1 = alpha0 * 8.0
    alpha_burst_2 = alpha0 * 25.0
    
    alpha_base = jnp.where(cycle_idx == 0.0, alpha_base_0,
                   jnp.where(cycle_idx == 1.0, alpha_base_1, alpha_base_2))
                   
    alpha_burst = jnp.where(cycle_idx == 0.0, alpha_burst_0,
                    jnp.where(cycle_idx == 1.0, alpha_burst_1, alpha_burst_2))
                    
    # Sharply ramp penalty toward the end of each cycle
    alpha_main = alpha_base + (alpha_burst - alpha_base) * (cycle_progress ** 4.0)

    # --- 3. Cyclic Adam Moments ---
    # beta1 decreases (momentum drops) during feasibility bursts
    # beta2 increases during bursts to dampen constraint-curvature oscillations
    b1_base, b1_burst = 0.15, 0.04
    b2_base, b2_burst = 0.15, 0.90
    
    beta1_main = b1_base + (b1_burst - b1_base) * (cycle_progress ** 2.0)
    beta2_main = b2_base + (b2_burst - b2_base) * (cycle_progress ** 2.0)

    # --- 4. Terminal Feasibility Spike ---
    # The last 5% enforces absolute compliance and convergence.
    is_terminal = progress >= 0.95
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2