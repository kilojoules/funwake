import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR with Warm Restarts (Multi-Cycle Cosine) ---
    # 3 cycles of decreasing length and max learning rate.
    # Allows multiple chances to escape local minima while progressively 
    # narrowing the search space and forcing feasibility at each cycle's end.
    
    warmup_end = 0.05
    c1_end = 0.40
    c2_end = 0.75
    c3_end = 0.90
    
    is_c1 = progress < c1_end
    is_c2 = (progress >= c1_end) & (progress < c2_end)
    is_c3 = (progress >= c2_end) & (progress < c3_end)
    is_term = progress >= c3_end
    
    # Progress within the current cycle (0.0 to 1.0)
    # Cycle 1 delays its cosine decay until after the warmup period
    c1_prog = jnp.maximum(0.0, (progress - warmup_end) / (c1_end - warmup_end))
    c2_prog = (progress - c1_end) / (c2_end - c1_end)
    c3_prog = (progress - c2_end) / (c3_end - c2_end)
    
    cycle_prog = jnp.where(is_c1, c1_prog,
                 jnp.where(is_c2, c2_prog,
                 jnp.where(is_c3, c3_prog, 1.0)))
                 
    # Standard cosine decay within the cycle (from 1.0 down to 0.0)
    cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_prog))
    
    # Peak exploration rates shrink per cycle to encourage convergence
    lr_max_c1 = 1.50 * D_f
    lr_max_c2 = 0.60 * D_f
    lr_max_c3 = 0.15 * D_f
    
    lr_max = jnp.where(is_c1, lr_max_c1, 
               jnp.where(is_c2, lr_max_c2, lr_max_c3))
               
    # Linear warmup only applies during the very beginning
    is_warmup = progress < warmup_end
    lr_warmup = gamma_min_f + (lr_max_c1 - gamma_min_f) * (progress / warmup_end)
    lr_cycle = gamma_min_f + (lr_max - gamma_min_f) * cosine_decay
    
    lr_main = jnp.where(is_warmup, lr_warmup, lr_cycle)
    lr = jnp.where(is_term, gamma_min_f, lr_main)

    # --- 2. Cyclical, Progressively Stricter Alpha ---
    # Sync alpha with the learning rate cycles: when lr is high (exploration),
    # alpha drops to allow constraint violation. As lr cools within each cycle,
    # alpha rises to force feasibility ("mid-run feasibility-restoration bursts").
    # Across cycles, the penalty bands become progressively stricter.
    
    alpha_min_c1 = alpha0 * 0.1
    alpha_max_c1 = alpha0 * 5.0
    
    alpha_min_c2 = alpha0 * 0.5
    alpha_max_c2 = alpha0 * 15.0
    
    alpha_min_c3 = alpha0 * 2.0
    alpha_max_c3 = alpha0 * 30.0
    
    alpha_min = jnp.where(is_c1, alpha_min_c1, 
                  jnp.where(is_c2, alpha_min_c2, alpha_min_c3))
    alpha_max = jnp.where(is_c1, alpha_max_c1, 
                  jnp.where(is_c2, alpha_max_c2, alpha_max_c3))
                  
    # Alpha rises as cosine_decay falls (from min to max over the cycle)
    alpha_main = alpha_max - (alpha_max - alpha_min) * cosine_decay
    
    # Terminal feasibility spike to guarantee constraints at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_term, alpha_terminal, alpha_main)

    # --- 3. Synchronized Adam Moments ---
    # High momentum/low beta2 for exploration (start of cycle, cosine_decay~1)
    # Low momentum/high beta2 for feasibility restoration (end of cycle, cosine_decay~0)
    b1_exp, b2_exp = 0.12, 0.15
    b1_feas, b2_feas = 0.04, 0.85
    
    beta1_main = b1_feas + (b1_exp - b1_feas) * cosine_decay
    beta2_main = b2_feas + (b2_exp - b2_feas) * cosine_decay
    
    # Extreme damping during terminal phase
    beta1 = jnp.where(is_term, 0.01, beta1_main)
    beta2 = jnp.where(is_term, 0.99, beta2_main)

    return lr, alpha, beta1, beta2