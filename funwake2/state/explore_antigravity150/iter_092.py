import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- Structurally Different Approach: SGDR with Cyclic Penalty ---
    # We use a 3-cycle Cosine Annealing with Warm Restarts (SGDR) approach.
    # Each cycle starts with a high LR and soft penalty to massively perturb 
    # the layout and escape poor local minima. As the cycle progresses, LR decays 
    # to gamma_min via a cosine curve, while the penalty quadratically ramps up 
    # to enforce constraints. We do this 3 times with diminishing peak amplitudes.
    
    c1_end = 0.40
    c2_end = 0.75
    c3_end = 0.92  # 0.92 to 1.00 is reserved for the pure terminal feasibility spike
    
    # Local progress within each cycle [0.0, 1.0]
    p1 = jnp.clip(progress / c1_end, 0.0, 1.0)
    p2 = jnp.clip((progress - c1_end) / (c2_end - c1_end), 0.0, 1.0)
    p3 = jnp.clip((progress - c2_end) / (c3_end - c2_end), 0.0, 1.0)
    
    # 1. Cyclic Learning Rate (Cosine Decay per cycle)
    lr_max1 = 1.25 * D_f  # Massive initial exploration
    lr_max2 = 0.60 * D_f  # Moderate secondary search
    lr_max3 = 0.20 * D_f  # Local fine-tuning
    
    lr1 = gamma_min_f + 0.5 * (lr_max1 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p1))
    lr2 = gamma_min_f + 0.5 * (lr_max2 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p2))
    lr3 = gamma_min_f + 0.5 * (lr_max3 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p3))
    
    lr_cyclic = jnp.where(progress < c1_end, lr1,
                  jnp.where(progress < c2_end, lr2, lr3))
                  
    # 2. Cyclic Alpha (Quadratic penalty ramp per cycle)
    # Allows "breathing": relax constraints -> explore -> lock down -> repeat.
    # Each successive cycle enforces a higher terminal penalty plateau.
    a1 = alpha0_f * (0.1 + (5.0 - 0.1) * (p1 ** 2.0))
    a2 = alpha0_f * (1.0 + (15.0 - 1.0) * (p2 ** 2.0))
    a3 = alpha0_f * (5.0 + (30.0 - 5.0) * (p3 ** 2.0))
    
    alpha_cyclic = jnp.where(progress < c1_end, a1,
                     jnp.where(progress < c2_end, a2, a3))
                     
    # 3. Cyclic Adam Moments
    # Synchronize momentum with the cycles: low betas at the start of a cycle
    # (high momentum/volatility) and standard high-damping betas at the end.
    b1_start, b1_end = 0.20, 0.04
    b2_start, b2_end = 0.15, 0.90
    
    b1_1 = b1_start + (b1_end - b1_start) * p1
    b1_2 = b1_start + (b1_end - b1_start) * p2
    b1_3 = b1_start + (b1_end - b1_start) * p3
    
    b2_1 = b2_start + (b2_end - b2_start) * p1
    b2_2 = b2_start + (b2_end - b2_start) * p2
    b2_3 = b2_start + (b2_end - b2_start) * p3
    
    beta1_cyclic = jnp.where(progress < c1_end, b1_1,
                     jnp.where(progress < c2_end, b1_2, b1_3))
    beta2_cyclic = jnp.where(progress < c1_end, b2_1,
                     jnp.where(progress < c2_end, b2_2, b2_3))
                     
    # --- 4. Terminal Feasibility Spike ---
    # Absolute strict compliance guaranteed in the final 8% of the optimization.
    is_terminal = progress >= c3_end
    
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cyclic)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyclic)
    beta1 = jnp.where(is_terminal, 0.01, beta1_cyclic)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cyclic)
    
    return lr, alpha, beta1, beta2