import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR with Cyclic Alpha ---
    # We use 3 cycles of cosine annealing for the learning rate (SGDR).
    # Synchronously, the penalty (alpha) ramps up during each cycle,
    # forcing local feasibility, before being relaxed at the next warm restart.
    # This "breathes" the constraints, allowing macro-shifts out of bad local optima.
    
    c1_end = 0.40
    c2_end = 0.75
    c3_end = 0.92
    
    p_c1 = jnp.clip(progress / c1_end, 0.0, 1.0)
    p_c2 = jnp.clip((progress - c1_end) / (c2_end - c1_end), 0.0, 1.0)
    p_c3 = jnp.clip((progress - c2_end) / (c3_end - c2_end), 0.0, 1.0)
    
    # Learning rates: High exploration -> Medium -> Fine tuning
    lr_max_1 = 1.6 * D_f
    lr_max_2 = 0.8 * D_f
    lr_max_3 = 0.3 * D_f
    
    # Cosine decays
    lr_c1 = gamma_min_f + 0.5 * (lr_max_1 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_c1))
    lr_c2 = gamma_min_f + 0.5 * (lr_max_2 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_c2))
    lr_c3 = gamma_min_f + 0.5 * (lr_max_3 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_c3))
    
    lr_main = jnp.where(progress < c1_end, lr_c1,
                jnp.where(progress < c2_end, lr_c2, lr_c3))
                
    is_terminal = progress >= c3_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Cyclic Alpha Penalty ---
    # Relax constraints at the start of each cycle to let the high LR shift the layout.
    # Tighten constraints (inverted cosine) by the end of each cycle.
    # We progressively increase both the base and max penalty across cycles.
    
    a_base_1 = alpha0 * 0.02
    a_max_1  = alpha0 * 5.0
    
    a_base_2 = alpha0 * 0.50
    a_max_2  = alpha0 * 15.0
    
    a_base_3 = alpha0 * 2.00
    a_max_3  = alpha0 * 40.0
    
    # Inverted cosine for smooth ramp up within each cycle
    alpha_c1 = a_base_1 + 0.5 * (a_max_1 - a_base_1) * (1.0 - jnp.cos(jnp.pi * p_c1))
    alpha_c2 = a_base_2 + 0.5 * (a_max_2 - a_base_2) * (1.0 - jnp.cos(jnp.pi * p_c2))
    alpha_c3 = a_base_3 + 0.5 * (a_max_3 - a_base_3) * (1.0 - jnp.cos(jnp.pi * p_c3))
    
    alpha_main = jnp.where(progress < c1_end, alpha_c1,
                   jnp.where(progress < c2_end, alpha_c2, alpha_c3))
                   
    # Terminal feasibility spike (filter method)
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Cyclic Adam Moments ---
    # Low beta2 + high beta1 at the start of cycles for fast reaction & momentum.
    # High beta2 + low beta1 at the end to damp oscillations in the constrained space.
    
    b1_start, b2_start = 0.15, 0.20
    b1_end, b2_end = 0.02, 0.90
    
    # Use the same inverted cosine profile as alpha for the moments
    ramp_c1 = 0.5 * (1.0 - jnp.cos(jnp.pi * p_c1))
    ramp_c2 = 0.5 * (1.0 - jnp.cos(jnp.pi * p_c2))
    ramp_c3 = 0.5 * (1.0 - jnp.cos(jnp.pi * p_c3))
    
    b1_c1 = b1_start + (b1_end - b1_start) * ramp_c1
    b2_c1 = b2_start + (b2_end - b2_start) * ramp_c1
    
    b1_c2 = b1_start + (b1_end - b1_start) * ramp_c2
    b2_c2 = b2_start + (b2_end - b2_start) * ramp_c2
    
    b1_c3 = b1_start + (b1_end - b1_start) * ramp_c3
    b2_c3 = b2_start + (b2_end - b2_start) * ramp_c3
    
    beta1_main = jnp.where(progress < c1_end, b1_c1,
                   jnp.where(progress < c2_end, b1_c2, b1_c3))
    beta2_main = jnp.where(progress < c1_end, b2_c1,
                   jnp.where(progress < c2_end, b2_c2, b2_c3))
                   
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2