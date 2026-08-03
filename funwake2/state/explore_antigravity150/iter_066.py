import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- SGDR with Cyclic Alpha (Multi-cycle warm restarts) ---
    # Instead of a single warmup/decay, we use 3 shrinking cycles.
    # Each cycle starts with a high LR (exploration) and low alpha (constraint relaxation),
    # then smoothly anneals LR to gamma_min while quadratically ramping alpha 
    # to restore feasibility. This provides mid-run feasibility-restoration bursts.
    
    c1_end = 0.40
    c2_end = 0.75
    c3_end = 0.95

    # Normalized progress within each active cycle [0.0 -> 1.0]
    p_c1 = progress / c1_end
    p_c2 = (progress - c1_end) / (c2_end - c1_end)
    p_c3 = (progress - c2_end) / (c3_end - c2_end)

    # 1. Cyclic Cosine Learning Rates
    # Successive cycles have decaying peak learning rates
    lr_m1 = 1.50 * D_f
    lr_m2 = 1.00 * D_f
    lr_m3 = 0.50 * D_f

    lr_c1 = gamma_min_f + 0.5 * (lr_m1 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_c1))
    lr_c2 = gamma_min_f + 0.5 * (lr_m2 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_c2))
    lr_c3 = gamma_min_f + 0.5 * (lr_m3 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_c3))

    # 2. Cyclic Decoupled Penalties (Alpha)
    # Drop penalty at the start of each cycle to free the turbines, 
    # then ramp to a higher plateau than the previous cycle.
    a1_min, a1_max = 0.1 * alpha0_f, 5.0 * alpha0_f
    a2_min, a2_max = 0.5 * alpha0_f, 20.0 * alpha0_f
    a3_min, a3_max = 2.0 * alpha0_f, 60.0 * alpha0_f

    # Quadratic ramp delays the penalty enforcement until the end of the cycle
    alpha_c1 = a1_min + (a1_max - a1_min) * (p_c1 ** 2.0)
    alpha_c2 = a2_min + (a2_max - a2_min) * (p_c2 ** 2.0)
    alpha_c3 = a3_min + (a3_max - a3_min) * (p_c3 ** 2.0)

    # 3. Phase-Transition Adam Moments
    # Sync moments with the cycle phase: high momentum (beta1) when exploring,
    # heavy damping (beta2) when the cycle's feasibility penalty peaks.
    b1_start, b1_end = 0.15, 0.04
    b2_start, b2_end = 0.20, 0.90

    p_cyc_active = jnp.where(progress < c1_end, p_c1,
                   jnp.where(progress < c2_end, p_c2,
                   jnp.where(progress < c3_end, p_c3, 1.0)))

    beta1_main = b1_start + (b1_end - b1_start) * p_cyc_active
    beta2_main = b2_start + (b2_end - b2_start) * p_cyc_active

    # 4. Terminal Feasibility Spike (Filter Method equivalent)
    is_terminal = progress >= c3_end
    
    lr_term = gamma_min_f
    alpha_term = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    # Combine phases safely for JAX tracing
    lr = jnp.where(is_terminal, lr_term,
         jnp.where(progress < c1_end, lr_c1,
         jnp.where(progress < c2_end, lr_c2, lr_c3)))
         
    alpha = jnp.where(is_terminal, alpha_term,
            jnp.where(progress < c1_end, alpha_c1,
            jnp.where(progress < c2_end, alpha_c2, alpha_c3)))
            
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2