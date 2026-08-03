import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # Helper functions for cyclic transitions
    def cos_decay(prog, start, end, val_start, val_end):
        p = jnp.clip((prog - start) / (end - start), 0.0, 1.0)
        return val_end + 0.5 * (val_start - val_end) * (1.0 + jnp.cos(jnp.pi * p))

    def cos_growth(prog, start, end, val_start, val_end):
        p = jnp.clip((prog - start) / (end - start), 0.0, 1.0)
        return val_start + 0.5 * (val_end - val_start) * (1.0 - jnp.cos(jnp.pi * p))

    # --- 1. Multi-Cycle Cosine Learning Rate (SGDR) ---
    # Three cycles of decreasing length and magnitude to explore multiple 
    # basins, followed by a strict terminal phase.
    c1_end = 0.45
    c2_end = 0.80
    c3_end = 0.95
    
    lr_max1 = 1.40 * D_f
    lr_max2 = 0.70 * D_f
    lr_max3 = 0.25 * D_f
    
    lr_c1 = cos_decay(progress, 0.00, c1_end, lr_max1, gamma_min_f)
    lr_c2 = cos_decay(progress, c1_end, c2_end, lr_max2, gamma_min_f)
    lr_c3 = cos_decay(progress, c2_end, c3_end, lr_max3, gamma_min_f)
    
    lr_main = jnp.where(progress < c1_end, lr_c1,
                jnp.where(progress < c2_end, lr_c2, lr_c3))
                
    is_terminal = progress >= c3_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Cyclic Alpha with Mid-Run Feasibility Bursts ---
    # Alpha drops at the start of each warm restart to allow layout exploration, 
    # then spikes as the cycle ends. This creates 'feasibility bursts' that 
    # periodically pull the layout back to valid configurations before the next jump.
    alpha_base = alpha0 * 0.05
    alpha_peak1 = alpha0 * 4.0
    alpha_peak2 = alpha0 * 8.0
    alpha_peak3 = alpha0 * 15.0
    
    a_c1 = cos_growth(progress, 0.00, c1_end, alpha_base, alpha_peak1)
    a_c2 = cos_growth(progress, c1_end, c2_end, alpha_base, alpha_peak2)
    a_c3 = cos_growth(progress, c2_end, c3_end, alpha_base, alpha_peak3)
    
    alpha_main = jnp.where(progress < c1_end, a_c1,
                   jnp.where(progress < c2_end, a_c2, a_c3))
                   
    # Terminal feasibility spike ensures absolute compliance at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Phase-Transition Adam Moments (Cyclic) ---
    # Synchronize moments with the cyclic alpha bursts:
    # Low beta2 (high momentum effect) during warm restarts to vault out of local minima.
    # High beta2 (strong damping) during feasibility bursts to absorb constraint curvature.
    b1_start, b2_start = 0.15, 0.10
    b1_end, b2_end = 0.02, 0.90
    
    b1_c1 = cos_growth(progress, 0.00, c1_end, b1_start, b1_end)
    b2_c1 = cos_growth(progress, 0.00, c1_end, b2_start, b2_end)
    
    b1_c2 = cos_growth(progress, c1_end, c2_end, b1_start, b1_end)
    b2_c2 = cos_growth(progress, c1_end, c2_end, b2_start, b2_end)
    
    b1_c3 = cos_growth(progress, c2_end, c3_end, b1_start, b1_end)
    b2_c3 = cos_growth(progress, c2_end, c3_end, b2_start, b2_end)
    
    b1_main = jnp.where(progress < c1_end, b1_c1,
                jnp.where(progress < c2_end, b1_c2, b1_c3))
    b2_main = jnp.where(progress < c1_end, b2_c1,
                jnp.where(progress < c2_end, b2_c2, b2_c3))
                
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, b1_main)
    beta2 = jnp.where(is_terminal, 0.99, b2_main)

    return lr, alpha, beta1, beta2