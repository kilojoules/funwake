import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)
    
    # --- Multi-Cycle SGDR with Cyclic Alpha & Warm Restarts ---
    # We divide the schedule into 3 cycles + 1 terminal phase.
    # Each cycle starts with high LR (exploration) and low alpha,
    # and ends with low LR and high alpha (feasibility-restoration burst).
    # This prevents the solver from getting stuck while guaranteeing mid-run check-ins.
    
    T0 = 0.15
    T1 = 0.30
    T2 = 0.45
    # Terminal phase is the last 0.10
    
    c1_cond = progress < T0
    c2_cond = (progress >= T0) & (progress < T0 + T1)
    c3_cond = (progress >= T0 + T1) & (progress < T0 + T1 + T2)
    is_terminal = progress >= T0 + T1 + T2
    
    t_c = jnp.where(c1_cond, progress / T0,
          jnp.where(c2_cond, (progress - T0) / T1,
          jnp.where(c3_cond, (progress - (T0 + T1)) / T2, 1.0)))
          
    # 1. SGDR Cosine Annealing Learning Rate
    lr_max_1 = 1.30 * D_f
    lr_max_2 = 0.85 * D_f
    lr_max_3 = 0.50 * D_f
    
    lr_max = jnp.where(c1_cond, lr_max_1,
             jnp.where(c2_cond, lr_max_2, lr_max_3))
             
    lr_main = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * t_c))
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    # 2. Cyclic Alpha: Decoupled Feasibility Bursts
    # Alpha peaks at the end of each cycle when LR is lowest.
    alpha_base_1 = alpha0 * 0.1
    alpha_base_2 = alpha0 * 0.3
    alpha_base_3 = alpha0 * 0.8
    
    alpha_peak_1 = alpha0 * 2.5
    alpha_peak_2 = alpha0 * 5.0
    alpha_peak_3 = alpha0 * 10.0
    
    alpha_base_c = jnp.where(c1_cond, alpha_base_1,
                   jnp.where(c2_cond, alpha_base_2, alpha_base_3))
    alpha_peak_c = jnp.where(c1_cond, alpha_peak_1,
                   jnp.where(c2_cond, alpha_peak_2, alpha_peak_3))
                   
    # Smooth ramp up from base to peak over the cycle
    alpha_ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * t_c))
    alpha_main = alpha_base_c + (alpha_peak_c - alpha_base_c) * alpha_ramp
    
    # Strong terminal alpha restoration to guarantee strict layout feasibility
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    # 3. Phase-Transition Adam Moments
    # β1 drops (0.15 -> 0.02) reducing momentum as constraints tighten.
    # β2 rises (0.10 -> 0.85) increasing variance smoothing as penalties dominate.
    b1_start, b1_end = 0.15, 0.02
    b2_start, b2_end = 0.10, 0.85
    
    beta1_main = b1_start + (b1_end - b1_start) * alpha_ramp
    beta2_main = b2_start + (b2_end - b2_start) * alpha_ramp
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)
    
    return lr, alpha, beta1, beta2