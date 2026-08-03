import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- Dual WSD with Mid-Run Feasibility Burst ---
    # We break the schedule into two WSD (Warmup-Stable-Decay) phases, separated 
    # by a "feasibility burst". The burst drops LR to the minimum and spikes the 
    # penalty (alpha) to force an intermediate feasible state. This prevents the 
    # optimizer from getting irreversibly tangled in infeasible regions. The 
    # second WSD phase explores locally from that newly untangled state.

    # 1. Cycle 1: Global Exploration
    c1_warmup_end = 0.05
    c1_stable_end = 0.35
    c1_decay_end = 0.45
    c1_lr_max = 1.40 * D_f
    c1_alpha_base = alpha0 * 0.05
    c1_alpha_peak = alpha0 * 1.50
    
    # 2. Mid-Run Burst: Feasibility Restoration
    burst_end = 0.50
    burst_alpha = alpha0 * 30.0
    
    # 3. Cycle 2: Local Refinement
    c2_warmup_end = 0.55
    c2_stable_end = 0.75
    c2_decay_end = 0.88
    c2_lr_max = 0.60 * D_f
    c2_alpha_base = alpha0 * 2.0
    c2_alpha_peak = alpha0 * 15.0
    
    # 4. Terminal Phase: Strict Compliance (0.88 - 1.0)
    term_alpha = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)

    # Compute Continuous LR (M-shape)
    lr_c1 = jnp.where(progress < c1_warmup_end,
                      gamma_min_f + (c1_lr_max - gamma_min_f) * (progress / c1_warmup_end),
                      jnp.where(progress < c1_stable_end,
                                c1_lr_max,
                                gamma_min_f + (c1_lr_max - gamma_min_f) * (1.0 - (progress - c1_stable_end) / (c1_decay_end - c1_stable_end))))
                                
    lr_c2 = jnp.where(progress < c2_warmup_end,
                      gamma_min_f + (c2_lr_max - gamma_min_f) * ((progress - burst_end) / (c2_warmup_end - burst_end)),
                      jnp.where(progress < c2_stable_end,
                                c2_lr_max,
                                gamma_min_f + (c2_lr_max - gamma_min_f) * (1.0 - (progress - c2_stable_end) / (c2_decay_end - c2_stable_end))))

    lr = jnp.where(progress < c1_decay_end, lr_c1,
           jnp.where(progress < burst_end, gamma_min_f,
             jnp.where(progress < c2_decay_end, lr_c2, gamma_min_f)))

    # Compute Piecewise Alpha with Quadratic Ramps inside cycles
    alpha_c1 = c1_alpha_base + (c1_alpha_peak - c1_alpha_base) * (progress / c1_decay_end)**2
    alpha_c2 = c2_alpha_base + (c2_alpha_peak - c2_alpha_base) * ((progress - burst_end) / (c2_decay_end - burst_end))**2
    
    alpha = jnp.where(progress < c1_decay_end, alpha_c1,
              jnp.where(progress < burst_end, burst_alpha,
                jnp.where(progress < c2_decay_end, alpha_c2, term_alpha)))

    # Compute Phase-Synchronized Betas (Adam Moments)
    # Fast momentum (low b2) during exploration; heavy damping (high b2) during bursts/terminal
    beta1 = jnp.where(progress < c1_decay_end, 0.15,
              jnp.where(progress < burst_end, 0.05,
                jnp.where(progress < c2_decay_end, 0.10, 0.01)))
                
    beta2 = jnp.where(progress < c1_decay_end, 0.20,
              jnp.where(progress < burst_end, 0.95,
                jnp.where(progress < c2_decay_end, 0.60, 0.99)))

    return lr, alpha, beta1, beta2