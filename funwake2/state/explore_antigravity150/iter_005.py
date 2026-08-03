import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceable conversions (step and total_steps may be traced)
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    gamma_min_f = float(gamma_min)
    D_f = float(D)

    # 3-Cycle SGDR (Stochastic Gradient Descent with Warm Restarts) 
    # coupled with Cyclic Feasibility Penalty bursts.
    # This design tests the hypothesis that periodic constraint relaxation 
    # followed by forced restoration yields better global layouts than a monotonic schedule.
    
    c1_end = 0.50
    c2_end = 0.75
    c3_end = 0.90
    
    p1 = jnp.clip(progress / c1_end, 0.0, 1.0)
    p2 = jnp.clip((progress - c1_end) / (c2_end - c1_end), 0.0, 1.0)
    p3 = jnp.clip((progress - c2_end) / (c3_end - c2_end), 0.0, 1.0)
    
    # Custom plateau lengths for each cycle's exploration phase
    # Cycle 1: 70% plateau, 30% decay (allows a long, high initial peak)
    p1_decay = jnp.clip((p1 - 0.70) / 0.30, 0.0, 1.0)
    # Cycle 2: 20% plateau, 80% decay
    p2_decay = jnp.clip((p2 - 0.20) / 0.80, 0.0, 1.0)
    # Cycle 3: 10% plateau, 90% decay
    p3_decay = jnp.clip((p3 - 0.10) / 0.90, 0.0, 1.0)
    
    decay1 = 0.5 * (1.0 + jnp.cos(jnp.pi * p1_decay))
    decay2 = 0.5 * (1.0 + jnp.cos(jnp.pi * p2_decay))
    decay3 = 0.5 * (1.0 + jnp.cos(jnp.pi * p3_decay))
    
    decay = jnp.where(progress < c1_end, decay1, 
              jnp.where(progress < c2_end, decay2, decay3))
              
    # --- 1. Cyclic Learning Rate ---
    lr_max1 = 1.15 * D_f  # Peak higher than parent (1.04) to push AEP
    lr_max2 = 0.60 * D_f  # Mid-level exploration
    lr_max3 = 0.20 * D_f  # Fine refinement
    
    lr_decay_val1 = gamma_min_f + (lr_max1 - gamma_min_f) * decay1
    lr_decay_val2 = gamma_min_f + (lr_max2 - gamma_min_f) * decay2
    lr_decay_val3 = gamma_min_f + (lr_max3 - gamma_min_f) * decay3
    
    lr_decay_val = jnp.where(progress < c1_end, lr_decay_val1,
                     jnp.where(progress < c2_end, lr_decay_val2, lr_decay_val3))
                     
    # Warmup up to 5% progress
    lr_warmup = lr_max1 * (0.1 + 0.9 * jnp.clip(progress / 0.05, 0.0, 1.0))
    lr_main = jnp.where(progress < 0.05, lr_warmup, lr_decay_val)
    
    is_terminal = progress >= c3_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    # --- 2. Cyclic Feasibility Penalty (Alpha) ---
    # Alpha bursts as LR decays, enforcing constraints periodically.
    burst = 1.0 - decay
    
    # Cycle plateaus get progressively stricter
    alpha_min1 = alpha0 * 0.5  # Softest start for max exploration
    alpha_min2 = alpha0 * 2.0
    alpha_min3 = alpha0 * 5.0
    
    # Cycle bursts force feasibility mid-run
    alpha_max1 = alpha0 * 5.0
    alpha_max2 = alpha0 * 15.0
    alpha_max3 = alpha0 * 30.0
    
    alpha_cycle1 = alpha_min1 + (alpha_max1 - alpha_min1) * burst
    alpha_cycle2 = alpha_min2 + (alpha_max2 - alpha_min2) * burst
    alpha_cycle3 = alpha_min3 + (alpha_max3 - alpha_min3) * burst
    
    alpha_main = jnp.where(progress < c1_end, alpha_cycle1, 
                   jnp.where(progress < c2_end, alpha_cycle2, alpha_cycle3))
                   
    # Terminal feasibility spike (preserved from the successful parent)
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    # --- 3. Phase-transition Adam moments ---
    # As alpha bursts up, we drop momentum (beta1) and increase curvature absorption (beta2)
    b1_base1, b2_base1 = 0.10, 0.20
    b1_base2, b2_base2 = 0.08, 0.40
    b1_base3, b2_base3 = 0.06, 0.70
    
    b1_burst, b2_burst = 0.05, 0.90
    
    beta1_c1 = b1_base1 + (b1_burst - b1_base1) * burst
    beta2_c1 = b2_base1 + (b2_burst - b2_base1) * burst
    
    beta1_c2 = b1_base2 + (b1_burst - b1_base2) * burst
    beta2_c2 = b2_base2 + (b2_burst - b2_base2) * burst
    
    beta1_c3 = b1_base3 + (b1_burst - b1_base3) * burst
    beta2_c3 = b2_base3 + (b2_burst - b2_base3) * burst
    
    beta1_main = jnp.where(progress < c1_end, beta1_c1,
                   jnp.where(progress < c2_end, beta1_c2, beta1_c3))
    beta2_main = jnp.where(progress < c1_end, beta2_c1,
                   jnp.where(progress < c2_end, beta2_c2, beta2_c3))
                   
    beta1 = jnp.where(is_terminal, 0.05, beta1_main)
    beta2 = jnp.where(is_terminal, 0.90, beta2_main)
    
    return lr, alpha, beta1, beta2