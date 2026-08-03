import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)
    
    # --- 3-Cycle SGDR (Warm Restarts) with Cyclic Penalty ---
    # A structural departure: multi-cycle cosine annealing (SGDR). 
    # At the end of each cycle, alpha peaks to forcibly restore feasibility 
    # (mid-run feasibility-restoration bursts). A warm restart then spikes LR 
    # and drops alpha to aggressively escape constraint-induced local optima.
    
    warmup_end = 0.05
    t1 = 0.25
    t2 = 0.60
    t3 = 0.95
    
    # --- Phase Progress Trackers ---
    # Cycle 1: Linear warmup to 0.05, then cosine decay to t1
    p_c1_warmup = jnp.clip(progress / warmup_end, 0.0, 1.0)
    p_c1_decay = jnp.clip((progress - warmup_end) / (t1 - warmup_end), 0.0, 1.0)
    cos1 = 0.5 * (1.0 + jnp.cos(jnp.pi * p_c1_decay))
    
    # Cycle 2: Cosine decay from t1 to t2
    p_c2 = jnp.clip((progress - t1) / (t2 - t1), 0.0, 1.0)
    cos2 = 0.5 * (1.0 + jnp.cos(jnp.pi * p_c2))
    
    # Cycle 3: Cosine decay from t2 to t3
    p_c3 = jnp.clip((progress - t2) / (t3 - t2), 0.0, 1.0)
    cos3 = 0.5 * (1.0 + jnp.cos(jnp.pi * p_c3))
    
    inv_cos1 = 1.0 - cos1
    inv_cos2 = 1.0 - cos2
    inv_cos3 = 1.0 - cos3
    
    # --- 1. Learning Rate ---
    # Decaying peak heights per cycle (1.50 -> 0.75 -> 0.35)
    lr1_max = 1.50 * D_f
    lr2_max = 0.75 * D_f
    lr3_max = 0.35 * D_f
    
    lr_c1 = jnp.where(progress < warmup_end,
                      gamma_min_f + (lr1_max - gamma_min_f) * p_c1_warmup,
                      gamma_min_f + (lr1_max - gamma_min_f) * cos1)
    lr_c2 = gamma_min_f + (lr2_max - gamma_min_f) * cos2
    lr_c3 = gamma_min_f + (lr3_max - gamma_min_f) * cos3
    
    lr_main = jnp.where(progress < t1, lr_c1,
                jnp.where(progress < t2, lr_c2, lr_c3))
                
    is_terminal = progress >= t3
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    # --- 2. Cyclic Alpha Penalty ---
    # Alpha starts soft in each cycle to allow layout shifting, then ramps up 
    # sharply as learning rate decays to enforce feasibility constraints before restart.
    a1_base, a1_peak = alpha0 * 0.1, alpha0 * 4.0
    a2_base, a2_peak = alpha0 * 0.5, alpha0 * 12.0
    a3_base, a3_peak = alpha0 * 2.0, alpha0 * 30.0
    
    alpha_c1 = a1_base + (a1_peak - a1_base) * inv_cos1
    alpha_c2 = a2_base + (a2_peak - a2_base) * inv_cos2
    alpha_c3 = a3_base + (a3_peak - a3_base) * inv_cos3
    
    alpha_main = jnp.where(progress < t1, alpha_c1,
                   jnp.where(progress < t2, alpha_c2, alpha_c3))
                   
    # Terminal absolute feasibility filter preserves the parent's reliable success
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    # --- 3. Cyclic Adam Moments ---
    # Synchronized with the cycle: high momentum (beta1) and low variance smoothing 
    # (beta2) during restarts, transitioning to damped, high-smoothing phases at cycle ends.
    b1_high, b1_low = 0.15, 0.02
    beta1_c1 = b1_low + (b1_high - b1_low) * cos1
    beta1_c2 = b1_low + (b1_high - b1_low) * cos2
    beta1_c3 = b1_low + (b1_high - b1_low) * cos3
    
    beta1_main = jnp.where(progress < t1, beta1_c1,
                   jnp.where(progress < t2, beta1_c2, beta1_c3))
                   
    b2_low, b2_high = 0.15, 0.90
    beta2_c1 = b2_low + (b2_high - b2_low) * inv_cos1
    beta2_c2 = b2_low + (b2_high - b2_low) * inv_cos2
    beta2_c3 = b2_low + (b2_high - b2_low) * inv_cos3
    
    beta2_main = jnp.where(progress < t1, beta2_c1,
                   jnp.where(progress < t2, beta2_c2, beta2_c3))
                   
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)
    
    return lr, alpha, beta1, beta2