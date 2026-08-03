import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. WSD (Warmup-Stable-Decay) Learning Rate ---
    # Extend the stable exploration phase up to 60%, allowing broad layout shifts.
    # We increase peak LR to 1.35 * D for stronger exploration capability.
    warmup_end = 0.05
    stable_end = 0.60
    decay_end = 0.90
    
    lr_max = 1.35 * D_f
    
    # Warmup
    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (progress / warmup_end)
    
    # Stable
    lr_stable = lr_max
    
    # WSD decay: use a cosine cooldown instead of linear for a gentler touchdown
    decay_progress = jnp.clip((progress - stable_end) / (decay_end - stable_end), 0.0, 1.0)
    lr_decay = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * decay_progress))
    
    lr_main_smooth = jnp.where(progress < warmup_end, lr_warmup,
                               jnp.where(progress < stable_end, lr_stable, lr_decay))

    # --- 2. Decoupled Alpha with Logistic Ramp & Mid-Run Feasibility Bursts ---
    alpha_base = alpha0 * 0.05
    alpha_plateau = alpha0 * 20.0
    
    # Delayed logistic ramp (centered at 0.45, so main ramp is ~0.35 to 0.55)
    k = 25.0
    p0 = 0.45
    logistic_ramp = 1.0 / (1.0 + jnp.exp(-k * (progress - p0)))
    
    alpha_main_smooth = alpha_base + (alpha_plateau - alpha_base) * logistic_ramp
    
    # Mid-run bursts (filter method inspired):
    # Two distinct "restoration bursts" where penalty shoots up and LR drops. 
    # This acts as a periodic feasibility checkpoint that pulls turbines out 
    # of severe violation without terminating exploration.
    burst_1 = jnp.where((progress >= 0.20) & (progress < 0.23), 1.0, 0.0)
    burst_2 = jnp.where((progress >= 0.35) & (progress < 0.38), 1.0, 0.0)
    burst_mask = jnp.clip(burst_1 + burst_2, 0.0, 1.0)
    
    # Apply bursts to alpha and learning rate
    alpha_main = jnp.where(burst_mask > 0.5, alpha_plateau, alpha_main_smooth)
    lr_burst = 0.20 * lr_max
    lr_main = jnp.where(burst_mask > 0.5, lr_burst, lr_main_smooth)

    # --- 3. Terminal Feasibility Phase ---
    # Final 10% strictly enforces the constraint with an extreme penalty and gamma_min LR.
    is_terminal = progress >= decay_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 4. Phase-Synchronized Adam Moments ---
    b1_start, b2_start = 0.15, 0.10
    b1_plateau, b2_plateau = 0.03, 0.88
    
    beta1_main_smooth = b1_start + (b1_plateau - b1_start) * logistic_ramp
    beta2_main_smooth = b2_start + (b2_plateau - b2_start) * logistic_ramp
    
    # During bursts, Adam moments must snap to the plateau (feasibility) settings 
    # to drop velocity and adapt instantly to the constraint gradients.
    beta1_main = jnp.where(burst_mask > 0.5, b1_plateau, beta1_main_smooth)
    beta2_main = jnp.where(burst_mask > 0.5, b2_plateau, beta2_main_smooth)
    
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2