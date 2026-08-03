import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- 1. Baseline WSD (Warmup-Stable-Decay) Profile ---
    # Hold a high learning rate continuously for a long exploration phase,
    # then cool down. This is the structural foundation of the best schedules.
    warmup_end = 0.10
    stable_end = 0.85
    decay_end = 0.92
    
    lr_max = 1.35 * D_f
    
    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (progress / warmup_end)
    lr_stable = lr_max
    
    decay_progress = jnp.clip((progress - stable_end) / (decay_end - stable_end), 0.0, 1.0)
    # Cosine decay for the cooldown
    lr_decay = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * decay_progress))
    
    base_lr = jnp.where(progress < warmup_end, lr_warmup,
              jnp.where(progress < stable_end, lr_stable, lr_decay))
              
    # ADMM-style Baseline Alpha: Moderate and constant, allowing fluid exploration
    alpha_mod = alpha0_f * 0.3
    alpha_rise = alpha0_f * 6.0
    base_alpha = jnp.where(progress < stable_end, alpha_mod,
                           alpha_mod + (alpha_rise - alpha_mod) * decay_progress)
                           
    # Baseline Moments (low beta2 is critical for rapid layout shifting)
    base_beta1 = 0.12
    base_beta2 = 0.15

    # --- 2. Mid-Run Feasibility Restoration Bursts (Filter Method) ---
    # STRUCTURAL DEPARTURE: Instead of a single logistic ramp or cyclic resets,
    # we inject two smooth Gaussian "pulses" of extreme penalty mid-run.
    # During a pulse, LR drops sharply and Alpha spikes. This projects the 
    # layout back to the feasible boundary without losing the overall WSD momentum.
    
    pulse1_c, pulse1_w = 0.35, 0.03
    pulse2_c, pulse2_w = 0.65, 0.03
    
    g1 = jnp.exp(-0.5 * ((progress - pulse1_c) / pulse1_w)**2)
    g2 = jnp.exp(-0.5 * ((progress - pulse2_c) / pulse2_w)**2)
    
    pulse_profile = jnp.clip(g1 + g2, 0.0, 1.0)
    
    # During pulses, LR drops to 15% of its base value to ensure stable projection
    lr_main = base_lr * (1.0 - 0.85 * pulse_profile)
    
    # During pulses, Alpha spikes to enforce feasibility
    alpha_spike = alpha0_f * 30.0
    alpha_main = base_alpha + alpha_spike * pulse_profile
    
    # During pulses, drop beta1 (momentum) and raise beta2 to absorb constraint curvature
    beta1_main = base_beta1 * (1.0 - 0.90 * pulse_profile) # drops to ~0.01
    beta2_main = base_beta2 + (0.95 - base_beta2) * pulse_profile # rises to ~0.95

    # --- 3. Terminal Feasibility Spike ---
    # Absolute strict constraint enforcement in the final 8%
    is_terminal = progress >= decay_end
    
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    beta1_terminal = 0.01
    beta2_terminal = 0.99
    
    lr = jnp.where(is_terminal, lr_terminal, lr_main)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, beta1_terminal, beta1_main)
    beta2 = jnp.where(is_terminal, beta2_terminal, beta2_main)

    return lr, alpha, beta1, beta2