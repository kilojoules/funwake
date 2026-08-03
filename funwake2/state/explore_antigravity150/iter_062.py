import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. One-Cycle Learning Rate ---
    # Structural departure: A single continuous cosine wave (warmup then decay)
    # Reaches a higher peak than WSD or SGDR to force deeper exploration,
    # then cools down smoothly to fine-tune the layout.
    peak_progress = 0.35
    decay_end = 0.90
    
    lr_initial = 0.10 * D_f
    lr_peak = 1.40 * D_f
    
    # Phase 1: Cosine warmup
    phase1 = progress / peak_progress
    lr_up = lr_initial + 0.5 * (lr_peak - lr_initial) * (1.0 - jnp.cos(jnp.pi * phase1))
    
    # Phase 2: Cosine decay
    phase2 = jnp.clip((progress - peak_progress) / (decay_end - peak_progress), 0.0, 1.0)
    lr_down = gamma_min_f + 0.5 * (lr_peak - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * phase2))
    
    lr_main = jnp.where(progress < peak_progress, lr_up, lr_down)
    
    # --- 2. Decoupled, Delayed Alpha Ramp (Exact Penalty / ALM) ---
    # We delay the penalty until the learning rate is already cooling down.
    # This ensures the penalty doesn't prematurely restrict the layout during the
    # high-LR exploration phase, avoiding early local minima entrapment.
    alpha_base = alpha0 * 0.05
    alpha_plateau = alpha0 * 20.0
    
    # Steep logistic ramp centered at progress = 0.60
    k_alpha = 30.0
    p0_alpha = 0.60
    logistic_ramp = 1.0 / (1.0 + jnp.exp(-k_alpha * (progress - p0_alpha)))
    
    alpha_main = alpha_base + (alpha_plateau - alpha_base) * logistic_ramp
    
    # --- 3. Split-Synchronized Adam Moments ---
    # We split the moment schedules based on their mechanical roles:
    # beta1 (momentum) follows the LR phase (1-cycle policy): drops during warmup to 
    #   prevent overshoot when taking massive steps, then recovers during cooldown.
    # beta2 (adaptive scale) follows the alpha phase: jumps up precisely when the 
    #   penalty kicks in to damp oscillations and absorb stiff boundary curvature.
    
    b1_up = 0.15 - 0.13 * phase1  # 0.15 -> 0.02
    b1_down = 0.02 + 0.08 * phase2 # 0.02 -> 0.10
    beta1_lr = jnp.where(progress < peak_progress, b1_up, b1_down)
    
    b2_base = 0.15
    b2_plateau = 0.90
    beta2_main = b2_base + (b2_plateau - b2_base) * logistic_ramp
    
    # --- 4. Terminal Feasibility Spike (Filter Method) ---
    # Final 10% crushes LR to gamma_min and spikes penalty to guarantee 
    # 100% strict feasibility of the optimized layout.
    is_terminal = progress >= decay_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_lr)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2