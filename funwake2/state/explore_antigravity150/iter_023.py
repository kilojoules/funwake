import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. One-Cycle Cosine Learning Rate ---
    # Upward warmup, then smooth downward cosine decay.
    warmup_end = 0.15
    decay_end = 0.90
    lr_max = 1.35 * D_f
    
    phase_warmup = jnp.clip(progress / warmup_end, 0.0, 1.0)
    lr_warmup = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 - jnp.cos(jnp.pi * phase_warmup))
    
    phase_decay = jnp.clip((progress - warmup_end) / (decay_end - warmup_end), 0.0, 1.0)
    lr_decay = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * phase_decay))
    
    lr_base = jnp.where(progress < warmup_end, lr_warmup, lr_decay)
    
    # --- 2. Mid-Run Feasibility-Restoration Bursts ---
    # Introduce two narrow Gaussian pulses that temporarily spike the penalty
    # and damp the learning rate. This acts as a "filter method" to periodically
    # force the layout to respect constraints, evaluating viability without getting stuck.
    sigma = 0.015
    pulse1 = jnp.exp(-0.5 * ((progress - 0.30) / sigma)**2)
    pulse2 = jnp.exp(-0.5 * ((progress - 0.60) / sigma)**2)
    burst_envelope = jnp.clip(pulse1 + pulse2, 0.0, 1.0)
    
    # LR drops strongly during bursts to allow the penalty to fine-tune layout
    lr = lr_base * (1.0 - 0.85 * burst_envelope)
    
    # --- 3. Delayed Smoothstep Alpha Ramp ---
    alpha_early = alpha0 * 0.10
    alpha_late = alpha0 * 18.0
    
    # Main ramp delayed until mid-run (0.40 to 0.85)
    ramp_phase = jnp.clip((progress - 0.40) / 0.45, 0.0, 1.0)
    smooth_ramp = ramp_phase * ramp_phase * (3.0 - 2.0 * ramp_phase)
    
    alpha_base = alpha_early + (alpha_late - alpha_early) * smooth_ramp
    
    # Add the burst penalty on top of the base alpha envelope
    alpha_burst_peak = alpha0 * 12.0
    alpha_main = alpha_base + alpha_burst_peak * burst_envelope
    
    # --- 4. Synchronized Adam Moments ---
    # Moments transition from exploration (high var) to refinement (low var)
    beta1_base = 0.12 - 0.09 * smooth_ramp  # 0.12 -> 0.03
    beta2_base = 0.15 + 0.70 * smooth_ramp  # 0.15 -> 0.85
    
    # Bursts temporarily shift moments to strongly damp oscillations
    beta1_main = beta1_base * (1.0 - 0.8 * burst_envelope)
    beta2_main = beta2_base + (0.95 - beta2_base) * burst_envelope
    
    # --- 5. Terminal Feasibility Phase ---
    # Final 10% strictly enforces the constraint with minimal learning rate
    is_terminal = progress >= decay_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2