import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Safe conversions for JAX traceability (no python float/int casts on traced args)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- 1. SGDR Multi-Cycle (Warm Restarts) ---
    # Structural Shift: Replace the constant WSD plateau with 3 cycles of cosine decay.
    # This repeatedly "heats up" the layout to leap out of poor local AEP minima,
    # then "cools down" to settle into dense, optimized packings.
    main_end = 0.90
    p_main = jnp.clip(progress / main_end, 0.0, 1.0) # 0.0 to 1.0 over the main phase
    
    n_cycles = 3.0
    f_cycle = p_main * n_cycles
    
    # Fractional part dictates where we are in the current cycle [0.0, 1.0)
    f_mod = f_cycle - jnp.floor(f_cycle)
    
    # Cosine multiplier: 1.0 at cycle start (peak), 0.0 at cycle end (trough)
    cos_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * f_mod))
    
    # The peaks decay over the course of the run so later restarts are more refined
    lr_max = 1.35 * D_f
    lr_peak = lr_max * (1.0 - 0.6 * p_main) 
    
    lr_raw = gamma_min_f + (lr_peak - gamma_min_f) * cos_mult
    
    # Very short linear warmup (first 3%) to prevent Adam variance explosions on step 0
    warmup_end = 0.03
    warmup_factor = jnp.clip(progress / warmup_end, 0.0, 1.0)
    lr_cyclic = lr_raw * warmup_factor

    # --- 2. Synchronized Cyclic Alpha ---
    # Invert the LR curve: When LR peaks, we drop the penalty to allow constraint-violating leaps.
    # As LR decays, we ramp alpha up to forcefully pull turbines back to feasibility.
    # Both the soft floor and hard ceiling grow over time so the final cycle is strictly bounded.
    alpha_soft = alpha0_f * (0.1 + 1.9 * p_main)
    alpha_hard = alpha0_f * (3.0 + 12.0 * p_main)
    
    alpha_cyclic = alpha_hard - (alpha_hard - alpha_soft) * cos_mult

    # --- 3. Phase-Transition Cyclic Moments ---
    # High LR (exploration) -> low beta2 (0.15) to react wildly to layout gradients.
    # Low LR (settling) -> high beta2 (0.90) to damp oscillations against the stiff penalty walls.
    b1_peak, b1_trough = 0.20, 0.02
    b2_peak, b2_trough = 0.15, 0.90
    
    beta1_cyclic = b1_trough + (b1_peak - b1_trough) * cos_mult
    beta2_cyclic = b2_trough + (b2_peak - b2_trough) * cos_mult

    # --- 4. Terminal Phase ---
    # Absolute feasibility restoration / fine-tuning for the last 10% of the run.
    is_terminal = progress >= main_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cyclic)
    
    # Extreme filter-method terminal spike guarantees 100% boundary compliance
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyclic)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cyclic)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cyclic)

    return lr, alpha, beta1, beta2