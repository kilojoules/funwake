import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. Multi-Cycle SGDR Phase (0.0 to 0.90) ---
    # We use 3 full cycles. Each cycle starts with high LR / low alpha (exploration)
    # and ends with low LR / high alpha (mid-run feasibility-restoration burst).
    main_end = 0.90
    is_main = progress < main_end
    
    # Scale progress for the main phase to [0, 1]
    # We cap at 0.99999 to prevent cycle_idx from bumping to 3.0 exactly at the boundary
    p_main = jnp.minimum(progress / main_end, 0.99999) 
    
    num_cycles = 3.0
    # cycle_progress goes from 0 to 1 within each cycle
    cycle_progress = jnp.mod(p_main * num_cycles, 1.0)
    # cycle_idx is 0, 1, 2
    cycle_idx = jnp.floor(p_main * num_cycles)

    # 1a. Multi-Cycle Cosine Learning Rate
    # The peak learning rate decays across cycles (1.5x D, 0.9x D, 0.54x D)
    lr_max_start = 1.5 * D_f
    lr_max = lr_max_start * (0.6 ** cycle_idx)
    lr_min = gamma_min_f
    
    lr_main = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * cycle_progress))

    # 1b. Cyclic Alpha (Decoupled Penalty)
    # Alpha peaks at the end of each cycle to force constraint compliance,
    # with the peak intensity increasing each cycle (5x, 10x, 15x).
    alpha_base = alpha0 * 0.1
    alpha_peak = alpha0 * (5.0 + 5.0 * cycle_idx)
    
    # Ramp from 0 to 1 over the cycle
    alpha_ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * cycle_progress))
    alpha_main = alpha_base + (alpha_peak - alpha_base) * alpha_ramp

    # 1c. Phase-Transition Adam Moments
    # Synchronize momentum with the cycles: high momentum early in the cycle for 
    # exploration, dropping to low momentum during the feasibility bursts to damp oscillations.
    b1_high, b2_low = 0.15, 0.10
    b1_low, b2_high = 0.02, 0.90
    
    beta1_main = b1_low + (b1_high - b1_low) * (0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress)))
    beta2_main = b2_low + (b2_high - b2_low) * alpha_ramp

    # --- 2. Terminal Feasibility Spike (0.90 to 1.0) ---
    # Strict compliance for the final 10% of the run, preserving the exact 
    # exact-penalty filter logic from the best parent.
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    beta1_terminal = 0.01
    beta2_terminal = 0.99

    lr = jnp.where(is_main, lr_main, lr_terminal)
    alpha = jnp.where(is_main, alpha_main, alpha_terminal)
    beta1 = jnp.where(is_main, beta1_main, beta1_terminal)
    beta2 = jnp.where(is_main, beta2_main, beta2_terminal)

    return lr, alpha, beta1, beta2