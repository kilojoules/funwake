import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    terminal_start = 0.96
    is_terminal = progress >= terminal_start

    # --- 1. Cyclic LR with SGDR Warm Restarts ---
    # 3 cycles in the main run. Each cycle starts with a high LR (exploration)
    # and cosine-anneals down to gamma_min (exploitation/feasibility).
    num_cycles = 3.0
    phi = num_cycles * (progress / terminal_start)
    cycle_progress = jnp.remainder(phi, 1.0)
    cycle_idx = jnp.clip(jnp.floor(phi), 0.0, num_cycles - 1.0)
    
    # Cosine annealing multiplier: 1.0 at cycle start, 0.0 at cycle end
    cos_val = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    # The peak learning rate decays linearly across the whole run
    lr_max = 1.25 * D_f
    lr_peak = gamma_min_f + (lr_max - gamma_min_f) * (1.0 - progress)
    
    lr_cycle = gamma_min_f + (lr_peak - gamma_min_f) * cos_val
    
    # Short linear warmup only at the very beginning to prevent initial shock
    warmup_end = 0.05
    warmup_progress = jnp.clip(progress / warmup_end, 0.0, 1.0)
    lr_main = jnp.where(progress < warmup_end, 
                        gamma_min_f + (lr_cycle - gamma_min_f) * warmup_progress, 
                        lr_cycle)
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Cyclic Decoupled Penalty (Mid-Run Feasibility Bursts) ---
    # Alpha is inversely coupled to the cyclic LR:
    # When LR is high (warm restart), alpha drops to allow layout exploration.
    # When LR bottoms out, alpha spikes to restore feasibility.
    # The peak penalty of these bursts escalates with each cycle.
    
    alpha_soft = alpha0 * 0.1
    alpha_hard = alpha0 * (5.0 + 5.0 * cycle_idx)  # Peaks at 5x, 10x, 15x alpha0
    
    # As cos_val goes 1 -> 0, alpha goes soft -> hard
    alpha_main = alpha_soft + (alpha_hard - alpha_soft) * (1.0 - cos_val)
    
    # Terminal filter-method feasibility spike
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Phase-Transition Adam Moments ---
    # Momentum (beta1) tracks LR: high during exploration, drops for feasibility bursts.
    # Curvature (beta2) tracks alpha: rises to absorb stiff constraint landscapes.
    
    beta1_main = 0.02 + 0.13 * cos_val  # Ramps from 0.02 (hard) to 0.15 (soft)
    beta2_main = 0.98 - 0.78 * cos_val  # Ramps from 0.98 (hard) to 0.20 (soft)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2