import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Cyclic SGDR Phase (0% to 90% progress) ---
    # We use 3 warm-restart cycles. In each cycle, LR decays via cosine annealing,
    # while the penalty (alpha) ramps up. Across cycles, the max LR drops exponentially
    # and the base penalty rises exponentially. This provides structured exploration-exploitation.
    n_cycles = 3.0
    terminal_start = 0.90
    
    cyclic_progress = jnp.clip(progress / terminal_start, 0.0, 1.0)
    cycle_val = cyclic_progress * n_cycles
    cycle_idx = jnp.clip(jnp.floor(cycle_val), 0.0, n_cycles - 1.0)
    t_cycle = cycle_val - cycle_idx  # Progress within current cycle [0.0, 1.0]
    
    # Cosine interpolation factor from 0.0 to 1.0 within the cycle
    t_cos = 0.5 * (1.0 - jnp.cos(jnp.pi * t_cycle))
    cycle_ratio = cycle_idx / (n_cycles - 1.0)

    # --- 2. Multi-Cycle Learning Rate ---
    lr_max_initial = 1.5 * D_f
    lr_max_final = 0.25 * D_f
    
    lr_max = lr_max_initial * jnp.power(lr_max_final / lr_max_initial, cycle_ratio)
    lr_min = gamma_min_f
    
    # Cosine annealing from lr_max to lr_min
    lr_cyclic = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    is_terminal = progress >= terminal_start
    lr = jnp.where(is_terminal, gamma_min_f, lr_cyclic)

    # --- 3. Cyclic Breathing Penalty (Alpha) ---
    # Alpha increases across cycles. Within each cycle, it "breathes":
    # ramping up as LR cools to consolidate feasibility, then dropping at the next warm restart.
    alpha_base_initial = alpha0 * 0.05
    alpha_base_final = alpha0 * 20.0
    
    alpha_base = alpha_base_initial * jnp.power(alpha_base_final / alpha_base_initial, cycle_ratio)
    
    cycle_multiplier = 4.0  # Peak alpha in a cycle is 4x its base for that cycle
    alpha_cyclic = alpha_base * (1.0 + (cycle_multiplier - 1.0) * t_cos)
    
    # Terminal feasibility spike (filter method) to guarantee strict compliance
    alpha_terminal = alpha0 * (D_f / jnp.maximum(gamma_min_f, 1e-30))
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyclic)

    # --- 4. Cyclic Phase-Transition Adam Moments ---
    # Beta parameters escalate across cycles for increasing stability as penalty stiffens.
    # Within a cycle, momentum (beta1) drops and variance damping (beta2) rises as LR cools.
    b1_start_init, b1_start_fin = 0.20, 0.10
    b1_end_init, b1_end_fin     = 0.10, 0.02
    
    b2_start_init, b2_start_fin = 0.15, 0.60
    b2_end_init, b2_end_fin     = 0.60, 0.95
    
    b1_start_cycle = b1_start_init + (b1_start_fin - b1_start_init) * cycle_ratio
    b1_end_cycle   = b1_end_init   + (b1_end_fin - b1_end_init) * cycle_ratio
    beta1_cyclic   = b1_start_cycle + (b1_end_cycle - b1_start_cycle) * t_cos
    
    b2_start_cycle = b2_start_init + (b2_start_fin - b2_start_init) * cycle_ratio
    b2_end_cycle   = b2_end_init   + (b2_end_fin - b2_end_init) * cycle_ratio
    beta2_cyclic   = b2_start_cycle + (b2_end_cycle - b2_start_cycle) * t_cos
    
    # Terminal absolute damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_cyclic)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cyclic)

    return lr, alpha, beta1, beta2