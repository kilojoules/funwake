import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Multi-Cycle SGDR (Warm Restarts) ---
    # Structurally different from the single WSD phase: we use 3 cycles 
    # (warm restarts) for the first 90% of the run. This prevents settling 
    # in local optima early on and allows multiple structural explorations.
    terminal_start = 0.90
    main_progress = jnp.clip(progress / terminal_start, 0.0, 1.0)
    
    total_cycles = 3.0
    # Calculate cycle index (0, 1, or 2) and intra-cycle progress (0.0 to 1.0)
    cycle_idx = jnp.minimum(jnp.floor(main_progress * total_cycles), total_cycles - 1.0)
    cycle_progress = (main_progress * total_cycles) - cycle_idx
    
    # Decay the peak learning rate each cycle to gradually fine-tune the layout
    lr_max_initial = 1.5 * D_f
    lr_max = lr_max_initial * jnp.power(0.5, cycle_idx)
    
    # Cosine annealing within the cycle
    lr_cycle = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    # --- 2. Cyclic Alpha with Mid-Run Feasibility Bursts ---
    # Instead of a monotonic plateau, we use a cyclic penalty. 
    # At the start of each cycle, alpha is low to allow rapid turbine movement.
    # Towards the end of each cycle, alpha spikes to enforce feasibility,
    # pushing overlapping turbines apart before the next restart.
    alpha_base = alpha0 * 0.05
    # The peak penalty grows stronger with each cycle
    alpha_peak = alpha0 * (10.0 * jnp.power(1.5, cycle_idx))
    
    # A sharp power-4 ramp ensures alpha stays low for exploration, then spikes
    alpha_cycle = alpha_base + (alpha_peak - alpha_base) * jnp.power(cycle_progress, 4.0)
    
    # --- 3. Cyclic Adam Moments ---
    # Synchronized with the cycle: high momentum during exploration, 
    # high beta2 damping during the feasibility-restoration bursts.
    b1_start, b2_start = 0.15, 0.10
    b1_end, b2_end = 0.02, 0.90
    
    beta1_cycle = b1_start + (b1_end - b1_start) * cycle_progress
    beta2_cycle = b2_start + (b2_end - b2_start) * cycle_progress
    
    # --- 4. Terminal Strict Feasibility Phase ---
    # The final 10% strictly restores and locks in feasibility.
    is_terminal = progress >= terminal_start
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    # Terminal filter method spike
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2