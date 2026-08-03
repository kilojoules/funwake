import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    # Use jnp.asarray as requested to avoid float() casting issues with traced JAX arrays
    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. Cyclic Cosine Annealing with Warm Restarts (SGDR) ---
    # Structural shift: We break the main optimization phase (first 90%) into 3 cycles.
    # At the start of each cycle, the learning rate jumps up (warm restart)
    # to encourage exploration and escape local minima, then decays following
    # a cosine curve.
    main_end = 0.90
    main_progress = jnp.clip(progress / main_end, 0.0, 1.0)
    
    num_cycles = 3.0
    raw_idx = jnp.floor(main_progress * num_cycles)
    cycle_idx = jnp.minimum(raw_idx, num_cycles - 1.0)
    cycle_progress = (main_progress * num_cycles) - cycle_idx
    
    # Cosine decay factor: 1.0 at start of cycle, 0.0 at end of cycle
    cos_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    # Peak LR decays across cycles: e.g., 2.0 * D, 1.0 * D, 0.5 * D
    lr_peak = (2.0 * D_f) * (0.5 ** cycle_idx)
    lr_main = gamma_min_f + (lr_peak - gamma_min_f) * cos_decay
    
    # --- 2. Cyclic Decoupled Alpha (Penalty) ---
    # Synchronize the penalty weight with the SGDR cycles.
    # When LR spikes (exploration), we RELAX the penalty to allow turbines
    # to jump over constraints. As LR decays (exploitation), we TIGHTEN the 
    # penalty to restore feasibility. Overall severity increases with each cycle.
    
    alpha_base = (alpha0 * 0.1) * (4.0 ** cycle_idx)  # base rises: 0.1, 0.4, 1.6
    alpha_peak = (alpha0 * 5.0) * (3.0 ** cycle_idx)  # peak rises: 5.0, 15.0, 45.0
    
    # Alpha rises as cos_decay falls
    alpha_main = alpha_base + (alpha_peak - alpha_base) * (1.0 - cos_decay)
    
    # --- 3. Cyclic Phase-Transition Adam Moments ---
    # Beta1 (momentum) drops as we transition from exploration to feasibility in each cycle.
    # Beta2 rises to dampen oscillations in the steep penalty valleys.
    
    b1_start = 0.15 * (0.5 ** cycle_idx)
    b1_end   = 0.05 * (0.5 ** cycle_idx)
    beta1_main = b1_end + (b1_start - b1_end) * cos_decay
    
    b2_start = 0.20 + 0.25 * cycle_idx  # 0.20, 0.45, 0.70
    b2_end   = 0.80 + 0.08 * cycle_idx  # 0.80, 0.88, 0.96
    beta2_main = b2_end + (b2_start - b2_end) * cos_decay
    
    # --- 4. Terminal Feasibility Spike ---
    # The last 10% of steps are dedicated strictly to local constraint satisfaction.
    is_terminal = progress >= main_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    # Severe penalty for absolute compliance (filter method style)
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    # Maximum damping at the end
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2