import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR with Cyclic Penalty (Multi-Cycle One-Cycle Policy) ---
    # We break the main run into 3 warm-restart cycles. Each cycle does a short 
    # warmup to a high learning rate (exploration), then a cosine decay while 
    # smoothly ramping up the penalty (alpha) to pull into a feasible local minimum.
    # Successive cycles have lower peak LRs but higher peak penalties.
    
    decay_end = 0.90
    n_cycles = 3.0
    
    # Calculate cycle index [0, 1, 2] and relative progress [0, 1] within it
    cycle_progress = (progress / decay_end) * n_cycles
    cycle_idx = jnp.clip(jnp.floor(cycle_progress), 0.0, n_cycles - 1.0)
    p_cycle = cycle_progress - cycle_idx
    
    warmup_frac = 0.15  # 15% of each cycle is warmup (exploration burst)
    is_warmup = p_cycle < warmup_frac
    
    p_warmup = p_cycle / warmup_frac
    p_decay = (p_cycle - warmup_frac) / (1.0 - warmup_frac)
    
    # --- Learning Rate ---
    lr_peak_start = 1.5 * D_f
    lr_peak_c = lr_peak_start * (0.6 ** cycle_idx)  # Peaks: 1.5, 0.9, 0.54
    
    lr_warmup = gamma_min_f + (lr_peak_c - gamma_min_f) * p_warmup
    lr_decay = gamma_min_f + 0.5 * (lr_peak_c - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_decay))
    lr_cycle = jnp.where(is_warmup, lr_warmup, lr_decay)
    
    # --- Penalty (Alpha) ---
    # Starts low for exploration, ramps to enforce constraints.
    alpha_base = alpha0 * 0.1
    alpha_peak_start = alpha0 * 3.0
    alpha_peak_c = alpha_peak_start * (2.0 ** cycle_idx)  # Peaks: 3, 6, 12
    
    alpha_decay = alpha_peak_c - 0.5 * (alpha_peak_c - alpha_base) * (1.0 + jnp.cos(jnp.pi * p_decay))
    alpha_cycle = jnp.where(is_warmup, alpha_base, alpha_decay)
    
    # --- Phase-Transition Adam Moments ---
    # Sync with phase: highly adaptive / low momentum during exploration,
    # damped / higher memory during feasibility tightening.
    b1_expl, b2_expl = 0.15, 0.15
    b1_feas, b2_feas = 0.04, 0.85
    
    b1_decay = b1_feas + 0.5 * (b1_expl - b1_feas) * (1.0 + jnp.cos(jnp.pi * p_decay))
    b2_decay = b2_feas - 0.5 * (b2_feas - b2_expl) * (1.0 + jnp.cos(jnp.pi * p_decay))
    
    b1_cycle = jnp.where(is_warmup, b1_expl, b1_decay)
    b2_cycle = jnp.where(is_warmup, b2_expl, b2_decay)
    
    # --- 2. Terminal Feasibility Spike ---
    # Final 10% strictly restores absolute compliance at the end
    is_terminal = progress >= decay_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    beta1 = jnp.where(is_terminal, 0.01, b1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, b2_cycle)

    return lr, alpha, beta1, beta2