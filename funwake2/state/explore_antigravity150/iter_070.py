import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. SGDR with Warm Restarts (Multi-Cycle) ---
    # We use 3 cycles of exploration, each ending with a feasibility-restoration burst,
    # followed by a final strict terminal phase.
    
    n_cycles = 3.0
    terminal_phase_start = 0.90
    
    # Map progress into cycles for the first 90% of the run
    # Each cycle is 30% of total steps
    cycle_length = terminal_phase_start / n_cycles
    
    cycle_idx = jnp.floor(progress / cycle_length)
    cycle_idx = jnp.clip(cycle_idx, 0.0, n_cycles - 1.0)
    
    cycle_progress = (progress - cycle_idx * cycle_length) / cycle_length
    cycle_progress = jnp.clip(cycle_progress, 0.0, 1.0)
    
    # Peak learning rate decays across cycles (e.g. 1.5D, 0.9D, 0.54D)
    lr_max_base = 1.5 * D_f
    lr_max_cycle = lr_max_base * (0.6 ** cycle_idx)
    
    # Short linear warmup per cycle, then cosine decay
    warmup_frac = 0.10
    in_warmup = cycle_progress < warmup_frac
    
    lr_warmup = gamma_min_f + (lr_max_cycle - gamma_min_f) * (cycle_progress / warmup_frac)
    
    decay_progress = jnp.clip((cycle_progress - warmup_frac) / (1.0 - warmup_frac), 0.0, 1.0)
    cosine_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress))
    lr_decay = gamma_min_f + (lr_max_cycle - gamma_min_f) * cosine_mult
    
    lr_cycle = jnp.where(in_warmup, lr_warmup, lr_decay)
    
    # --- 2. Cyclic Alpha with Mid-Run Feasibility Bursts ---
    # Alpha drops low at the start of each cycle to permit layout exploration,
    # then spikes powerfully at the end of each cycle to pull violating turbines back.
    alpha_start = alpha0 * 0.1
    # The burst intensity increases with each cycle
    alpha_burst = alpha0 * 15.0 * (1.5 ** cycle_idx)
    
    # Cubic ramp ensures alpha stays moderate during the middle of the cycle, spiking late
    alpha_cycle = alpha_start + (alpha_burst - alpha_start) * (cycle_progress ** 3.0)
    
    # --- 3. Cyclic Adam Moments ---
    # Sync momentum with the alpha cycles:
    # High momentum (b1=0.15) / low beta2 (0.1) when alpha is low allows rapid movement.
    # Low momentum (b1=0.04) / high beta2 (0.9) during the alpha burst absorbs constraint stiffness.
    b1_start, b2_start = 0.15, 0.10
    b1_end, b2_end = 0.04, 0.90
    
    mom_progress = cycle_progress ** 2.0
    beta1_cycle = b1_start + (b1_end - b1_start) * mom_progress
    beta2_cycle = b2_start + (b2_end - b2_start) * mom_progress
    
    # --- 4. Terminal Feasibility Phase ---
    # The final 10% is purely for tightening to exact gamma_min tolerances.
    is_terminal = progress >= terminal_phase_start
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2