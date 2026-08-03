import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Multi-Cycle SGDR (Warm Restarts) ---
    # A structural departure from the WSD parent: we use 2 cycles for the main 
    # optimization phase, followed by a terminal feasibility phase.
    decay_end = 0.90
    num_cycles = 2.0
    cycle_len = decay_end / num_cycles
    
    # Calculate which cycle we are in (0 or 1) and progress within the cycle (0.0 to 1.0)
    cycle_idx = jnp.floor(progress / cycle_len)
    cycle_idx = jnp.minimum(cycle_idx, num_cycles - 1.0)
    cycle_prog = (progress - cycle_idx * cycle_len) / cycle_len
    
    is_cycle_0 = cycle_idx < 0.5
    
    # Warm restarts: learning rate jumps back up, but to a lower peak in cycle 1
    # Cycle 0 explores globally (1.3D); Cycle 1 searches locally (0.8D)
    current_lr_max = jnp.where(is_cycle_0, 1.3 * D_f, 0.8 * D_f)
    
    # Cosine annealing within each cycle to drop smoothly to gamma_min
    lr_main = gamma_min_f + 0.5 * (current_lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_prog))
    
    # --- 2. Cyclic Decoupled Penalty (Alpha) ---
    # Alpha starts low in each cycle to allow exploration (layout shifting).
    # It ramps up to a peak at the end of each cycle (mid-run feasibility-restoration bursts).
    # The base and peak increase in the second cycle to ensure progressive strictness.
    alpha_base = jnp.where(is_cycle_0, alpha0 * 0.1, alpha0 * 1.0)
    alpha_peak = jnp.where(is_cycle_0, alpha0 * 15.0, alpha0 * 25.0)
    
    # Using a cubic curve to delay the penalty ramp until the second half of each cycle,
    # letting the optimizer exploit the low penalty phase for longer.
    alpha_main = alpha_base + (alpha_peak - alpha_base) * (cycle_prog ** 3.0)
    
    # --- 3. Cyclic Phase-Transition Adam Moments ---
    # Moments synchronize with the warm restarts:
    # Low beta2 / moderate beta1 early in the cycle (with high LR) for rapid movement.
    # High beta2 / low beta1 late in the cycle to damp oscillations as penalty peaks.
    b1_start = jnp.where(is_cycle_0, 0.12, 0.10)
    b1_end = 0.04
    b2_start = jnp.where(is_cycle_0, 0.15, 0.25)
    b2_end = 0.85
    
    beta1_main = b1_start + (b1_end - b1_start) * cycle_prog
    # Square the cycle_prog for beta2 so damping kicks in just slightly before 
    # the cubic alpha penalty ramps up, smoothing the entry into stiff constraints
    beta2_main = b2_start + (b2_end - b2_start) * (cycle_prog ** 2.0)
    
    # --- 4. Terminal Feasibility Phase ---
    # Absolute compliance enforcement (filter method spike) in the final 10%
    is_terminal = progress >= decay_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)
    
    return lr, alpha, beta1, beta2