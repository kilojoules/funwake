import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. Multi-Cycle Cosine Annealing (SGDR) Learning Rate ---
    # We use 3 cycles. Each cycle starts with a high LR and decays to gamma_min.
    # The peak LR also decays across cycles to focus on local refinement later.
    decay_end = 0.90
    num_cycles = 3.0
    
    p_active = jnp.clip(progress / decay_end, 0.0, 1.0)
    
    # Calculate cycle index (0, 1, 2)
    cycle_idx = jnp.minimum(jnp.floor(p_active * num_cycles), num_cycles - 1.0)
    cycle_progress = (p_active * num_cycles) - cycle_idx
    
    lr_peak_start = 1.5 * D_f
    lr_peak_end = 0.5 * D_f
    # Linear decay of the peaks across the 3 cycles
    current_lr_peak = lr_peak_start - (lr_peak_start - lr_peak_end) * (cycle_idx / jnp.maximum(1.0, num_cycles - 1.0))
    
    cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    lr_main = gamma_min_f + (current_lr_peak - gamma_min_f) * cosine_decay
    
    # --- 2. Cyclic Alpha (Synchronized with SGDR) ---
    # Alpha cycles inversely to the LR. At the start of a cycle, LR is high and 
    # alpha drops to allow exploration. As the cycle ends, LR drops and alpha 
    # surges to act as a "mid-run feasibility-restoration burst".
    # The peak of these bursts increases across cycles to guarantee compliance.
    
    alpha_base = alpha0 * 0.1
    alpha_peak_start = alpha0 * 5.0
    alpha_peak_end = alpha0 * 20.0
    
    current_alpha_peak = alpha_peak_start + (alpha_peak_end - alpha_peak_start) * (cycle_idx / jnp.maximum(1.0, num_cycles - 1.0))
    alpha_cycle = alpha_base + (current_alpha_peak - alpha_base) * (1.0 - cosine_decay)
    
    # --- 3. Phase-Transition Adam Moments ---
    # Start of cycle (exploration): low beta1, low beta2
    # End of cycle (feasibility burst): higher beta2 to damp oscillations against stiff constraints
    b1_start, b2_start = 0.12, 0.15
    b1_end, b2_end = 0.04, 0.85
    
    beta1_cycle = b1_start + (b1_end - b1_start) * (1.0 - cosine_decay)
    beta2_cycle = b2_start + (b2_end - b2_start) * (1.0 - cosine_decay)

    # --- 4. Terminal Feasibility Spike ---
    # The last 10% of the run guarantees strict constraints with a massive penalty
    is_terminal = progress >= decay_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2