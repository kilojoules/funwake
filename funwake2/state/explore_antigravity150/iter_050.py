import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Multi-Cycle SGDR (Cosine Annealing with Warm Restarts) ---
    # 3 cycles of length 0.30, followed by a 0.10 terminal feasibility phase.
    # Each cycle aggressively explores with a high LR, then cools down.
    cycle_length = 0.30
    terminal_start = 0.90
    
    # Determine cycle index (0, 1, or 2)
    cycle_idx = jnp.floor(progress / cycle_length)
    cycle_idx = jnp.clip(cycle_idx, 0, 2)
    
    # Progress within the current cycle [0.0, 1.0]
    p_local = (progress - cycle_idx * cycle_length) / cycle_length
    p_local = jnp.clip(p_local, 0.0, 1.0)
    
    # Cosine annealing factor: 1.0 at cycle start, 0.0 at cycle end
    cos_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * p_local))
    inv_cos = 1.0 - cos_decay
    
    # Decay peak learning rate across cycles
    lr_max_start = 1.5 * D_f
    lr_max_end = 0.5 * D_f
    lr_max_current = lr_max_start - (lr_max_start - lr_max_end) * (cycle_idx / 2.0)
    
    lr_main = gamma_min_f + (lr_max_current - gamma_min_f) * cos_decay

    # --- 2. Cyclic Alpha with Mid-Run Feasibility Bursts ---
    # Decoupled from 1/lr: alpha starts low to permit layout shifts, 
    # then bursts at the end of each cycle to force a locally feasible state.
    # The peak of the burst scales up in later cycles.
    alpha_base = alpha0 * 0.1
    alpha_peak_start = alpha0 * 5.0
    alpha_peak_end = alpha0 * 30.0
    alpha_peak_current = alpha_peak_start + (alpha_peak_end - alpha_peak_start) * (cycle_idx / 2.0)
    
    alpha_main = alpha_base * cos_decay + alpha_peak_current * inv_cos

    # --- 3. Phase-Transition Adam Moments ---
    # High momentum / low beta2 during early-cycle exploration.
    # Low momentum / high beta2 during late-cycle feasibility restoration.
    b1_start, b2_start = 0.12, 0.15
    b1_end, b2_end = 0.02, 0.95
    
    beta1_main = b1_start * cos_decay + b1_end * inv_cos
    beta2_main = b2_start * cos_decay + b2_end * inv_cos

    # --- 4. Terminal Feasibility Phase ---
    # Final 10% of steps apply an absolute penalty spike to ensure constraint satisfaction.
    is_terminal = progress >= terminal_start
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2