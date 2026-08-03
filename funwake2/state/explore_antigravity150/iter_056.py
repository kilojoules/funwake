import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR Cyclic Exploration with Mid-Run Feasibility Bursts ---
    # Structurally different from WSD/Plateau: we use 3 cycles of Cosine Annealing.
    # At the start of each cycle, LR restarts high and Penalty (alpha) drops low,
    # allowing massive layout shifts and tunneling through constraint barriers.
    # As the cycle progresses, LR cools and Penalty spikes, pulling the farm 
    # back into a locally feasible, refined configuration.
    
    p_cyclic_end = 0.85  # Dedicate 85% of the run to 3 search cycles
    n_cycles = 3.0
    
    # Map progress into [0, 1] for the cyclic phase
    p_cyclic = jnp.clip(progress / p_cyclic_end, 0.0, 1.0)
    
    # Current cycle index (0, 1, 2)
    cycle_idx = jnp.minimum(jnp.floor(p_cyclic * n_cycles), n_cycles - 1.0)
    
    # Local progress within the current cycle [0, 1]
    local_p = (p_cyclic * n_cycles) - cycle_idx
    
    # SGDR Cosine multiplier (1.0 at start of cycle, 0.0 at end)
    cosine_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * local_p))
    
    # Cycle-dependent peaks: LR decays across cycles, Alpha grows across cycles
    # Cycle 0: Highest LR, lowest peak penalty. Cycle 2: Lower LR, tighter penalty.
    lr_peak = (1.5 * D_f) * (0.8 ** cycle_idx)
    lr_min_cycle = 0.1 * D_f
    
    alpha_base = alpha0 * 0.05
    alpha_peak_cycle = (alpha0 * 5.0) * (1.5 ** cycle_idx)
    
    # LR follows the cosine decay
    lr_cyclic = lr_min_cycle + (lr_peak - lr_min_cycle) * cosine_mult
    
    # Alpha is inverted relative to LR: highest when LR is lowest (end of cycle)
    alpha_cyclic = alpha_base + (alpha_peak_cycle - alpha_base) * (1.0 - cosine_mult)
    
    # Cyclic Moments: high momentum (low beta2) when LR is high to encourage movement
    b1_peak, b1_low = 0.15, 0.05
    b2_peak, b2_low = 0.15, 0.70
    beta1_cyclic = b1_low + (b1_peak - b1_low) * cosine_mult
    beta2_cyclic = b2_low + (b2_peak - b2_low) * cosine_mult
    
    # --- 2. Terminal Strict Feasibility Cooldown ---
    # The final 15% freezes exploration and aggressively enforces constraints.
    # We linearly decay LR to gamma_min and exponentially surge alpha.
    
    terminal_p = jnp.clip((progress - p_cyclic_end) / (1.0 - p_cyclic_end), 0.0, 1.0)
    
    lr_terminal = lr_min_cycle - (lr_min_cycle - gamma_min_f) * terminal_p
    
    # Terminal absolute feasibility spike
    alpha_super = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    # Power 4 delays the massive spike until the very end, preventing early freezing
    alpha_terminal = alpha_peak_cycle + (alpha_super - alpha_peak_cycle) * (terminal_p ** 4.0)
    
    beta1_terminal = b1_low * (1.0 - terminal_p) + 0.01 * terminal_p
    beta2_terminal = b2_low * (1.0 - terminal_p) + 0.99 * terminal_p
    
    # --- 3. Combination and Final Guard ---
    is_cyclic = progress < p_cyclic_end
    
    lr_main = jnp.where(is_cyclic, lr_cyclic, lr_terminal)
    alpha_main = jnp.where(is_cyclic, alpha_cyclic, alpha_terminal)
    beta1_main = jnp.where(is_cyclic, beta1_cyclic, beta1_terminal)
    beta2_main = jnp.where(is_cyclic, beta2_cyclic, beta2_terminal)
    
    # Hard terminal guard for the absolute final steps to guarantee feasibility
    is_very_end = progress >= 0.98
    lr = jnp.where(is_very_end, gamma_min_f, lr_main)
    alpha = jnp.where(is_very_end, alpha_super, alpha_main)
    beta1 = jnp.where(is_very_end, 0.01, beta1_main)
    beta2 = jnp.where(is_very_end, 0.99, beta2_main)
    
    return lr, alpha, beta1, beta2