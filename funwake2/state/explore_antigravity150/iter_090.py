import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Cyclic LR with SGDR Warm Restarts ---
    # We divide the first 90% of the optimization into 3 equal cycles.
    # Each cycle starts with a high learning rate (warm restart) to escape local minima,
    # then decays via cosine annealing to fine-tune.
    T_phase = 0.90
    num_cycles = 3.0
    cycle_len = T_phase / num_cycles

    # current_cycle is 0, 1, or 2
    current_cycle = jnp.minimum(jnp.floor(progress / cycle_len), num_cycles - 1.0)
    cycle_prog = (progress - current_cycle * cycle_len) / cycle_len

    # Peak LR decays across cycles to gradually zero in on the global layout structure
    lr_max_start = 1.25 * D_f
    lr_max_cycle = lr_max_start * (0.6 ** current_cycle)
    
    # Cosine annealing within the cycle
    lr_cycle = gamma_min_f + 0.5 * (lr_max_cycle - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_prog))

    # --- 2. Cyclic Decoupled Penalty (Alpha) ---
    # At the start of each cycle, the penalty is softened to allow free layout movement.
    # In the second half of the cycle, it ramps sharply to a plateau to enforce feasibility.
    alpha_base_cycle = alpha0 * (0.2 + 0.3 * current_cycle)
    alpha_plateau_cycle = alpha0 * (4.0 * (1.5 ** current_cycle)) 
    
    # Logistic ramp centered at 65% of the cycle
    k = 25.0
    p0 = 0.65
    cyclic_alpha_ramp = 1.0 / (1.0 + jnp.exp(-k * (cycle_prog - p0)))
    alpha_cycle = alpha_base_cycle + (alpha_plateau_cycle - alpha_base_cycle) * cyclic_alpha_ramp

    # --- 3. Phase-Transition Adam Moments ---
    # Synchronize momentum with the feasibility enforcement:
    # High momentum (low beta2) during exploration; clamped momentum (high beta2) during feasibility
    b1_explore, b2_explore = 0.15, 0.10
    b1_feas, b2_feas = 0.02, 0.90

    beta1_cycle = b1_explore + (b1_feas - b1_explore) * cyclic_alpha_ramp
    beta2_cycle = b2_explore + (b2_feas - b2_explore) * cyclic_alpha_ramp

    # --- 4. Terminal Feasibility Spike ---
    # The final 10% strictly enforces absolute constraint compliance.
    is_terminal = progress >= T_phase
    
    # LR drops immediately to the tolerance scale
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    # Alpha and moments ramp to their terminal limits over the first half of the terminal phase (from 0.90 to 0.95)
    alpha_terminal_target = alpha0 * (D_f / jnp.maximum(gamma_min_f, 1e-30))
    terminal_ramp = jnp.clip((progress - T_phase) / 0.05, 0.0, 1.0)
    
    alpha_terminal = alpha_plateau_cycle + (alpha_terminal_target - alpha_plateau_cycle) * terminal_ramp
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)

    beta1_terminal = b1_feas - (b1_feas - 0.01) * terminal_ramp
    beta2_terminal = b2_feas + (0.99 - b2_feas) * terminal_ramp
    
    beta1 = jnp.where(is_terminal, beta1_terminal, beta1_cycle)
    beta2 = jnp.where(is_terminal, beta2_terminal, beta2_cycle)

    return lr, alpha, beta1, beta2