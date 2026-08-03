import jax.numpy as jnp

_C = 250.0 / 240.0  # Maintain the strong early peak multiplier

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceable conversions
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    lr0 = _C * float(D)
    gamma_min_f = float(gamma_min)
    
    # Terminal phase (last 10% of steps) for strict feasibility
    is_terminal = progress >= 0.90
    
    # Normalize progress for the exploration phase (0.0 to 0.9)
    p_explore = jnp.minimum(progress / 0.90, 1.0)
    
    # 1. Multi-cycle Cosine Annealing (SGDR) Learning Rate
    # 2 cycles to allow broad exploration then localized refinement
    n_cycles = 2.0
    cycle_position = p_explore * n_cycles
    cycle_idx = jnp.minimum(jnp.floor(cycle_position), n_cycles - 1.0)
    cycle_progress = cycle_position - cycle_idx
    
    # Peak learning rate decays each cycle (1.0x for first, 0.6x for second)
    lr_peak = lr0 * jnp.power(0.6, cycle_idx)
    
    # Cosine annealing within the cycle down to gamma_min
    lr_cycle = gamma_min_f + 0.5 * (lr_peak - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    # 2. Decoupled Cyclic Penalty with Mid-Run Bursts
    # Delay the alpha ramp until the second half of each cycle
    alpha_exploration = alpha0 * 1.0
    alpha_burst = alpha0 * 6.0
    
    ramp_factor = jnp.clip((cycle_progress - 0.5) / 0.5, 0.0, 1.0)
    alpha_cycle = alpha_exploration + (alpha_burst - alpha_exploration) * jnp.power(ramp_factor, 2.0)
    
    # Terminal feasibility spike to enforce absolute constraints at the end
    alpha_terminal = alpha0 * float(D) / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    # 3. Phase-transition Adam moments during the alpha bursts
    # Drops momentum (beta1) and increases curvature absorption (beta2) when alpha spikes
    beta1_cycle = 0.1 - 0.05 * jnp.power(ramp_factor, 2.0)
    beta2_cycle = 0.2 + 0.7 * jnp.power(ramp_factor, 2.0)
    
    beta1_terminal = 0.05
    beta2_terminal = 0.90
    
    beta1 = jnp.where(is_terminal, beta1_terminal, beta1_cycle)
    beta2 = jnp.where(is_terminal, beta2_terminal, beta2_cycle)
    
    return lr, alpha, beta1, beta2