import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR Multi-Cycle Schedule (Warm Restarts) ---
    # We compress 3 complete cycles of cosine annealing into the first 90% 
    # of the run. Each cycle explores with high LR and low alpha, then settles
    # with low LR and high alpha (a mid-run feasibility-restoration burst).
    
    phase_end = 0.90
    effective_progress = jnp.clip(progress / phase_end, 0.0, 1.0)
    
    num_cycles = 3.0
    cycle_idx = jnp.minimum(jnp.floor(effective_progress * num_cycles), num_cycles - 1.0)
    cycle_progress = (effective_progress * num_cycles) - cycle_idx
    
    # --- 2. Cyclic Learning Rate ---
    # LR peak drops with each cycle: 1.8 D -> 1.2 D -> 0.6 D
    lr_max = (1.8 - 0.6 * cycle_idx) * D_f
    lr_cycle = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    # --- 3. Cyclic Decoupled Alpha ---
    # Alpha mirrors LR: low during exploration, rising as the cycle cools.
    # The peak penalty increases with each cycle: 10 alpha0 -> 15 alpha0 -> 20 alpha0
    alpha_base = alpha0 * 0.1
    alpha_peak = alpha0 * (10.0 + 5.0 * cycle_idx)
    alpha_cycle = alpha_base + 0.5 * (alpha_peak - alpha_base) * (1.0 - jnp.cos(jnp.pi * cycle_progress))
    
    # --- 4. Cyclic Adam Moments ---
    # Phase-transition the moments with the cycle:
    # High LR (exploration): high momentum, low beta2 (rapid adaptation to new gradients)
    # Low LR (feasibility): drop momentum, raise beta2 (damping for stiff penalty walls)
    b1_start, b1_end = 0.15, 0.02
    b2_start, b2_end = 0.10, 0.90
    
    beta1_cycle = b1_end + 0.5 * (b1_start - b1_end) * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    beta2_cycle = b2_start + 0.5 * (b2_end - b2_start) * (1.0 - jnp.cos(jnp.pi * cycle_progress))
    
    # --- 5. Terminal Feasibility Spike ---
    # The last 10% of the run guarantees strict absolute constraint compliance
    is_terminal = progress >= phase_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2