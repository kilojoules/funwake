import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR Multi-Cycle Cosine Learning Rate ---
    # We allocate the first 90% of steps to 3 exploration/feasibility cycles.
    main_phase_end = 0.90
    main_progress = jnp.minimum(progress / main_phase_end, 1.0)
    
    num_cycles = 3.0
    cycle = jnp.floor(main_progress * num_cycles)
    cycle = jnp.minimum(cycle, num_cycles - 1.0)  # bounds to [0, 1, 2]
    
    # Fractional progress within the current cycle [0, 1]
    frac = (main_progress * num_cycles) - cycle
    
    # Warm restarts: peak LR starts high and decays slightly each cycle
    lr_max = 1.25 * D_f * (0.8 ** cycle) 
    lr_min = gamma_min_f
    
    # Cosine annealing for LR: high at cycle start, low at cycle end
    lr_cos = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * frac))
    
    is_terminal = progress >= main_phase_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_cos)

    # --- 2. Cyclic Alpha with Mid-Run Feasibility Bursts ---
    # Alpha anti-correlates with LR: soft during exploration, surging at the end
    # of each cycle to force layout convergence and feasibility restoration.
    alpha_base = alpha0 * 1.5
    
    # The feasibility burst gets progressively stricter each cycle
    alpha_peak = alpha0 * 12.0 * (1.5 ** cycle)
    
    # Inverse cosine for alpha: starts low, ends high (mid-run bursts)
    alpha_cyc = alpha_base + 0.5 * (alpha_peak - alpha_base) * (1.0 - jnp.cos(jnp.pi * frac))
    
    # Terminal feasibility spike (filter method) ensures absolute compliance at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyc)
    
    # --- 3. Phase-Transition Adam Moments ---
    # Synchronize moments with the SGDR cycles:
    # beta2 up / beta1 down in the feasibility phase of each cycle.
    b1_start, b1_end = 0.12, 0.02
    b2_start, b2_end = 0.15, 0.90
    
    beta1_cyc = b1_start + 0.5 * (b1_end - b1_start) * (1.0 - jnp.cos(jnp.pi * frac))
    beta2_cyc = b2_start + 0.5 * (b2_end - b2_start) * (1.0 - jnp.cos(jnp.pi * frac))
    
    # Extreme damping for the terminal spike
    beta1 = jnp.where(is_terminal, 0.01, beta1_cyc)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cyc)
    
    return lr, alpha, beta1, beta2