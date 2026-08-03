import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. Multi-Cycle Cosine Annealing (SGDR) ---
    # We use 3 cycles of length 0.30, reserving the last 10% for the terminal phase.
    decay_end = 0.90
    cycle_length = 0.30
    
    cycle_progress = progress / cycle_length
    current_cycle = jnp.floor(cycle_progress)
    
    # The phase goes from 0.0 to 1.0 within each cycle.
    # We clip it for the terminal phase to avoid undefined behavior.
    phase_raw = cycle_progress - current_cycle
    phase = jnp.where(progress < decay_end, phase_raw, 1.0)
    
    # Peak learning rate decays across cycles (100% -> 70% -> 49%)
    lr_peak_initial = 1.30 * D_f
    lr_peak = lr_peak_initial * (0.70 ** current_cycle)
    
    # Cosine annealing within the cycle
    lr_cosine = gamma_min_f + 0.5 * (lr_peak - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * phase))
    
    # --- 2. Cyclic & Growing Alpha ---
    # A global alpha envelope that grows linearly throughout the main run
    alpha_base = alpha0 * 0.1
    alpha_target = alpha0 * 15.0
    global_alpha = alpha_base + (alpha_target - alpha_base) * (progress / decay_end)
    
    # Alpha dips at the start of each cycle to allow exploration (anti-correlated with LR)
    # The dip severity decreases over cycles (drops to 10% -> 45% -> 80% of global_alpha)
    dip_factor = 0.10 + 0.35 * current_cycle
    dip_factor = jnp.clip(dip_factor, 0.0, 1.0)
    
    alpha_cyclic = global_alpha * (dip_factor + (1.0 - dip_factor) * phase)
    
    # --- 3. Phase-Synchronized Adam Moments ---
    # Moments also transition within each cycle. 
    # Early cycles have extreme momentum drops; later cycles are more conservative.
    b1_start_cycle = jnp.maximum(0.15 - 0.05 * current_cycle, 0.02)
    b2_start_cycle = jnp.minimum(0.15 + 0.30 * current_cycle, 0.85)
    
    b1_end, b2_end = 0.02, 0.85
    
    beta1_cyclic = b1_start_cycle + (b1_end - b1_start_cycle) * phase
    beta2_cyclic = b2_start_cycle + (b2_end - b2_start_cycle) * phase
    
    # --- 4. Terminal Feasibility Spike ---
    # Final 10% strictly enforces the constraint with minimal learning rate.
    is_terminal = progress >= decay_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cosine)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyclic)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cyclic)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cyclic)

    return lr, alpha, beta1, beta2