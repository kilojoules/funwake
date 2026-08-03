import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. Multi-Cycle SGDR (Warm Restarts) ---
    # We break the optimization into 3 expanding cycles before the terminal phase.
    # Each cycle starts with high LR (exploration) and decays to gamma_min (fine-tuning).
    # Cycle 0: [0.00, 0.15) - Violent exploration
    # Cycle 1: [0.15, 0.40) - Basin search
    # Cycle 2: [0.40, 0.90) - Deep fine-tuning
    
    is_cycle_0 = progress < 0.15
    is_cycle_1 = (progress >= 0.15) & (progress < 0.40)
    
    cycle_progress = jnp.where(
        is_cycle_0, progress / 0.15,
        jnp.where(
            is_cycle_1, (progress - 0.15) / 0.25,
            (progress - 0.40) / 0.50
        )
    )
    cycle_progress = jnp.clip(cycle_progress, 0.0, 1.0)
    
    # Peak LR decays across cycles to focus the search
    lr_max = jnp.where(is_cycle_0, 1.50 * D_f,
             jnp.where(is_cycle_1, 0.80 * D_f, 0.40 * D_f))
             
    # Cosine annealing within cycle
    lr_cycle = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    # --- 2. Cyclic Mid-Run Feasibility Bursts ---
    # Alpha (penalty) is anti-correlated with LR. 
    # When LR drops at the end of a cycle, alpha spikes to pull the layout into the feasible space.
    # The spikes get progressively stronger across cycles to solidify constraint satisfaction.
    alpha_base = jnp.where(is_cycle_0, alpha0 * 0.05,
                 jnp.where(is_cycle_1, alpha0 * 0.50, alpha0 * 2.0))
                 
    alpha_peak = jnp.where(is_cycle_0, alpha0 * 3.0,
                 jnp.where(is_cycle_1, alpha0 * 10.0, alpha0 * 40.0))
                 
    # Inverse cosine ramp-up within cycle
    alpha_cycle = alpha_base + 0.5 * (alpha_peak - alpha_base) * (1.0 - jnp.cos(jnp.pi * cycle_progress))
    
    # --- 3. Phase-Transition Adam Moments ---
    # High momentum (low b2, high b1) at start of cycle for maximum unrestrained movement.
    # Low momentum (high b2, low b1) at end of cycle to absorb stiffness of the cyclic penalty spike.
    b1_start = jnp.where(is_cycle_0, 0.15, jnp.where(is_cycle_1, 0.10, 0.05))
    b1_end = 0.01
    beta1_cycle = b1_start + 0.5 * (b1_end - b1_start) * (1.0 - jnp.cos(jnp.pi * cycle_progress))
    
    b2_start = jnp.where(is_cycle_0, 0.10, jnp.where(is_cycle_1, 0.25, 0.50))
    b2_end = 0.95
    beta2_cycle = b2_start + 0.5 * (b2_end - b2_start) * (1.0 - jnp.cos(jnp.pi * cycle_progress))
    
    # --- 4. Terminal Feasibility Lock ---
    # The last 10% strictly enforces compliance with zero remaining exploration.
    is_terminal = progress >= 0.90
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    # Filter-method absolute feasibility spike (from successful parent)
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    # Heavily damped moments for stability against massive penalty gradients
    beta1 = jnp.where(is_terminal, 0.001, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.999, beta2_cycle)
    
    return lr, alpha, beta1, beta2