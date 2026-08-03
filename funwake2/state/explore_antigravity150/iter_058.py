import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- 1. SGDR: Multi-cycle Cosine Annealing ---
    # We divide the first 90% of the optimization into 3 equal cycles.
    # Warm restarts help escape local minima and explore different layouts.
    n_cycles = 3.0
    cyclic_phase_end = 0.90
    
    # Progress within the cyclic phase [0, 1]
    cyclic_progress = jnp.clip(progress / cyclic_phase_end, 0.0, 1.0)
    
    # Fractional progress within the current cycle [0, 1)
    # Using modulo to automatically loop through cycles
    cycle_frac = (cyclic_progress * n_cycles) % 1.0
    
    # The current cycle index: 0, 1, or 2
    # Ensure it doesn't exceed 2 even at exactly cyclic_progress = 1.0
    cycle_idx = jnp.minimum(jnp.floor(cyclic_progress * n_cycles), n_cycles - 1.0)
    
    # Peak learning rate decays across cycles to encourage convergence
    # Cycle 0: 1.5 * D, Cycle 1: 1.0 * D, Cycle 2: 0.75 * D
    lr_max_initial = 1.5 * D_f
    lr_max = lr_max_initial / (1.0 + cycle_idx * 0.5)
    
    # Cosine decay within each cycle
    lr_cyclic = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_frac))
    
    # --- 2. Cyclic Decoupled Penalty (Alpha) ---
    # Alpha drops at the start of each cycle to allow layout rearrangement, 
    # then hardens quadratically as the cycle ends to push turbines apart.
    # This creates mid-run feasibility-restoration bursts.
    alpha_base = alpha0_f * 0.1
    alpha_peak = alpha0_f * 20.0 * (1.0 + cycle_idx * 0.5)
    
    # Quadratic ramp within the cycle to delay the strong penalty until the end
    alpha_cyclic = alpha_base + (alpha_peak - alpha_base) * (cycle_frac ** 2)
    
    # --- 3. Cyclic Adam Moments ---
    # Synchronize Adam moments with the cycles:
    # Highly reactive (low beta2) during exploration to jump around.
    # More stable (higher beta2) as penalty kicks in to absorb stiff gradients.
    b1_start, b1_end = 0.12, 0.02
    b2_start, b2_end = 0.15, 0.85
    
    beta1_cyclic = b1_end + 0.5 * (b1_start - b1_end) * (1.0 + jnp.cos(jnp.pi * cycle_frac))
    beta2_cyclic = b2_end + 0.5 * (b2_start - b2_end) * (1.0 + jnp.cos(jnp.pi * cycle_frac))
    
    # --- 4. Terminal Feasibility Spike ---
    # In the final 10% of steps, freeze exploration. Drop LR to gamma_min and
    # spike the penalty dramatically to guarantee strict constraint compliance.
    is_terminal = progress >= cyclic_phase_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cyclic)
    
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyclic)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cyclic)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cyclic)

    return lr, alpha, beta1, beta2