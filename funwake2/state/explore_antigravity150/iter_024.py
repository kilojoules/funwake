import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR: Multi-Cycle Cosine Learning Rate ---
    # We use 3 cycles for the first 85% of the run.
    # This provides repeated exploration bursts (warm restarts), allowing the
    # optimizer to escape local minima multiple times.
    phase_end = 0.85
    cycles = 3.0
    
    # Scale progress to the cyclic phase
    cyclic_progress = jnp.clip(progress / phase_end, 0.0, 1.0) * cycles
    
    # local_progress goes from 0 to 1 within each cycle.
    # The modulo operator naturally creates the warm restarts (jumps back to 0).
    local_progress = jnp.where(progress >= phase_end, 0.0, cyclic_progress % 1.0)
    
    # Cosine annealing per cycle
    lr_max = 1.2 * D_f
    lr_min = gamma_min_f * 5.0
    
    lr_cosine = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * local_progress))
    
    # --- 2. Cyclic Alpha with Mid-Run Feasibility Bursts ---
    # Alpha starts low in each cycle to allow the LR burst to move turbines freely,
    # then spikes to a moderate peak at the end of the cycle. This acts as a 
    # mid-run feasibility-restoration burst, consolidating constraints before the next jump.
    alpha_base = alpha0 * 0.1
    alpha_peak = alpha0 * 10.0
    
    # Power curve for alpha ramp within the cycle (spikes late in the cycle)
    alpha_cyclic = alpha_base + (alpha_peak - alpha_base) * (local_progress ** 4.0)
    
    # --- 3. Terminal Feasibility Spike ---
    # The final 15% of the run is dedicated solely to feasibility and fine-tuning.
    is_terminal = progress >= phase_end
    
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    lr = jnp.where(is_terminal, lr_terminal, lr_cosine)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyclic)
    
    # --- 4. Phase-Transition Adam Moments ---
    # Sync moments with the cycles:
    # High momentum (low beta2, high beta1) during exploration (early cycle).
    # Low momentum (high beta2, low beta1) during feasibility (late cycle & terminal).
    b1_start, b2_start = 0.15, 0.10
    b1_end, b2_end = 0.05, 0.90
    
    b1_cyclic = b1_start + (b1_end - b1_start) * (local_progress ** 2.0)
    b2_cyclic = b2_start + (b2_end - b2_start) * (local_progress ** 2.0)
    
    beta1 = jnp.where(is_terminal, 0.01, b1_cyclic)
    beta2 = jnp.where(is_terminal, 0.99, b2_cyclic)
    
    return lr, alpha, beta1, beta2