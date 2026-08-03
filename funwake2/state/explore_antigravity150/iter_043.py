import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Multi-cycle Cosine Annealing (SGDR) with Flat Peaks ---
    # A structural departure to break out of local optima: we use 3 cycles 
    # (warm restarts) over the first 90% of the optimization. Each cycle holds 
    # a flat, high exploration peak before cosine-decaying.
    cycle_end = 0.90
    is_terminal = progress >= cycle_end
    
    # Map progress to 3 discrete cycles
    cyc_progress = jnp.minimum(progress / cycle_end, 1.0)
    n_cycles = 3.0
    cycle_position = cyc_progress * n_cycles
    cycle_id = jnp.minimum(jnp.floor(cycle_position), n_cycles - 1.0)
    local_progress = cycle_position - cycle_id
    
    # Decaying peak for each warm restart (starts at 1.6*D, higher than parent)
    lr_max = D_f * (1.6 - 0.4 * cycle_id)
    lr_min = gamma_min_f + 1e-6
    
    # Flat peak for the first 25% of each cycle, then cosine decay
    decay_phase = jnp.maximum(0.0, (local_progress - 0.25) / 0.75)
    lr_main = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * decay_phase))
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Decoupled Alpha with Mid-Run Feasibility Bursts ---
    # Overall trend: Logistic ramp centered at mid-run
    k = 15.0
    p0 = 0.50
    logistic_ramp = 1.0 / (1.0 + jnp.exp(-k * (cyc_progress - p0)))
    
    alpha_base = alpha0 * 0.1
    alpha_plateau = alpha0 * 12.0
    alpha_trend = alpha_base + (alpha_plateau - alpha_base) * logistic_ramp
    
    # Mid-run feasibility-restoration bursts: at the end of each cycle (last 20%),
    # as LR bottoms out, we rapidly spike Alpha. This snaps the layout back to 
    # feasibility before the next explosive warm restart.
    burst_gate = jnp.clip((local_progress - 0.80) / 0.20, 0.0, 1.0)
    
    # Bursts get progressively stronger each cycle
    burst_multiplier = alpha0 * 20.0 * (cycle_id + 1.0) / 3.0
    alpha_main = alpha_trend + burst_multiplier * burst_gate
    
    # Terminal feasibility spike ensures absolute compliance in the final 10%
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Phase-Transition Adam Moments ---
    # Synchronize baseline moments with the logistic ramp.
    b1_start, b2_start = 0.12, 0.15
    b1_plateau, b2_plateau = 0.04, 0.85
    
    beta1_trend = b1_start + (b1_plateau - b1_start) * logistic_ramp
    beta2_trend = b2_start + (b2_plateau - b2_start) * logistic_ramp
    
    # During burst phases (constraint snapping), strongly damp momentum to 
    # absorb constraint stiffness and avoid violent oscillations.
    beta1_main = beta1_trend - (beta1_trend - 0.01) * burst_gate
    beta2_main = beta2_trend + (0.95 - beta2_trend) * burst_gate
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2