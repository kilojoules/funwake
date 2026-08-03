import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Cyclic SGDR with Warm Restarts and Feasibility Bursts ---
    # We use 3 cycles of length 0.30, followed by a 0.10 terminal phase.
    # This directly tests the multi-cycle cosine LR + cyclic alpha hypotheses,
    # giving the optimizer repeated chances to blast out of local minima 
    # before re-enforcing constraints.
    cycle_length = 0.30
    terminal_start = 0.90
    
    is_terminal = progress >= terminal_start
    
    # Identify cycle index (0, 1, or 2)
    cycle_idx = jnp.minimum(jnp.floor(progress / cycle_length), 2.0)
    
    # Progress within the current cycle [0.0, 1.0]
    p_cycle = (progress - cycle_idx * cycle_length) / cycle_length
    
    # --- 2. Learning Rate: Decaying Multi-Cycle Cosine ---
    # Cycle 0 peak: 1.25 * D, Cycle 1: 0.90 * D, Cycle 2: 0.55 * D
    lr_max = D_f * (1.25 - 0.35 * cycle_idx)
    lr_min = gamma_min_f * 5.0
    
    # Cosine annealing within the cycle
    lr_cyclic = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * p_cycle))
    
    # --- 3. Decoupled Penalty (Alpha): Mid-run Feasibility Bursts ---
    # Within each cycle, we start with a soft penalty for broad exploration,
    # then linearly ramp to a 'burst' penalty in the final 30% of the cycle
    # to restore feasibility. The base and burst penalties stiffen across cycles.
    alpha_base = alpha0 * (0.2 + 2.0 * cycle_idx)
    alpha_burst = alpha0 * (8.0 + 12.0 * cycle_idx)
    
    # Linear ramp for the burst in the last 30% of the cycle (p_cycle >= 0.70)
    burst_ramp = jnp.clip((p_cycle - 0.70) / 0.30, 0.0, 1.0)
    alpha_cyclic = alpha_base + (alpha_burst - alpha_base) * burst_ramp
    
    # --- 4. Phase-Transition Adam Moments ---
    # Synchronize with the feasibility bursts: drop momentum and raise beta2 
    # when the penalty peaks, absorbing the stiff constraint curvature.
    b1_expl = 0.12
    b2_expl = 0.15
    
    b1_burst = 0.04
    b2_burst = 0.85
    
    beta1_cyclic = b1_expl + (b1_burst - b1_expl) * burst_ramp
    beta2_cyclic = b2_expl + (b2_burst - b2_expl) * burst_ramp
    
    # --- 5. Terminal Phase ---
    # Absolute feasibility restoration and strict cooling for the final 10%.
    lr = jnp.where(is_terminal, gamma_min_f, lr_cyclic)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyclic)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cyclic)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cyclic)

    return lr, alpha, beta1, beta2