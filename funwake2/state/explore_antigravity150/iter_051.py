import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    progress = step_f / total_f

    # --- 1. Structurally Different: SGDR Multi-Cycle (Warm Restarts) ---
    # We break from the WSD/monotonic paradigm and run 3 distinct cycles.
    # Each cycle uses a massive initial LR to explode the layout, then smoothly 
    # anneals into a mid-run feasibility-restoration burst, before restarting.
    main_end = 0.90
    n_cycles = 3.0
    
    # Map progress to [0.0, 1.0] for the multi-cycle phase
    main_progress = jnp.clip(progress / main_end, 0.0, 1.0)
    cycle_phase = main_progress * n_cycles
    
    # Determine which cycle we are in (0, 1, or 2)
    cycle_idx = jnp.minimum(jnp.floor(cycle_phase), n_cycles - 1.0)
    
    # Progress within the current cycle: [0.0, 1.0]
    cycle_tau = cycle_phase - cycle_idx
    
    # Cosine shapes for smooth transitions within each cycle
    decay_shape = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_tau))  # 1.0 -> 0.0
    rise_shape = 1.0 - decay_shape                           # 0.0 -> 1.0

    # --- 2. Cyclic Learning Rate ---
    # Decaying peak learning rates across the 3 restarts (1.5, 1.2, 0.9 * D)
    lr_max_c = D_f * (1.5 - 0.3 * cycle_idx)
    lr_main = gamma_min_f + (lr_max_c - gamma_min_f) * decay_shape

    # --- 3. Cyclic Alpha Penalty with Mid-Run Restorations ---
    # Alpha ramps up strongly at the end of every cycle. This pulls the layout 
    # back into the feasible region to evaluate a valid structure before the 
    # next warm restart kicks it out again.
    alpha_low = alpha0_f * 0.1
    # Peaks get progressively stricter: 8x, 12x, 16x
    alpha_high_c = alpha0_f * (8.0 + 4.0 * cycle_idx) 
    alpha_main = alpha_low + (alpha_high_c - alpha_low) * rise_shape

    # --- 4. Cyclic Adam Moments ---
    # Sync moments with the cycles to match the required optimization mode:
    # Start of cycle (explore): low b1, low b2 (highly reactive, unconstrained)
    # End of cycle (restore): very low b1, high b2 (damped, absorbs penalty gradients)
    b1_start, b2_start = 0.15, 0.15
    b1_end, b2_end = 0.05, 0.95
    
    beta1_main = b1_start + (b1_end - b1_start) * rise_shape
    beta2_main = b2_start + (b2_end - b2_start) * rise_shape

    # --- 5. Terminal Feasibility Phase ---
    # A guaranteed pure feasibility mode for the final 10% of steps.
    # Drops LR to tolerance, spikes penalty to maximum, and freezes momentum.
    is_terminal = progress >= main_end
    
    alpha_terminal = alpha0_f * (D_f / jnp.maximum(gamma_min_f, 1e-30))
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2