import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- Structural Change: SGDR Multi-cycle with Cyclic Penalty ---
    # Instead of a single monotonic cooldown, we use 3 distinct cycles.
    # At the start of each cycle, LR is restarted high and Alpha is dropped low
    # to allow massive layout reorganization and escape local minima.
    # As the cycle progresses, LR cosine-decays and Alpha cosine-ramps up 
    # to enforce feasibility, creating mid-run feasibility-restoration bursts.
    
    terminal_start = 0.90
    is_terminal = progress >= terminal_start

    # We fit exactly 3 cycles into the first 90% of the optimization
    num_cycles = 3.0
    cycle_len = terminal_start / num_cycles
    
    # Identify which cycle we are in (0, 1, or 2)
    cycle_idx = jnp.minimum(jnp.floor(progress / cycle_len), num_cycles - 1.0)
    
    # Progress within the current cycle [0.0, 1.0]
    cycle_progress = jnp.clip((progress - cycle_idx * cycle_len) / cycle_len, 0.0, 1.0)

    # --- 1. Cyclic Learning Rate (SGDR) ---
    # lr_max decays across cycles: 1.5*D -> 0.9*D -> 0.54*D
    lr_max_initial = 1.5 * D_f
    lr_max = lr_max_initial * (0.6 ** cycle_idx)
    lr_min = gamma_min_f
    
    # Cosine annealing within the cycle
    lr_cycle = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * cycle_progress))

    # --- 2. Cyclic Alpha (Feasibility Bursts) ---
    # Start each cycle extremely loose (0.1 * alpha0) to allow free movement.
    # Peak penalty strictly increases across cycles: 5x -> 10x -> 20x alpha0
    alpha_base = alpha0 * 0.1
    alpha_peak = alpha0 * (5.0 * (2.0 ** cycle_idx))
    
    # Reversed cosine annealing: ramps up from base to peak as the cycle ends
    alpha_cycle = alpha_base + 0.5 * (alpha_peak - alpha_base) * (1.0 - jnp.cos(jnp.pi * cycle_progress))

    # --- 3. Cyclic Adam Moments ---
    # Synchronize Adam with the feasibility bursts.
    # High momentum (low beta2) during loose exploration;
    # Low momentum (high beta2) during tight constraint enforcement.
    b1_start, b2_start = 0.15, 0.20
    b1_end, b2_end = 0.02, 0.90
    
    beta1_cycle = b1_start + 0.5 * (b1_end - b1_start) * (1.0 - jnp.cos(jnp.pi * cycle_progress))
    beta2_cycle = b2_start + 0.5 * (b2_end - b2_start) * (1.0 - jnp.cos(jnp.pi * cycle_progress))

    # --- 4. Terminal Phase (Strict Feasibility Clamping) ---
    # The final 10% throws away cycles and forces absolute exact-penalty compliance.
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2