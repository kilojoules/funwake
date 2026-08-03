import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR Multi-Cycle Cosine Learning Rate ---
    # We employ 3 warm-restart cycles (SGDR). 
    # Each cycle starts with a high exploratory LR and cools to gamma_min.
    # This provides multiple "warm restarts" to escape local minima in complex layouts.
    
    lr_max = 1.35 * D_f
    main_phase_end = 0.90
    
    # Scale progress for the main cyclic phase (0 to 0.90)
    phase_main = jnp.clip(progress / main_phase_end, 0.0, 1.0)
    
    num_cycles = 3.0
    cycle_position = (phase_main * num_cycles) % 1.0
    
    # Cosine decay from 1.0 to 0.0 within each cycle
    lr_cycle_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_position))
    
    # The peak of each cycle decays over time (envelope) to gradually converge
    lr_peak = lr_max * (1.0 - 0.5 * phase_main)
    
    lr_main = gamma_min_f + (lr_peak - gamma_min_f) * lr_cycle_decay
    
    # Global linear warmup for the first 3% of steps to prevent initial blowout
    warmup_end = 0.03
    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (progress / warmup_end)
    lr_main = jnp.where(progress < warmup_end, lr_warmup, lr_main)

    # --- 2. Cyclic Alpha with Mid-Run Feasibility Bursts ---
    # Alpha (penalty) operates inversely to the LR cycles. 
    # At the start of a cycle (high LR), alpha is low to allow unconstrained exploration.
    # At the end of a cycle (low LR), alpha bursts to pull the layout back to feasibility.
    
    # Base penalty slowly rises across the whole run
    alpha_base = alpha0 * (0.5 + 4.5 * phase_main) 
    
    # Burst magnitude grows with each cycle as we demand stricter compliance
    alpha_burst_mag = alpha0 * 15.0 * phase_main 
    
    # Burst shape is the inverse of the LR cycle, squared to sharpen the spike
    # so it only kicks in right at the end of each cycle.
    burst_shape = (1.0 - lr_cycle_decay) ** 2
    
    alpha_main = alpha_base + alpha_burst_mag * burst_shape

    # --- 3. Phase-Transition Adam Moments ---
    # Synchronize moments with the SGDR and Alpha bursts.
    # High momentum / low beta2 during the exploratory start of a cycle.
    # Drop momentum / raise beta2 during the alpha burst to absorb stiff curvature.
    
    beta1_main = 0.15 - 0.13 * burst_shape  # Drops from 0.15 down to 0.02 during bursts
    beta2_main = 0.20 + 0.75 * burst_shape  # Rises from 0.20 up to 0.95 during bursts

    # --- 4. Terminal Feasibility Spike ---
    # The final 10% strictly enforces the absolute constraint by holding
    # maximum penalty and minimum step size, ensuring a valid final layout.
    is_terminal = progress >= main_phase_end
    
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0 * (D_f / jnp.maximum(gamma_min_f, 1e-30))
    
    lr = jnp.where(is_terminal, lr_terminal, lr_main)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2