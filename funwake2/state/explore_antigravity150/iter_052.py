import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- SGDR with Cyclic Alpha (3 Cycles) ---
    # A structural shift to Stochastic Gradient Descent with Warm Restarts (SGDR).
    # We use multi-cycle cosine annealing coupled with a "cyclic alpha".
    # This provides mid-run feasibility-restoration bursts, followed by sudden 
    # relaxations (warm restarts) to escape poor local constraint boundaries.
    
    # Cycle boundaries (increasing lengths: ~1x, 2x, 4x)
    c0_end = 0.15
    c1_end = 0.45
    c2_end = 1.00
    
    cycle = jnp.where(progress < c0_end, 0.0,
            jnp.where(progress < c1_end, 1.0, 2.0))
            
    cycle_progress = jnp.where(progress < c0_end, progress / c0_end,
                     jnp.where(progress < c1_end, (progress - c0_end) / (c1_end - c0_end),
                                                  (progress - c1_end) / (c2_end - c1_end)))
    
    cycle_progress = jnp.clip(cycle_progress, 0.0, 1.0)
                                                  
    # 1. Cyclic Learning Rate
    # Start with high exploration, decaying peak LR across cycles
    lr_max_cycle = jnp.where(cycle == 0.0, 1.50 * D_f,
                   jnp.where(cycle == 1.0, 0.75 * D_f, 
                                           0.30 * D_f))
                                           
    # Cosine annealing from lr_max down to gamma_min within each cycle
    lr_main = gamma_min_f + 0.5 * (lr_max_cycle - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    # 2. Cyclic Alpha (Decoupled Penalty)
    # Alpha ramps up smoothly via inverse-cosine over each cycle as LR drops.
    # At cycle boundaries, the penalty suddenly drops to allow massive restructuring,
    # before tightening down harder in the next cycle.
    alpha_base_cycle = jnp.where(cycle == 0.0, 0.1 * alpha0_f,
                       jnp.where(cycle == 1.0, 0.5 * alpha0_f, 
                                               2.0 * alpha0_f))
                                               
    alpha_peak_cycle = jnp.where(cycle == 0.0,  5.0 * alpha0_f,
                       jnp.where(cycle == 1.0, 15.0 * alpha0_f, 
                                               50.0 * alpha0_f))
                                               
    alpha_ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * cycle_progress))
    alpha_main = alpha_base_cycle + (alpha_peak_cycle - alpha_base_cycle) * alpha_ramp
    
    # 3. Phase-Transition Adam Moments
    # Reset momentum at warm restarts to allow rapid directional changes,
    # then heavily damp oscillations as the penalty peaks.
    b1_start, b1_end = 0.15, 0.02
    b2_start, b2_end = 0.15, 0.95
    
    beta1_main = b1_start + (b1_end - b1_start) * alpha_ramp
    beta2_main = b2_start + (b2_end - b2_start) * alpha_ramp
    
    # 4. Terminal Feasibility Spike
    # Final 5% of steps lock down absolute compliance (filter method)
    is_terminal = progress >= 0.95
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2