import jax.numpy as jnp

_C = 250.0 / 240.0  # exploration scaling factor

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceable conversions (step and total_steps may be traced)
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    lr_max = _C * float(D)
    gamma_min_f = float(gamma_min)

    # 1. Multi-cycle Cosine Annealing (SGDR)
    # 3 cycles of exploration, progressively shorter and cooler, ending at 90%
    is_cycle1 = progress < 0.40
    is_cycle2 = (progress >= 0.40) & (progress < 0.70)
    is_cycle3 = (progress >= 0.70) & (progress < 0.90)
    is_terminal = progress >= 0.90
    
    # Normalized progress within the current cycle [0, 1]
    c1_prog = progress / 0.40
    c2_prog = (progress - 0.40) / 0.30
    c3_prog = (progress - 0.70) / 0.20
    
    cp = jnp.where(is_cycle1, c1_prog,
         jnp.where(is_cycle2, c2_prog,
         jnp.where(is_cycle3, c3_prog, 1.0)))

    # Cycle-specific max learning rates (decaying peaks)
    lr_max_c1 = lr_max
    lr_max_c2 = lr_max * 0.50
    lr_max_c3 = lr_max * 0.20
    
    current_lr_max = jnp.where(is_cycle1, lr_max_c1,
                     jnp.where(is_cycle2, lr_max_c2,
                     jnp.where(is_cycle3, lr_max_c3, gamma_min_f)))

    # Cosine decay for learning rate within each cycle
    lr_cyclic = gamma_min_f + 0.5 * (current_lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cp))
    
    # Short linear warmup in the first 3% of the run to stabilize initial gradients
    lr_cyclic = jnp.where(progress < 0.03,
                          gamma_min_f + (lr_cyclic - gamma_min_f) * (progress / 0.03),
                          lr_cyclic)
                          
    lr = jnp.where(is_terminal, gamma_min_f, lr_cyclic)

    # 2. Cyclic Alpha (Penalty) with Mid-run Restoration Bursts
    # At the start of each cycle, the penalty is relaxed to a moderate base level to allow 
    # unhindered exploration. As the learning rate cools, the penalty aggressively ramps up 
    # to enforce constraints and restore feasibility before the next exploration burst.
    base_c1 = alpha0 * 1.0
    base_c2 = alpha0 * 3.0
    base_c3 = alpha0 * 6.0
    
    current_alpha_base = jnp.where(is_cycle1, base_c1,
                         jnp.where(is_cycle2, base_c2,
                         jnp.where(is_cycle3, base_c3, alpha0 * 10.0)))
    
    burst_max_c1 = alpha0 * 15.0
    burst_max_c2 = alpha0 * 30.0
    burst_max_c3 = alpha0 * 60.0
    
    current_burst_max = jnp.where(is_cycle1, burst_max_c1,
                        jnp.where(is_cycle2, burst_max_c2,
                        jnp.where(is_cycle3, burst_max_c3, current_alpha_base)))

    # Smooth polynomial ramp that concentrates the burst at the end of the cycle
    burst_profile = cp ** 4.0
    
    alpha_main = current_alpha_base + (current_burst_max - current_alpha_base) * burst_profile
    
    # Terminal feasibility spike to forcibly resolve any remaining violation tolerance
    alpha_terminal = alpha0 * float(D) / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # 3. Phase-transition Adam moments
    # Drop momentum (beta1) and increase curvature absorption (beta2) during the feasibility bursts
    # to handle the stiff penalty gradients without overshooting.
    beta1 = jnp.where(is_terminal, 0.05, 0.10 - 0.05 * burst_profile)
    beta2 = jnp.where(is_terminal, 0.90, 0.20 + 0.70 * burst_profile)

    return lr, alpha, beta1, beta2