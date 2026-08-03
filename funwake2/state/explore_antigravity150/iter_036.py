import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # Multi-cycle SGDR (Warm Restarts) with Cyclic Alpha
    # 3 cycles of exploration and feasibility restoration, followed by terminal absolute feasibility
    c1_end = 0.40
    c2_end = 0.75
    c3_end = 0.90

    c1 = progress < c1_end
    c2 = (progress >= c1_end) & (progress < c2_end)
    c3 = (progress >= c2_end) & (progress < c3_end)
    is_terminal = progress >= c3_end

    # Local progress within the current cycle/phase [0.0, 1.0]
    p_in_cycle = jnp.where(c1, progress / c1_end,
                 jnp.where(c2, (progress - c1_end) / (c2_end - c1_end),
                 jnp.where(c3, (progress - c2_end) / (c3_end - c2_end),
                 (progress - c3_end) / (1.0 - c3_end))))
                 
    p_in_cycle = jnp.clip(p_in_cycle, 0.0, 1.0)

    # Cosine annealing factor for cycles (1.0 down to 0.0)
    cos_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * p_in_cycle))
    # Inverted cosine factor for alpha and betas (0.0 up to 1.0)
    cos_rise = 1.0 - cos_decay

    # Learning rate peaks for each cycle, decaying over the run
    lr_max = jnp.where(c1, 1.5 * D_f,
             jnp.where(c2, 1.0 * D_f,
             jnp.where(c3, 0.5 * D_f,
             gamma_min_f)))

    # Cycle LR: cosine decay from lr_max to gamma_min
    lr_cycle = gamma_min_f + (lr_max - gamma_min_f) * cos_decay
    
    # Terminal phase: linear decay from 2*gamma_min down to gamma_min to cool down completely
    lr_terminal = gamma_min_f + gamma_min_f * (1.0 - p_in_cycle)
    lr = jnp.where(is_terminal, lr_terminal, lr_cycle)

    # Alpha (penalty) schedule: soft at start of cycle (exploration), 
    # spikes at end of cycle (feasibility restoration burst)
    alpha_low = jnp.where(c1, alpha0 * 0.1,
                jnp.where(c2, alpha0 * 0.5,
                jnp.where(c3, alpha0 * 2.0,
                alpha0)))
                
    alpha_high = jnp.where(c1, alpha0 * 5.0,
                 jnp.where(c2, alpha0 * 15.0,
                 jnp.where(c3, alpha0 * 30.0,
                 alpha0)))

    alpha_cycle = alpha_low + (alpha_high - alpha_low) * cos_rise
    
    # Terminal absolute feasibility (like the native final step, but held)
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)

    # Beta scheduling: phase-transition with alpha
    # Low beta2 / High beta1 during exploration allows escaping local minima
    # High beta2 / Low beta1 during feasibility phases absorbs curvature from penalty boundaries
    b1_start = 0.15
    b1_end = 0.04
    b2_start = 0.10
    b2_end = 0.90

    beta1_cycle = b1_start + (b1_end - b1_start) * cos_rise
    beta2_cycle = b2_start + (b2_end - b2_start) * cos_rise

    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2