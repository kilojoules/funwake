import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR with Cyclic Alpha & Feasibility Bursts ---
    # A multi-cycle cosine learning rate (SGDR) with warm restarts.
    # Synchronously, the penalty (alpha) dips at each restart to allow 
    # layout shifts, then smoothly ramps up to a "burst" of feasibility 
    # at the end of the cycle (cyclic alpha).
    
    # Define three structural cycles: Exploration, Refinement, Convergence
    c0_end = 0.40
    c1_end = 0.75
    
    is_cycle_0 = progress < c0_end
    is_cycle_1 = (progress >= c0_end) & (progress < c1_end)
    
    cycle_progress = jnp.where(
        is_cycle_0, progress / c0_end,
        jnp.where(is_cycle_1, (progress - c0_end) / (c1_end - c0_end),
                  (progress - c1_end) / (1.0 - c1_end))
    )

    # Normalised decay progress (0 to 1) for the cosine part
    cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))

    # --- Learning Rate Schedule ---
    lr_max = jnp.where(is_cycle_0, 1.25 * D_f,
               jnp.where(is_cycle_1, 0.60 * D_f, 0.25 * D_f))
    
    lr_min = jnp.where(is_cycle_0, 0.20 * D_f,
               jnp.where(is_cycle_1, 0.05 * D_f, gamma_min_f))

    lr_main = lr_min + (lr_max - lr_min) * cosine_decay

    # Global Warmup Override (first 5% of total steps) to prevent initial explosion
    warmup_end = 0.05
    in_warmup = progress < warmup_end
    lr_warmup = gamma_min_f + (1.25 * D_f - gamma_min_f) * (progress / warmup_end)
    lr_main = jnp.where(in_warmup, lr_warmup, lr_main)

    # --- Alpha (Penalty) Schedule ---
    # (1.0 - cosine_decay) forms a smooth Haversine S-curve from 0 to 1
    alpha_min = jnp.where(is_cycle_0, alpha0 * 0.1,
                  jnp.where(is_cycle_1, alpha0 * 0.5, alpha0 * 2.0))
    
    alpha_max = jnp.where(is_cycle_0, alpha0 * 2.5,
                  jnp.where(is_cycle_1, alpha0 * 10.0, alpha0 * 25.0))

    alpha_main = alpha_min + (alpha_max - alpha_min) * (1.0 - cosine_decay)

    # --- Phase-Transition Adam Moments ---
    # Tie momentum to the cycle phase. 
    # High LR (start of cycle) -> high momentum (b1=0.15) and reactive b2 (0.15).
    # Low LR (end of cycle) -> low momentum (b1=0.02) and smooth b2 (0.85) to absorb penalty curvature.
    b1_max, b1_min = 0.15, 0.02
    b2_min, b2_max = 0.15, 0.85

    beta1_main = b1_min + (b1_max - b1_min) * cosine_decay
    beta2_main = b2_max - (b2_max - b2_min) * cosine_decay

    # --- Terminal Feasibility Spike ---
    # Absolute compliance in the final 8% of steps.
    is_terminal = progress >= 0.92
    
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)

    lr = jnp.where(is_terminal, lr_terminal, lr_main)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2