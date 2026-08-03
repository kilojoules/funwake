import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Two-Cycle SGDR (Warm Restarts) ---
    # Cycle 1: Aggressive global exploration (0% to 45%)
    c1_end = 0.45
    c1_prog = jnp.clip(progress / c1_end, 0.0, 1.0)
    lr_max_1 = 1.25 * D_f
    lr_min_1 = 0.10 * D_f
    lr_c1 = lr_min_1 + 0.5 * (lr_max_1 - lr_min_1) * (1.0 + jnp.cos(jnp.pi * c1_prog))
    
    # Cycle 2: Refined local exploration (45% to 90%)
    c2_end = 0.90
    c2_prog = jnp.clip((progress - c1_end) / (c2_end - c1_end), 0.0, 1.0)
    lr_max_2 = 0.75 * D_f
    lr_min_2 = gamma_min_f
    lr_c2 = lr_min_2 + 0.5 * (lr_max_2 - lr_min_2) * (1.0 + jnp.cos(jnp.pi * c2_prog))
    
    lr_main = jnp.where(progress < c1_end, lr_c1, lr_c2)
    
    # Terminal phase: strict convergence (90% to 100%)
    is_terminal = progress >= c2_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Decoupled Penalty with Mid-Run Feasibility Burst ---
    # Smooth logistic transition from exploration to constraint refinement.
    # We decouple alpha from 1/lr to avoid extreme gradients early on.
    transition = 1.0 / (1.0 + jnp.exp(-50.0 * (progress - c1_end)))
    
    alpha_explore = alpha0 * 0.15
    alpha_refine = alpha0 * 12.0
    alpha_base = alpha_explore + (alpha_refine - alpha_explore) * transition
    
    # Mid-run feasibility-restoration burst aligned exactly with the LR restart.
    # Forces rapid decoupling of any overlapping turbines before Cycle 2 cools down.
    burst_shape = jnp.exp(-1000.0 * (progress - c1_end)**2)
    alpha_burst = alpha0 * 15.0 * burst_shape
    alpha_main = alpha_base + alpha_burst
    
    # Terminal feasibility spike ensures absolute layout validity at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Phase-Transition Adam Moments ---
    # Correlate momentum/curvature dynamically with the logistic penalty phase.
    # Momentum (beta1) drops and curvature-absorption (beta2) rises to handle 
    # the stiff constraint boundaries as we transition into the refinement phase.
    b1_start, b2_start = 0.12, 0.15
    b1_end, b2_end = 0.02, 0.92
    
    beta1_main = b1_start + (b1_end - b1_start) * transition
    beta2_main = b2_start + (b2_end - b2_start) * transition
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2