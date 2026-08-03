import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Cyclic Progress (Multi-cycle SGDR) ---
    # We partition the run into 3 cycles: Initial exploration (35%), Refinement (40%), and Terminal (25%).
    # This allows for "mid-run feasibility-restoration bursts" before full convergence.
    is_c1 = progress < 0.35
    is_c2 = (progress >= 0.35) & (progress < 0.75)
    is_c3 = progress >= 0.75

    cp_c1 = progress / 0.35
    cp_c2 = (progress - 0.35) / 0.40
    cp_c3 = (progress - 0.75) / 0.25

    local_progress = jnp.where(is_c1, cp_c1, 
                         jnp.where(is_c2, cp_c2, cp_c3))

    # --- 2. Cyclic Cosine Learning Rate ---
    # Warm restarts at the beginning of each cycle, decaying to gamma_min.
    # The starting LR for each cycle steps down as we transition from global to local search.
    lr_max_c1 = 1.5 * D_f
    lr_max_c2 = 1.0 * D_f
    lr_max_c3 = 0.5 * D_f

    lr_max_current = jnp.where(is_c1, lr_max_c1, 
                         jnp.where(is_c2, lr_max_c2, lr_max_c3))

    cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * local_progress))
    lr_main = gamma_min_f + (lr_max_current - gamma_min_f) * cosine_decay
    
    is_terminal = progress >= 0.98
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 3. Cyclic Feasibility Bursts (ADMM-style Penalty) ---
    # We hold an ADMM-style moderate base penalty, bursting it heavily at the end 
    # of each cycle. This periodically forces the layout into strict compliance (feasibility bursts)
    # when LR is low, before dropping the penalty back down upon the next warm restart.
    alpha_base = alpha0 * 0.5
    
    burst_c1 = alpha0 * 5.0
    burst_c2 = alpha0 * 15.0
    burst_c3 = alpha0 * (D_f / jnp.maximum(gamma_min_f, 1e-30))

    burst_current = jnp.where(is_c1, burst_c1, 
                        jnp.where(is_c2, burst_c2, burst_c3))

    # Power shape: stays near base for most of the cycle, then sharply spikes
    burst_shape = local_progress ** 8
    alpha_main = alpha_base + (burst_current - alpha_base) * burst_shape
    
    # Absolute terminal clamp to guarantee strict compliance in the final 2%
    alpha_terminal = alpha0 * (D_f / jnp.maximum(gamma_min_f, 1e-30))
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 4. Cyclic Moment Phase-Transitions ---
    # We sync the Adam moments with the cyclic feasibility rhythm:
    # Exploration phase (low penalty): high momentum (beta1), low curvature tracking (beta2)
    # Burst phase (high penalty): drop momentum, raise beta2 to absorb extreme penalty curvature
    b1_start, b1_end = 0.15, 0.02
    b2_start, b2_end = 0.20, 0.98

    beta1_main = b1_start + (b1_end - b1_start) * burst_shape
    beta2_main = b2_start + (b2_end - b2_start) * burst_shape
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2