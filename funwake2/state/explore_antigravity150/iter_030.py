import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Safe float32 casting without using python float() or int() on tracers
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # We partition the run into an Exploration Phase (0 to 85%) and a Terminal Phase (85%+).
    T_max = 0.85
    is_terminal = progress >= T_max
    
    # --- 1. Cyclic Exploration Phase (SGDR with Warm Restarts & Cyclic Alpha) ---
    # We use 3 cycles of cosine annealing for learning rate, coupled with 
    # an INVERTED cyclic schedule for the penalty (alpha).
    # High LR + Low Alpha = Rapid layout exploration (warm restart)
    # Low LR + High Alpha = Settling into local feasible configurations
    
    n_cycles = 3.0
    cycle_len = T_max / n_cycles
    
    # Calculate progress within the current cycle (0.0 to 1.0)
    phi = progress / cycle_len
    cycle_idx = jnp.floor(phi)
    tau = phi - cycle_idx
    tau = jnp.clip(tau, 0.0, 1.0)
    
    # Decaying peak learning rates for successive cycles (c.f. SGDR)
    lr_max_cycle = jnp.where(cycle_idx <= 0.0, 1.50 * D_f,
                     jnp.where(cycle_idx == 1.0, 1.00 * D_f, 
                               0.70 * D_f))
    lr_min = 0.05 * D_f
    
    # Escalating penalty peaks for successive cycles to strictly enforce constraints
    alpha_base = alpha0 * 0.1
    alpha_peak_cycle = jnp.where(cycle_idx <= 0.0, 3.0 * alpha0,
                         jnp.where(cycle_idx == 1.0, 10.0 * alpha0, 
                                   25.0 * alpha0))
                                   
    # Cosine annealing curves (1.0 -> -1.0)
    cos_val = jnp.cos(jnp.pi * tau)
    
    # LR cools down, Penalty ramps up
    lr_explore = lr_min + 0.5 * (lr_max_cycle - lr_min) * (1.0 + cos_val)
    alpha_explore = alpha_base + 0.5 * (alpha_peak_cycle - alpha_base) * (1.0 - cos_val)
    
    # Phase-Transition Adam Moments
    # Momentum (beta1) is high during exploration, low during settling
    beta1_explore = 0.02 + 0.5 * (0.25 - 0.02) * (1.0 + cos_val)
    # Variance damping (beta2) is high when penalty is high, to absorb curvature
    beta2_explore = 0.40 + 0.5 * (0.90 - 0.40) * (1.0 - cos_val)
    
    # --- 2. Terminal Feasibility Restoration Phase ---
    term_progress = jnp.clip((progress - T_max) / (1.0 - T_max), 0.0, 1.0)
    
    # Linear decay of LR down to the spatial tolerance
    lr_terminal = lr_min - (lr_min - gamma_min_f) * term_progress
    
    # Extreme spike in penalty to guarantee spacing constraints are met.
    # Note: alpha_term_start is precisely continuous with the end of the 3rd cycle
    alpha_term_start = 25.0 * alpha0
    alpha_term_target = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    # Cubic ramp avoids an immediate numerical shock at the phase transition
    alpha_terminal = alpha_term_start + (alpha_term_target - alpha_term_start) * (term_progress ** 3.0)
    
    # Terminal damping to squeeze into the exact-penalty minimum
    beta1_terminal = 0.02 * (1.0 - term_progress)
    beta2_terminal = 0.90 + (0.99 - 0.90) * term_progress
    
    # --- 3. Final Selection ---
    lr = jnp.where(is_terminal, lr_terminal, lr_explore)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_explore)
    beta1 = jnp.where(is_terminal, beta1_terminal, beta1_explore)
    beta2 = jnp.where(is_terminal, beta2_terminal, beta2_explore)
    
    return lr, alpha, beta1, beta2