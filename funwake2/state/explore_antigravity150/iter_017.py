import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- SGDR Multi-Cycle Cosine Annealing with Cyclic Alpha ---
    # This represents a structural shift from the WSD (Warmup-Stable-Decay) approach.
    # We use 3 explicit cycles (warm restarts) to allow the optimizer to explore 
    # different global topologies, effectively escaping local minima. Each restart 
    # drops the penalty (alpha) and spikes the learning rate.
    
    c1_end = 0.40  # Cycle 1: Massive exploration
    c2_end = 0.80  # Cycle 2: Refinement
    c3_end = 0.95  # Cycle 3: Local convergence
    
    # Fractional progress within each cycle [0.0, 1.0]
    prog_c1 = jnp.clip(progress / c1_end, 0.0, 1.0)
    prog_c2 = jnp.clip((progress - c1_end) / (c2_end - c1_end), 0.0, 1.0)
    prog_c3 = jnp.clip((progress - c2_end) / (c3_end - c2_end), 0.0, 1.0)
    
    # --- 1. Learning Rate ---
    # Cosine annealing in each cycle.
    lr_c1 = gamma_min_f + 0.5 * (1.60 * D_f - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * prog_c1))
    lr_c2 = gamma_min_f + 0.5 * (0.80 * D_f - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * prog_c2))
    lr_c3 = gamma_min_f + 0.5 * (0.25 * D_f - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * prog_c3))
    
    lr_main = jnp.where(progress < c1_end, lr_c1,
              jnp.where(progress < c2_end, lr_c2, lr_c3))
         
    # --- 2. Penalty (Alpha) ---
    # Alpha drops at the start of each cycle to permit layout restructuring,
    # then ramps up quadratically to enforce constraints as the cycle cools.
    
    # Cycle 1: extremely loose penalty
    alpha_c1 = alpha0 * 0.05 + (alpha0 * 3.0 - alpha0 * 0.05) * (prog_c1 ** 2)
    
    # Cycle 2: moderate penalty (pulls into feasible regions)
    alpha_c2 = alpha0 * 0.50 + (alpha0 * 15.0 - alpha0 * 0.50) * (prog_c2 ** 2)
    
    # Cycle 3: strict penalty leading up to terminal feasibility
    terminal_alpha = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha_c3 = alpha0 * 5.0 + (terminal_alpha * 0.1 - alpha0 * 5.0) * (prog_c3 ** 2)
    
    alpha_main = jnp.where(progress < c1_end, alpha_c1,
                 jnp.where(progress < c2_end, alpha_c2, alpha_c3))

    # --- 3. Adam Moments ---
    # High momentum (low beta1) and low variance damping (low beta2) 
    # when LR is high and alpha is low. As cycles progress, beta2 rises 
    # to absorb stiff constraint gradients.
    
    b1_c1 = 0.15 - 0.10 * prog_c1
    b2_c1 = 0.15 + 0.60 * prog_c1
    
    b1_c2 = 0.12 - 0.08 * prog_c2
    b2_c2 = 0.30 + 0.55 * prog_c2
    
    b1_c3 = 0.06 - 0.05 * prog_c3
    b2_c3 = 0.70 + 0.25 * prog_c3
    
    beta1_main = jnp.where(progress < c1_end, b1_c1,
                 jnp.where(progress < c2_end, b1_c2, b1_c3))
            
    beta2_main = jnp.where(progress < c1_end, b2_c1,
                 jnp.where(progress < c2_end, b2_c2, b2_c3))

    # --- 4. Terminal Feasibility Spike ---
    # Unconditional exact compliance for the final 5% of steps.
    is_terminal = progress >= c3_end
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    alpha = jnp.where(is_terminal, terminal_alpha, alpha_main)
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2