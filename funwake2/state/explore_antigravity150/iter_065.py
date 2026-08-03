import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = jnp.clip(step_f / total_f, 0.0, 1.0)

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR (Cosine Annealing with Warm Restarts) ---
    # We replace the single WSD plateau with 3 cycles of increasing length: 
    # 20%, 30%, 40% of the run. This allows multiple phases of layout disruption
    # followed by fine-tuning, helping escape local optima better than a single decay.
    c1 = 0.20
    c2 = 0.50
    c3 = 0.90
    
    prog_c1 = jnp.clip(progress / c1, 0.0, 1.0)
    prog_c2 = jnp.clip((progress - c1) / (c2 - c1), 0.0, 1.0)
    prog_c3 = jnp.clip((progress - c2) / (c3 - c2), 0.0, 1.0)
    
    # Peak LR decays across cycles to allow sequential fine-tuning of the layout.
    lr_max1 = 1.6 * D_f
    lr_max2 = 1.0 * D_f
    lr_max3 = 0.5 * D_f
    lr_min_c = 0.02 * D_f
    
    # Cosine annealing curves for LR
    lr_c1 = lr_min_c + 0.5 * (lr_max1 - lr_min_c) * (1.0 + jnp.cos(jnp.pi * prog_c1))
    lr_c2 = lr_min_c + 0.5 * (lr_max2 - lr_min_c) * (1.0 + jnp.cos(jnp.pi * prog_c2))
    lr_c3 = gamma_min_f + 0.5 * (lr_max3 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * prog_c3))
    
    lr_main = jnp.where(progress < c1, lr_c1,
                jnp.where(progress < c2, lr_c2, lr_c3))

    # --- 2. Cyclic Alpha (Synchronized with LR) ---
    # Alpha starts low at the beginning of each cycle (high LR) to allow massive 
    # exploration and turbine shifting. It then ramps up as LR decays to enforce 
    # constraints strictly on the newly found local topology. Base/peak penalties grow.
    a_base1, a_max1 = alpha0 * 0.1, alpha0 * 2.0
    a_base2, a_max2 = alpha0 * 1.0, alpha0 * 10.0
    a_base3, a_max3 = alpha0 * 5.0, alpha0 * 30.0
    
    # Cosine annealing for Alpha (inverted: goes from base to max)
    a_c1 = a_max1 - 0.5 * (a_max1 - a_base1) * (1.0 + jnp.cos(jnp.pi * prog_c1))
    a_c2 = a_max2 - 0.5 * (a_max2 - a_base2) * (1.0 + jnp.cos(jnp.pi * prog_c2))
    a_c3 = a_max3 - 0.5 * (a_max3 - a_base3) * (1.0 + jnp.cos(jnp.pi * prog_c3))
    
    alpha_main = jnp.where(progress < c1, a_c1,
                   jnp.where(progress < c2, a_c2, a_c3))

    # --- 3. Cyclic Adam Moments ---
    # High momentum (beta1) and low variance damping (beta2) during high LR phases.
    b1_high, b2_low = 0.15, 0.10
    b1_low, b2_high = 0.02, 0.90
    
    b1_c1 = b1_low + 0.5 * (b1_high - b1_low) * (1.0 + jnp.cos(jnp.pi * prog_c1))
    b1_c2 = b1_low + 0.5 * (b1_high - b1_low) * (1.0 + jnp.cos(jnp.pi * prog_c2))
    b1_c3 = b1_low + 0.5 * (b1_high - b1_low) * (1.0 + jnp.cos(jnp.pi * prog_c3))
    
    beta1_main = jnp.where(progress < c1, b1_c1,
                   jnp.where(progress < c2, b1_c2, b1_c3))
                   
    b2_c1 = b2_high - 0.5 * (b2_high - b2_low) * (1.0 + jnp.cos(jnp.pi * prog_c1))
    b2_c2 = b2_high - 0.5 * (b2_high - b2_low) * (1.0 + jnp.cos(jnp.pi * prog_c2))
    b2_c3 = b2_high - 0.5 * (b2_high - b2_low) * (1.0 + jnp.cos(jnp.pi * prog_c3))
    
    beta2_main = jnp.where(progress < c1, b2_c1,
                   jnp.where(progress < c2, b2_c2, b2_c3))

    # --- 4. Terminal Feasibility Phase ---
    # The final 10% strictly enforces constraints with a massive alpha and minimal LR,
    # ensuring absolute feasibility after the 3 cycles finish.
    is_terminal = progress >= c3
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2