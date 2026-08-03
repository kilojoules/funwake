import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- 1. Multi-Cycle Cosine Learning Rate with Warm Restarts (SGDR) ---
    # We use 3 complete cycles. Warm restarts help escape local minima and 
    # explore structurally different layouts across cycles.
    n_cycles = 3.0
    safe_progress = jnp.minimum(progress, 0.999)
    cycle_idx = jnp.floor(safe_progress * n_cycles)
    phi = (safe_progress * n_cycles) - cycle_idx  # Phase within cycle: [0.0, 1.0)
    
    # Decay the peak learning rate for each successive cycle to settle into a good optimum
    lr_peak = 1.25 * D_f / (1.0 + cycle_idx * 0.4)
    
    # 10% of each cycle is a linear warmup to avoid exploding gradients on restart
    warmup_fraction = 0.10
    is_warmup = phi < warmup_fraction
    
    lr_warmup = gamma_min_f + (lr_peak - gamma_min_f) * (phi / warmup_fraction)
    cos_phi = jnp.clip((phi - warmup_fraction) / (1.0 - warmup_fraction), 0.0, 1.0)
    lr_cos = gamma_min_f + 0.5 * (lr_peak - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cos_phi))
    
    lr_cycle = jnp.where(is_warmup, lr_warmup, lr_cos)

    # --- 2. Cyclic Alpha (Mid-run Feasibility Bursts) ---
    # Alpha stays relatively soft during the high-LR exploration phase of each cycle,
    # then surges polynomially at the end of the cycle to pull the layout into feasibility.
    alpha_base = alpha0 * 0.1
    alpha_peak = alpha0 * 12.0 * (1.0 + cycle_idx) # Constraints tighten with each cycle
    
    # phi**5 ensures the penalty only ramps up in the final ~25% of the cycle
    alpha_cycle = alpha_base + (alpha_peak - alpha_base) * (phi ** 5.0)

    # --- 3. Phase-Transition Adam Moments ---
    # Synchronized with the cycle: high momentum (low beta2) when exploring (low phi),
    # low momentum (high beta2) during the feasibility bursts (high phi) to damp oscillations.
    b1_start, b2_start = 0.15, 0.15
    b1_end, b2_end = 0.03, 0.90
    
    beta1_cycle = b1_start + (b1_end - b1_start) * (phi ** 3.0)
    beta2_cycle = b2_start + (b2_end - b2_start) * (phi ** 3.0)

    # --- 4. Terminal Feasibility Spike ---
    # The final 6% of optimization is STRICTLY dedicated to feasibility restoration.
    # We override the cyclic behavior to freeze the layout and crank the exact penalty.
    is_terminal = progress >= 0.94
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    # Terminal exact-penalty spike ensures absolute constraint compliance
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2