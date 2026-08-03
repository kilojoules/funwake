import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Cyclic LR with SGDR Warm Restarts ---
    # Two full cycles: 
    # Cycle 1 (0.0 to 0.55): Broad exploration
    # Cycle 2 (0.55 to 0.90): Local refinement with a warm restart
    # Terminal (0.90 to 1.0): Feasibility lock-in
    
    c1_end = 0.55
    terminal_start = 0.90
    
    is_c1 = progress < c1_end
    is_term = progress >= terminal_start
    
    p1 = progress / c1_end
    p2 = jnp.clip((progress - c1_end) / (terminal_start - c1_end), 0.0, 1.0)
    
    cycle_prog = jnp.where(is_c1, p1, p2)
    
    # Peak LR for each cycle (second cycle is a "warm restart" at lower LR)
    lr_peak_1 = 1.25 * D_f
    lr_peak_2 = 0.40 * D_f
    
    lr_peak = jnp.where(is_c1, lr_peak_1, lr_peak_2)
    
    # SGDR Cosine Annealing
    lr_cycle = gamma_min_f + 0.5 * (lr_peak - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_prog))
    lr = jnp.where(is_term, gamma_min_f, lr_cycle)
    
    # --- 2. Cyclic Alpha (Mid-run Feasibility-Restoration Burst) ---
    # Alpha starts low to let the high LR reshape the layout. It then ramps up 
    # aggressively at the end of Cycle 1 to restore feasibility (the burst).
    # At Cycle 2, alpha drops again to permit local shifting, then ramps to a 
    # higher plateau to ensure strong feasibility before the terminal phase.
    
    a_base_1 = alpha0 * 0.02
    a_peak_1 = alpha0 * 5.0
    
    a_base_2 = alpha0 * 0.20
    a_peak_2 = alpha0 * 15.0
    
    a_base = jnp.where(is_c1, a_base_1, a_base_2)
    a_peak = jnp.where(is_c1, a_peak_1, a_peak_2)
    
    # Power-4 ramp keeps penalty low early in the cycle, then sharply rises
    alpha_cycle = a_base + (a_peak - a_base) * (cycle_prog ** 4.0)
    
    # Terminal feasibility spike (filter method logic)
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_term, alpha_terminal, alpha_cycle)
    
    # --- 3. Cyclic Adam Moments ---
    # Beta1 (momentum) drops as alpha rises; Beta2 (RMS dampening) increases.
    # The cyclic reset of Beta2 allows the optimizer to forget the constraint 
    # walls of Cycle 1 and freely navigate the warm restart of Cycle 2.
    
    b1_start, b1_end = 0.12, 0.02
    b2_start, b2_end = 0.10, 0.85
    
    beta1_cycle = b1_start + (b1_end - b1_start) * (cycle_prog ** 2.0)
    beta2_cycle = b2_start + (b2_end - b2_start) * (cycle_prog ** 2.0)
    
    beta1 = jnp.where(is_term, 0.01, beta1_cycle)
    beta2 = jnp.where(is_term, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2