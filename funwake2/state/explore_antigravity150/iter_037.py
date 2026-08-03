import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- Structurally Different: SGDR (Warm Restarts) + Cyclical Delayed Penalties ---
    # Cycle 1: Deep exploration & massive layout rearrangement (65% of steps)
    # Cycle 2: Micro-refinement & strict feasibility (25% of steps)
    # Terminal: Absolute constraint satisfaction (10% of steps)
    
    c1_end = 0.65
    c2_end = 0.90
    
    m1 = progress < c1_end
    m2 = (progress >= c1_end) & (progress < c2_end)
    is_terminal = progress >= c2_end
    
    # Local progress in [0, 1] for each cycle
    tau1 = progress / c1_end
    tau2 = (progress - c1_end) / (c2_end - c1_end)
    
    # --- 1. Cosine Annealing Learning Rate ---
    def cosine_decay(start, end, tau):
        return end + 0.5 * (start - end) * (1.0 + jnp.cos(jnp.pi * tau))
        
    lr1 = cosine_decay(1.5 * D_f, 0.05 * D_f, tau1)
    # Warm restart with a lower peak for local refinement
    lr2 = cosine_decay(0.5 * D_f, gamma_min_f, tau2)
    
    lr = jnp.where(m1, lr1, 
           jnp.where(m2, lr2, gamma_min_f))
           
    # --- 2. Cyclical Delayed Alpha (Penalty) ---
    # We use a delayed exponential ramp: tau^3 keeps the penalty low for the first 
    # half of each cycle, allowing free layout rearrangement, then smoothly spikes it 
    # to enforce feasibility before the next restart.
    def delayed_exp_ramp(start, end, tau, power=3.0):
        log_s = jnp.log(jnp.maximum(start, 1e-30))
        log_e = jnp.log(jnp.maximum(end, 1e-30))
        return jnp.exp(log_s + (log_e - log_s) * (tau ** power))
        
    # Cycle 1: soft penalty to allow constraint violations while exploring
    a1 = delayed_exp_ramp(alpha0_f * 0.05, alpha0_f * 5.0, tau1)
    # Cycle 2: higher base and aggressive plateau for tight constraint adherence
    a2 = delayed_exp_ramp(alpha0_f * 0.20, alpha0_f * 30.0, tau2)
    
    # Terminal Feasibility Spike (filter method)
    a_term = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    alpha = jnp.where(m1, a1, 
              jnp.where(m2, a2, a_term))
              
    # --- 3. Cyclical Adam Moments ---
    # Momentum (beta1) drops and variance damping (beta2) rises as we approach
    # the end of each cycle. This syncs with the penalty spike to absorb constraint 
    # curvature and prevent oscillations.
    def linear_ramp(start, end, tau):
        return start + (end - start) * tau
        
    b1_1 = linear_ramp(0.20, 0.05, tau1)
    b2_1 = linear_ramp(0.10, 0.85, tau1)
    
    b1_2 = linear_ramp(0.12, 0.02, tau2)
    b2_2 = linear_ramp(0.40, 0.95, tau2)
    
    beta1 = jnp.where(m1, b1_1, 
              jnp.where(m2, b1_2, 0.01))
              
    beta2 = jnp.where(m1, b2_1, 
              jnp.where(m2, b2_2, 0.99))
              
    return lr, alpha, beta1, beta2