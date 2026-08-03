import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- Structural Change: 3-Cycle SGDR with Synchronized Alpha Restarts ---
    # Instead of a single plateau, we use multi-cycle cosine annealing for the learning rate,
    # coupled with synchronized "feasibility-restoration bursts" for alpha.
    # At the start of each cycle, LR is high (exploration) and alpha drops to allow turbines 
    # to move freely out of constraint-bound local minima. As the cycle cools, alpha rises 
    # to enforce feasibility. Each subsequent cycle is stricter than the last.
    
    cycle_1_end = 0.50  # Global layout search
    cycle_2_end = 0.80  # Local refinement
    cycle_3_end = 0.95  # Micro-adjustments
    
    in_cycle_1 = progress < cycle_1_end
    in_cycle_2 = (progress >= cycle_1_end) & (progress < cycle_2_end)
    in_cycle_3 = (progress >= cycle_2_end) & (progress < cycle_3_end)
    is_terminal = progress >= cycle_3_end
    
    # Normalized progress within each cycle [0, 1]
    prog_c1 = jnp.clip(progress / cycle_1_end, 0.0, 1.0)
    prog_c2 = jnp.clip((progress - cycle_1_end) / (cycle_2_end - cycle_1_end), 0.0, 1.0)
    prog_c3 = jnp.clip((progress - cycle_2_end) / (cycle_3_end - cycle_2_end), 0.0, 1.0)
    
    # Cycle 1 has a short linear warmup for LR (first 5% of total run)
    warmup_frac = 0.10  # 10% of cycle 1
    c1_is_warmup = prog_c1 < warmup_frac
    c1_warmup_prog = prog_c1 / warmup_frac
    c1_decay_prog = jnp.clip((prog_c1 - warmup_frac) / (1.0 - warmup_frac), 0.0, 1.0)
    
    # --- 1. Cyclic Learning Rate ---
    lr_max_c1 = 1.60 * D_f
    lr_max_c2 = 0.75 * D_f
    lr_max_c3 = 0.25 * D_f
    
    lr_max = jnp.where(in_cycle_1, lr_max_c1,
             jnp.where(in_cycle_2, lr_max_c2, lr_max_c3))
             
    c1_mult = jnp.where(c1_is_warmup, 
                        c1_warmup_prog, 
                        0.5 * (1.0 + jnp.cos(jnp.pi * c1_decay_prog)))
    c2_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * prog_c2))
    c3_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * prog_c3))
    
    lr_mult = jnp.where(in_cycle_1, c1_mult,
              jnp.where(in_cycle_2, c2_mult, c3_mult))
              
    lr_main = gamma_min_f + (lr_max - gamma_min_f) * lr_mult
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    # --- 2. Dynamic Cyclic Alpha ---
    # S-curve from 0 to 1 over the decay phase of each cycle
    alpha_prog_c1 = 0.5 * (1.0 - jnp.cos(jnp.pi * c1_decay_prog))
    alpha_prog_c2 = 0.5 * (1.0 - jnp.cos(jnp.pi * prog_c2))
    alpha_prog_c3 = 0.5 * (1.0 - jnp.cos(jnp.pi * prog_c3))
    
    # Hold alpha at its lowest during cycle 1 warmup to maximize initial structural exploration
    alpha_prog_c1 = jnp.where(c1_is_warmup, 0.0, alpha_prog_c1)
    
    alpha_prog = jnp.where(in_cycle_1, alpha_prog_c1,
                 jnp.where(in_cycle_2, alpha_prog_c2, alpha_prog_c3))
                 
    # Progressively tighten the constraint bands cycle by cycle
    alpha_low_c1 = alpha0 * 0.05
    alpha_high_c1 = alpha0 * 5.0
    
    alpha_low_c2 = alpha0 * 0.50
    alpha_high_c2 = alpha0 * 15.0
    
    alpha_low_c3 = alpha0 * 2.00
    alpha_high_c3 = alpha0 * 50.0
    
    alpha_low = jnp.where(in_cycle_1, alpha_low_c1,
                jnp.where(in_cycle_2, alpha_low_c2, alpha_low_c3))
                
    alpha_high = jnp.where(in_cycle_1, alpha_high_c1,
                 jnp.where(in_cycle_2, alpha_high_c2, alpha_high_c3))
                 
    alpha_main = alpha_low + (alpha_high - alpha_low) * alpha_prog
    
    # Terminal feasibility spike ensures absolute compliance (Filter method mechanism)
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    # --- 3. Phase-Transition Adam Moments ---
    # Sync with alpha_prog: exploratory (low momentum / low beta2) at the cycle start,
    # transitioning to heavily damped (stable) moments as the penalty kicks in.
    b1_expl, b2_expl = 0.15, 0.15
    b1_feas, b2_feas = 0.02, 0.90
    
    beta1_main = b1_expl + (b1_feas - b1_expl) * alpha_prog
    beta2_main = b2_expl + (b2_feas - b2_expl) * alpha_prog
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)
    
    return lr, alpha, beta1, beta2