import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR (Cosine Annealing with Warm Restarts) ---
    # Structurally different from WSD: we use 2 cycles of exploration and 
    # exploitation. Each cycle has a high initial learning rate that decays 
    # following a cosine curve. This creates a mid-run "shakeup" to escape 
    # local minima and find better global layouts before final settling.
    
    T_main = 0.90
    num_cycles = 2.0
    
    # Calculate which cycle we are in and the progress within that cycle
    progress_main = jnp.clip(progress / T_main, 0.0, 1.0)
    
    # cycle_idx will be 0.0 or 1.0
    cycle_idx = jnp.minimum(jnp.floor(progress_main * num_cycles), num_cycles - 1.0)
    
    # cycle_progress will be 0.0 to 1.0 within the current cycle
    cycle_progress = (progress_main * num_cycles) - cycle_idx
    
    # Learning rate peaks decay each cycle: 1.6*D, 0.8*D
    lr_max = 1.6 * D_f * jnp.power(0.5, cycle_idx)
    
    # Cosine decay within cycle
    cos_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    lr_main = gamma_min_f + (lr_max - gamma_min_f) * cos_decay
    
    # --- 2. Cyclic Ratcheting Alpha ---
    # Alpha also cycles: dropping at the warm restart to allow the high LR to 
    # rapidly shift the layout (re-exploration), then ramping up as LR cools 
    # to restore feasibility. The peak penalty ratchets up across cycles to 
    # enforce tighter compliance on the second pass.
    
    # Bases: 0.2 * alpha0, 0.5 * alpha0
    alpha_base = alpha0 * 0.2 * jnp.power(2.5, cycle_idx)
    # Peaks: 5.0 * alpha0, 20.0 * alpha0
    alpha_peak = alpha0 * 5.0 * jnp.power(4.0, cycle_idx)
    
    # Alpha increases as cosine decays (inverse of LR)
    alpha_main = alpha_base + (alpha_peak - alpha_base) * (1.0 - cos_decay)
    
    # --- 3. Cyclic Adam Moments ---
    # Beta1 (momentum) mirrors LR: high when exploring, low when exploiting.
    # Beta2 (RMS memory) mirrors Alpha: low when exploring, high to damp 
    # oscillations when the penalty is high and the landscape is stiff.
    
    b1_start, b1_end = 0.15, 0.02
    b2_start, b2_end = 0.10, 0.90
    
    beta1_main = b1_end + (b1_start - b1_end) * cos_decay
    beta2_main = b2_start + (b2_end - b2_start) * (1.0 - cos_decay)
    
    # --- 4. Terminal Feasibility Spike ---
    # Absolute compliance at the very end (last 10% of steps)
    is_terminal = progress >= T_main
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)
    
    return lr, alpha, beta1, beta2