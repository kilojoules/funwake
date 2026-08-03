import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- SGDR Multi-Cycle Cosine Annealing & Cyclic Alpha ---
    # We use 3 cycles of exploration and feasibility-restoration.
    # In each cycle, the learning rate decays via cosine annealing, while 
    # alpha (penalty) ramps up sharply to pull turbines out of violation.
    # The cycles become successively less aggressive in LR, and more strict in alpha.
    
    num_cycles = 3.0
    
    # Compute which cycle we are in (0, 1, or 2) and progress within it [0.0, 1.0]
    cycle_idx = jnp.floor(progress * num_cycles)
    cycle_idx = jnp.clip(cycle_idx, 0.0, num_cycles - 1.0)
    cycle_progress = (progress * num_cycles) - cycle_idx

    # 1. Cosine Annealing Learning Rate with Warm Restarts
    # Peak LR decays each cycle: 1.5*D -> 0.75*D -> 0.375*D
    lr_max = 1.5 * D_f * (0.5 ** cycle_idx)
    lr_cycle = gamma_min_f + 0.5 * (lr_max - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    # 2. Cyclic Alpha (Penalty)
    # Starts very loose in each cycle to allow layouts to scramble, then
    # tightens rapidly at the end of the cycle to enforce constraints.
    alpha_low = alpha0 * 0.1
    alpha_high = alpha0 * 5.0 * (2.0 ** cycle_idx)  # Gets stricter each cycle (5, 10, 20)
    
    # Sharp polynomial ramp in the second half of the cycle
    alpha_cycle = alpha_low + (alpha_high - alpha_low) * (cycle_progress ** 4.0)
    
    # 3. Phase-Transition Adam Moments
    # Start of cycle (exploration): higher beta1, low beta2 (allows rapid layout shifts)
    # End of cycle (feasibility): low beta1, high beta2 (damps oscillations around constraints)
    b1_start, b2_start = 0.15, 0.10
    b1_end, b2_end = 0.02, 0.85
    
    beta1_cycle = b1_start + (b1_end - b1_start) * (cycle_progress ** 3.0)
    beta2_cycle = b2_start + (b2_end - b2_start) * (cycle_progress ** 3.0)

    # 4. Terminal Feasibility Spike (Final 5% of total steps)
    # Override the final cycle's end with an absolute constraint enforcement phase.
    is_terminal = progress >= 0.95
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)
    
    # Exact-penalty terminal enforcement
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2