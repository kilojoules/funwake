import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- Phase 1: SGDR (Cosine Annealing with Warm Restarts) & Cyclic Alpha ---
    # Implementing a structurally different approach: multiple exploration cycles
    # separated by mid-run feasibility-restoration bursts. 
    # Each cycle abruptly drops the penalty and spikes LR to reorganize the layout, 
    # then smoothly cools down LR and ramps up penalty to find new feasible pockets.
    
    cycle_phase_end = 0.90
    num_cycles = 3.0
    cycle_length = cycle_phase_end / num_cycles  # 0.30 per cycle
    
    # Identify the current cycle (0.0, 1.0, or 2.0)
    cycle_idx = jnp.clip(jnp.floor(progress / cycle_length), 0.0, num_cycles - 1.0)
    
    # Normalized progress within the current cycle [0.0, 1.0]
    cycle_progress = (progress - cycle_idx * cycle_length) / cycle_length
    
    # Cosine annealing multiplier (1.0 at start, 0.0 at end of each cycle)
    cosine_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    # 1. Cyclic Learning Rate
    # Peak LR decays across cycles: 1.5 D -> 1.0 D -> 0.5 D
    lr_max = D_f * (1.5 - 0.5 * cycle_idx)
    lr_min = D_f * 0.05
    lr_cycles = lr_min + (lr_max - lr_min) * cosine_mult
    
    # 2. Cyclic Decoupled Alpha
    # Penalty grows inversely to LR within each cycle, creating a feasibility burst.
    # Cycle base penalties (explore): 0.1 a0 -> 0.3 a0 -> 0.5 a0
    # Cycle peak penalties (burst):   5.0 a0 -> 10.0 a0 -> 15.0 a0
    alpha_min = alpha0 * (0.1 + 0.2 * cycle_idx)
    alpha_max = alpha0 * (5.0 + 5.0 * cycle_idx)
    
    alpha_cycles = alpha_min + (alpha_max - alpha_min) * (1.0 - cosine_mult)
    
    # 3. Cyclic Phase-Transition Moments
    # High momentum / low beta2 during exploration (start of cycle)
    # Dropping momentum / high beta2 during the feasibility burst (end of cycle)
    b1_start, b1_end = 0.15, 0.02
    b2_start, b2_end = 0.20, 0.90
    
    beta1_cycles = b1_end + (b1_start - b1_end) * cosine_mult
    beta2_cycles = b2_end + (b2_start - b2_end) * cosine_mult
    
    # --- Phase 2: Terminal Feasibility Spike ---
    # The final 10% strictly enforces absolute constraints to ensure 5/5 valid seeds.
    is_terminal = progress >= cycle_phase_end
    
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    lr = jnp.where(is_terminal, lr_terminal, lr_cycles)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycles)
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycles)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycles)
    
    return lr, alpha, beta1, beta2