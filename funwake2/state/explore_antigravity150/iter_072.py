import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- SGDR: Multi-Cycle Cosine Annealing with Warm Restarts ---
    # Phase 1: Warmup [0.0, 0.04]      - Prevent catastrophic jumps
    # Phase 2: Cycle 0  [0.04, 0.40]   - Broad layout exploration
    # Phase 3: Cycle 1  [0.40, 0.92]   - Warm restart for fine-tuning
    # Phase 4: Terminal [0.92, 1.00]   - Strict feasibility restoration
    
    warmup_end = 0.04
    cycle_0_end = 0.40
    cycle_1_end = 0.92
    
    is_warmup = progress < warmup_end
    is_cycle_0 = progress < cycle_0_end
    is_terminal = progress >= cycle_1_end
    
    # Progress within the current cosine cycle [0, 1]
    cycle_0_prog = jnp.clip((progress - warmup_end) / (cycle_0_end - warmup_end), 0.0, 1.0)
    cycle_1_prog = jnp.clip((progress - cycle_0_end) / (cycle_1_end - cycle_0_end), 0.0, 1.0)
    
    cycle_prog = jnp.where(is_cycle_0, cycle_0_prog, cycle_1_prog)
    
    # --- Learning Rate ---
    lr_peak = jnp.where(is_cycle_0, 1.50 * D_f, 0.80 * D_f)
    lr_warmup = gamma_min_f + (lr_peak - gamma_min_f) * (progress / warmup_end)
    
    lr_cosine = gamma_min_f + 0.5 * (lr_peak - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * cycle_prog))
    
    lr_main = jnp.where(is_warmup, lr_warmup, lr_cosine)
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    # --- Decoupled, Cyclic Alpha (Delayed Ramp & Mid-Run Burst) ---
    # Alpha drops at the start of each cycle (decoupling penalty from LR to explore),
    # then ramps up with a power curve to enforce feasibility before the next cycle.
    alpha_start = jnp.where(is_cycle_0, alpha0 * 0.1, alpha0 * 0.2)
    alpha_end   = jnp.where(is_cycle_0, alpha0 * 6.0, alpha0 * 15.0)
    
    alpha_cyclic = alpha_start + (alpha_end - alpha_start) * (cycle_prog ** 3.0)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyclic)
    
    # --- Phase-Transition Adam Moments ---
    # Low beta2 / High beta1 at cycle start to allow rapid layout shifting.
    # High beta2 / Low beta1 at cycle end to absorb constraint boundary stiffness.
    b1_start = jnp.where(is_cycle_0, 0.15, 0.12)
    b1_end   = 0.02
    b2_start = jnp.where(is_cycle_0, 0.10, 0.15)
    b2_end   = 0.90
    
    # Moments transition slightly faster than alpha (power 2 vs 3)
    beta1_cyclic = b1_start + (b1_end - b1_start) * (cycle_prog ** 2.0)
    beta2_cyclic = b2_start + (b2_end - b2_start) * (cycle_prog ** 2.0)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_cyclic)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cyclic)

    return lr, alpha, beta1, beta2