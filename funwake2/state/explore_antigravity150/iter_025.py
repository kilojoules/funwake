import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Multi-Cycle SGDR Learning Rate ---
    # 3 distinct cycles: Macro-Exploration, Intermediate-Refinement, Micro-Tuning.
    # We use cosine annealing with warm restarts to escape local basins repeatedly.
    num_cycles = 3.0
    
    # cycle_idx: 0, 1, or 2. cycle_progress: 0.0 -> 1.0 within the current cycle.
    cycle_idx = jnp.clip(jnp.floor(progress * num_cycles), 0.0, num_cycles - 1.0)
    cycle_progress = (progress * num_cycles) - cycle_idx
    
    # Peak learning rate decays sharply across cycles to fine-tune the layout
    # Peaks: 1.50 * D, 0.45 * D, 0.135 * D
    lr_peak = 1.5 * D_f * (0.3 ** cycle_idx)
    
    # Cosine annealing to gamma_min within each cycle
    cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    lr_main = gamma_min_f + (lr_peak - gamma_min_f) * cosine_decay

    # --- 2. Cyclic Alpha with Mid-Run Feasibility Bursts ---
    # Instead of a single monotonic ramp, we enforce a penalty burst at the 
    # end of EACH cycle. This forces the layout into intermediate valid structures 
    # before the next cycle's exploration restarts.
    
    # Base penalty steps up each cycle (0.5x, 5.0x, 9.5x alpha0)
    alpha_base = alpha0 * (0.5 + 4.5 * cycle_idx)
    
    # Non-linear burst multiplier (1x at start -> 26x at end of cycle)
    burst_multiplier = 1.0 + 25.0 * (cycle_progress ** 4.0)
    alpha_main = alpha_base * burst_multiplier

    # --- 3. Synchronized Phase-Transition Moments ---
    # Restart momentum at the beginning of each cycle for free exploration, 
    # damp it heavily at the end of the cycle to absorb the burst curvature.
    beta1_main = 0.15 - 0.10 * cycle_progress
    beta2_main = 0.10 + 0.85 * cycle_progress

    # --- 4. Terminal Feasibility Spike ---
    # Overriding absolute guarantee of constraint compliance in the final 5%.
    is_terminal = progress >= 0.95
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2