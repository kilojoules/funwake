import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- Structural Change: Multi-Cycle SGDR with Feasibility Bursts ---
    # Directions untried: cyclic alpha with SGDR warm restarts.
    # We use 3 cycles of cosine annealing for the learning rate.
    # At the end of each cycle, the learning rate drops and the penalty (alpha)
    # spikes, creating a "feasibility burst" that pulls the layout back to validity
    # before the next exploration cycle (warm restart) begins.
    
    phase_end = 0.85
    n_cycles = 3.0
    cycle_length = phase_end / n_cycles
    
    # Safe cycle calculation
    is_main = progress < phase_end
    cycle_idx = jnp.floor(progress / cycle_length)
    cycle_idx = jnp.minimum(cycle_idx, n_cycles - 1.0)
    
    local_prog = (progress - cycle_idx * cycle_length) / cycle_length
    local_prog = jnp.clip(local_prog, 0.0, 1.0)
    
    # Cosine decay from 1.0 down to 0.0 within each cycle
    cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * local_prog))
    
    # --- 1. Learning Rate (SGDR) ---
    # Initial peak of each cycle decays so we explore less wildly over time.
    lr_max_c = 1.5 * D_f / (1.0 + 0.5 * cycle_idx)
    lr_min_c = gamma_min_f * 5.0
    lr_cycle = lr_min_c + (lr_max_c - lr_min_c) * cosine_decay
    
    # Terminal LR: linear decay from lr_min_c down to strict gamma_min
    terminal_prog = jnp.clip((progress - phase_end) / (1.0 - phase_end), 0.0, 1.0)
    lr_terminal = lr_min_c - (lr_min_c - gamma_min_f) * terminal_prog
    
    lr = jnp.where(is_main, lr_cycle, lr_terminal)
    
    # --- 2. Penalty (Cyclic Alpha) ---
    # Alpha moves inversely to LR: low during exploration (warm restarts), 
    # high during consolidation (feasibility bursts).
    alpha_min_c = alpha0_f * 0.1
    # Increase the peak penalty of each subsequent cycle to slowly tighten constraints
    alpha_max_c = alpha0_f * 8.0 * (1.5 ** cycle_idx) 
    
    alpha_cycle = alpha_min_c + (alpha_max_c - alpha_min_c) * (1.0 - cosine_decay)
    
    # Terminal Alpha: strict feasibility spike (Filter method)
    alpha_terminal_start = alpha0_f * 8.0 * (1.5 ** (n_cycles - 1.0))
    alpha_terminal_end = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    # Quadratic ramp for terminal alpha to ensure absolute compliance at the end
    alpha_terminal = alpha_terminal_start + (alpha_terminal_end - alpha_terminal_start) * (terminal_prog ** 2.0)
    
    alpha = jnp.where(is_main, alpha_cycle, alpha_terminal)
    
    # --- 3. Phase-Transition Adam Moments ---
    # Synchronize moments with the cycles:
    # High momentum (weight on current grad) / low beta2 during exploration.
    # Drop momentum / raise beta2 as the penalty kicks in to absorb curvature.
    b1_expl, b2_expl = 0.12, 0.15
    b1_cons, b2_cons = 0.04, 0.85
    
    beta1_cycle = b1_cons + (b1_expl - b1_cons) * cosine_decay
    beta2_cycle = b2_cons + (b2_expl - b2_cons) * cosine_decay
    
    b1_term_end = 0.01
    b2_term_end = 0.99
    
    beta1_terminal = b1_cons + (b1_term_end - b1_cons) * terminal_prog
    beta2_terminal = b2_cons + (b2_term_end - b2_cons) * terminal_prog
    
    beta1 = jnp.where(is_main, beta1_cycle, beta1_terminal)
    beta2 = jnp.where(is_main, beta2_cycle, beta2_terminal)
    
    return lr, alpha, beta1, beta2