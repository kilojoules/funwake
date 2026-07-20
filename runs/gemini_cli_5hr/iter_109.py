import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 109: 5-cycle Cosine Annealing with adaptive Beta and Alpha.
    - 5 cycles of 1600 steps each.
    - LR peak decays exponentially over cycles.
    - Alpha base increases cubically over total steps.
    - Alpha is coupled to 1/LR with a local dip at the start of each cycle.
    - Beta1 and Beta2 ramp up from (0.1, 0.2) to (0.5, 0.9) to stabilize late.
    """
    n_cycles = 5
    cycle_len = total_steps // n_cycles
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak LR decays from 10*lr0 to ~5*lr0
    lr_peak = lr0 * 10.0 * (0.5 ** t_global)
    lr_min = lr0 * 0.001
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global alpha scale increases from 1 to 501
    alpha_global_scale = 1.0 + 500.0 * (t_global ** 3)
    alpha_global = alpha0 * alpha_global_scale
    
    # Local coupling: higher LR -> lower Alpha
    # Using a reference LR of 5*lr0 for the coupling baseline
    alpha_coupled = alpha_global * (lr0 * 5.0 / jnp.maximum(lr, 1e-10))
    
    # Dip alpha at the start of each cycle to allow repositioning
    # The dip is stronger in early cycles
    dip_magnitude = 0.95 * (1.0 - 0.5 * t_global)
    dip_width = 0.15
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    # Extreme penalty and tiny LR at the very end to ensure feasibility
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    # Momentum and scaling ramp up to stabilize the final solution
    beta1 = 0.1 + 0.4 * (t_global ** 2)
    beta2 = 0.2 + 0.7 * (t_global ** 2)
    
    return lr, alpha, beta1, beta2
