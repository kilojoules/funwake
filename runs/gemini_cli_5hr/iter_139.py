import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 139: Refined 10-cycle strategy with Beta-annealing.
    - 10 equal cycles of 800 steps.
    - LR: Cosine annealing in each cycle, peak decaying from 8.0*lr0 to 3.0*lr0.
    - Alpha: Coupled to 1/LR, global cubic ramp (150x), deep dips (0.96) for movement.
    - Beta1: 0.1 -> 0.5 within each cycle.
    - Beta2: 0.2 -> 0.99 within each cycle for better adaptive scaling.
    - Squeeze: last 1.5% for final feasibility.
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions (10 cycles) ──────────────────────────
    n_cycles = 10
    steps_per_cycle = total_steps // n_cycles
    t_cycle = (step % steps_per_cycle) / (steps_per_cycle - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak LR decays globally
    lr_peak = lr0 * (8.0 * (1.0 - 0.6 * t_global) + 3.0 * 0.6 * t_global)
    lr_min = lr0 * 0.002
    
    # Cosine annealing
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global scale ramp
    alpha_global_scale = 1.0 + 150.0 * (t_global ** 3.0)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR
    alpha_coupled = alpha_global * (lr0 * 8.0 / jnp.maximum(lr, 1e-10))
    
    # Cycle dip
    dip_mag = 0.96 * (1.0 - 0.4 * t_global)
    dip = dip_mag * jnp.exp(- (t_cycle**2) / (2 * 0.15**2))
    
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Beta (Annealed within cycle) ────────────────────────────
    beta1 = 0.1 + 0.4 * t_cycle
    beta2 = 0.2 + 0.79 * t_cycle
    
    # ── Final Squeeze (last 1.5%) ───────────────────────────────
    is_squeeze = (t_global > 0.985)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e9, alpha)
    
    return lr, alpha, beta1, beta2
