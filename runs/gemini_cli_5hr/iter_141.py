import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 141: 20-cycle "High-Frequency Shaking".
    - 20 short cycles of 400 steps each.
    - High-frequency exploration to find global optima.
    - LR peak decays with power law (t^0.5).
    - Alpha global ramp: t^3.0.
    - Alpha: extra pressure at cycle ends (t_cycle^4).
    - Beta: Ramps within cycles for stability.
    - Final Squeeze: last 1%.
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions (20 cycles of 400 steps) ─────────────
    n_cycles = 20
    cycle_len = total_steps // n_cycles
    t_cycle = (step % cycle_len) / (cycle_len - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak LR decays globally (square root for slower initial decay)
    lr_peak_global = lr0 * 15.0 * (1.0 - t_global)**0.5
    lr_min = lr0 * 0.001
    
    # Cosine annealing in each cycle
    lr_cyclic = lr_min + 0.5 * (lr_peak_global - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global Alpha Scale
    alpha_global_scale = 1.0 + 200.0 * (t_global ** 3.0)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR (reference peak 15.0)
    alpha_coupled = alpha_global * (lr0 * 15.0 / jnp.maximum(lr_cyclic, 1e-10))
    
    # Extra pressure at cycle ends to force feasibility before next shake
    cycle_end_pressure = 1.0 + 2.0 * (t_cycle**4)
    
    # Dip at start of cycle
    dip_mag = 0.98 * (1.0 - 0.4 * t_global)
    dip_width = 0.12
    dip = dip_mag * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    
    alpha_dip = alpha_coupled * cycle_end_pressure * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1 + 0.5 * t_cycle
    beta2 = 0.2 + 0.79 * t_cycle
    
    # ── Final Selection ─────────────────────────────────────────
    is_squeeze = (t_global > 0.99)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr_cyclic)
    alpha = jnp.where(is_squeeze, alpha0 * 1e9, alpha_dip)
    
    return lr, alpha, beta1, beta2
