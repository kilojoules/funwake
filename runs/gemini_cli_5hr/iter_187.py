import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 187: 5-cycle Cosine Annealing with adaptive Beta2.
    - 5 cycles of LR cosine annealing with peak decay.
    - Beta2 transitions from smooth (0.9) to fast (0.1).
    - Alpha ramp that increases globally and dips locally in each cycle.
    """
    n_cycles = 5
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1.0)
    t_global = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Decay the peak of each cycle
    lr_peak_start = lr0 * 10.0
    lr_peak_end = lr0 * 1.5
    lr_peak = lr_peak_start * (1.0 - t_global**0.5) + lr_peak_end * t_global**0.5
    
    lr_min = lr0 * 0.001
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Beta ────────────────────────────────────────────────────
    # Start with standard Adam-ish beta2 (0.9) to get stable scaling.
    # End with TopFarm-ish beta2 (0.1) to adapt to local constraints.
    beta1 = 0.1
    beta2 = 0.9 * (1.0 - t_global) + 0.1 * t_global
    
    # ── Alpha (Penalty Weight) ──────────────────────────────────
    # Global alpha scale starts low, ends very high.
    # We use a power law to keep it low for a while then ramp up.
    alpha_global = alpha0 * (1.0 + 1000.0 * t_global**3.0)
    
    # Local dip in alpha at the start of each cycle to explore AEP
    # The dip magnitude decreases as we progress.
    dip_magnitude = 0.9 * (1.0 - 0.7 * t_global)
    dip_width = 0.15
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    
    # Coupling to 1/LR is essential for feasibility at the end of cycles.
    alpha_coupled = alpha_global * (lr_peak / jnp.maximum(lr, 1e-10))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Final Squeeze ───────────────────────────────────────────
    # In the last 1% of the total steps, force feasibility at all costs.
    is_squeeze = (t_global > 0.99)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e12, alpha)
    
    return lr, alpha, beta1, beta2
