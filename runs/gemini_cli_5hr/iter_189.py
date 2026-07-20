import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 189: 4-cycle Cosine Annealing with Adaptive Momentum and Sigmoid Alpha.
    - 4 cycles of LR cosine annealing.
    - Adaptive Beta1: low for movement, higher for stabilization at cycle ends.
    - Alpha ramp: Sigmoid-based surge after 75% of run.
    - Beta2: Constant 0.2 (TopFarm style).
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1.0)
    t_global = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Decay the peak of each cycle
    lr_peak = lr0 * 14.0 * (1.0 - 0.75 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Beta ────────────────────────────────────────────────────
    # Beta1 starts at 0.1, ramps to 0.5 at the end of each cycle to stabilize.
    beta1 = 0.1 + 0.4 * t_cycle**2.0
    beta2 = 0.2
    
    # ── Alpha (Penalty Weight) ──────────────────────────────────
    # Sigmoid surge after 70% of global run
    alpha_global_scale = 1.0 + 800.0 * (jax.nn.sigmoid(20.0 * (t_global - 0.70)))
    alpha_global = alpha0 * alpha_global_scale
    
    # Local dip in alpha at the start of each cycle (LR peaks)
    # The dip magnitude stays relatively high for the first 3 cycles.
    dip_magnitude = 0.98 * (1.0 - 0.5 * t_global)
    dip_width = 0.18 # Wider dip to allow more exploration during high LR
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    
    # Coupled LR-penalty
    alpha_local = alpha_global * (lr0 * 10.0 / jnp.maximum(lr, 1e-10))
    alpha = alpha_local * (1.0 - dip)
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_squeeze = (t_global > 0.985)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e11, alpha)
    beta1 = jnp.where(is_squeeze, 0.9, beta1)
    beta2 = jnp.where(is_squeeze, 0.999, beta2)
    
    return lr, alpha, beta1, beta2
