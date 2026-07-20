import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 096: Refined iter_087 with higher early exploration.
    - 4 cycles of cosine decay.
    - Initial peak LR increased to 8x.
    - Global alpha ramp increased to 150x.
    - Reactive but stable beta1/beta2.
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Start high, decay peaks more slowly than 087
    lr_peak = lr0 * 8.0 * (1.0 - 0.4 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Very strong global ramp for late-stage enforcement
    alpha_global = alpha0 * (1.0 + 150.0 * (t_global ** 3))
    
    # Coupling (use the constant 8.0 as reference)
    alpha_local = alpha_global * (lr0 * 8.0 / jnp.maximum(lr, 1e-10))
    
    # Sharp dip at the start of each cycle
    dip_mag = 0.95 * (1.0 - 0.4 * t_global)
    dip = dip_mag * jnp.exp(-(t_cycle**2) / (2 * 0.08**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.15
    beta2 = 0.35
    
    return lr, alpha, beta1, beta2
