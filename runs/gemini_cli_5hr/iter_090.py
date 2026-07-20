import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 090: Exponential Decay with Warm Restarts and Adaptive Beta.
    - 4 cycles of exponential decay.
    - Alpha ramped up globally and locally within cycles.
    - Beta1 and Beta2 ramp up to become more stable.
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Exponential decay within cycle: lr = peak * base^t
    lr_peak = lr0 * 10.0 * (0.8 ** cycle_idx)
    # Decay to 0.01 of peak within each cycle
    base = 0.01
    lr = lr_peak * (base ** t_cycle)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global cubic ramp
    alpha_global = alpha0 * (1.0 + 100.0 * (t_global ** 3))
    
    # Within cycle, we want alpha to start low and end high
    # but also coupled to the decaying LR.
    # alpha_cycle_ramp = 0.1 + 0.9 * (t_cycle ** 2)
    alpha_cycle_ramp = jax.nn.sigmoid(12.0 * (t_cycle - 0.5))
    
    alpha = alpha_global * (0.2 + 0.8 * alpha_cycle_ramp) * (lr_peak / jnp.maximum(lr, 1e-10))
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    # Ramp from TopFarm-like (0.1, 0.2) to standard-Adam-like (0.9, 0.999)
    beta1 = 0.1 + 0.4 * t_global
    beta2 = 0.2 + 0.799 * t_global
    
    return lr, alpha, beta1, beta2
