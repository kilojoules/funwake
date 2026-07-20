import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 128: Aggressive Shaking and Fast Settling.
    - 10 cycles of 800 steps.
    - First 25% of each cycle: SHAKE (High LR, Low Alpha, Low Beta).
    - Remaining 75% of each cycle: SETTLE (Decay LR, High Alpha, High Beta).
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions ──────────────────────────────────────
    cycle_len = 800
    c_step = step % cycle_len
    t_cycle = c_step / (cycle_len - 1)
    
    is_shake = (t_cycle < 0.25)
    t_settle = (t_cycle - 0.25) / 0.75
    t_settle = jnp.maximum(t_settle, 0.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Decay the peak shake LR over global time
    lr_peak = lr0 * (15.0 * (1.0 - 0.8 * t_global) + 3.0 * 0.8 * t_global)
    
    lr_shake = lr_peak
    # Exponential decay during settle
    lr_settle = lr_peak * (0.001 ** t_settle)
    
    lr = jnp.where(is_shake, lr_shake, lr_settle)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_min = alpha0 * 1.0
    alpha_max = alpha0 * 1000000.0
    
    alpha_shake = alpha_min
    alpha_settle = alpha_min * ( (alpha_max / alpha_min) ** t_settle )
    
    alpha = jnp.where(is_shake, alpha_shake, alpha_settle)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = jnp.where(is_shake, 0.1, 0.9)
    beta2 = jnp.where(is_shake, 0.1, 0.999)
    
    # ── Final Squeeze (Last 100 steps) ─────────────────────────
    is_squeeze = (step > total_steps - 100)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
