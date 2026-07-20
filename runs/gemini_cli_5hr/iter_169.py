import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 169: High-Peak Warmup + Cosine.
    - 5% linear warmup to 15.0 * lr0.
    - 95% cosine decay to 0.001 * lr0.
    - Alpha: Coupled to 1/LR, starting at 75 * alpha0.
    - Beta: (0.1, 0.9).
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 15.0
    lr_min = lr0 * 0.001
    
    warmup_steps = 0.05
    lr_warmup = lr_peak * t / warmup_steps
    
    cosine_t = (t - warmup_steps) / (1.0 - warmup_steps)
    lr_cosine = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * cosine_t))
    
    lr = jnp.where(t < warmup_steps, lr_warmup, lr_cosine)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    alpha_base = 75.0 * alpha0
    
    # Couple to LR and increase globally
    alpha = alpha_base * (lr_peak / jnp.maximum(lr, 1e-10)) * (1.0 + 15.0 * t**2.5)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.9
    
    # ── Final Squeeze ───────────────────────────────────────────
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
