import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 083: Balanced escape-and-settle cycle.
    - Two large LR bumps at 0.5 and 0.8 to escape local optima.
    - A single, deep alpha relaxation between them (t=0.65).
    - Moderate, stable beta values (0.3/0.6).
    - Coupled alpha with a smoother late ramp.
    """
    t = step / (total_steps - 1)
    lr_init = 4.0 * lr0
    lr_min = lr_init / 5000.0
    
    # ── LR: Warmup + Cosine ──────────────────────────────────────
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    
    cosine_t = jnp.clip((t - warmup_end) / (0.95 - warmup_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
    
    # ── 2 LR Bumps ──────────────────────────────────────────────
    def gaussian(t, center, width, height):
        return height * jnp.exp(-0.5 * ((t - center) / width) ** 2)
    
    bump1 = gaussian(t, 0.50, 0.04, 0.35 * lr_init)
    bump2 = gaussian(t, 0.80, 0.05, 0.45 * lr_init)
    
    lr = lr_base + bump1 + bump2
    
    # ── Alpha: Coupled + Ramp + Dip ─────────────────────────────
    # Standard coupling
    alpha_coupled = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    
    # Late-stage ramp
    alpha_ramp = 5.0 * alpha0 * (jnp.maximum(t - 0.4, 0.0) / 0.6)**2
    
    # Dip between bumps (t=0.65)
    dip = 0.7 * gaussian(t, 0.65, 0.06, 1.0)
    
    alpha = (alpha_coupled + alpha_ramp) * (1.0 - dip)
    
    # ── SQUEEZE ───────────────────────────────────────────────────
    is_squeeze = (t > 0.985)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr)
    alpha = jnp.where(is_squeeze, 1000000.0 * alpha0, alpha)
    
    # ── Beta: Moderate ───────────────────────────────────────────
    beta1 = 0.3
    beta2 = 0.6
    
    return lr, alpha, beta1, beta2
