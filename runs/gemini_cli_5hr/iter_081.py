import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 081: Triple-bump discovery with coordinated alpha relaxation.
    - Warmup phase to stabilize early Adam moments.
    - 3 LR bumps to escape local optima (0.4, 0.65, 0.85).
    - Alpha dips coordinated with bumps to allow easier rearranging.
    - Surgical squeeze (99%+) to lock in feasibility.
    - Reactive Beta1/Beta2 (0.3/0.5) as discovered in top schedules.
    """
    t = step / (total_steps - 1)
    lr_init = 4.0 * lr0
    lr_min = lr_init / 1000.0
    
    # ── LR: Warmup + Cosine ──────────────────────────────────────
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    
    cosine_t = jnp.clip((t - warmup_end) / (0.95 - warmup_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
    
    # ── 3 LR Bumps ───────────────────────────────────────────────
    def gaussian(t, center, width, height):
        return height * jnp.exp(-0.5 * ((t - center) / width) ** 2)
    
    bump1 = gaussian(t, 0.40, 0.03, 0.25 * lr_init)
    bump2 = gaussian(t, 0.65, 0.04, 0.35 * lr_init)
    bump3 = gaussian(t, 0.85, 0.04, 0.45 * lr_init)
    
    lr = lr_base + bump1 + bump2 + bump3
    
    # ── Alpha: Coupled + Ramp + Dips ────────────────────────────
    # Stronger coupling than baseline
    alpha_coupled = 6.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    
    # Quadratic ramp for late-stage enforcement
    alpha_ramp = 5.0 * alpha0 * (jnp.maximum(t - 0.4, 0.0) / 0.6)**2
    
    # Dips during/after bumps
    dip1 = 0.6 * gaussian(t, 0.42, 0.04, 1.0)
    dip2 = 0.7 * gaussian(t, 0.67, 0.05, 1.0)
    dip3 = 0.8 * gaussian(t, 0.87, 0.05, 1.0)
    
    alpha = (alpha_coupled + alpha_ramp) * (1.0 - jnp.maximum(jnp.maximum(dip1, dip2), dip3))
    
    # ── SQUEEZE ───────────────────────────────────────────────────
    is_squeeze = (t > 0.985)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr)
    alpha = jnp.where(is_squeeze, 1000000.0 * alpha0, alpha)
    
    # ── Beta: Reactive ───────────────────────────────────────────
    beta1 = 0.3
    beta2 = 0.5
    
    return lr, alpha, beta1, beta2
