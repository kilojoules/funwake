import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    fraction = step / total_steps
    
    # Constant for first 84% of steps, then cosine decay to 1e-6 * lr0
    const_end = 0.84
    decay_frac = jnp.clip((fraction - const_end) / (1.0 - const_end), 0.0, 1.0)
    cos_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_frac))
    lr = jnp.where(fraction < const_end, lr0, lr0 * (1e-6 + (1.0 - 1e-6) * cos_decay))
    
    # Alpha: sigmoid centered at 0.80, steepness 15, range 0.01*alpha0 to 1e7*alpha0
    sig = 1.0 / (1.0 + jnp.exp(-15.0 * (fraction - 0.80)))
    alpha = alpha0 * (0.01 + (1e7 - 0.01) * sig)
    
    # Betas: smooth sigmoid transition centered at 0.835, steepness 50
    beta_sig = 1.0 / (1.0 + jnp.exp(-50.0 * (fraction - 0.835)))
    beta1 = 0.95 * (1.0 - beta_sig) + 0.008 * beta_sig
    beta2 = 0.999 * (1.0 - beta_sig) + 0.008 * beta_sig
    
    return lr, alpha, beta1, beta2