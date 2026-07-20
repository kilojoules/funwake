import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Increased initial LR, coupled alpha with cubic boost.
    """
    # Keeping TopFarm-style betas
    beta1 = 0.1
    beta2 = 0.2
    
    # Higher initial LR to explore more
    lr_initial = 1.5 * lr0
    
    # Progress
    t = step / (total_steps - 1)
    
    # LR schedule: Cosine decay from lr_initial to a small value
    lr_final = 0.01 * lr0
    lr = lr_final + (lr_initial - lr_final) * 0.5 * (1 + jnp.cos(jnp.pi * t))
    
    # Alpha schedule: Coupled to 1/lr with an additional boost towards the end
    # Starts at 1.5 * alpha0 (since lr_initial/lr_initial * (1+0))
    # Wait, lr_initial/lr at t=0 is 1.0. 
    # Let's adjust to start at exactly alpha0
    alpha = alpha0 * (lr_initial / lr) * (1.0 + 9.0 * t**3) / 1.5
    
    return lr, alpha, beta1, beta2
