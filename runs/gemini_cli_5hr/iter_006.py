import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Smooth cosine LR, oscillating alpha ramp.
    """
    # TopFarm-style beta values
    beta1 = 0.1
    beta2 = 0.2
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # Learning rate schedule: smooth cosine decay from lr0 to 0.01*lr0
    lr = 0.01 * lr0 + 0.99 * lr0 * 0.5 * (1 + jnp.cos(jnp.pi * t))
    
    # Alpha schedule: generally increasing quadratic ramp
    # Base alpha goes from alpha0 to 1000*alpha0
    alpha_base = alpha0 * (1.0 + 999.0 * t**2)
    
    # Add oscillations to alpha to help escape local minima
    # 4 cycles over the course of optimization
    oscillation = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * t * 4)
    alpha = alpha_base * oscillation
    
    return lr, alpha, beta1, beta2
