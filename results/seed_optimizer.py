"""Baseline optimizer with EXPOSED optimization internals.

Uses topfarm_sgd_solve but with all settings visible and editable.
The LLM can modify hyperparameters, initialization, multi-start
strategy, or replace the solver entirely with custom code.

Key levers to pull:
- learning_rate, max_iter, additional_constant_lr_iterations
- beta1, beta2 (Adam momentum)
- boundary_weight, spacing_weight, ks_rho (constraint handling)
- gamma_min_factor (LR decay target)
- Initialization strategy (grid, random, wind-aware, hex, etc.)
- Multi-start count and perturbation strategy
- Or: replace topfarm_sgd_solve entirely with jax.grad + custom loop
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    # ── Objective ──────────────────────────────────────────────────
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # ── Initial layout: grid inside polygon ────────────────────────
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    nx = int(jnp.ceil((x_max - x_min) / min_spacing))
    ny = int(jnp.ceil((y_max - y_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(x_min + min_spacing/2, x_max - min_spacing/2, nx),
        jnp.linspace(y_min + min_spacing/2, y_max - min_spacing/2, ny))
    cand_x, cand_y = gx.flatten(), gy.flatten()

    n_verts = boundary.shape[0]
    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)
    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
    inside_x, inside_y = cand_x[inside], cand_y[inside]

    if len(inside_x) >= n_target:
        idx = jnp.round(jnp.linspace(0, len(inside_x) - 1, n_target)).astype(int)
        init_x, init_y = inside_x[idx], inside_y[idx]
    else:
        key = jax.random.PRNGKey(0)
        init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    # ── Optimizer settings (ALL editable) ──────────────────────────
    # Modify these to improve performance:
    settings = SGDSettings(
        learning_rate=50.0,              # initial LR — try 75, 100, 150
        max_iter=4000,                   # gradient iterations after const phase
        additional_constant_lr_iterations=2000,  # iterations at constant LR
        tol=1e-6,                        # convergence tolerance
        beta1=0.1,                       # Adam 1st moment — try 0.9
        beta2=0.2,                       # Adam 2nd moment — try 0.999
        gamma_min_factor=0.01,           # final LR = this × initial LR
        ks_rho=100.0,                    # KS aggregation sharpness
        spacing_weight=1.0,              # penalty weight for spacing
        boundary_weight=1.0,             # penalty weight for boundary
    )

    # ── Run optimizer ──────────────────────────────────────────────
    # This calls the Adam + penalty ramping solver. You can:
    # 1. Change settings above
    # 2. Add multi-start (loop with different init_x/init_y)
    # 3. Add two-stage (feasibility then AEP)
    # 4. Replace entirely with a custom jax.grad loop:
    #
    #    grad_obj = jax.grad(objective, argnums=(0, 1))
    #    grad_con = jax.grad(lambda x, y: boundary_penalty(x, y, boundary)
    #                        + spacing_penalty(x, y, min_spacing), argnums=(0, 1))
    #    # ... your custom Adam/SGD/L-BFGS loop here ...

    opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y,
                                      boundary, min_spacing, settings)
    return opt_x, opt_y
