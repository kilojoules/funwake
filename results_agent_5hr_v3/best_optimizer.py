
"""Optimizer with iterative multi-start refinement: multiple rounds of optimization, using the best result from the previous round as the starting point for the next, with decreasing perturbation scale, with slightly adjusted learning rate, ks_rho, and increased iterations (bug fix)."""

import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
import jax # For random number generation

def optimize(sim, init_x, init_y, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # Optimized settings from Attempt 53, with slight increases in iterations
    settings = SGDSettings(learning_rate=115.0, max_iter=1750, # Increased max_iter
                           additional_constant_lr_iterations=875, # Increased additional_constant_lr_iterations
                           tol=1e-6, ks_rho=1500.0, gamma_min_factor=0.005)

    best_aep_overall = -jnp.inf
    best_opt_x_overall, best_opt_y_overall = init_x, init_y

    num_rounds = 3 # Number of iterative refinement rounds
    num_restarts_per_round = 6 # Increased number of restarts in each round

    master_key = jax.random.PRNGKey(42) # Fixed master seed for reproducibility

    current_round_init_x, current_round_init_y = init_x, init_y

    for round_idx in range(num_rounds):
        # Adjust perturbation scale for each round, decreasing it to focus refinement
        perturbation_scale = min_spacing * (1.0 / (1.0 + round_idx * 0.5))
        perturbation_scale = jnp.maximum(perturbation_scale, min_spacing * 0.1)

        for restart_idx in range(num_restarts_per_round):
            master_key, subkey_init = jax.random.split(master_key)

            current_init_x_run, current_init_y_run = None, None

            if restart_idx == 0 and round_idx == 0:
                # First run of the first round uses original init_x, init_y
                current_init_x_run, current_init_y_run = init_x, init_y
            elif restart_idx == 0:
                # First run of subsequent rounds uses the best from previous round
                current_init_x_run, current_init_y_run = current_round_init_x, current_round_init_y
            else:
                # Other runs in the round are perturbed from the round's initial point
                noise_x = jax.random.uniform(subkey_init, shape=init_x.shape, minval=-perturbation_scale, maxval=perturbation_scale)
                master_key, subkey_init = jax.random.split(master_key) # Split key again for y noise
                noise_y = jax.random.uniform(subkey_init, shape=init_y.shape, minval=-perturbation_scale, maxval=perturbation_scale)
                current_init_x_run = current_round_init_x + noise_x
                current_init_y_run = current_round_init_y + noise_y # FIX: Changed from current_init_y to current_round_init_y

            opt_x, opt_y = topfarm_sgd_solve(objective, current_init_x_run, current_init_y_run,
                                              boundary, min_spacing, settings)

            current_aep = -objective(opt_x, opt_y)

            if current_aep > best_aep_overall:
                best_aep_overall = current_aep
                best_opt_x_overall, best_opt_y_overall = opt_x, opt_y

        # After each round, update the starting point for the next round to the best layout found so far
        current_round_init_x, current_round_init_y = best_opt_x_overall, best_opt_y_overall

    return best_opt_x_overall, best_opt_y_overall
