
"""Optimizer with grid-based multi-start, using reduced iterations per run and a larger shift magnitude for broader exploration within time limits."""

import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
import jax # For random number generation

def optimize(sim, init_x, init_y, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # Optimized settings from Attempt 30, with reduced iterations for faster restarts
    settings = SGDSettings(learning_rate=110.0, max_iter=2000,
                           additional_constant_lr_iterations=1000, tol=1e-6, ks_rho=1400.0,
                           gamma_min_factor=0.005)

    best_aep = -jnp.inf
    best_opt_x, best_opt_y = init_x, init_y # Initialize with provided init_x, init_y

    # Grid-based multi-start parameters
    num_shifts_per_dim = 3 # Total 9 restarts
    shift_magnitude = min_spacing * 0.75 # Shift by 75% of min_spacing in each direction

    # Create 1D arrays of shifts for x and y
    shifts_1d = jnp.linspace(-shift_magnitude, shift_magnitude, num_shifts_per_dim)

    # Generate all combinations of 2D shifts
    shift_x_grid, shift_y_grid = jnp.meshgrid(shifts_1d, shifts_1d)
    all_shifts = jnp.stack([shift_x_grid.flatten(), shift_y_grid.flatten()], axis=-1)

    for shift_idx in range(len(all_shifts)):
        current_shift_x = all_shifts[shift_idx, 0]
        current_shift_y = all_shifts[shift_idx, 1]

        # Apply the current shift to the initial layout
        current_init_x = init_x + current_shift_x
        current_init_y = init_y + current_shift_y

        # Run the optimization for the current shifted initial layout
        opt_x, opt_y = topfarm_sgd_solve(objective, current_init_x, current_init_y,
                                          boundary, min_spacing, settings)

        # Calculate AEP for the current optimized layout
        current_aep = -objective(opt_x, opt_y) # objective returns negative AEP

        if current_aep > best_aep:
            best_aep = current_aep
            best_opt_x, best_opt_y = opt_x, opt_y

    return best_opt_x, best_opt_y
