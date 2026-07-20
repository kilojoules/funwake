# Agent Memory — 05:48:45

## Time Budget
- Elapsed: 169.1 min
- Remaining: 10.9 min
- Budget: 180 min total
- Phase: explore


## Performance
- Baseline: 5540.7 GWh
- Best so far: 0.0 GWh (attempt 0, gap: -5540.7)
- Attempts: 0 (0 success, 0 errors)
- Strategies tried: none

## Strategy Registry

*Mode: schedule. Close-after: 3 attempts.*

### UNEXPLORED — try these FIRST (0 remaining)
- *(all taxonomy families tried at least once — free exploration mode)*

### PARTIALLY EXPLORED (10): adam_high_beta2 (5556.1), adam_standard (5556.1), alpha_anti_phase_dip (5562.1), alpha_quadratic_ramp (5562.0), lr_constant (5562.0), lr_cyclical_triangular (5529.4), lr_linear_decay (5529.2), lr_polynomial_decay (5542.5), lr_sgdr_warm_restarts (5529.0), schedule_two_phase (5560.0)

### CLOSED (12, do not revisit): adam_topfarm_low (5564.5), adam_zero_momentum (5564.5), alpha_coupled_inverse_lr (5563.4), alpha_cyclic (5563.3), alpha_linear_ramp (5564.5), lr_cosine (5561.4), lr_exponential_decay (5564.5), lr_gaussian_bumps (5564.5), lr_noise_injection (5564.5), lr_one_cycle (5542.5), lr_sinusoidal_shake (5560.3), warmup (5563.4)

### MANDATORY NEXT ACTION

All taxonomy families have been tried. Free exploration mode: the most promising avenue is PARTIAL entries with the highest best_train, or combinations across families not yet tried together.

## Recent Attempts
- #76: AEP=5559.5 (+18.8) [custom] t=26s
- #77: AEP=5562.7 (+22.0) [custom] t=26s
- #78: AEP=5559.4 (+18.7) [custom] t=26s
- #79: AEP=5562.7 (+22.0) [custom] t=26s
- #80: AEP=5561.4 (+20.7) [custom] t=26s
- #81: AEP=5560.9 (+20.2) [custom] t=26s
- #82: AEP=5564.5 (+23.8) [custom] t=26s
- #83: AEP=5564.3 (+23.6) [custom] t=26s
- #84: AEP=5564.5 (+23.8) [custom] t=26s
- #85: AEP=5564.5 (+23.8) [custom] t=26s

## Phase 2: Exploration

You've been running for a while. Consider:
- Custom gradient descent with jax.grad (not topfarm_sgd_solve)
- Wind-direction-aware grid initialization
- Different penalty schedules (alpha should INCREASE as lr decays)
- Diverse multi-start with varied initialization strategies
