# Agent Memory — 02:39:36

## Time Budget
- Elapsed: 179.2 min
- Remaining: 0.8 min
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

### PARTIALLY EXPLORED (3): lr_constant, lr_exponential_decay, lr_noise_injection

### CLOSED (19, do not revisit): adam_high_beta2, adam_standard, adam_topfarm_low, adam_zero_momentum (5567.9), alpha_anti_phase_dip, alpha_coupled_inverse_lr, alpha_cyclic (5567.1), alpha_linear_ramp, alpha_quadratic_ramp, lr_cosine (5567.9), lr_cyclical_triangular, lr_gaussian_bumps (5567.9), lr_linear_decay, lr_one_cycle, lr_polynomial_decay, lr_sgdr_warm_restarts, lr_sinusoidal_shake (5560.2), schedule_two_phase, warmup (5567.9)

### MANDATORY NEXT ACTION

All taxonomy families have been tried. Free exploration mode: the most promising avenue is PARTIAL entries with the highest best_train, or combinations across families not yet tried together.

## Recent Attempts
- #100: AEP=5567.9 (+27.1) [custom] t=26s
- #101: ERROR — Traceback (most recent call last):
  File "/Users/julianquick/portfolio_copy/fun
- #102: AEP=5559.0 (+18.3) [custom] t=26s
- #103: AEP=5560.2 (+19.4) [custom] t=26s
- #104: AEP=5561.4 (+20.7) [custom] t=26s
- #105: AEP=5563.0 (+22.3) [custom] t=26s
- #106: AEP=5563.0 (+22.3) [custom] t=26s
- #107: AEP=5567.9 (+27.1) [custom] t=26s
- #108: AEP=5567.9 (+27.1) [custom] t=26s
- #109: AEP=5567.9 (+27.1) [custom] t=26s

## Phase 2: Exploration

You've been running for a while. Consider:
- Custom gradient descent with jax.grad (not topfarm_sgd_solve)
- Wind-direction-aware grid initialization
- Different penalty schedules (alpha should INCREASE as lr decays)
- Diverse multi-start with varied initialization strategies
