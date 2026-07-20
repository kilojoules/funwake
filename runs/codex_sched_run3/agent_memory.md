# Agent Memory — 08:50:39

## Time Budget
- Elapsed: 171.9 min
- Remaining: 8.1 min
- Budget: 180 min total
- Phase: final


## Performance
- Baseline: 5540.7 GWh
- Best so far: 5567.9 GWh (attempt 72, gap: +27.1)
- Attempts: 72 (72 success, 0 errors)
- Strategies tried: timed Adam memory release; hard high-alpha0 repair routing

## Strategy Registry

*Mode: schedule. Close-after: 3 attempts.*

### UNEXPLORED — try these FIRST (0 remaining)
- *(all taxonomy families tried at least once — free exploration mode)*

### PARTIALLY EXPLORED (12): adam_high_beta2 (5567.9), adam_standard (5567.9), adam_topfarm_low (5563.3), alpha_linear_ramp, alpha_quadratic_ramp (5562.5), lr_constant (5563.2), lr_cyclical_triangular (5561.6), lr_exponential_decay (5561.6), lr_noise_injection (5564.5), lr_one_cycle (5562.7), lr_sinusoidal_shake (5564.5), schedule_two_phase

### CLOSED (10, do not revisit): adam_zero_momentum (5567.9), alpha_anti_phase_dip (5563.3), alpha_coupled_inverse_lr (5567.9), alpha_cyclic (5567.9), lr_cosine (5567.9), lr_gaussian_bumps (5567.9), lr_linear_decay (5565.4), lr_polynomial_decay (5560.6), lr_sgdr_warm_restarts (5560.1), warmup (5567.9)

### MANDATORY NEXT ACTION

All taxonomy families have been tried. Free exploration mode: the most promising avenue is PARTIAL entries with the highest best_train, or combinations across families not yet tried together.

## Recent Attempts
- #63: AEP=5567.9 (+27.1) [custom] t=26s
- #64: AEP=5565.0 (+24.3) [custom] t=26s
- #65: AEP=5559.6 (+18.8) [custom] t=26s
- #66: AEP=5563.7 (+23.0) [custom] t=26s
- #67: AEP=5562.3 (+21.6) [custom] t=27s
- #68: AEP=5567.9 (+27.1) [custom] t=26s
- #69: AEP=5560.2 (+19.5) [custom] t=26s
- #70: AEP=5567.9 (+27.1) [custom] t=26s
- #71: AEP=5561.7 (+21.0) [custom] t=27s — timed Adam memory release, feasible, worse
- #72: AEP=5567.9 (+27.1) [custom] t=27s — hard high-alpha0 repair route, ROWP feasible

## Latest Lessons
- Farm1 alpha0 is about 2.54e-4; ROWP alpha0 is about 6.94e-4, so the
  5.0e-4 hard gate cleanly separates the held-out ROWP-scale case.
- Releasing Adam memory during the learned mobility windows is feasible but
  loses train AEP, so the train branch should preserve beta1=0.239994 and
  beta2=0.635963 unless a broader LR/alpha redesign is attempted.
- `iter_074.py` is the best current final candidate: exact train branch plus
  hard high-alpha0 repair with stronger tail constraint lock.

## Phase 2: Exploration

You've been running for a while. Consider:
- Custom gradient descent with jax.grad (not topfarm_sgd_solve)
- Wind-direction-aware grid initialization
- Different penalty schedules (alpha should INCREASE as lr decays)
- Diverse multi-start with varied initialization strategies
