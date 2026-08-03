import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): no lr cycling and
# no burst/plateau alpha machinery at all. Two untried prior-art bets fused:
#
#   lr    — WSD / one-cycle trapezoid (menu row 1, §2/§6): 3% linear warmup,
#           then a SUSTAINED HOT SLANTED PLATEAU (1.5*D -> 1.1*D over the
#           first 58%) instead of cosine restarts. Time-averaged exploration
#           heat (~1.3*D) exceeds every cyclic parent (~1.0*D avg) — the
#           "hold near c*D, then near-linear cool-down beats cosine/product"
#           hypothesis — followed by the proven straight linear tail landing
#           exactly on gamma_min at the last step (42% of the run, longer
#           polish/repair window than the best's 38% to offset the extra heat).
#
#   alpha — epsilon-CONSTRAINED SHRINKING VIOLATION BAND (§7.9, flagged
#           untried): alpha is DECOUPLED from lr and set by a metre-valued
#           tolerance band, alpha = alpha0 * D / band(t). The band starts
#           wide (2.5*D, i.e. the proven 0.4*alpha0 exploration floor) and
#           contracts GEOMETRICALLY (log-space), cubically back-loaded from
#           the cool-down start, reaching gamma_min/6 only at the final step —
#           so the enforced violation band hits gamma_min exactly at the end.
#           One monotone contraction law replaces floor+bursts+logistic+
#           terminal-climb, yet its tail reproduces (and slightly exceeds)
#           the 5/5-seed-feasible terminal trajectory: ~6*alpha0 at 85%,
#           ~40*alpha0 at 90%, ending at 6*alpha0*D/gamma_min — a stronger
#           terminal feasibility spike than the proven 5x gain, as insurance
#           for the hotter plateau. Before 80% the band stays lax, buying
#           mid-cool-down AEP freedom the bursty best never had.
#
#   betas — the proven feasible transitions, unchanged: beta2 0.2 -> 0.9
#           logistic at the cool-down start (variance control for the
#           alpha-dominated tail), beta1 0.1 -> 0.02 gated into the terminal
#           spike so momentum never carries turbines back across the boundary.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.58        # plateau ends here; linear decay to gamma_min at 100%
_HI0 = 1.5            # plateau start lr, in units of D — sustained-hot WSD bet
_HI1 = 1.1            # plateau end lr; the linear tail starts from here
_BAND_HI = 2.5        # initial violation band, in units of D (=> 0.4*alpha0 floor)
_TERM_GAIN = 6.0      # final band = gamma_min/6 => alpha ends 6*alpha0*D/gamma_min
_A_POW = 3.0          # cubic back-loading of the band contraction (proven shape)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum inside the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> slanted hot plateau -> linear tail onto gamma_min ---
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL          # freezes at 1 past _F_COOL
    lr_plate = (_HI0 + (_HI1 - _HI0) * fc) * Dj          # 1.5*D -> 1.1*D slant
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_plate - gmin) * (1.0 - p)        # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)              # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: geometric contraction of the enforced violation band ---
    # band: 2.5*D -> gamma_min/6, log-space, cubically back-loaded from the
    # cool-down start; alpha = alpha0 * D / band recovers the native scale
    # (alpha0 = mean|grad J|/D) with the band, not 1/lr, setting strictness.
    s = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0) ** _A_POW
    log_hi = jnp.log(_BAND_HI * Dj)
    log_lo = jnp.log(gmin / _TERM_GAIN)
    band = jnp.exp(log_hi + s * (log_lo - log_hi))
    alpha = alpha0 * Dj / jnp.maximum(band, 1e-30)       # ends at 6*alpha0*D/gmin

    # --- betas: proven feasible transitions, unchanged ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2