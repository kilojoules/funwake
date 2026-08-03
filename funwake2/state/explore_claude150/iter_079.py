import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the last
# search-state direction untouched by the whole lineage — an ADMM-STYLE
# CONSTANT MODERATE PENALTY (prior-art §10, "constant moderate penalty";
# ALM/§7.2) under a WSD (warmup -> STABLE HOT HOLD -> decay) learning rate
# (§6). Every parent so far modulates alpha during exploration (native 1/lr
# coupling, logistic ramps, anti-phased bursts); here alpha is held FLAT at
# the lineage's *effective average* penalty while lr is held FLAT at a hot
# plateau — the ADMM bet that a fixed, well-conditioned penalty lets the
# hot phase equilibrate the AEP/violation trade once, instead of re-fighting
# a moving penalty landscape every cycle.
#
#   lr    — one-cycle/WSD, not restarts: 4% linear warmup, then a LONG hot
#           hold at 1.30*D through 55% of the run — more total hot steps
#           than any SGDR parent (whose cyclic mean was ~1.0*D) — answering
#           the parent guidance "higher/longer lr peak early". A FADING
#           FAST DITHER (6 cycles, amplitude 0.25*D -> 0, peak 1.55*D at
#           the start) rides the plateau: two-timescale exploration kicks
#           without ever paying a full restart's cold trough. After 55%,
#           the proven straight linear tail lands exactly on gamma_min at
#           the last step.
#   alpha — DECOUPLED and PIECEWISE-CONSTANT-IN-SPIRIT: a flat 2.2*alpha0
#           through the entire hot hold (the parent's exploration-phase
#           *mean* penalty, delivered as a constant instead of bursts —
#           same average pressure, totally different conditioning), then
#           the proven logistic lift to the bounded 6*alpha0 ALM plateau
#           centered just after the cool-down starts, then the proven
#           cubic-delayed geometric climb from 78% landing on the
#           5/5-seed-feasible terminal 5*alpha0*D/gamma_min. The terminal
#           feasibility restoration is preserved verbatim.
#   betas — the proven, feasibility-critical transitions untouched:
#           beta2 0.2 -> 0.9 logistic at the cool-down boundary (absorbs
#           the constraint-curvature conditioning as alpha lifts), beta1
#           gated 0.1 -> 0.02 inside the terminal spike so momentum never
#           carries turbines back across the boundary at the end.
_F_WARM = 0.04        # linear lr warmup (proven range)
_F_HOLD = 0.55        # hot hold ends here; linear decay to gamma_min at 100%
_LR_HI = 1.30         # stable plateau lr, in units of D
_RIP = 0.25           # initial dither amplitude, in units of D (peak 1.55*D)
_N_RIP = 6.0          # fast dither cycles across the hold (two-timescale)
_A_CONST = 2.2        # ADMM-style flat penalty during the hold, alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.60      # logistic alpha lift centered just after hold end
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the hold end
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> stable hot hold with fading fast dither -> linear tail ---
    # h freezes at 1 past _F_HOLD, where the dither has fully faded, so the
    # decay leg starts exactly from the clean plateau _LR_HI * D.
    h = jnp.clip(frac, 0.0, _F_HOLD) / _F_HOLD
    dither = _RIP * (1.0 - h) * jnp.sin(2.0 * jnp.pi * _N_RIP * h)
    lr_hot = (_LR_HI + dither) * Dj
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (lr_hot - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM flat penalty -> bounded plateau -> proven terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_CONST + (_A_PLAT - _A_CONST) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven feasibility-critical transitions, unchanged ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2