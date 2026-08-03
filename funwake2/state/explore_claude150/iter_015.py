import jax.numpy as jnp

# STRUCTURALLY NEW vs the +0.0450% best: the two menu directions the search
# state marks UNTRIED — CYCLIC ALPHA SYNCHRONIZED WITH THE SGDR RESTARTS and
# MID-RUN FEASIBILITY-RESTORATION BURSTS — replace the best's single smooth
# logistic ramp. The best repays all hot-phase violation debt in one late
# ramp; here the debt is repaid INSIDE the exploration phase, at every lr
# trough, so each warm restart launches from a (near-)feasible layout and the
# hot peaks can be pushed HOTTER than anything tried without risking the
# endgame.
#
#   lr    — proven SGDR decaying-peak engine, made hotter: three cosine
#           cycles with peaks 1.65*D -> 1.05*D (best used 1.55*D), bounded
#           troughs at 0.65*D, 3% linear warmup, and from 60% the proven WSD
#           linear tail landing lr exactly on gamma_min at the last step.
#   alpha — ANTI-PHASE RESTORATION BURSTS (structurally new): a 0.4*alpha0
#           exploration floor, plus a burst term sin^6-localized at each lr
#           trough whose height GROWS across cycles (~5, ~7, ~9 * alpha0) —
#           early violations are repaired gently (cheap AEP freedom), late
#           ones firmly, and every burst fires exactly when lr is at its
#           bounded trough so the repair is a controlled projection, not a
#           blow-up. Past 60% the bursts vanish by construction (cyc pins at
#           its peak) and the proven endgame takes over: logistic rise to the
#           bounded 6*alpha0 ALM plateau, then the cubic-delayed geometric
#           climb from 78% to the proven terminal 5*alpha0*D/gamma_min —
#           strict 5/5-seed feasibility preserved.
#   betas — proven transitions only: beta2 0.2 -> 0.9 at the cool-down
#           start; beta1 0.1 -> 0.02 during the terminal spike so diverging
#           alpha never rides momentum.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.60        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three warm restarts inside the exploration phase
_HI0 = 1.65           # first peak — hotter than the best's 1.55*D, affordable
_HI1 = 1.05           # last peak; the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven; never floors)
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_B0 = 4.0           # first-trough burst amplitude, in alpha0 units
_A_B1 = 10.0          # final-trough burst amplitude, in alpha0 units
_Q = 3.0              # (1-cyc)^3 = sin^6 localization of bursts at troughs
_A_PLAT = 6.0         # bounded ALM-style polish plateau, in alpha0 units
_A_CENTER = 0.62      # logistic plateau ramp centered just after cool-down
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down start
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

    # --- lr: warmup -> 3 hotter decaying-peak cosine restarts -> linear tail ---
    # fc freezes at 1 past _F_COOL, so cos(2*pi*N) = 1 pins the cool-down
    # start at the final (coolest) peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * fc))   # 1 at peaks, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + anti-phase trough bursts -> plateau -> terminal climb ---
    # Bursts: (1-cyc)^3 is ~0 at every lr peak and 1 at every trough, so each
    # restoration fires exactly when lr sits at the bounded 0.65*D trough.
    # Amplitude grows across cycles; the term is identically 0 past _F_COOL.
    amp = _A_B0 + (_A_B1 - _A_B0) * fc
    burst = amp * (1.0 - cyc) ** _Q
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    plateau = _A_LO + (_A_PLAT - _A_LO) * ramp                # in alpha0 units
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * (plateau + burst) * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2