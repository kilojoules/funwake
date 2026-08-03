import jax.numpy as jnp

# STRUCTURALLY NEW vs the decoupled-ramp best (+0.0450%): the two remaining
# UNTRIED menu directions — CYCLIC ALPHA synchronized with the SGDR restarts,
# i.e. MID-RUN FEASIBILITY-RESTORATION BURSTS — composed as an open-loop ALM
# outer loop, on top of the proven skeleton (3% warmup, linear tail landing
# exactly on gamma_min, beta2 phase transition, terminal alpha spike).
#
#   lr    — HOTTER/LONGER decaying-peak restarts (the explicit AEP push):
#           three cosine cycles, peaks decaying 1.70*D -> 1.05*D (vs 1.55*D)
#           over a LONGER exploration window (62% vs 60%), troughs at 0.60*D.
#           The hotter start is safe *because* each trough now repairs its
#           own violation debt (see alpha). From 62% the proven WSD linear
#           tail lands lr exactly on gamma_min at the last step.
#   alpha — ANTI-PHASE FEASIBILITY BURSTS + plateau + proven terminal spike:
#           during exploration alpha sits at a 0.4*alpha0 floor at every lr
#           PEAK (free basin-hopping) and rises sharply — cubic in (1-cyc),
#           so the burst concentrates near the troughs — to 7*alpha0 at every
#           lr TROUGH. Each (hot peak -> cool trough) cycle is one ALM outer
#           iteration: explore infeasibly, then repair while lr is low and
#           repair is cheap, so every restart launches from a near-feasible
#           layout instead of accumulating debt for the endgame. Past the
#           cool-down the bursts vanish (cyc pins at 1) and a logistic ramp
#           lifts alpha onto the proven bounded 6*alpha0 plateau; the proven
#           cubic-delayed geometric climb from 78% ends at exactly
#           5*alpha0*D/gamma_min, preserving the 5/5-seed feasible endgame.
#   betas — proven transitions only: beta2 0.2 -> 0.9 at the cool-down
#           start; beta1 flat 0.1, gated to 0.02 during the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # longer exploration than the best (0.60); tail to gamma_min
_N_CYC = 3.0          # three restarts; troughs at frac ~ 0.10, 0.31, 0.52
_HI0 = 1.70           # hotter first peak than anything tried (best used 1.55)
_HI1 = 1.05           # final peak — the linear tail starts from here (proven)
_LO = 0.60            # trough lr, in D units; low lr makes burst repair cheap
_A_LO = 0.4           # penalty floor at lr peaks, in alpha0 units (proven)
_A_BURST = 7.0        # burst height at lr troughs, in alpha0 units
_B_POW = 3.0          # cubic sharpening: burst concentrates near troughs
_A_PLAT = 6.0         # bounded ALM plateau through the polish (proven)
_A_CENTER = 0.64      # logistic ramp onto the plateau, just after cool-down
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
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

    # --- alpha: anti-phase trough bursts -> plateau ramp -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    burst = (_A_BURST - _A_LO) * (1.0 - cyc) ** _B_POW * (1.0 - ramp)
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