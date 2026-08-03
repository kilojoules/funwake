import jax.numpy as jnp

# STRUCTURALLY NEW vs the SGDR-decaying-peak best (+0.0450%): the one listed
# direction still untried — FULLY CYCLIC ALPHA ANTI-PHASED WITH THE SGDR WARM
# RESTARTS — replaces both the best's flat 0.4*alpha0 exploration floor and
# the parent's narrow von-Mises bursts. Alpha is no longer an envelope with
# spikes bolted on: it is a periodic explore/repair wave locked in exact
# anti-phase with lr, with GRADUATED amplitude (each repair wave stricter than
# the last), so every restart is a complete annealed round: hot kick -> broad
# repair while lr is cool. Mid-slope the wave passes through ~1.3*alpha0 — an
# ADMM-style moderate penalty band — so violation debt is repaid continuously,
# not only at trough centers.
#   lr    — proven SGDR decaying-peak restarts, pushed slightly HOTTER early
#           (first peak 1.65*D vs the best's 1.55*D — the "higher early peak"
#           bet), peaks decaying to 1.00*D over three cycles in the first 60%;
#           troughs pinned at 0.62*D (never the drop-to-floor failure mode);
#           proven 3% warmup and linear tail landing exactly on gamma_min.
#           The hotter first kick is safe BECAUSE the anti-phased alpha wave
#           repairs its damage before the next kick.
#   alpha — log-space anti-phased wave on the 0.4*alpha0 floor: repair crests
#           at the lr troughs grow 4*alpha0 -> ~15*alpha0 across the three
#           cycles (graduated tightening: cheap violations early, strict
#           late), zero at every lr peak so exploration stays free. Past 60%
#           the wave shuts off (anti-phase pins it at a peak) and the BEST's
#           proven bounded logistic plateau (6*alpha0, center 0.62) carries
#           the polish; the identical cubic-delayed terminal climb from 78%
#           lands alpha on the proven 5*alpha0*D/gamma_min, preserving the
#           5/5-seed feasible endgame.
#   betas — proven beta2 0.2 -> 0.9 at the cool-down start; beta1 0.1 while
#           exploring, anti-correlated with the repair wave (down to 0.04 at
#           each trough so momentum cannot drag turbines back across the
#           boundary mid-repair), gated to 0.02 during the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.60        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three explore/repair rounds inside the exploration phase
_HI0 = 1.65           # first restart peak — hotter than anything scored so far
_HI1 = 1.00           # last restart peak; the linear tail starts from here
_LO = 0.62            # bounded trough lr, in units of D
_A_LO = 0.4           # exploration penalty floor at every lr peak, alpha0 units
_A_REP0 = 4.0         # first repair crest, alpha0 units (gentle: cheap early debt)
_A_REP1 = 20.0        # final repair-crest scale — crests grow geometrically
_A_PLAT = 6.0         # bounded ALM-style polish plateau (proven), alpha0 units
_A_CENTER = 0.62      # logistic plateau ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum at the hot lr peaks
_B1_REP = 0.04        # low momentum at each repair crest (anti-correlated w/ lr)
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> 3 decaying-peak cosine restarts -> linear tail ---
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

    # --- alpha: anti-phased graduated repair wave -> plateau -> terminal climb ---
    # repair = 1 exactly at the lr troughs, 0 at every peak, and identically 0
    # once fc pins at 1 — the wave hands off cleanly to the logistic plateau.
    # Crest height grows geometrically with fc: floor * (REP0/A_LO) * (REP1/REP0)^fc.
    repair = 1.0 - cyc
    rep_log = jnp.log(_A_REP0 / _A_LO) + fc * jnp.log(_A_REP1 / _A_REP0)
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    plateau = _A_LO + (_A_PLAT - _A_LO) * ramp                # in alpha0 units
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * plateau * jnp.exp(repair * rep_log + s * log_term)

    # --- betas: beta2 up for the polish; beta1 rides the repair wave down ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    b1_expl = _B1_HI - (_B1_HI - _B1_REP) * repair
    beta1 = b1_expl + (_B1_LO - b1_expl) * b1r

    return lr, alpha, beta1, beta2