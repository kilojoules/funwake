import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%): the change is not in
# the lr/alpha waveforms (those are kept — they are the proven, 5/5-feasible
# machinery) but in the OPTIMIZER'S DYNAMICAL REGIME. Every schedule in this
# lineage explored with native low momentum (beta1 ~ 0.1): each Adam step
# re-estimates its direction from the current noisy multi-directional wake
# gradient, so turbines DITHER through the hot phase. This attempt inverts
# that — the prior-art menu's untouched beta1 row (§2 one-cycle
# anti-correlation, §4 Sutskever/Demon ramps, "momentum as implicit ALM
# multiplier"):
#
#   HIGH-MOMENTUM GLIDE: beta1 = 0.8 during the hot holds. In Adam, raising
#   beta1 does NOT enlarge the step (m_hat/sqrt(v_hat) stays ~O(1)); it
#   low-pass-filters the DIRECTION, so turbines migrate coherently down long
#   shallow wake valleys (global rearrangements) instead of jittering in
#   place — more basin reach at the same, feasibility-safe step size. The
#   momentum buffer also integrates the boundary/spacing gradient across
#   steps, acting as a poor-man's ALM multiplier at the exploration floor.
#
#   Anti-correlated collapse: beta1 drops 0.8 -> 0.04 inside each repair
#   notch (a smoothed direction estimate must never carry turbines back
#   across the boundary mid-repair — the parent's proven dip, now with far
#   deeper contrast), and glides 0.8 -> 0.1 at cool-down so the proven
#   native-momentum polish and gated 0.02 terminal spike are unchanged.
#
# The proven waveforms are preserved with only the hinted heat increase:
#   lr    — 3% warmup -> flat-top hold, envelope 1.6*D -> 1.15*D (slightly
#           hotter than 1.5 -> 1.1, as momentum smoothing tolerates it),
#           three narrow sin^12 repair notches to 0.35*D, exploration
#           stretched 62% -> 64%, then the proven straight linear tail
#           landing exactly on gamma_min at the last step.
#   alpha — unchanged architecture: 0.4*alpha0 floor, growing notch-locked
#           restoration bursts (3 -> 8 alpha0), logistic ramp to the bounded
#           6*alpha0 ALM plateau just after cool-down, cubic-delayed
#           geometric climb from 78% to the terminal 5*alpha0*D/gamma_min
#           feasibility spike.
#   beta2 — proven 0.2 -> 0.9 transition, re-aligned to the 64% cool-down.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.64        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three repair notches inside the exploration phase
_Q = 6.0              # notch = sin^(2Q); ~80% of each cycle at full hold lr
_HI0 = 1.6            # initial hold level, in units of D — hotter, momentum-smoothed
_HI1 = 1.15           # final hold level; the linear tail starts from here
_LO = 0.35            # notch-bottom lr — deep, surgical repair windows (proven)
_A_LO = 0.4           # exploration penalty floor at full heat, in alpha0 units
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.68      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.64     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_EXPL = 0.8        # high-momentum glide during the hot holds (the new bet)
_B1_POL = 0.1         # native momentum for the proven cool-down polish
_B1_MID = 0.64        # glide -> polish transition center (at cool-down)
_B1_MIDW = 0.035
_B1_NOTCH = 0.04      # momentum collapses inside each repair notch
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat-top hold with three deep narrow notches -> tail ---
    # notch = sin(pi*N*fc)^(2Q): 0 at fc=0 and fc=1 (so the tail launches from
    # the clean hold level _HI1*D), ~0 for ~80% of each cycle, 1 briefly at
    # each cycle midpoint. fc freezes at 1 past _F_COOL.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    notch = jnp.sin(jnp.pi * _N_CYC * fc) ** (2.0 * _Q)
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    lr_hold = (_LO + (hi - _LO) * (1.0 - notch)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + notch-synchronized growing bursts -> plateau -> climb ---
    # Bursts fire exactly inside the lr notches (repair when steps are small)
    # and vanish for frac >= _F_COOL; the proven bounded endgame then takes over.
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- beta2: proven 0.2 -> 0.9 transition at cool-down ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    # --- beta1: one-cycle momentum inversion, anti-correlated with repairs ---
    # High-momentum glide (0.8) across the hot holds -> native 0.1 for the
    # polish; collapses to 0.04 inside every repair notch; gated to 0.02
    # during the terminal alpha spike (proven endgame untouched).
    b1m = 1.0 / (1.0 + jnp.exp(-(frac - _B1_MID) / _B1_MIDW))
    b1_base = _B1_EXPL + (_B1_POL - _B1_EXPL) * b1m           # glide -> polish
    b1_exp = b1_base - (b1_base - _B1_NOTCH) * notch          # collapse while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2