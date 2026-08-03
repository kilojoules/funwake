import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the two menu
# directions the lineage has never combined — an ADMM-STYLE CONSTANT MODERATE
# PENALTY (prior-art §7.2/§7.9) under a WSD/ONE-CYCLE lr (§2/§6). Every
# schedule in the lineage explores at a near-zero alpha floor and then repays
# the accumulated violation debt (once at the end, or per-cycle via bursts).
# This schedule inverts the bet: NEVER INCUR THE DEBT. A constant 2*alpha0
# penalty keeps the layout near-feasible throughout exploration, and that is
# precisely what licenses more sustained heat than any restart train can
# deliver — a long FLAT HOT HOLD instead of oscillating peaks with cold
# troughs wasted at low lr.
#
#   lr    — one-cycle/WSD trapezoid: 4% linear warmup -> flat hold at 1.45*D
#           until 55% (far more total hot exposure than the 1.65->1.05*D
#           decaying peaks, whose troughs at 0.65*D spend half the exploration
#           phase cold) -> the proven straight linear tail landing exactly on
#           gamma_min at the last step. No restarts: with feasibility held by
#           alpha, re-annealing buys nothing and the cold trough time is
#           reclaimed as hot search time.
#   alpha — ADMM-style constant 2*alpha0 through the entire hot phase (high
#           enough that boundary/spacing violations stay small and local, low
#           enough that turbines still slide between basins), then the PROVEN
#           endgame preserved verbatim in structure: logistic lift to the
#           bounded 6*alpha0 ALM plateau as the cool-down begins, and the
#           cubic-delayed geometric climb from 78% landing on the 5/5-seed-
#           feasible terminal 5*alpha0*D/gamma_min. Because violation never
#           accumulates, the terminal spike arrives with almost nothing to
#           collect and cannot shred AEP structure late.
#   betas — one-cycle momentum, anti-correlated with lr (menu bet, §2/§4):
#           beta1 = 0.05 during the hot hold so momentum never carries
#           turbines through the boundary at 1.45*D steps, native 0.1 during
#           the cool-down polish, and the proven gate to 0.02 inside the
#           terminal spike. beta2 keeps the proven 0.2 -> 0.9 transition at
#           the start of the cool-down.
_F_WARM = 0.04        # linear lr warmup over the first 4%
_F_HOLD = 0.55        # hot hold ends here; linear decay to gamma_min at 100%
_HOLD = 1.45          # hold lr, in units of D — sustained, not peaked
_A_ADMM = 2.0         # constant moderate penalty during the hot phase, alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau for the polish phase (proven)
_A_CENTER = 0.60      # logistic alpha lift centered just after the hold ends
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with the start of the decay
_B2_WIDTH = 0.05
_B1_HOT = 0.05        # low momentum while lr is hot (one-cycle anti-correlation)
_B1_MID = 0.1         # native momentum for the cool-down polish
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat hot hold at 1.45*D -> linear tail to gamma_min ---
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (_HOLD * Dj - gmin) * (1.0 - p)           # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> logistic lift to plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_ADMM + (_A_PLAT - _A_ADMM) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: one-cycle beta1 (low-hot -> native-polish -> spike gate) ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_mid = _B1_HOT + (_B1_MID - _B1_HOT) * b2r              # rises as lr cools
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1r

    return lr, alpha, beta1, beta2