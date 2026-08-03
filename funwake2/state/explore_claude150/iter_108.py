import jax.numpy as jnp

# STRUCTURALLY NEW vs the SGDR-burst best (+0.0533%): the exploration phase is
# rebuilt around the two menu directions the lineage has NOT tried — a
# WSD/one-cycle lr (warmup -> hot STABLE HOLD -> long pure-LINEAR cool-down,
# prior-art §2/§6) and an ADMM-STYLE CONSTANT MODERATE PENALTY (search-state
# list; §7.2) instead of cyclic bursts. The proven 5/5-feasible endgame
# (logistic ramp to the bounded 6*alpha0 ALM plateau + cubic-delayed geometric
# climb landing on 5*alpha0*D/gamma_min, beta2 0.2->0.9, terminal beta1->0.02)
# is preserved verbatim.
#
#   lr    — 3% linear warmup, then a sustained hold at 1.40*D for half the
#           run. Integrated hot-lr time (~0.70 D-fractions) exceeds the SGDR
#           best's cycle average (~0.62): the layout spends MORE total time
#           above 1*D than any cosine schedule tried, without ever touching
#           the risky 1.65*D transient. From 50% a straight linear tail lands
#           exactly on gamma_min at the last step (§6: near-linear cool-down
#           beats cosine/product decay).
#   alpha — DECOUPLED and flat: a constant 1.5*alpha0 through the entire hold
#           (ADMM-style moderate penalty — higher than the best's 0.4 floor,
#           so violation debt stays bounded WITHOUT restoration bursts, which
#           is what licenses the long hot hold). Then the proven logistic
#           ramp (centered just after cool-down start) to the bounded
#           6*alpha0 plateau, and the proven cubic terminal climb from 78% to
#           5*alpha0*D/gamma_min.
#   betas — beta2: proven 0.2 -> 0.9 logistic at the cool-down start.
#           beta1: NEW rise-then-fall — held at native 0.1 while hot, ramped
#           UP to 0.4 during the enforcement phase (momentum as an implicit
#           ALM multiplier: accumulated constraint gradients act like a
#           multiplier estimate, letting the bounded plateau enforce more
#           than its size suggests — menu beta1 bet), then the proven gated
#           drop to 0.02 inside the terminal spike so momentum never carries
#           turbines back across the boundary at the end.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.50        # stable hot hold ends here; linear tail to gamma_min
_HOLD = 1.40          # hold lr in units of D — sustained, not a transient peak
_A_ADMM = 1.5         # constant moderate penalty during the hold, alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.60      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.05
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.50     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum during the hot hold
_B1_MID = 0.4         # ALM-multiplier momentum during enforcement
_B1_MID_CENTER = 0.62
_B1_MID_WIDTH = 0.05
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> stable hold at 1.40*D -> pure linear tail to gamma_min ---
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (_HOLD * Dj - gmin) * (1.0 - p)   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)           # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> bounded plateau -> terminal geometric climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_ADMM + (_A_PLAT - _A_ADMM) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; beta1 rises for enforcement, then gates low ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1m = 1.0 / (1.0 + jnp.exp(-(frac - _B1_MID_CENTER) / _B1_MID_WIDTH))
    b1_mid = _B1_HI + (_B1_MID - _B1_HI) * b1m            # 0.1 -> 0.4 while enforcing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1r              # -> 0.02 in the spike

    return lr, alpha, beta1, beta2