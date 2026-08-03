import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the lr leaves the
# SGDR cosine-restart family entirely and moves to the prior-art §6/§2 top
# untried lr bet — a WSD/one-cycle HOLD: warmup -> sustained hot plateau near
# c*D -> linear cool-down to gamma_min. The best's exploration spends most of
# its time transiting a cosine between peak and trough; here the layout sits
# at FULL heat for ~80% of the exploration phase, and the feasibility debt is
# repaid in three brief, scheduled COLD NOTCHES carved into the hold — the
# exact dual of warm restarts (sustained heat + short cold repairs, instead of
# short peaks + long lukewarm transitions). The proven restoration principle
# is preserved: alpha bursts fire exactly inside the notches, when lr is at
# its coldest and repair cannot destroy AEP structure.
#
#   lr    — 3% linear warmup; then a gently tilted hold 1.45*D -> 1.25*D
#           (hotter time-integral than any cosine lineage member, yet below
#           the tried 1.65*D peak amplitude) punctuated by three sharpened
#           raised-cosine notches down to 0.45*D — colder than the best's
#           0.65*D troughs, so each repair bites harder and licenses the
#           sustained heat between them. The notch shape is zero at both ends
#           of the exploration phase, pinning the cool-down start at the full
#           hold value; from 62% the proven straight linear tail lands exactly
#           on gamma_min at the last step.
#   alpha — proven machinery kept intact end-to-end: 0.4*alpha0 exploration
#           floor while hot; growing restoration bursts (3 -> 8 alpha0) fired
#           inside the lr notches; the proven logistic ramp to the bounded
#           6*alpha0 ALM plateau after cool-down begins; and the 5/5-seed
#           terminal cubic-delayed geometric climb from 78% landing on
#           5*alpha0*D/gamma_min. Terminal feasibility restoration preserved.
#   betas — proven transitions unchanged: beta2 0.2 -> 0.9 at cool-down start;
#           beta1 0.1 with a 0.05 dip inside each restoration notch (momentum
#           never carries turbines back over the boundary mid-repair) and the
#           gated drop to 0.02 during the terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_NOTCH = 3.0        # three cold restoration notches inside the hold
_HOLD0 = 1.45         # hold height at exploration start, in units of D
_HOLD1 = 1.25         # hold height at cool-down start (tilted trapezoid)
_LO_N = 0.45          # notch floor lr, in units of D — colder than tried troughs
_QN = 5.0             # sharpens notches: ~80% of exploration at full heat
_A_LO = 0.4           # exploration penalty floor while hot, in alpha0 units
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_NOTCH = 0.05      # reduced momentum inside each restoration notch
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hold with three cold notches -> linear tail ---
    # notch is 0 at fc = 0 and fc = 1, so the hold value (not a notch) is what
    # the cool-down inherits; past _F_COOL, fc freezes and notch stays 0.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    bump = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * _N_NOTCH * fc))   # 1 at notch centers
    notch = bump ** _QN                                          # sharpened: brief repairs
    hold = _HOLD0 + (_HOLD1 - _HOLD0) * fc                       # gentle tilt of the plateau
    lr_cyc = (hold - (hold - _LO_N) * notch) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)                  # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                      # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + growing in-notch bursts -> plateau -> terminal climb ---
    # bursts share the notch shape, so restoration fires exactly when lr is
    # coldest; notch = 0 past _F_COOL, handing over to the proven endgame.
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                     # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)         # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-notch beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * notch               # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2