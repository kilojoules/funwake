import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-with-notches best (+0.0577%): the MACRO
# architecture changes from "one exploration block with three transient repair
# notches" to a TWO-BLOCK WARM-RESTART with a single WIDE MID-RUN FEASIBILITY
# RESTORATION TRENCH and an ALM MULTIPLIER RATCHET — two of the explicitly
# untried directions (mid-run restoration; decoupled penalty whose level is
# updated like an augmented-Lagrangian multiplier, not pulsed).
#
# Mechanism, and why it is not another notch tweak:
#   * Block A (3%-30%) holds the lr HOTTER and LONGER than anything tried
#     (1.7*D sustained — the requested higher/longer early peak) with a LOW
#     0.4*alpha0 penalty: pure basin-hopping, deliberately running up
#     constraint debt that a narrow notch could never repay.
#   * One WIDE trench (30%-38%, ~640 steps) then repays it all at once:
#     lr sinks to 0.25*D while alpha holds a genuine 8*alpha0 restoration
#     PLATEAU (not a spike) and beta1 drops to 0.04 — a full mid-run
#     feasibility restoration, per the filter/funnel prior art.
#   * ALM RATCHET: after the trench the alpha floor steps PERMANENTLY from
#     0.4 to 1.2*alpha0 (the multiplier estimate is retained, not reset), so
#     Block B (38%-62%, warm-restarted at 1.3*D decaying to 1.05*D) explores
#     the repaired layout while staying near-feasible. The parent's bursts
#     always relaxed back to the same floor; this schedule learns from repair.
#   * The 5/5-seed-feasible ENDGAME is preserved verbatim: linear lr tail
#     landing exactly on gamma_min, logistic alpha ramp to the bounded
#     6*alpha0 plateau at 66%, cubic-delayed geometric climb from 78% to the
#     terminal 5*alpha0*D/gamma_min spike, beta2 0.2 -> 0.9 at cool-down,
#     gated beta1 drop to 0.02 under the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear tail to gamma_min at 100%
_T0 = 0.30            # restoration trench opens
_T1 = 0.38            # restoration trench closes
_TW = 0.008           # trench edge sharpness (smooth logistic box, traceable)
_RW = 0.02            # ratchet transition width, centered mid-trench
_H1 = 1.7             # Block A hold level, in units of D — hotter and longer
_H2A = 1.3            # Block B warm-restart hold level
_H2B = 1.05           # Block B end level; the linear tail launches from here
_LO_T = 0.25          # trench-bottom lr — small steps, repair only
_A_LO1 = 0.4          # Block A penalty floor, in alpha0 units (proven value)
_A_LO2 = 1.2          # Block B ratcheted floor — retained ALM multiplier
_A_TRENCH = 8.0       # restoration plateau height inside the trench
_A_PLAT = 6.0         # bounded ALM endgame plateau, in alpha0 units (proven)
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
_B1_TRENCH = 0.04     # momentum nearly off during the mid-run restoration
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- smooth indicators (all traceable; no branches on step) ---
    # trench: logistic box ~1 on [_T0, _T1], ~0 elsewhere
    box = (1.0 / (1.0 + jnp.exp(-(frac - _T0) / _TW))) \
        * (1.0 / (1.0 + jnp.exp(-(_T1 - frac) / _TW)))
    # ratchet: 0 -> 1 across the trench center; switches hold level and alpha floor
    post = 1.0 / (1.0 + jnp.exp(-(frac - 0.5 * (_T0 + _T1)) / _RW))

    # --- lr: warmup -> hot hold A -> trench -> warm-restart hold B -> tail ---
    fB = jnp.clip((frac - _T1) / (_F_COOL - _T1), 0.0, 1.0)
    h2 = _H2A + (_H2B - _H2A) * fB                           # Block B envelope
    hold = _H1 * (1.0 - post) + h2 * post                    # macro warm restart
    lr_x = (hold * (1.0 - box) + _LO_T * box) * Dj           # trench carved in
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_x - gmin) * (1.0 - p)                # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                  # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: low floor -> wide restoration plateau -> RATCHETED floor ->
    #     bounded ALM plateau -> proven terminal geometric climb ---
    floor = _A_LO1 + (_A_LO2 - _A_LO1) * post                # multiplier retained
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = floor + (_A_PLAT - floor) * ramp + _A_TRENCH * box
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)     # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + trench beta1 cut ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_TRENCH) * box            # momentum off mid-repair
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2