import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-with-notches best (+0.0577%): the PERIODIC
# repair cadence (three notch/burst cycles) is replaced by a TWO-BLOCK
# "GREEDY, THEN CONSOLIDATE" plan built around a single GRAND MID-RUN
# FEASIBILITY-RESTORATION TRENCH (the filter/funnel restoration-phase idea
# from the prior-art menu, deployed mid-run at full strength instead of as
# small distributed bursts) plus an ALPHA RATCHET.
#
# Why this is a different mechanism, not a re-tune:
#  * Block A (warmup -> 43%) is a single UNINTERRUPTED hold at 1.7*D — hotter
#    and longer than any tried exploration phase (the guidance's "higher/
#    longer lr peak early"), with alpha pinned at the 0.4*alpha0 floor. Pure
#    AEP ascent; violation debt is allowed to accumulate.
#  * ONE grand restoration trench (~43%-57%, flat-bottomed super-Gaussian):
#    lr crashes to 0.25*D while alpha bursts to 10*alpha0 and beta1 drops to
#    0.04 — roughly the combined repair capacity of all three parent notches,
#    spent in one sustained window where the debt is paid down in full.
#  * ALPHA RATCHET: after the trench, alpha does NOT fall back to the floor
#    (the parent's bursts did). It steps up permanently to a moderate
#    2*alpha0 ALM level, so Block B (57%-70%, cooler 1.0*D hold) refines the
#    layout without re-accruing the debt just repaid.
#  * The 5/5-seed-feasible endgame is preserved verbatim in structure:
#    linear lr tail landing exactly on gamma_min, logistic alpha ramp to the
#    bounded 6*alpha0 plateau at cool-down, cubic-delayed geometric climb to
#    the terminal 5*alpha0*D/gamma_min spike, beta2 0.2 -> 0.9 at cool-down,
#    gated beta1 drop to 0.02 during the spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.70        # exploration ends LATER than the parent's 0.62 (more heat)
_TR_C = 0.50          # grand restoration trench center
_TR_W = 0.06          # trench width scale (flat core ~0.46-0.54)
_HB_W = 0.03          # width of the A->B hold step-down (hidden inside trench)
_HI_A = 1.7           # Block A hold level, units of D — sustained, hotter than 1.5
_HI_B = 1.0           # Block B consolidation hold level
_LR_TR = 0.25         # trench-bottom lr — deepest repair window tried
_A_LO = 0.4           # exploration penalty floor, alpha0 units (proven)
_A_MID = 2.0          # post-trench ratchet level — never returns to the floor
_A_PLAT = 6.0         # bounded ALM plateau, alpha0 units (proven)
_A_TR = 10.0          # grand-trench burst height, alpha0 units
_STEP_C = 0.53        # ratchet step centered at the trench exit
_STEP_W = 0.025
_A_CENTER = 0.74      # logistic alpha ramp just after cool-down start (proven offset)
_A_WIDTH = 0.04
_F_TERM = 0.80        # terminal geometric alpha climb start
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.70     # beta2 transition aligned with cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_TR = 0.04         # near-zero momentum inside the restoration trench
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.90
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- trench gate: flat-bottomed super-Gaussian, ~1 inside, ~0 outside ---
    u = (frac - _TR_C) / _TR_W
    u2 = u * u
    g = jnp.exp(-(u2 * u2 * u2))

    # --- lr: warmup -> hot Block A hold -> grand trench -> Block B hold -> tail ---
    # The A->B step-down is a logistic hidden under the trench, so the visible
    # waveform is: flat 1.7*D, one deep 0.25*D trench, flat 1.0*D, straight tail.
    hi = _HI_A + (_HI_B - _HI_A) / (1.0 + jnp.exp(-(frac - _TR_C) / _HB_W))
    lr_exp = (_LR_TR + (hi - _LR_TR) * (1.0 - g)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_exp - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor -> grand burst in the trench -> RATCHET to 2*alpha0 ---
    # --- -> logistic ramp to bounded plateau -> proven terminal climb ---
    step_up = 1.0 / (1.0 + jnp.exp(-(frac - _STEP_C) / _STEP_W))
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = (_A_LO + (_A_MID - _A_LO) * step_up
                   + (_A_PLAT - _A_MID) * ramp + _A_TR * g)
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + deep beta1 cut inside the trench ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_TR) * g                   # no momentum mid-repair
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2