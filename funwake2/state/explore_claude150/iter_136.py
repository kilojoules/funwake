import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top/notch best (+0.0577%): the exploration
# phase is reorganized from three brief periodic repair notches into ONE
# MACRO-CYCLE of three contiguous regimes — a graduated continuation /
# filter-method "restoration phase" design at a timescale no prior attempt
# has used:
#
#   Phase A (0-30%):  GREEDY HEAT. A single uninterrupted hold at 1.75*D —
#       both hotter and longer-contiguous than any tried hold (parent: 1.5*D
#       broken by notches) — with the proven 0.4*alpha0 penalty floor. All
#       budget goes to basin hopping; constraint debt is allowed to build.
#   Phase B (30-42%): THE GREAT REPAIR. lr drops to a 0.35*D valley for ~960
#       steps while a single long Gaussian alpha burst (peak 10*alpha0)
#       performs a genuine feasibility-restoration solve — unlike the
#       parent's fleeting notches, this window is long enough for the
#       constraint subproblem to actually converge. beta1 dips so momentum
#       cannot drag turbines back across the boundary mid-repair.
#   Phase C (42-68%): REFINEMENT REHEAT. lr returns to a medium 1.15*D hold
#       to explore *within* the repaired basin, with the alpha floor
#       RATCHETED UP to 1.0*alpha0 (graduated penalty: never re-borrow the
#       debt just paid off).
#
# The proven endgame is preserved intact: linear lr tail from 68% landing
# exactly on gamma_min, logistic alpha ramp to the bounded 6*alpha0 ALM
# plateau, the 5/5-seed-feasible cubic-delayed geometric climb from 78% to
# the terminal 5*alpha0*D/gamma_min spike, beta2 0.2 -> 0.9 at cool-down,
# and the gated beta1 drop to 0.02 under the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_REPAIR0 = 0.30     # hot phase ends; the great repair begins
_F_REPAIR1 = 0.42     # repair ends; refinement reheat begins
_F_COOL = 0.68        # exploration ends; linear decay to gamma_min at 100%
_W_TRANS = 0.015      # width of the smooth sigmoid level transitions
_LR_HOT = 1.75        # phase-A hold, in units of D — hotter AND longer than tried
_LR_REPAIR = 0.35     # repair-valley lr, in units of D (proven surgical depth)
_LR_REFINE = 1.15     # phase-C refinement hold, in units of D
_A_FLOOR_HOT = 0.4    # phase-A penalty floor, in alpha0 units (proven)
_A_FLOOR_REF = 1.0    # ratcheted phase-C floor — graduated penalty continuation
_A_BURST = 10.0       # peak of the single long restoration burst, alpha0 units
_A_BURST_C = 0.36     # burst centered in the repair valley
_A_BURST_W = 0.045    # Gaussian width — spans the whole valley
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.72      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.68     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_REPAIR = 0.04     # low momentum throughout the repair valley
_B1_REPAIR_W = 0.05   # width of the beta1 repair window
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> hot hold -> repair valley -> refinement hold -> tail ---
    # Two smooth level transitions chain the three exploration regimes; both
    # sigmoids are fully saturated by _F_COOL, so the linear tail launches
    # cleanly from the refinement level and lands exactly on gamma_min.
    sig_dn = 1.0 / (1.0 + jnp.exp(-(frac - _F_REPAIR0) / _W_TRANS))
    sig_up = 1.0 / (1.0 + jnp.exp(-(frac - _F_REPAIR1) / _W_TRANS))
    lr_u = _LR_HOT + (_LR_REPAIR - _LR_HOT) * sig_dn \
                   + (_LR_REFINE - _LR_REPAIR) * sig_up
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_u * Dj - gmin) * (1.0 - p)            # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor -> single long restoration burst -> ratcheted floor
    #     -> bounded plateau -> proven terminal geometric climb ---
    burst = _A_BURST * jnp.exp(-(((frac - _A_BURST_C) / _A_BURST_W) ** 2))
    floor = _A_FLOOR_HOT + (_A_FLOOR_REF - _A_FLOOR_HOT) * sig_up
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = floor + burst + (_A_PLAT - _A_FLOOR_REF) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + sustained beta1 dip in the repair valley ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    repair_win = jnp.exp(-(((frac - _A_BURST_C) / _B1_REPAIR_W) ** 2))
    b1_exp = _B1_HI - (_B1_HI - _B1_REPAIR) * repair_win      # low momentum while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2