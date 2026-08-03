import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top+3-notch best (+0.0577%): a MACRO-PHASE
# INVERSION of the repair cadence. The parent interleaves three narrow repair
# notches into the hot hold, so exploration is interrupted every ~1650 steps
# and the hold envelope must decay (1.5*D -> 1.1*D) to stay controllable.
# Here the exploration budget is consolidated into ONE long, UNINTERRUPTED,
# HOTTER hold (a sustained 1.6*D — cumulatively hotter and longer-contiguous
# than anything tried), and ALL mid-run repair is consolidated into a single
# WIDE FLAT-BOTTOMED RESTORATION VALLEY at mid-run — the prior-art menu's
# "mid-run feasibility-restoration burst" (filter/funnel restoration phase)
# implemented at macro scale instead of as micro-notches. The valley is a
# super-Gaussian window (~10% of the run, ~500 steps at full depth): lr sinks
# to 0.30*D while a single large alpha burst (9*alpha0) repays the entire
# accumulated constraint debt in one dedicated restoration phase. The valley
# doubles as the phase transition: the hold level steps 1.6*D -> 1.05*D
# inside it, so the post-valley warm hold refines the repaired layout before
# the proven endgame takes over.
#
#   lr    — 3% linear warmup (proven) -> hot hold 1.6*D (3%..~45%) -> wide
#           super-Gaussian valley to 0.30*D centered at 50% -> warm hold
#           1.05*D (~55%..68%) -> proven straight linear tail landing exactly
#           on gamma_min at the last step.
#   alpha — 0.4*alpha0 exploration floor (proven) + single 9*alpha0
#           restoration burst filling the valley, logistic ramp to the
#           bounded 6*alpha0 ALM plateau centered at 70%, then the proven
#           5/5-seed-feasible cubic-delayed geometric climb from 78% to the
#           terminal 5*alpha0*D/gamma_min spike.
#   betas — proven transitions: beta2 0.2 -> 0.9 at cool-down; beta1 0.1 with
#           a dip to 0.05 across the restoration valley (momentum must not
#           drag turbines back over the boundary mid-repair) and the gated
#           drop to 0.02 during the terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_C_VAL = 0.50         # center of the mid-run restoration valley
_W_VAL = 0.05         # super-Gaussian half-width; flat bottom ~47%..53%
_F_COOL = 0.68        # exploration ends here; linear decay to gamma_min at 100%
_HI0 = 1.6            # hot-hold level, in units of D — sustained and uninterrupted
_HI1 = 1.05           # post-valley warm-hold level; the linear tail starts here
_W_HI = 0.03          # width of the hold-level step hidden inside the valley
_LO = 0.30            # valley-floor lr — cold, dedicated repair steps
_A_LO = 0.4           # exploration penalty floor at full heat, in alpha0 units
_A_BURST = 9.0        # single consolidated restoration burst, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.70      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.68     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_VAL = 0.05        # reduced momentum inside the restoration valley
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- restoration valley: super-Gaussian flat-bottom window at mid-run ---
    # exp(-z^4) is ~1 across a flat bottom (~47%..53%), falls off sharply at
    # the shoulders, and is numerically 0 at warmup start, cool-down start,
    # and the final step — so the tail launches cleanly from _HI1*D and the
    # terminal alpha scale is untouched.
    z = (frac - _C_VAL) / _W_VAL
    valley = jnp.exp(-(z ** 4))

    # --- lr: warmup -> hot hold -> valley -> warm hold -> linear tail ---
    # The hold level steps from 1.6*D down to 1.05*D via a sigmoid centered
    # inside the valley, so the transition is hidden where lr is smallest.
    hgate = 1.0 / (1.0 + jnp.exp(-(_C_VAL - frac) / _W_HI))
    hi = _HI1 + (_HI0 - _HI1) * hgate
    lr_hold = (_LO + (hi - _LO) * (1.0 - valley)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + single valley-synchronized burst -> plateau -> climb ---
    # The consolidated burst fills the cold valley (repair happens where steps
    # are small); the proven bounded ALM endgame then takes over.
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + _A_BURST * valley
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + valley-wide beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_VAL) * valley             # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2