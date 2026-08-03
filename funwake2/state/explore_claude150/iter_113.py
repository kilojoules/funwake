import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-with-notches best (+0.0577%): the schedule
# is rebuilt as a MACRO TWO-BLOCK WSD — two long holds at distinct spatial
# scales separated by ONE wide Gaussian RESTORATION VALLEY — instead of one
# decaying hold punctured by three narrow repair notches. This is the untried
# "mid-run feasibility-restoration" direction at macro scale, and it directly
# implements the requested hotter/longer early peak.
#
# Mechanism: the parent's three ~130-step notches interrupt exploration often
# but repair briefly; each burst barely finishes before the lr snaps back to
# full heat. Here the exploration budget is reorganized into
#   BLOCK A (3%..~42%):  a sustained 1.75*D hold — hotter than any level the
#                        search has ever SUSTAINED (1.5*D was the prior max) —
#                        with alpha pinned at a 0.35*alpha0 floor so basin
#                        hopping is unimpeded;
#   VALLEY  (~40%..53%): a single wide Gaussian sink to 0.30*D synchronized
#                        with one large 8*alpha0 restoration burst and a
#                        momentum cut to beta1=0.04 — hundreds of consecutive
#                        small, penalty-dominated steps that fully re-establish
#                        feasibility mid-run instead of patching it thrice;
#   BLOCK B (~53%..70%): a consolidation hold at 1.00*D with a raised
#                        1.5*alpha0 floor — re-heated refinement launched from
#                        a repaired, near-feasible layout;
#   TAIL    (70%..100%): the proven straight linear descent landing exactly on
#                        gamma_min at the last step.
# alpha keeps the proven endgame verbatim: logistic ramp to the bounded
# 6*alpha0 ALM plateau just after cool-down, then the 5/5-seed-feasible
# cubic-delayed geometric climb from 80% to the terminal 5*alpha0*D/gamma_min
# spike. betas keep the proven transitions: beta2 0.2 -> 0.9 at cool-down,
# beta1 0.1 with the valley dip and the gated drop to 0.02 in the spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.70        # exploration ends here; linear decay to gamma_min at 100%
_HA = 1.75            # block-A hold level, in units of D — hottest sustained yet
_HB = 1.00            # block-B consolidation hold level, in units of D
_LV = 0.30            # valley-bottom lr, in units of D — deep single repair
_V_CENTER = 0.46      # center of the Gaussian restoration valley
_V_SIGMA = 0.035      # valley width; meaningful dip spans roughly 40%..53%
_STEP_W = 0.02        # smooth hold-A -> hold-B step, hidden under the valley
_A_LO_A = 0.35        # block-A penalty floor, in alpha0 units (full heat)
_A_LO_B = 1.5         # block-B raised floor — keep the repaired layout honest
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_BURST = 8.0        # single mid-run restoration burst height (proven scale)
_A_STEP_CENTER = 0.55 # floor raise engages as the valley releases
_A_STEP_W = 0.03
_A_CENTER = 0.73      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.03
_F_TERM = 0.80        # terminal geometric alpha climb starts here
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_WIDTH = 0.05      # beta2 transition aligned with the cool-down start
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_VALLEY = 0.04     # momentum cut inside the restoration valley
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.89
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> hot hold A -> wide Gaussian valley -> hold B -> tail ---
    # valley is ~0 outside +-3 sigma, so block A sits cleanly at _HA*D and the
    # tail launches cleanly from _HB*D; the hold-A -> hold-B step is centered
    # under the valley so no extra transient appears at full lr.
    valley = jnp.exp(-(((frac - _V_CENTER) / _V_SIGMA) ** 2))
    stepdown = 1.0 / (1.0 + jnp.exp(-(frac - _V_CENTER) / _STEP_W))
    base = _HA + (_HB - _HA) * stepdown                       # two-level hold envelope
    lr_hold = (base - (base - _LV) * valley) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: low floor -> single valley-synchronized burst -> raised floor
    # --- -> bounded plateau -> proven terminal geometric spike ---
    floor_shift = 1.0 / (1.0 + jnp.exp(-(frac - _A_STEP_CENTER) / _A_STEP_W))
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = (_A_LO_A + (_A_LO_B - _A_LO_A) * floor_shift
                   + (_A_PLAT - _A_LO_B) * ramp
                   + _A_BURST * valley)                       # one big mid-run repair
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + momentum cut inside the valley ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _F_COOL) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_VALLEY) * valley          # no momentum mid-repair
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2