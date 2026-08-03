import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst/SGDR best (+0.0533%): the lineage is saturated
# with cosine cycling (restarts, anti-phased bursts, decaying peaks) and every
# recent cyclic tweak scored at or below the best. This attempt abandons
# cycling entirely and takes the two strongest UNTRIED rows of the prior-art
# menu simultaneously:
#
#   lr    — WSD / one-cycle (menu row 1, §2/§6): 3% linear warmup, then a HOT
#           CONSTANT HOLD at 1.30*D for over half the run — the time-integral
#           of exploration lr is far larger than any restart schedule tried
#           (parent averages ~1.0*D over its cycles; this holds 1.30*D flat),
#           which is the "higher/longer early lr" the search state asks for,
#           delivered as a stable phase instead of yet another hotter peak.
#           From 55% the proven straight linear tail lands EXACTLY on
#           gamma_min at the last step.
#   alpha — ε-CONSTRAINED SHRINKING TOLERANCE (§7.9, untried): fully decoupled
#           from lr. One smooth geometric law replaces the floor + logistic +
#           plateau + spike stack: alpha sits at a 0.45*alpha0 exploration
#           floor for the whole hot hold (AEP trades freely, like the parent's
#           peak floor), then from the cool-down start a back-loaded power
#           ramp (p=2.2) contracts the enforced violation band continuously —
#           alpha climbs geometrically and lands on the 5/5-seed-proven
#           terminal 5*alpha0*D/gamma_min at the last step. Because the climb
#           starts at 55% (vs the parent's spike at 78%), the ENTIRE 45%
#           cool-down is one long feasibility restoration: by the terminal
#           steps alpha is already near-strict while lr is near gamma_min, so
#           the debt of the hot hold is repaid gradually, not in a cliff.
#   betas — beta2 keeps the proven 0.2 -> 0.9 transition at the cool-down
#           start. beta1 is ONE-CYCLE ANTI-CORRELATED with lr (§2/§4, untried
#           — no ancestor ever exceeded 0.1): reduced to 0.08 during the hot
#           hold (tempers the sustained large steps), a smooth Gaussian bump
#           to 0.35 mid-cool-down accelerates convergence through the smooth
#           polish region where the landscape is quiet, then the proven gate
#           drops it to 0.02 for the terminal feasibility phase so momentum
#           never carries turbines back across the boundary.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_DECAY = 0.55       # hot hold ends here; linear decay to gamma_min at 100%
_HOLD = 1.30          # constant exploration lr, in units of D
_A_LO = 0.45          # exploration penalty floor during the hold, alpha0 units
_A_TERM = 5.0         # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_A_POW = 2.2          # back-loading of the geometric alpha climb
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_BASE = 0.08       # reduced momentum during the hot constant hold
_B1_MID = 0.35        # one-cycle momentum bump in the smooth polish region
_B1_BUMP_C = 0.70     # bump centered mid-cool-down
_B1_BUMP_W = 0.08     # Gaussian width; negligible by the terminal gate
_B1_LO = 0.02         # near-zero momentum in the terminal feasibility phase
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> hot constant hold -> linear tail to gamma_min ---
    p = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0)
    lr_env = gmin + (_HOLD * Dj - gmin) * (1.0 - p)   # flat hold, exact landing
    warm = jnp.minimum(frac / _F_WARM, 1.0)           # damps the start; lr only
    lr = lr_env * warm

    # --- alpha: floor, then one geometric shrinking-tolerance climb ---
    # s runs 0 -> 1 over the cool-down with power back-loading; alpha moves
    # smoothly from the exploration floor to the proven terminal magnitude.
    s = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0) ** _A_POW
    log_climb = jnp.log(jnp.maximum(_A_TERM * Dj / (gmin * _A_LO), 1.0))
    alpha = alpha0 * _A_LO * jnp.exp(s * log_climb)   # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition + one-cycle beta1 bump ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    bump = jnp.exp(-((frac - _B1_BUMP_C) / _B1_BUMP_W) ** 2)
    b1_cycle = _B1_BASE + (_B1_MID - _B1_BASE) * bump
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_cycle + (_B1_LO - b1_cycle) * b1r

    return lr, alpha, beta1, beta2