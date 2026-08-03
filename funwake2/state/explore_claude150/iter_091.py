import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): drops the SGDR
# restart machinery entirely and composes the TWO menu bets no lineage member
# embodies — a WSD/one-cycle lr (warmup -> hot tilted HOLD -> straight linear
# cool-down, prior-art §6/§2: "hold near c*D then near-linear decay beats
# cosine/product") with an ADMM-STYLE CONSTANT MODERATE PENALTY through
# exploration that hands off to an EPSILON-CONSTRAINED SHRINKING-TOLERANCE
# alpha (§7.9): the enforced violation band contracts geometrically and only
# reaches gamma_min at the last step, which reproduces the proven terminal
# 5*alpha0*D/gamma_min restoration as the *endpoint of one smooth law* rather
# than a bolted-on spike.
#
#   lr    — 3% linear warmup, then a tilted hold decaying 1.40*D -> 1.10*D
#           over [3%, 55%]: hotter AVERAGE exploration than any restart
#           schedule tried (cycles average ~1.0*D; this holds ~1.25*D), with
#           the early-hottest tilt the decaying-peak lineage proved out. From
#           55% the proven straight linear tail lands EXACTLY on gamma_min at
#           the final step — and it is LONGER (45% of the run) than the best's
#           38%, buying back polish/feasibility time for the hotter hold.
#   alpha — ADMM-style constant 1.2*alpha0 while exploring (no floor, no
#           bursts, no plateau: one moderate multiplier, the clean untried
#           ablation). From 50% the tolerance band contracts: cubic-delayed
#           geometric climb alpha = 1.2*alpha0 * exp(u * log(5*D/(1.2*gmin))),
#           u = clip((frac-0.5)/0.5)^3. This passes ~6*alpha0 near 78% (the
#           proven plateau level at the proven climb-start point) and ends at
#           the 5/5-seed-feasible terminal 5*alpha0*D/gamma_min — strict
#           feasibility is enforced by the same magnitude that made every
#           feasible ancestor feasible.
#   betas — beta2 keeps the proven 0.2 -> 0.9 transition at the cool-down
#           start. beta1 is ONE-CYCLE ANTI-CORRELATED with lr (§2, untried):
#           low momentum (0.06) during the hot hold so big steps never ride
#           momentum through the boundary, more momentum (0.16) in the small-
#           step tail to accelerate along the valley floor, then the proven
#           gate to 0.02 during the terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.55        # hold ends here; straight linear tail to gamma_min at 100%
_HOLD0 = 1.40         # hold start level, in units of D (early-hottest tilt)
_HOLD1 = 1.10         # hold end level; the linear tail starts from here
_A_BASE = 1.2         # ADMM-style constant moderate penalty, in alpha0 units
_F_CONTRACT = 0.50    # epsilon-band contraction (alpha climb) starts here
_POW = 3.0            # cubic back-loading of the contraction (proven shape)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.06        # low momentum while lr is high (one-cycle anti-phase)
_B1_COOL = 0.16       # higher momentum in the small-step polish
_B1_XCENTER = 0.60    # momentum crossover just after the cool-down starts
_B1_XWIDTH = 0.05
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88     # proven terminal gate position
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hold -> straight linear tail onto gamma_min ---
    h = jnp.clip(frac, 0.0, _F_HOLD) / _F_HOLD
    lr_hold = (_HOLD0 + (_HOLD1 - _HOLD0) * h) * Dj           # 1.40*D -> 1.10*D
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> epsilon-constrained geometric contraction ---
    # u = 0 through exploration (constant 1.2*alpha0), then the enforced
    # tolerance band shrinks geometrically; alpha hits 5*alpha0*D/gmin only
    # at the final step, exactly the proven terminal restoration magnitude.
    u = jnp.clip((frac - _F_CONTRACT) / (1.0 - _F_CONTRACT), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_BASE), 1.0))
    alpha = alpha0 * _A_BASE * jnp.exp(u * log_term)

    # --- betas: proven beta2 transition; one-cycle beta1 with terminal gate ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1x = 1.0 / (1.0 + jnp.exp(-(frac - _B1_XCENTER) / _B1_XWIDTH))
    b1_base = _B1_HOT + (_B1_COOL - _B1_HOT) * b1x            # anti-correlated with lr
    b1g = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_base + (_B1_LO - b1_base) * b1g                # gated for the spike

    return lr, alpha, beta1, beta2