import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): drops the entire
# SGDR-restart + burst machinery and instead combines the three untried menu
# bets in one clean architecture:
#
#   lr    — WSD / one-cycle (prior-art §2/§6, untried): 4% linear warmup, then
#           a STABLE HOT HOLD at 1.35*D until 58% — the time-integrated hot-lr
#           dose is far larger than the burst-best's decaying cosine cycles
#           (which average only ~1.0*D over their exploration window), i.e.
#           the "higher/longer peak" the guidance asks for, delivered as a
#           plateau instead of spikes — followed by the proven straight linear
#           cool-down landing exactly on gamma_min at the last step.
#   alpha — ADMM-style CONSTANT moderate penalty (untried) during the hold:
#           a flat 0.7*alpha0, no bursts, no coupling to lr — violation is
#           priced steadily while basins are explored. From 45% an
#           EPSILON-CONSTRAINED SHRINKING TOLERANCE BAND (§7.9, untried)
#           takes over: alpha = 0.7*alpha0*(D_eff/gamma_t) with the enforced
#           band gamma_t contracting GEOMETRICALLY from ~D down to gamma_min,
#           back-loaded quartically so mid-cool-down alpha passes through the
#           proven plateau range (~1-6*alpha0 near 75-85%) and only then
#           accelerates. At the final step it lands exactly on the
#           5/5-seed-feasible terminal scale 5*alpha0*D/gamma_min — the
#           terminal feasibility restoration is preserved, but as the natural
#           endpoint of one smooth contraction instead of a bolted-on spike.
#   beta1 — one-cycle anti-correlation + Sutskever INCREASING ramp (§2/§4,
#           untried): low momentum (0.06) while lr is hot so steps stay local
#           and reversible, RISING to 0.28 through the cool-down — momentum
#           integrates the growing constraint gradient like an implicit ALM
#           multiplier, letting a moderate alpha enforce feasibility — then
#           the proven gated drop to 0.02 inside the terminal contraction so
#           momentum never carries turbines back across the boundary.
#   beta2 — the proven 0.2 -> 0.9 logistic transition, centered on the end of
#           the hold, absorbing the alpha-driven curvature growth.
_F_WARM = 0.04        # linear lr warmup fraction
_F_HOLD = 0.58        # stable hot phase ends here; linear decay to gamma_min after
_LR_HOLD = 1.35       # hold-phase lr, in units of D
_A_CONST = 0.7        # ADMM-style constant penalty during exploration, alpha0 units
_F_ALPHA = 0.45       # tolerance-band contraction starts here
_P_ALPHA = 4.0        # quartic back-loading of the contraction
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven feasible scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOLD = 0.06       # low momentum while lr is hot (one-cycle anti-correlation)
_B1_MID = 0.28        # Sutskever ramp peak during cool-down (implicit ALM multiplier)
_B1_UP_CENTER = 0.70
_B1_UP_WIDTH = 0.06
_B1_LO = 0.02         # near-zero momentum inside the terminal contraction
_B1_DN_CENTER = 0.90
_B1_DN_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> stable hold at 1.35*D -> linear tail to gamma_min ---
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (_LR_HOLD * Dj - gmin) * (1.0 - p)   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)              # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate penalty -> geometric tolerance contraction ---
    # s runs 0 -> 1 quartically from _F_ALPHA; alpha climbs in log space from the
    # constant 0.7*alpha0 to exactly _TERM_GAIN*alpha0*D/gamma_min at the last
    # step, i.e. the enforced violation band shrinks geometrically to gamma_min.
    s = jnp.clip((frac - _F_ALPHA) / (1.0 - _F_ALPHA), 0.0, 1.0) ** _P_ALPHA
    log_ratio = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_CONST), 1.0))
    alpha = alpha0 * _A_CONST * jnp.exp(s * log_ratio)

    # --- beta2: proven low -> high transition at the cool-down start ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    # --- beta1: low in the hot hold -> rising ramp in cool-down -> terminal drop ---
    b1_up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_mid = _B1_HOLD + (_B1_MID - _B1_HOLD) * b1_up
    b1_dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1_dn

    return lr, alpha, beta1, beta2