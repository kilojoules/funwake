import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the lineage is
# saturated with cyclic/SGDR shapes — every recent tweak of peaks, ramps and
# plateaus landed at or below the best. This attempt removes ALL cycles and
# combines the two prior-art directions the lineage has never embodied:
#
#   lr    — WSD / ONE-CYCLE TRAPEZOID (prior-art §2/§6, the explicitly untried
#           "hold near c*D then (near-)linear cool-down beats cosine" bet).
#           3% linear warmup -> a LONG HOT HOLD tilting 1.40*D -> 1.20*D over
#           the first 55% -> the proven straight linear tail landing exactly
#           on gamma_min at the last step. A cosine restart spends most of its
#           cycle far below its peak; this hold keeps the layout at basin-
#           hopping temperature for the ENTIRE exploration phase — strictly
#           more sustained hot time than any restart schedule in the lineage.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY (untried direction) fused
#           with the eps-CONSTRAINED SHRINKING TOLERANCE (§7.9). During the
#           hold alpha is a flat 1.2*alpha0 — 3x the burst-best's exploration
#           floor, so the long hot hold cannot accumulate unbounded violation
#           debt, yet violations are still tradeable for AEP. From 55% the
#           tolerated violation band contracts GEOMETRICALLY from ~D down to
#           gamma_min: alpha climbs smoothly (back-loaded, power 2.8) through
#           every decade, landing on the proven 5/5-seed-feasible terminal
#           5*alpha0*D/gamma_min at the final step. One continuous
#           contraction replaces the parent's burst/plateau/spike stack —
#           the enforced feasibility band shrinks to gamma_min only at the
#           end, exactly the eps-constraint mechanism.
#   betas — proven transitions kept verbatim: beta2 logistic 0.2 -> 0.9 at
#           the cool-down boundary (absorbs the growing alpha-curvature);
#           beta1 held at the native 0.1 through hold and polish, gated down
#           to 0.02 inside the terminal contraction so momentum never carries
#           turbines back across the boundary at the finish.
_F_WARM = 0.03       # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.55       # hot hold ends here; linear decay to gamma_min at 100%
_H0 = 1.40           # hold entry temperature, in units of D
_H1 = 1.20           # hold exit temperature, in units of D (tail starts here)
_A_HOLD = 1.2        # ADMM-style constant penalty during the hold, in alpha0 units
_POW = 2.8           # back-loads the geometric band contraction
_TERM_GAIN = 5.0     # terminal alpha = 5*alpha0*D/gamma_min (proven feasible scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55    # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1         # native momentum through hold and polish
_B1_LO = 0.02        # near-zero momentum during the terminal contraction
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> linear tail to gamma_min ---
    # h freezes at 1 past _F_HOLD, so the tail peels off exactly from _H1 * D.
    h = jnp.clip(frac, 0.0, _F_HOLD) / _F_HOLD
    lr_hold = (_H0 + (_H1 - _H0) * h) * Dj                    # 1.40*D -> 1.20*D tilt
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate penalty -> geometric band contraction ---
    # u = 0 through the hold (alpha flat at _A_HOLD*alpha0), then rises to 1
    # at the last step; the exp climbs through every decade between the hold
    # penalty and the proven terminal 5*alpha0*D/gamma_min, i.e. the enforced
    # violation band contracts from ~D to gamma_min only at the very end.
    u = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0) ** _POW
    log_span = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_HOLD), 1.0))
    alpha = alpha0 * _A_HOLD * jnp.exp(u * log_span)

    # --- betas: proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2