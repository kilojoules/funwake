import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the two prior-art
# menu rows still untried anywhere in the lineage, fused into one schedule.
#
#   lr    — WSD / TRAPEZOID hold-then-linear (prior-art §6 / §2). After the
#           proven 3% warmup the lr does NOT oscillate: it holds a tilted hot
#           plateau (1.50*D -> 1.10*D) for over half the run, then takes one
#           straight linear cool-down landing exactly on gamma_min at the last
#           step. Integrated exploration lr (~1.3*D average across 52% of the
#           steps) exceeds every restart schedule tried — their cosine troughs
#           park half the hot phase near 0.65*D. This delivers the requested
#           "higher/longer peak early" as area under the curve rather than a
#           taller-but-momentary spike, and tests the menu's core lr bet:
#           hold near c*D then (near-)linear decay beats cosine/product decay.
#   alpha — ε-CONSTRAINED SHRINKING TOLERANCE (§7.9), fully decoupled from lr.
#           A low floor (0.5*alpha0) leaves basin formation unpenalized while
#           the plateau is hot; then one sustained GEOMETRIC climb spans the
#           whole second half, equivalent to contracting the enforced
#           violation band gamma(t) = D * (gamma_min/D)^(s^2) down to
#           gamma_min exactly at the last step via alpha = c*alpha0*D/gamma(t).
#           No mid-run bursts, no plateau, no separate terminal spike — the
#           violation debt is repaid continuously. The climb passes THROUGH
#           the proven plateau scale (~5*alpha0 near 75%) on its way to the
#           proven 5/5-seed-feasible terminal value 5*alpha0*D/gamma_min, and
#           its feasibility pressure over the 85-95% stretch is an order of
#           magnitude ABOVE the parent's cubic-delayed spike, so strict
#           feasibility gets safer while the hotter hold buys AEP.
#   betas — proven transitions, phase-locked to the new geometry: beta2
#           0.2 -> 0.9 at the hold->decay corner (adaptive scaling absorbs the
#           growing ~alpha constraint curvature, §4); beta1 gated 0.1 -> 0.02
#           late so momentum cannot carry turbines back across the boundary
#           while the contracting band squeezes out the last violations.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.55        # hot hold ends here; single linear decay to gamma_min
_HI0 = 1.50           # lr at end of warmup, in units of D (tilted-top start)
_HI1 = 1.10           # lr at the hold->decay corner, in units of D
_F_RAMP = 0.50        # geometric alpha climb spans [50%, 100%] of the run
_A_LO = 0.5           # decoupled exploration penalty floor, in alpha0 units
_P = 2.0              # back-loading of the tolerance contraction exponent
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the hold->decay corner
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and annealing
_B1_LO = 0.02         # near-zero momentum in the strict-feasibility endgame
_B1_CENTER = 0.85
_B1_WIDTH = 0.04


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> single linear tail to gamma_min ---
    # th freezes at 1 past _F_HOLD, so the decay starts from the cool end of
    # the tilt (_HI1 * D); p sweeps the linear tail, landing exactly on gmin.
    th = jnp.clip(frac / _F_HOLD, 0.0, 1.0)
    lr_hold = (_HI0 + (_HI1 - _HI0) * th) * Dj
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    lr = (gmin + (lr_hold - gmin) * (1.0 - p)) * warm

    # --- alpha: decoupled floor -> geometric shrinking-tolerance climb ---
    # s^_P contracts the enforced violation band from D to gamma_min; alpha is
    # the floor times (D/gamma(t)) in log-space, ending at 5*alpha0*D/gmin.
    s = jnp.clip((frac - _F_RAMP) / (1.0 - _F_RAMP), 0.0, 1.0) ** _P
    log_ratio = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_LO), 1.0))
    alpha = alpha0 * _A_LO * jnp.exp(s * log_ratio)

    # --- betas: proven transitions, phase-locked to the corner and endgame ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2