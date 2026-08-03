import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst/SGDR best (+0.0533%): the two search-state
# directions its lineage never combined — a WSD/ONE-CYCLE lr (warmup -> long
# HOT STABLE plateau -> linear cool-down; prior-art §6/§2 "hold near c*D then
# near-linear decay beats cosine/product") and an ADMM-STYLE CONSTANT MODERATE
# PENALTY (§ "ADMM-style constant" direction) whose endgame is an
# eps-CONSTRAINED SHRINKING TOLERANCE (§7.9): alpha is defined through an
# enforced-violation band eps(t) that contracts geometrically to gamma_min/5
# only at the very end.
#
#   lr    — 4% linear warmup, then a SUSTAINED hot plateau tilting gently
#           1.55*D -> 1.05*D over the first 55% (average exploration heat
#           ~1.3*D, higher AND longer than any restart schedule tried — the
#           parent guidance's "higher/longer peak early" taken to its WSD
#           limit), then the proven straight linear tail landing exactly on
#           gamma_min at the last step. No oscillation: heat is spent
#           continuously instead of in decaying pulses.
#   alpha — DECOUPLED from lr everywhere. During the hot phase it is an
#           ADMM-like constant 1.6*alpha0 (comparable to the time-average of
#           the burst best's exploration penalty, so violation debt stays
#           bounded without ever yanking the layout mid-hop). From 55% the
#           enforced band eps = alpha0*D/alpha contracts GEOMETRICALLY from
#           D/1.6 down to gamma_min/5, back-loaded (p^2.5) so most of the
#           polish runs at moderate stiffness and the contraction lands on
#           the proven 5/5-seed-feasible terminal 5*alpha0*D/gamma_min at
#           the final step — the terminal feasibility spike emerges as the
#           end of the tolerance contraction rather than a bolted-on ramp.
#   betas — the proven feasible endgame transitions, untouched: beta2
#           logistic 0.2 -> 0.9 at the cool-down boundary (absorbs the
#           rising constraint curvature ~alpha), beta1 held at native 0.1
#           through exploration and polish, logistic drop to 0.02 centered
#           at 88% so momentum cannot carry turbines back across the
#           boundary while the band snaps shut.
_F_WARM = 0.04        # linear lr warmup fraction
_F_STABLE = 0.55      # hot plateau ends / linear tail + eps-contraction begin
_HI0 = 1.55           # plateau start, in units of D
_HI1 = 1.05           # plateau end (tail starts here), in units of D (proven)
_A_PLAT = 1.6         # ADMM constant penalty, in alpha0 units (eps0 = D/1.6)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_POW = 2.5            # back-loading of the geometric band contraction
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum during the terminal contraction
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot plateau -> linear tail to gamma_min ---
    fs = jnp.clip(frac / _F_STABLE, 0.0, 1.0)
    hi = (_HI0 + (_HI1 - _HI0) * fs) * Dj                     # gentle plateau tilt
    p = jnp.clip((frac - _F_STABLE) / (1.0 - _F_STABLE), 0.0, 1.0)
    lr_env = gmin + (hi - gmin) * (1.0 - p)                   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant ADMM plateau, then geometric eps-band contraction ---
    # alpha = alpha0 * D / eps  (alpha0*D = mean|grad J|), with
    # eps: D/_A_PLAT -> gmin/_TERM_GAIN along p^_POW  ==>
    # alpha: _A_PLAT*alpha0 (hot phase, p=0) -> _TERM_GAIN*alpha0*D/gmin (p=1).
    s = p ** _POW
    log_eps0 = jnp.log(Dj / _A_PLAT)
    log_eps1 = jnp.log(gmin / _TERM_GAIN)
    eps = jnp.exp(log_eps0 + (log_eps1 - log_eps0) * s)
    alpha = alpha0 * Dj / jnp.maximum(eps, 1e-30)

    # --- betas: proven feasible transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2