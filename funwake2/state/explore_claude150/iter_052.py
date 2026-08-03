import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the two menu rows
# never tried anywhere in the lineage, combined —
#   (1) WSD / trapezoid lr (prior-art §6): NO cycles, NO restarts, NO troughs.
#       Warmup, then a HOT TILTED HOLD, then the proven straight linear tail.
#       Every ancestor (product, cosine, cyclic, SGDR, bursts) oscillates; this
#       tests the §6 hypothesis directly: "hold near c*D, then near-linear
#       cool-down beats cosine/product decay". The hold starts at 1.55*D (just
#       under the hottest peak ever tried) and never dips below 1.05*D, so its
#       exploration heat-INTEGRAL far exceeds the burst parent's restarts,
#       whose troughs sat at 0.65*D — this is the "higher/longer peak" push in
#       a shape the search has not touched.
#   (2) epsilon-CONSTRAINED SHRINKING TOLERANCE alpha (prior-art §7.9): one
#       smooth GEOMETRIC climb, alpha = alpha0 * A_LO / eps(t) in band units,
#       where the enforced violation band eps contracts geometrically from
#       ~2*D down to gamma_min/5 at the final step. No bursts, no plateau, no
#       separate terminal spike — the terminal feasibility restoration IS the
#       tail of the contraction, and it lands EXACTLY on the proven,
#       5/5-seed-feasible terminal scale 5*alpha0*D/gamma_min. The climb stays
#       below the old plateau until ~75% (a greedier mid-polish, where AEP is
#       won) and then DOMINATES every ancestor's alpha from ~85% on (a
#       stricter endgame, where feasibility is won).
#   betas — the proven transitions, unchanged: beta2 0.2 -> 0.9 as the hold
#       ends (adaptive scaling absorbs the growing ~alpha constraint
#       curvature), beta1 0.1 -> 0.02 late so momentum cannot carry turbines
#       back across the boundary while the band snaps shut.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.58        # tilted hold ends here; linear tail to gamma_min at 100%
_HI0 = 1.55           # hold start, in units of D — sustained, not a fleeting peak
_HI1 = 1.05           # hold end; the proven linear tail starts from 1.05*D
_A_LO = 0.5           # exploration penalty floor, in alpha0 units
_F_RAMP = 0.50        # band contraction (geometric alpha climb) starts here
_P = 1.6              # mild back-loading: below old plateau to ~75%, above by 85%
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven 5/5 scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with the hold end
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum during the final band snap
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> hot tilted hold (1.55*D -> 1.05*D) -> linear tail ---
    # fh freezes at 1 past _F_HOLD, pinning the tail's start at _HI1 * D; the
    # tail then lands exactly on gamma_min at the last step (proven landing).
    fh = jnp.clip(frac, 0.0, _F_HOLD) / _F_HOLD
    hold = (_HI0 + (_HI1 - _HI0) * fh) * Dj
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (hold - gmin) * (1.0 - p)                 # exact gamma_min landing
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant floor -> single geometric band contraction ---
    # s ramps 0 -> 1 over [_F_RAMP, 1]; alpha climbs log-linearly (power-_P
    # back-loaded) from the 0.5*alpha0 exploration floor to the proven
    # terminal 5*alpha0*D/gamma_min. Equivalent view: the enforced violation
    # band eps(t) = _A_LO*D/alpha_units contracts geometrically from 2*D to
    # gamma_min/5, reaching gamma_min-strictness only at the very end (§7.9).
    s = jnp.clip((frac - _F_RAMP) / (1.0 - _F_RAMP), 0.0, 1.0) ** _P
    log_span = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_LO), 1.0))
    alpha = alpha0 * _A_LO * jnp.exp(s * log_span)            # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions, aligned with the hold end / band snap ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2