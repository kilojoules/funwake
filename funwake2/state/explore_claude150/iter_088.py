import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst/SGDR best (+0.0533%): drop ALL cycles and
# bursts. This is the prior-art menu's row-1 + row-5 combination that no
# attempt in the lineage has tried yet:
#
#   lr    — WSD / one-cycle TRAPEZOID (§2, §6): short 4% warmup, then a LONG
#           TILTED HOT PLATEAU (1.45*D -> 1.15*D over the first 55%), then the
#           proven straight linear tail landing exactly on gamma_min at the
#           last step. Hypothesis (§6): holding near c*D beats cosine — the
#           SGDR parent spends most of its exploration budget near the 0.65*D
#           troughs (time-average lr ~ 1.0*D); this schedule sustains ~1.3*D
#           for the whole exploration phase, i.e. strictly hotter-for-longer
#           than any restart scheme can be, exactly what the parent guidance
#           licenses.
#   alpha — ADMM-style CONSTANT moderate penalty (untried direction) during
#           the hot plateau: a flat 1.5*alpha0 keeps violation debt bounded
#           the whole time (no 0.4*alpha0 free-violation floor, so the hotter
#           plateau is safe), while never spiking mid-run — basin hops are
#           never interrupted by restoration bursts. Then the §7.9
#           eps-CONSTRAINED SHRINKING TOLERANCE realized in closed form: from
#           55% the enforced violation band contracts geometrically in
#           lockstep with the lr cool-down, i.e. a single cubic-back-loaded
#           log-space climb (proven _POW = 3) from 1.5*alpha0 that lands
#           EXACTLY on the 5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#           By 78% this sits at ~5*alpha0 (matching the parent's proven
#           plateau there) and by 90% it is ~5x the parent's alpha — a
#           STRONGER terminal feasibility restoration, not a weaker one.
#   betas — only the proven transitions, nothing exotic: beta2 0.2 -> 0.9 at
#           the cool-down start (absorbs the growing ~alpha constraint
#           curvature, menu bet 4), beta1 gated 0.1 -> 0.02 at 90% so momentum
#           never carries turbines back across the boundary inside the
#           terminal spike.
_F_WARM = 0.04        # linear lr warmup over the first 4% (hotter start -> slightly longer)
_F_STAB = 0.55        # hot plateau ends here; linear decay to gamma_min at 100%
_HI0 = 1.45           # plateau entry lr, in units of D (hot edge of the tried range)
_HI1 = 1.15           # plateau exit lr; the linear tail starts from here
_A_CONST = 1.5        # ADMM-style constant penalty during the plateau, in alpha0 units
_POW = 3.0            # cubic back-loading of the eps-contraction climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven 5/5-feasible scale)
_B2_LO = 0.2          # native beta2 while exploring
_B2_HI = 0.9          # standard-Adam beta2 for the conditioning-heavy endgame
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum through exploration and polish
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.90
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot plateau -> linear tail to gamma_min ---
    # fc freezes at 1 past _F_STAB, so the tail decays from exactly _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_STAB) / _F_STAB
    lr_hot = (_HI0 + (_HI1 - _HI0) * fc) * Dj                 # 1.45*D -> 1.15*D tilt
    p = jnp.clip((frac - _F_STAB) / (1.0 - _F_STAB), 0.0, 1.0)
    lr_env = gmin + (lr_hot - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate -> geometric eps-band contraction ---
    # s = 0 through the whole plateau (alpha flat at 1.5*alpha0), then the
    # enforced tolerance band shrinks geometrically to gamma_min, back-loaded
    # cubically so mid-decay keeps AEP-polish freedom and the final ~15%
    # carries a restoration stronger than the parent's.
    s = p ** _POW
    log_ratio = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_CONST), 1.0))
    alpha = alpha0 * _A_CONST * jnp.exp(s * log_ratio)        # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions only ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2