import jax.numpy as jnp

# STRUCTURALLY NEW vs the whole SGDR-restart lineage (best +0.0533%, parent
# +0.0402%): NO cycles, NO bursts. This is the one prior-art menu row (§6/§2)
# plus the one search-state direction that NO attempt has combined yet:
#
#   lr    — ONE-CYCLE / WSD "tilted trapezoid": 3% linear warmup, then a
#           SUSTAINED hot hold — the exploration budget the restart family
#           only touches at its brief cosine peaks — sloping gently from
#           1.45*D down to 1.15*D across the first half of the run (hot while
#           basins are being chosen, cooler as structure forms), then the
#           proven straight linear tail landing EXACTLY on gamma_min at the
#           last step. Integrated hot-phase lr well exceeds any restart
#           parent without ever exceeding the 1.65*D peak that stayed
#           feasible.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY (explicitly listed as still
#           untried): a flat 1.2*alpha0 through the entire hold. Unlike the
#           0.3–0.4*alpha0 exploration floors, violation debt is opposed
#           continuously at moderate strength — with beta1 momentum acting as
#           the implicit ALM multiplier — so no mid-run bursts are needed and
#           the hot phase never banks a debt the endgame can't repay. Then
#           the PROVEN endgame, untouched: logistic ramp onto the bounded
#           6*alpha0 ALM plateau just after cool-down begins, and the
#           cubic-delayed geometric terminal climb from 78% ending at
#           5*alpha0*D/gamma_min — the restoration that made the lineage
#           5/5-seed feasible after even hotter exploration than this.
#   betas — proven transitions only: beta2 0.2 -> 0.9 logistic at the
#           cool-down start (absorbs the alpha-driven curvature change);
#           beta1 flat 0.1, gated to 0.02 during the terminal spike so the
#           diverging alpha never rides momentum.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD_END = 0.50    # hot hold ends here; linear decay to gamma_min at 100%
_HI_START = 1.45      # hold entry lr, in units of D (hot, but below 1.65 peak)
_HI_END = 1.15        # hold exit lr — tilted top cools as structure forms
_A_CONST = 1.2        # ADMM-style constant exploration penalty, alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau through the polish (proven)
_A_CENTER = 0.56      # logistic ramp centered just after cool-down start
_A_WIDTH = 0.04       # proven ramp width
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.50     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> linear tail onto gamma_min ---
    # f1 sweeps 0 -> 1 across the hold, so the hold slopes 1.45*D -> 1.15*D
    # and then freezes at 1.15*D, pinning the cool-down start there.
    f1 = jnp.clip((frac - _F_WARM) / (_F_HOLD_END - _F_WARM), 0.0, 1.0)
    hold = (_HI_START + (_HI_END - _HI_START) * f1) * Dj
    p = jnp.clip((frac - _F_HOLD_END) / (1.0 - _F_HOLD_END), 0.0, 1.0)
    lr_env = gmin + (hold - gmin) * (1.0 - p)                 # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate penalty -> plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_CONST + (_A_PLAT - _A_CONST) * ramp      # in alpha0 units
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2