import jax.numpy as jnp

# STRUCTURALLY NEW vs both the flat-top-notch best (+0.0577%) and the SGDR
# parent: ALL cyclic machinery is removed. This is the prior-art menu's
# ONE-CYCLE superconvergence bet (§2) fused with the §7.9 DYNAMIC-PENALTY /
# ε-CONSTRAINED contracting-tolerance bet — the two frontier rows the recent
# attempts have never combined — while keeping the terminal feasibility
# machinery that delivered 5/5-seed feasibility WITHOUT any mid-run bursts
# (the gen-117 parent proves bursts are not required for feasibility).
#
#   lr    — ONE CYCLE, NO RESTARTS, NO NOTCHES: a long linear ramp over the
#           first 18% up to a 1.9*D peak (hotter than any sustained level
#           tried; safe because the ramp is gradual, momentum is LOW at the
#           peak, and alpha is small there), then a single heat-skewed
#           half-cosine descent (t^1.2 inside the cosine holds heat longer
#           mid-run than a plain cosine) landing EXACTLY on gamma_min at the
#           last step. Cumulative heat exceeds the notched hold — with zero
#           steps wasted in transitions between hot and cold regimes.
#   alpha — CLASSIC DYNAMIC PENALTY alpha0*(1+C*frac)^4 (§7.9), the one
#           penalty law never tried: no floor, no bursts, no logistic jump.
#           It starts at 0.3*alpha0 (freest exploration yet during the hot
#           ramp) and grows CONTINUOUSLY to 8*alpha0, so the enforced
#           violation band ~lr/alpha contracts monotonically toward
#           gamma_min instead of being repaid in discrete windows. From 78%
#           the PROVEN cubic-delayed geometric climb carries alpha to the
#           proven terminal 5*alpha0*D/gamma_min spike — feasibility endgame
#           unchanged.
#   beta1 — ANTI-CORRELATED WITH lr (one-cycle momentum, §2/§4 — untried):
#           0.05 at the 1.9*D peak (huge steps must not compound through
#           momentum), rising to 0.45 as lr cools, where momentum averages
#           the now-dominant constraint gradient and acts as an implicit ALM
#           multiplier (§ momentum-as-multiplier hypothesis) — letting the
#           bounded 8*alpha0 penalty enforce constraints before the spike.
#           Gated to the proven 0.02 during the terminal alpha spike.
#   beta2 — the proven 0.2 -> 0.9 logistic transition, centered mid-descent
#           where the run pivots from exploration to feasibility.
_F_PEAK = 0.18        # one-cycle ramp ends here; single descent thereafter
_PEAK = 1.9           # peak lr in units of D — hottest sustained level yet
_SKEW = 1.2           # >1 skews the half-cosine to hold heat longer mid-run
_A0U = 0.3            # alpha at step 0, in alpha0 units (freest start tried)
_A_END = 8.0          # alpha just before the terminal spike, in alpha0 units
_P_DYN = 4.0          # dynamic-penalty exponent: alpha ~ (1 + C*frac)^4
_C_DYN = (_A_END / _A0U) ** (1.0 / _P_DYN) - 1.0
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # mid-descent exploration -> feasibility pivot
_B2_WIDTH = 0.06
_B1_MIN = 0.05        # momentum at peak lr (anti-correlated: hot => low)
_B1_MAX = 0.45        # momentum as lr -> 0: implicit ALM multiplier
_B1_END = 0.02        # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.85     # gate closes just before the spike gets steep
_B1_WIDTH = 0.04


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: one-cycle — linear ramp to 1.9*D, skewed half-cosine to gamma_min ---
    up = jnp.minimum(frac / _F_PEAK, 1.0)
    t = jnp.clip((frac - _F_PEAK) / (1.0 - _F_PEAK), 0.0, 1.0)
    down = 0.5 * (1.0 + jnp.cos(jnp.pi * t ** _SKEW))
    lr_shape = up * down                      # 0 -> 1 at the peak -> 0 at the end
    lr = gmin + (_PEAK * Dj - gmin) * lr_shape  # exact landing on gamma_min

    # --- alpha: continuous dynamic penalty -> proven terminal geometric spike ---
    alpha_units = _A0U * (1.0 + _C_DYN * frac) ** _P_DYN   # 0.3 -> 8 alpha0, smooth
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_END), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)   # ends at 5*alpha0*D/gmin

    # --- beta1: anti-correlated with lr, gated down for the terminal spike ---
    b1_anti = _B1_MAX - (_B1_MAX - _B1_MIN) * lr_shape
    b1g = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_anti * (1.0 - b1g) + _B1_END * b1g

    # --- beta2: proven low -> high logistic transition mid-descent ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2