import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the two menu
# directions the lineage has never combined — a WSD/one-cycle lr (§2/§6:
# warmup -> sustained HOT HOLD near c*D -> straight linear cool-down to
# gamma_min) replacing every cosine/SGDR skeleton, an ADMM-STYLE CONSTANT
# moderate penalty (§ search-state "constant moderate penalty", untried)
# replacing floors/bursts/plateaus, and an epsilon-CONSTRAINED SHRINKING
# TOLERANCE (§7.9) replacing the two-stage logistic+terminal climb.
#
#   lr    — 4% linear warmup, then HOLD at 1.30*D from 4% to 55%. No cycles:
#           the cosine parents only touch their hottest lr for a few steps
#           per restart; holding 1.30*D for half the run integrates far more
#           exploration displacement than any 1.65*D peak ever delivered,
#           which is the "longer lr peak" the parent guidance asks for.
#           From 55% the proven straight linear tail lands EXACTLY on
#           gamma_min at the last step.
#   alpha — ADMM-style CONSTANT 1.6*alpha0 through warmup+hold: one fixed,
#           moderate, decoupled price on violation the whole time the layout
#           is hot — no bursts, no ramp, so basin hops face a uniform market
#           instead of a fluctuating one. Then the epsilon-constrained
#           endgame: over the ENTIRE cool-down the enforced violation band
#           contracts geometrically, realized as a single continuous
#           power-2.5 back-loaded geometric climb from 1.6*alpha0 to the
#           5/5-seed-proven terminal 5*alpha0*D/gamma_min. The climb passes
#           the old 6*alpha0 plateau level well before 80% and exceeds the
#           parent's alpha everywhere in the final 15%, so the terminal
#           feasibility restoration is STRONGER than the proven one, not
#           weaker — the structural risk is spent mid-run, not at the end.
#   betas — one-cycle momentum anti-correlated with lr (§2/§4, untried):
#           beta1 LOW (0.06) during the hot hold so momentum never flings
#           turbines across the boundary at 1.30*D steps, RISES to 0.35 as
#           lr falls (momentum as implicit ALM multiplier, letting the
#           still-moderate alpha enforce constraints while polishing), then
#           the proven terminal gate drops it to 0.02 so nothing coasts
#           through the final spike. beta2 keeps the proven 0.2 -> 0.9
#           transition, centered at the hold->cool-down corner.
_F_WARM = 0.04        # linear lr warmup over the first 4%
_F_HOLD = 0.55        # hot hold ends here; linear decay to gamma_min at 100%
_HOLD = 1.30          # sustained exploration lr, in units of D
_A_CONST = 1.6        # ADMM-style constant penalty during warmup+hold, in alpha0 units
_POW = 2.5            # back-loading of the geometric band contraction
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven 5/5-feasible scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition at the hold -> cool-down corner
_B2_WIDTH = 0.05
_B1_HOT = 0.06        # near-zero momentum while lr is hot
_B1_COOL = 0.35       # raised momentum during cool-down (implicit ALM multiplier)
_B1_CENTER = 0.60     # one-cycle beta1 rise, just after lr starts falling
_B1_WIDTH = 0.05
_B1_LO = 0.02         # terminal momentum during the feasibility spike
_B1_TCENTER = 0.90
_B1_TWIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> sustained hot hold -> straight linear tail ---
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr = (gmin + (_HOLD * Dj - gmin) * (1.0 - p)) * warm      # exact landing on gamma_min

    # --- alpha: ADMM constant, then one continuous shrinking-band climb ---
    # s = 0 through the hold (alpha frozen at _A_CONST*alpha0), then rises
    # with power-2.5 back-loading over the whole cool-down; the geometric
    # climb lands exactly on the proven terminal 5*alpha0*D/gamma_min.
    s = p ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_CONST), 1.0))
    alpha = alpha0 * _A_CONST * jnp.exp(s * log_term)

    # --- betas: one-cycle beta1 anti-correlated with lr; proven beta2 ramp ---
    b1_up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    b1_mid = _B1_HOT + (_B1_COOL - _B1_HOT) * b1_up
    b1_dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_TCENTER) / _B1_TWIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1_dn
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2