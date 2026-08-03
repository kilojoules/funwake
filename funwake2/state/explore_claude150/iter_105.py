import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the last untried
# search-state direction — an ADMM-STYLE CONSTANT MODERATE PENALTY — fused with
# the one untried lr shape from the prior-art menu (§6): WSD / trapezoid
# (warmup -> STABLE HOLD -> linear decay). The whole lineage is oscillatory
# (cosine restarts, anti-phased bursts); this schedule is deliberately FLAT.
#
#   lr    — 3% linear warmup (proven), then a flat hold at 1.30*D for the
#           entire exploration phase (to 60%). This is "higher and longer"
#           than any restart lineage: the parent's cycles average only ~1.0*D
#           and waste steps in near-cold troughs, while its hottest 1.65*D
#           peak lasts moments. A constant 1.30*D delivers strictly more
#           cumulative basin-hopping heat with zero dead time. From 60% the
#           proven straight linear tail lands exactly on gamma_min at the
#           last step — the §6 hypothesis that hold-then-linear beats
#           cosine/product decay, tested cleanly for the first time.
#   alpha — open-loop ADMM: a constant moderate penalty rho = 1*alpha0 plus a
#           LINEAR MULTIPLIER PROXY growing to +1*alpha0 across the hold (a
#           crude dual update under roughly constant violation — prior-art
#           §7.9 dynamic penalty). Fully decoupled from lr, never zero, never
#           spiking mid-run: violations stay uniformly moderate instead of
#           swinging between the parent's 0.4*alpha0 free-fall and 8*alpha0
#           bursts, so AEP structure is never torn up by repair cycles. The
#           PROVEN endgame is preserved untouched: logistic ramp (centered
#           0.64) to the bounded 6*alpha0 ALM plateau, then the cubic-delayed
#           geometric climb from 78% landing on the 5/5-seed-feasible
#           terminal 5*alpha0*D/gamma_min feasibility spike.
#   betas — exactly the proven transitions, with the burst machinery removed:
#           beta2 0.2 -> 0.9 logistic at the hold->decay handoff (adaptive
#           scaling tames the alpha-driven curvature change), beta1 gated
#           0.1 -> 0.02 at 88% so momentum cannot carry turbines back across
#           the boundary during the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.60        # stable-phase end; linear decay to gamma_min at 100%
_LR_HOLD = 1.30       # flat hold height, in units of D — hotter-on-average
                      # and longer than any restart schedule in the lineage
_A_RHO = 1.0          # ADMM constant penalty during the hold, in alpha0 units
_A_DUAL = 1.0         # linear multiplier-proxy growth across the hold
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.64      # logistic alpha ramp centered just after the hold ends
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the hold->decay handoff
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum through hold and anneal
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat 1.30*D hold -> linear tail to gamma_min ---
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr = (gmin + (_LR_HOLD * Dj - gmin) * (1.0 - p)) * warm   # exact landing on gmin

    # --- alpha: ADMM hold (rho + linear dual proxy) -> plateau -> terminal climb ---
    fc = jnp.clip(frac, 0.0, _F_HOLD) / _F_HOLD               # freezes at 1 past hold
    a_hold = _A_RHO + _A_DUAL * fc                            # 1*alpha0 -> 2*alpha0
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = a_hold + (_A_PLAT - a_hold) * ramp          # smooth lift to plateau
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions, no burst machinery ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2