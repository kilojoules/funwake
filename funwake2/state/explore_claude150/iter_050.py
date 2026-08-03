import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): every schedule in
# the lineage explores with CYCLES (cosine anneals, SGDR restarts, bursty
# alpha). This one has NO cycles at all — it is the prior-art menu's strongest
# UNTRIED lr row (§6 WSD / §2 one-cycle: "hold near c*D, then cool to
# gamma_min beats cosine/product decay") fused with the one listed direction
# still untried anywhere: an ADMM-STYLE CONSTANT MODERATE PENALTY.
#
#   lr    — WSD HOT-HOLD: 3% linear warmup (proven), then a SUSTAINED hot
#           plateau tilted gently 1.5*D -> 1.25*D until 60% — far more total
#           exploration energy than the burst best (whose cosine troughs at
#           0.65*D waste ~half the exploration window near stall), honoring
#           the parent guidance "higher/LONGER lr peak early" by holding heat
#           instead of spiking it. Then a single half-cosine cool-down landing
#           exactly on gamma_min at the last step: it lingers near-hot just
#           after the hold (extra exploration) and near-gamma_min at the end
#           (extra polish + feasibility margin vs the proven linear tail).
#   alpha — ADMM-style CONSTANT 1.5*alpha0 through the whole hold. The burst
#           best's exploration alpha AVERAGES ~2.1*alpha0 (0.4 floor + strong
#           bursts) yet lets violation debt build between repayments; the
#           constant 1.5 is LOWER on average (more AEP freedom overall) while
#           continuously nudging turbines legal, so the long hot hold never
#           accumulates a debt spike the cool-down must burn steps repaying.
#           Endgame is the proven 5/5-seed-feasible machinery, verbatim:
#           logistic ramp (center 0.66) to the bounded 6*alpha0 ALM plateau,
#           then the cubic-delayed geometric climb from 78% landing on the
#           terminal 5*alpha0*D/gamma_min restoration spike.
#   betas — proven transitions only, so attribution stays clean: beta2
#           0.2 -> 0.9 at the hold end (absorbs the alpha-ramp curvature),
#           beta1 gated 0.1 -> 0.02 inside the terminal spike so momentum
#           cannot carry turbines back across the boundary.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.60        # hot hold ends here; single cool-down to gamma_min at 100%
_HI0 = 1.50           # hold lr at warmup end, in units of D
_HI1 = 1.25           # hold lr at hold end; cool-down starts from here
_A_HOLD = 1.5         # ADMM constant penalty during the hold, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after the hold ends
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the hold end
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

    # --- lr: warmup -> tilted hot hold -> single half-cosine cool-down ---
    # fh freezes at 1 past _F_HOLD, so the cool-down launches from _HI1 * D.
    fh = jnp.clip(frac, 0.0, _F_HOLD) / _F_HOLD
    hi = _HI0 + (_HI1 - _HI0) * fh                            # gentle hold tilt
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    cool = 0.5 * (1.0 + jnp.cos(jnp.pi * p))                  # 1 during hold, 0 at end
    lr_env = gmin + (hi * Dj - gmin) * cool                   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate hold -> bounded plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_HOLD + (_A_PLAT - _A_HOLD) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions, nothing else ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2