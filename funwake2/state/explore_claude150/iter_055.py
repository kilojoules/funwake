import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst/restart best (+0.0533%): abandon cyclic
# machinery entirely. This is a WSD HOLD-HOT schedule (prior-art §6 — the one
# lr family the lineage has never tried: no restarts, no troughs) paired with
# an ADMM-STYLE CONSTANT MODERATE PENALTY during exploration and an
# epsilon-CONSTRAINED GEOMETRIC CONTRACTION for the endgame (§7.9).
#
# Why this can beat the best: the SGDR parent spends roughly half of its
# exploration window at trough lr (0.65*D) under 3-8*alpha0 restoration
# bursts — cold, constrained steps that buy feasibility but no AEP. Here the
# ENTIRE exploration span (3%-60%) runs continuously hot (1.45*D tilting to
# 1.15*D, hotter on time-average than any parent) under one flat moderate
# penalty, so every exploration step is a full-strength basin hop.
#
#   lr    — 3% linear warmup (proven) -> long STABLE hold at 1.45*D tilting
#           linearly down to 1.15*D by 60% (hot early, as the guidance asks)
#           -> single straight linear decay landing exactly on gamma_min at
#           the last step (the proven tail shape, just longer: 40% of steps).
#   alpha — DECOUPLED and two-phase. Phase 1 (0-60%): ADMM-style CONSTANT
#           1.5*alpha0 — high enough to keep violations bounded, low enough
#           never to fight an AEP-improving move; no bursts, no ramps. Phase 2
#           (60-100%): the enforced violation band contracts geometrically to
#           gamma_min — realized as a quadratically back-loaded geometric
#           alpha climb from 1.5*alpha0 to the 5/5-seed-proven terminal
#           5*alpha0*D/gamma_min. Restoration spans 40% of the run (vs the
#           parent's 22% spike), so the debt from the hot hold is repaid
#           gently while lr is still large enough to actually move turbines.
#   betas — the proven feasibility-critical transitions, unchanged: beta2
#           0.2 -> 0.9 at decay start (absorbs the alpha-driven curvature);
#           beta1 gated 0.1 -> 0.02 late so momentum never carries turbines
#           back across the boundary during the final contraction.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_DECAY = 0.60       # hold ends here; decay + epsilon-contraction to 100%
_HI_START = 1.45      # hold entry lr, in units of D — hot from the start
_HI_END = 1.15        # hold exit lr; the linear tail starts from here
_A_HOLD = 1.5         # ADMM-style constant penalty during the hold, alpha0 units
_POW = 2.0            # quadratic back-loading of the epsilon contraction
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with decay start (proven)
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum through hold and early decay
_B1_LO = 0.02         # near-zero momentum during the final contraction (proven)
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> hot tilted hold -> single linear decay to gamma_min ---
    # h freezes at 1 past _F_DECAY, so the tail decays from _HI_END * D.
    h = jnp.clip(frac, 0.0, _F_DECAY) / _F_DECAY
    lr_hold = (_HI_START + (_HI_END - _HI_START) * h) * Dj
    p = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)        # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate hold -> geometric epsilon-contraction ---
    # u^_POW back-loads the climb: alpha stays near the hold level while lr is
    # still large (protecting AEP), then rises geometrically to the proven
    # terminal 5*alpha0*D/gamma_min as lr shrinks toward gamma_min.
    u = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0) ** _POW
    log_climb = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_HOLD), 1.0))
    alpha = alpha0 * _A_HOLD * jnp.exp(u * log_climb)

    # --- betas: proven transitions, no per-cycle machinery to interact with ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2