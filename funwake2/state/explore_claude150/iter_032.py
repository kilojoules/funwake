import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the two menu
# entries absent from the entire lineage, fused into one architecture.
#   (a) WSD / one-cycle lr (prior-art §2/§6): NO cycles, no restarts — a short
#       warmup, then the longest sustained hot hold any attempt has run
#       (1.30*D for ~54% of the steps), then the proven straight linear tail
#       landing exactly on gamma_min. The parent's restart train averages
#       ~1.0*D over exploration with brief 1.65*D spikes; a flat 1.30*D hold
#       delivers strictly more integrated exploration energy with none of the
#       cold troughs, directly testing the survey's "hold near c*D then
#       linear cool-down beats cosine/product decay" hypothesis.
#   (b) ADMM-constant + epsilon-contracting-band alpha (§7.9): instead of
#       floor -> bursts -> logistic plateau -> cubic climb (four mechanisms),
#       a SINGLE law. A moderate constant penalty alpha = 1.0*alpha0 for the
#       whole hold (the untried "ADMM-style constant moderate penalty"), then
#       one smooth cubic-back-loaded GEOMETRIC contraction of the enforced
#       violation band across the entire tail, landing on the proven
#       5/5-seed-feasible terminal 5*alpha0*D/gamma_min at the last step.
#       Relative to the parent this trades mid-tail pressure (≈3*alpha0 at
#       78% vs the parent's 6*alpha0 plateau — more AEP freedom in the
#       polish) for late pressure (≈60*alpha0 at 90% vs ≈20 — a longer,
#       harder feasibility restoration), so strict feasibility is preserved
#       by construction where it is actually decided.
#   betas — the proven feasibility-critical transitions only: beta2 0.2->0.9
#       logistic at decay onset (native low-beta2 exploration, adaptive
#       damping while alpha's curvature grows), beta1 gated 0.1->0.02 in the
#       terminal restoration so momentum never re-ejects repaired turbines.
_F_WARM = 0.04        # linear lr warmup over the first 4%
_F_HOLD = 0.58        # hot hold ends here; decay + alpha contraction begin
_HI = 1.30            # hold lr, in units of D — within the proven [0.65,1.65]
_A_LO = 1.0           # ADMM-style constant moderate penalty, in alpha0 units
_POW = 3.0            # cubic back-loading of the band contraction (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with decay onset
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum through warmup, hold, and polish
_B1_LO = 0.02         # near-zero momentum during the terminal restoration
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat hot hold at 1.30*D -> linear tail to gamma_min ---
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (_HI * Dj - gmin) * (1.0 - p)   # flat for frac<=_F_HOLD,
    warm = jnp.minimum(frac / _F_WARM, 1.0)         # exact gamma_min landing at p=1
    lr = lr_env * warm

    # --- alpha: constant moderate penalty -> geometric band contraction ---
    # One law: alpha = _A_LO*alpha0 * exp(s * log(ratio)). s=0 through the
    # hold (constant ADMM penalty); s->1 cubically across the tail, so the
    # enforced violation band gamma(t) ~ D*alpha0/alpha contracts smoothly
    # and reaches gamma_min-scale strictness only at the very end.
    s = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0) ** _POW
    log_ratio = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_LO), 1.0))
    alpha = alpha0 * _A_LO * jnp.exp(s * log_ratio)  # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions only ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2