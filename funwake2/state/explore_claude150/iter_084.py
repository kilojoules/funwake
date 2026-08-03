import jax.numpy as jnp

# STRUCTURALLY NEW vs the cyclic-burst best (+0.0533%): the prior-art menu's
# strongest UNTRIED lr bet — a WSD / one-cycle TRAPEZOID (warmup -> sustained
# tilted hot plateau -> linear cool-down, §2/§6) — paired with a fully
# DECOUPLED, MONOTONE GEOMETRIC penalty ramp (dynamic-penalty a0*(1+Ct)^p
# family, §7.9) in place of cycles, bursts, and logistic plateaus.
#
#   lr    — 3% linear warmup, then a SUSTAINED tilted plateau 1.55*D -> 1.05*D
#           over the first 62% (time-average ~1.3*D: hotter AND longer than
#           the best's cosine restarts, whose exploration phase averaged only
#           ~1.0*D — exactly the "higher/longer lr peak early" the guidance
#           asks for), then the proven straight linear tail landing exactly
#           on gamma_min at the last step. No cycles: exploration heat is
#           spent continuously instead of being repeatedly cooled and rebuilt.
#   alpha — DECOUPLED from lr and MONOTONE: a single delayed log-linear
#           (geometric) climb from a 0.5*alpha0 exploration floor to the
#           proven 6*alpha0 ALM level at 78% — continuous, gently growing
#           constraint pressure that keeps violation debt bounded under the
#           sustained hot lr (vs the best's episodic burst repayments) — then
#           the PROVEN terminal feasibility spike, unchanged: cubic
#           back-loaded geometric climb from 78% landing on the 5/5-seed-
#           feasible 5*alpha0*D/gamma_min at the final step.
#   betas — proven transitions only: beta2 0.2 -> 0.9 logistic at the
#           cool-down boundary (absorbs the growing ~alpha constraint
#           curvature), beta1 0.1 -> 0.02 logistic gate inside the terminal
#           spike so momentum cannot carry turbines back across the boundary
#           while the spike is pulling them in.
_F_WARM = 0.03    # linear lr warmup fraction (proven)
_F_COOL = 0.62    # exploration ends here; linear decay to gamma_min at 100%
_HI0 = 1.55       # plateau start, in units of D (sustained, not a fleeting peak)
_HI1 = 1.05       # plateau end at _F_COOL; the linear tail starts here (proven)
_A_LO = 0.5       # exploration alpha floor, in alpha0 units
_A_MID = 6.0      # proven bounded ALM level, reached exactly at _F_TERM
_PRAMP = 1.6      # >1 delays the geometric alpha climb toward mid/late run
_F_TERM = 0.78    # terminal geometric alpha climb starts here (proven)
_POW = 3.0        # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0  # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62  # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1      # native momentum while exploring and polishing
_B1_LO = 0.02     # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot plateau -> linear tail to gamma_min ---
    u = jnp.clip(frac / _F_COOL, 0.0, 1.0)               # position along plateau
    hi = (_HI0 + (_HI1 - _HI0) * u) * Dj                 # 1.55*D -> 1.05*D, then frozen
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (hi - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)              # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: delayed geometric climb -> proven cubic terminal spike ---
    r = jnp.clip(frac / _F_TERM, 0.0, 1.0) ** _PRAMP     # delayed ramp coordinate
    alpha_base = _A_LO * alpha0 * (_A_MID / _A_LO) ** r  # 0.5*alpha0 -> 6*alpha0 at 78%
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_MID), 1.0))
    alpha = alpha_base * jnp.exp(s * log_term)           # ends at 5*alpha0*D/gmin

    # --- betas: proven logistic transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2