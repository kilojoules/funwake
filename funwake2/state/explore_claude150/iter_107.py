import jax.numpy as jnp

# STRUCTURALLY NEW vs both the cyclic-burst best (+0.0533%) and the trapezoid
# parent (+0.0423%): MOMENTUM-DRIVEN EXPLORATION. Every schedule in the 107-
# generation lineage explores with near-zero momentum (beta1 <= 0.1, beta2
# <= 0.2 early) and buys exploration ONLY with raw lr heat — peaks of
# 1.55-1.65*D. The one axis never touched is the prior-art menu's beta1 row
# (Sutskever-increasing vs DEMON-decaying ramp, §4) and the gen-0 idea list's
# "standard Adam (0.9, ~0.999) vs TopFarm (0.1, 0.2)" ablation. Here the
# exploration phase runs as STANDARD ADAM: beta1 = 0.9 gives turbines
# ballistic, directionally-COHERENT motion (in Adam, high beta1 does not
# inflate step size — sqrt(v) renormalizes it — it adds persistence), so they
# sweep long distances through the wake field and cross shallow wake traps
# that big-but-memoryless native steps rattle around in. A Demon-style
# logistic then hands the moments back to the proven native polish pair at
# the cool-down boundary, and the ENTIRE proven feasibility machinery is
# preserved unchanged.
#
#   lr    — the parent's proven trapezoid, nudged hotter as the guidance
#           asks: 3% linear warmup, tilted sustained plateau 1.60*D -> 1.05*D
#           over the first 62%, then the proven straight linear tail landing
#           exactly on gamma_min at the last step.
#   alpha — the parent's proven 5/5-seed-feasible path, UNCHANGED: delayed
#           log-linear (geometric) climb from a 0.5*alpha0 exploration floor
#           to the bounded 6*alpha0 ALM level at 78%, then the proven cubic
#           back-loaded geometric terminal spike landing on
#           5*alpha0*D/gamma_min at the final step.
#   betas — the new bet. Exploration: standard Adam (0.9, 0.98) — beta2 must
#           sit ABOVE beta1 for stability, so the native 0.2 (invalid under
#           high momentum) is replaced by a long-memory 0.98 that also
#           uniformizes per-turbine step scales during the ballistic phase.
#           A Demon logistic at the cool-down boundary drops beta1 0.9 -> 0.1
#           (momentum dissipates in a few steps, native polishing resumes)
#           and beta2 settles on the proven late value 0.9 that absorbs the
#           growing ~alpha constraint curvature. The proven terminal gate
#           then pins beta1 at 0.02 inside the alpha spike so no residual
#           momentum can carry turbines back across the boundary.
_F_WARM = 0.03        # linear lr warmup fraction (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_HI0 = 1.60           # plateau start, in units of D (slightly hotter than parent)
_HI1 = 1.05           # plateau end at _F_COOL; the linear tail starts here (proven)
_A_LO = 0.5           # exploration alpha floor, in alpha0 units (proven)
_A_MID = 6.0          # proven bounded ALM level, reached exactly at _F_TERM
_PRAMP = 1.6          # >1 delays the geometric alpha climb toward mid/late run
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B1_EXPLORE = 0.9     # standard-Adam ballistic momentum while exploring — the bet
_B1_POLISH = 0.1      # proven native momentum for the polish phase
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike (proven)
_DEMON_CENTER = 0.62  # Demon beta1 hand-off aligned with the cool-down start
_DEMON_WIDTH = 0.05
_B1_CENTER = 0.88     # proven terminal beta1 gate
_B1_WIDTH = 0.03
_B2_EXPLORE = 0.98    # long-memory second moment; must exceed beta1 = 0.9
_B2_POLISH = 0.9      # proven late beta2 under the growing constraint curvature
_B2_CENTER = 0.62
_B2_WIDTH = 0.05


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot plateau -> linear tail to gamma_min (proven) ---
    u = jnp.clip(frac / _F_COOL, 0.0, 1.0)               # position along plateau
    hi = (_HI0 + (_HI1 - _HI0) * u) * Dj                 # 1.60*D -> 1.05*D, then frozen
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (hi - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)              # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: delayed geometric climb -> proven cubic terminal spike (unchanged) ---
    r = jnp.clip(frac / _F_TERM, 0.0, 1.0) ** _PRAMP     # delayed ramp coordinate
    alpha_base = _A_LO * alpha0 * (_A_MID / _A_LO) ** r  # 0.5*alpha0 -> 6*alpha0 at 78%
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_MID), 1.0))
    alpha = alpha_base * jnp.exp(s * log_term)           # ends at 5*alpha0*D/gmin

    # --- betas: standard-Adam exploration -> Demon hand-off -> proven endgame ---
    demon = 1.0 / (1.0 + jnp.exp(-(frac - _DEMON_CENTER) / _DEMON_WIDTH))
    b1_mid = _B1_EXPLORE + (_B1_POLISH - _B1_EXPLORE) * demon  # 0.9 -> 0.1 at cool-down
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * gate                  # -> 0.02 in the spike

    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_EXPLORE + (_B2_POLISH - _B2_EXPLORE) * b2r     # 0.98 -> proven 0.9

    return lr, alpha, beta1, beta2