import jax.numpy as jnp

# STRUCTURALLY NEW vs the SGDR/anti-phased-burst best (+0.0533%): a classical
# SUMT / AUGMENTED-LAGRANGIAN OUTER LOOP (prior-art §7.9 dynamic penalty +
# LANCELOT-style delayed growth), untried anywhere in the lineage. Instead of
# cyclic alpha fighting cyclic lr, the run is 4 OUTER STAGES: alpha is a
# MONOTONE GEOMETRIC STAIRCASE (piecewise-constant per stage, x3 per stage —
# the textbook penalty-method outer iteration), and the lr WARM-RESTARTS AT
# EACH ALPHA JUMP, then cosine-anneals to a low trough so each penalty
# subproblem is approximately SOLVED before the penalty tightens. Every prior
# schedule either coupled alpha to 1/lr, ramped it smoothly, or pulsed it
# anti-phase; none ever solved a sequence of fixed-penalty subproblems.
#
#   lr    — 3% warmup, then per-stage one-cycle: restart peaks decay linearly
#           1.7*D -> 0.9*D across stages (first peak hotter/longer than any
#           attempt, as the parent guidance asks — stage 1 is a long, hot,
#           lightly-penalized subproblem), each cosine-annealed to a 0.25*D
#           trough so the stage actually converges. After 78%: one modest
#           polish restart at 0.6*D, straight linear tail landing EXACTLY on
#           gamma_min at the last step (proven landing).
#   alpha — SUMT staircase 0.5 -> 1.5 -> 4.5 -> 13.5 alpha0 (delayed, bounded,
#           decoupled from lr). Crucially every hot restart arrives TOGETHER
#           with a stricter penalty, so basin hops can never re-spend the
#           feasibility already bought — the opposite of anti-phased bursts.
#           From 78% the proven cubic back-loaded geometric climb lifts alpha
#           from the 13.5*alpha0 plateau to the 5/5-seed-feasible terminal
#           5*alpha0*D/gamma_min. Terminal restoration fully preserved.
#   betas — one-cycle bet (§2, untried): beta1 ANTI-CORRELATED with lr inside
#           each stage (0.05 at hot restarts so momentum can't slingshot
#           turbines out of the polygon, 0.1 at converged troughs), gated to
#           the proven 0.02 in the terminal spike. beta2 does the proven
#           0.2 -> 0.9 logistic as the staircase enters its strict stages,
#           absorbing the alpha-driven curvature (menu bet 4).
_N_STAGE = 4.0        # outer penalty iterations over the exploration phase
_F_EXPL = 0.78        # staircase ends / terminal polish begins (proven start)
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_A0 = 0.5             # stage-0 penalty, in alpha0 units (light: buy AEP first)
_G = 3.0              # geometric penalty growth per stage: 0.5,1.5,4.5,13.5
_HI0 = 1.7            # first restart peak, units of D — hottest ever tried
_HI1 = 0.9            # last restart peak, units of D
_LO = 0.25            # per-stage trough — each subproblem is nearly solved
_POL_HI = 0.6         # modest polish restart, units of D
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 rises as the staircase turns strict
_B2_WIDTH = 0.05
_B1_HOT = 0.05        # low momentum at hot restarts (one-cycle anti-correlation)
_B1_COOL = 0.1        # native momentum at converged stage troughs
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.90
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- outer-loop coordinates: stage index + within-stage progress ---
    u = jnp.clip(frac / _F_EXPL, 0.0, 1.0)
    stage = jnp.minimum(jnp.floor(u * _N_STAGE), _N_STAGE - 1.0)   # 0..3
    w = jnp.clip(u * _N_STAGE - stage, 0.0, 1.0)                   # [0,1] in stage
    cyc_w = 0.5 * (1.0 + jnp.cos(jnp.pi * w))                      # 1 at restart, 0 at trough

    # --- lr: warmup -> per-stage restart+cosine-anneal -> linear polish tail ---
    hi = _HI0 + (_HI1 - _HI0) * stage / (_N_STAGE - 1.0)           # decaying restart peaks
    lr_stage = (_LO + (hi - _LO) * cyc_w) * Dj
    warm = jnp.minimum(frac / _F_WARM, 1.0)                        # damps the hot start
    p = jnp.clip((frac - _F_EXPL) / (1.0 - _F_EXPL), 0.0, 1.0)
    lr_pol = gmin + (_POL_HI * Dj - gmin) * (1.0 - p)              # exact gamma_min landing
    lr = jnp.where(frac <= _F_EXPL, lr_stage * warm, lr_pol)

    # --- alpha: geometric SUMT staircase -> proven terminal geometric climb ---
    units = _A0 * jnp.power(_G, stage)                             # 0.5,1.5,4.5,13.5 alpha0
    units_end = _A0 * _G ** (_N_STAGE - 1.0)                       # 13.5 (Python float)
    s = p ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * units_end), 1.0))
    alpha = alpha0 * units * jnp.exp(s * log_term)                 # ends at 5*alpha0*D/gmin

    # --- betas: one-cycle beta1 inside stages; proven beta2 ramp + terminal gate ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HOT + (_B1_COOL - _B1_HOT) * (1.0 - cyc_w)        # low when lr is hot
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2