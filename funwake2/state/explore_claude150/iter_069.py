import jax.numpy as jnp

# STRUCTURALLY NEW vs the SGDR+anti-phased-burst best (+0.0533%): every prior
# attempt explores by modulating the STEP SIZE (hot lr peaks, restarts, holds)
# while momentum stays pinned near the native 0.1. This schedule flips the
# exploration mechanism to the two untried menu directions at once:
#   betas — MOMENTUM-CYCLED EXPLORATION (one-cycle / Sutskever ramp, §2/§4):
#           three cosine momentum waves swing beta1 between 0.12 and 0.85
#           across the exploration phase. Under Adam the per-step size stays
#           ~lr regardless of beta1, so high-beta1 crests do not blow up the
#           step — they make motion DIRECTIONALLY COHERENT: turbines drift
#           ballistically across the farm for hundreds of steps instead of
#           jittering, a basin-hopping mode raw-lr peaks cannot produce.
#           lr is ANTI-CORRELATED with the wave (one-cycle style): diffusive
#           trough mode (lr 0.95*D, beta1 0.12) alternates with ballistic
#           crest mode (lr 0.42*D, beta1 0.85). The waves complete exactly at
#           the cool-down start, so polishing begins at a momentum trough.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY (the one menu direction no
#           lineage member has tried): a flat 1.3*alpha0 through the whole
#           exploration phase, fully DECOUPLED from lr and from the waves.
#           The mechanism is menu bet 4 verbatim: with beta1 high, the
#           one-sided boundary/spacing gradients ACCUMULATE in the momentum
#           buffer like an ALM multiplier estimate — the restoring push grows
#           with sustained violation — so a moderate constant alpha enforces
#           what previously needed 8-25x bursts. Note this floor is 3x
#           STRICTER than the best's proven-feasible 0.4*alpha0 floor.
#   endgame — the proven feasibility machinery is kept IDENTICALLY: logistic
#           ramp to the bounded 6*alpha0 ALM plateau after cool-down starts,
#           cubic-delayed geometric climb from 78% landing on the 5/5-seed-
#           feasible 5*alpha0*D/gamma_min, straight linear lr tail hitting
#           gamma_min exactly at the last step, beta2 0.2 -> 0.9 at the
#           cool-down, beta1 gated to 0.02 inside the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear lr decay to gamma_min at 100%
_N_WAVE = 3.0         # three momentum waves inside the exploration phase
_LR_BASE = 0.95       # diffusive-mode lr (momentum troughs), in units of D
_LR_CREST = 0.42      # ballistic-mode lr (momentum crests), in units of D
_B1_TROUGH = 0.12     # near-native momentum in the diffusive mode
_B1_CREST = 0.85      # heavy-ball momentum at each wave crest
_A_CONST = 1.3        # ADMM constant penalty during exploration, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- momentum wave: 0 at wave boundaries, 1 at crests; fc freezes at 1
    # past _F_COOL, where 3 full waves close (wave = 0), so the cool-down and
    # the terminal phase always run from a momentum trough.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    wave = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * _N_WAVE * fc))

    # --- lr: anti-correlated with the wave -> proven linear tail to gamma_min ---
    lr_expl = (_LR_BASE + (_LR_CREST - _LR_BASE) * wave) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_expl - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> logistic plateau -> terminal geometric climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_CONST + (_A_PLAT - _A_CONST) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: beta1 rides the wave, gated to 0.02 in the terminal spike;
    # beta2 steps up for the polish phase (proven transition) ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_expl = _B1_TROUGH + (_B1_CREST - _B1_TROUGH) * wave
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_expl + (_B1_LO - b1_expl) * b1r

    return lr, alpha, beta1, beta2