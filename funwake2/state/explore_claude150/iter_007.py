import jax.numpy as jnp

# STRUCTURAL REDESIGN vs the incumbent best (one-cycle hold + delayed logistic
# ramp + cosine cool-down). Three prior-art bets the search has NOT tried yet,
# composed, while keeping the PROVEN terminal feasibility restoration:
#
#   lr    — WSD trapezoid: short warmup -> HOTTER, LONGER hold at 1.12*D
#           (vs best's 1.0*D to 55%) -> LINEAR decay landing exactly on
#           gamma_min (prior-art §6: (near-)linear cool-down beats cosine).
#   alpha — ADMM-style CONSTANT moderate penalty (3*alpha0) for the whole
#           main run: no ramp, no plateau schedule. The layout is kept
#           near-feasible continuously instead of drifting and being
#           repaired, so the hotter lr peak is safe. A gated terminal
#           divergence (5*alpha0*D/lr_env -> 5*alpha0*D/gamma_min) restores
#           the native coupling over the final ~10% — the exact endgame that
#           has been 5/5-seed feasible in both the parent and the best.
#   beta1 — one-cycle style, anti-correlated with lr (§2/§4, untried):
#           0.1 while hot, rising to ~0.4 through the linear decay (momentum
#           as implicit ALM multiplier lets the MODERATE constant alpha
#           enforce constraints), then gated back to 0.1 for the terminal
#           spike so diverging alpha never rides high momentum.
#   beta2 — proven phase transition: 0.2 exploring -> 0.9 in the polish phase.
_C_PEAK = 1.12        # exploration lr = 1.12 * D — hotter than best's 1.0*D
_F_WARM = 0.03        # linear lr warmup over the first 3% of steps
_F_HOLD_END = 0.60    # hold at peak until 60% (longer than best's 55%)
_A_CONST = 3.0        # ADMM-style constant penalty, in units of alpha0
_GATE_CENTER = 0.90   # terminal restoration engages over the final ~10%
_GATE_WIDTH = 0.02
_TERM_GAIN = 5.0      # terminal alpha -> 5*alpha0*D/gamma_min (proven scale)
_B1_LO = 0.1
_B1_HI = 0.4          # momentum peak during the linear-decay polish phase
_B1_CENTER = 0.65     # beta1 rises shortly after the decay begins
_B1_WIDTH = 0.06
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the decay start
_B2_WIDTH = 0.05


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    D = jnp.asarray(D) * 1.0
    gmin = jnp.asarray(gamma_min) * 1.0
    lr0 = _C_PEAK * D

    frac = (step + 1.0) / total_steps        # traced; (0, 1], hits 1 at the end

    # --- lr: warmup -> hold at 1.12*D -> LINEAR decay to gamma_min (WSD) ---
    p = jnp.clip((frac - _F_HOLD_END) / (1.0 - _F_HOLD_END), 0.0, 1.0)
    lr_env = lr0 + (gmin - lr0) * p          # exact linear landing on gamma_min

    warm = jnp.minimum(frac / _F_WARM, 1.0)  # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate penalty + gated terminal divergence ---
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _GATE_CENTER) / _GATE_WIDTH))
    alpha_term = _TERM_GAIN * alpha0 * D / jnp.maximum(lr_env, 1e-30)
    alpha = _A_CONST * alpha0 + gate * alpha_term

    # --- beta1: anti-correlated with lr, released before the alpha spike ---
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * b1r * (1.0 - gate)

    # --- beta2: proven low->high phase transition into the polish phase ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2