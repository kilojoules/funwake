import jax.numpy as jnp

# STRUCTURALLY NEW vs both the SGDR-restart parent (-0.0134%) and the one-cycle
# best (+0.0349%). The restarts are removed — dropping lr to the floor mid-run
# demonstrably wasted exploration budget — and replaced by three untried menu
# directions composed into one design:
#   lr    — WSD TRAPEZOID with LINEAR cool-down (§6, untried): 3% warmup ->
#           long hot hold at 1.12*D (hotter AND longer than the best's 1.0*D
#           to 55%) -> straight LINEAR decay landing exactly on gamma_min.
#           Linear keeps materially more lr than cosine through the last ~20%,
#           giving the endgame real mobility for both AEP polish and repairs.
#   alpha — epsilon-CONSTRAINED SHRINKING TOLERANCE (§7.9, untried): one smooth
#           mechanism replaces the best's ramp+plateau+gated spike. The
#           enforced violation band contracts geometrically to gamma_min, i.e.
#           alpha grows LOG-LINEARLY from 0.5*alpha0 to the parent-proven
#           terminal magnitude 5*alpha0*D/gamma_min. A cubic delay back-loads
#           the growth: flat 0.5*alpha0 until 45% (longer free exploration than
#           the best's 35% ramp), a few*alpha0 near 80%, then an accelerating
#           blow-up that is STRONGER than the best's gate through 92-99% and
#           identical at the final step -> strict feasibility preserved.
#   betas — beta2 0.2 -> 0.9 at the cool-down start (proven phase transition);
#           NEW: beta1 0.1 -> 0.02 inside the terminal feasibility phase (menu
#           bet 4: beta1 DOWN with the alpha phase) so momentum cannot carry
#           turbines back across the boundary while alpha diverges.
_C_PEAK = 1.12         # hold lr = 1.12 * D (slightly hotter than the best)
_F_WARM = 0.03         # linear warmup over first 3% so the grid init survives
_F_HOLD_END = 0.58     # hold until 58%, then linear decay to gamma_min at 100%
_ALPHA_LO = 0.5        # exploration penalty (proven), in alpha0 units
_TERM_GAIN = 5.0       # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_F_CONTRACT = 0.45     # tolerance contraction starts mid-run (delayed ramp)
_POW = 3.0             # cubic back-loading of the contraction
_BETA2_LO = 0.2
_BETA2_HI = 0.9
_B2_CENTER = 0.58      # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1           # native momentum while exploring
_B1_LO = 0.02          # near-zero momentum during the terminal alpha blow-up
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> hold at 1.12*D -> LINEAR cool-down to gamma_min ---
    lr_peak = _C_PEAK * Dj
    p = jnp.clip((frac - _F_HOLD_END) / (1.0 - _F_HOLD_END), 0.0, 1.0)
    lr_env = gmin + (lr_peak - gmin) * (1.0 - p)
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    lr = lr_env * warm

    # --- alpha: geometric growth as the tolerance band shrinks to gamma_min ---
    # alpha(s) = 0.5*alpha0 * ratio**s, ratio = (5/0.5)*D/gamma_min, so s=0
    # gives the proven exploration penalty and s=1 the proven terminal
    # divergence; the cubic s keeps the run in the low-penalty regime longest.
    s = jnp.clip((frac - _F_CONTRACT) / (1.0 - _F_CONTRACT), 0.0, 1.0) ** _POW
    log_ratio = jnp.log((_TERM_GAIN / _ALPHA_LO) * Dj / gmin)
    alpha = _ALPHA_LO * alpha0 * jnp.exp(s * log_ratio)

    # --- betas: beta2 up for the polish phase, beta1 down for the endgame ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _BETA2_LO + (_BETA2_HI - _BETA2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2