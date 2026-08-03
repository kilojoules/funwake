import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): NO cosine cycles
# and NO piecewise alpha stages at all. The cyclic-restart family is saturated
# (8 straight attempts at or below best), so this composes the three menu
# directions the lineage has never tried, each as ONE smooth law:
#
#   lr    — WSD TILTED HOLD (prior-art §6/§2: "hold near c*D, then (near-)
#           linear cool-down beats cosine/product decay"). 3% linear warmup,
#           then a HOT STABLE PHASE: lr tilts linearly 1.45*D -> 1.10*D over
#           3%..58% — the layout spends the ENTIRE exploration phase at
#           basin-hopping heat instead of dipping to 0.65*D troughs half the
#           time (the best's cosine cycles average only ~1.0*D). More total
#           hot-phase displacement is exactly what the hotter-peak trend in
#           the lineage (+0.0450 -> +0.0533 by raising the peak) says pays.
#           From 58% the proven straight linear tail lands lr exactly on
#           gamma_min at the final step.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY + EPSILON-CONSTRAINED
#           SHRINKING TOLERANCE (§7.9, both untried). A flat 1.6*alpha0
#           through the hot phase (no floor/burst/plateau machinery): a
#           constant moderate multiplier that keeps violations bounded while
#           the hot lr trades them for AEP. Then ONE cubic-delayed geometric
#           contraction of the enforced violation band, D -> gamma_min:
#           alpha = 1.6*alpha0*(5*D/(1.6*gamma_min))^u with u = clipped
#           cubic progress from 58%. This shadows the proven feasible
#           envelope (~4*alpha0 at 78%, exploding only in the last ~15%) and
#           lands on the SAME proven terminal 5*alpha0*D/gamma_min that gave
#           5/5-seed feasibility — but as one smooth law with no plateau
#           discontinuity, and starting its climb at the cool-down (earlier
#           than the best's 78%) to repay the hotter hold's extra debt.
#   betas — menu bet: ANTI-CORRELATE beta1 WITH lr (one-cycle, §2). Momentum
#           is held LOW (0.05) through the hot hold so 1.45*D steps never
#           compound through the boundary, rises to the native 0.1 for the
#           polish where lr is small, then gates to 0.02 during the terminal
#           alpha explosion (proven). beta2 keeps the proven 0.2 -> 0.9
#           transition at the cool-down start.
_F_WARM = 0.03      # linear lr warmup over the first 3% (proven)
_F_COOL = 0.58      # hold ends here; linear decay to gamma_min at 100%
_HI0 = 1.45         # lr at the start of the hold, in units of D
_HI1 = 1.10         # lr at the end of the hold; the linear tail starts here
_A_BASE = 1.6       # ADMM-style constant penalty during the hold, in alpha0 units
_F_EPS = 0.58       # tolerance contraction starts at the cool-down
_P_EPS = 3.0        # cubic back-loading of the contraction (proven shape)
_TERM_GAIN = 5.0    # terminal alpha = 5*alpha0*D/gamma_min (proven 5/5-feasible)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58   # beta2 transition aligned with the cool-down start (proven)
_B2_WIDTH = 0.05
_B1_HOT = 0.05      # low momentum while lr is hot (one-cycle anti-correlation)
_B1_HI = 0.1        # native momentum for the polish
_B1_LO = 0.02       # near-zero momentum during the terminal alpha explosion
_B1_UP_CENTER = 0.58
_B1_UP_WIDTH = 0.05
_B1_CENTER = 0.88   # terminal beta1 gate (proven)
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold (1.45*D -> 1.10*D) -> linear tail ---
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    lr_hold = (_HI0 + (_HI1 - _HI0) * fc) * Dj                # frozen at 1.10*D past _F_COOL
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate penalty -> geometric tolerance contraction ---
    # One smooth law: the enforced violation band shrinks D -> gamma_min with
    # cubic-delayed progress u, so alpha climbs 1.6*alpha0 -> 5*alpha0*D/gmin.
    u = jnp.clip((frac - _F_EPS) / (1.0 - _F_EPS), 0.0, 1.0) ** _P_EPS
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_BASE), 1.0))
    alpha = alpha0 * _A_BASE * jnp.exp(u * log_term)

    # --- betas: beta1 anti-correlated with lr + proven terminal gate; proven beta2 ---
    b1u = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_base = _B1_HOT + (_B1_HI - _B1_HOT) * b1u              # low while hot, native in polish
    b1g = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_base + (_B1_LO - b1_base) * b1g
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2