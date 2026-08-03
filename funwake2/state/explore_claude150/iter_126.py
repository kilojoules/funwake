import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-with-notches best (+0.0577%): the entire
# cyclic machinery (repair notches, synchronized alpha bursts, logistic
# plateau) is REMOVED and replaced by the two prior-art directions the search
# has never combined:
#
#   1. ONE-CYCLE lr (menu §2, untried): a single cosine warmup from 0.08*D to
#      a 2.0*D apex at 30% — hotter than any sustained level tried (best held
#      1.5*D; hottest momentary peak tried was 1.65*D) — followed by ONE long
#      straight linear cool-down landing exactly on gamma_min at the last
#      step. No restarts, no notches: cumulative heat ~1.02 D·runs vs ~0.96
#      for the best, yet the lr trajectory over the critical final 15% almost
#      exactly matches the proven-feasible tail (0.31*D at 90%, 0.09*D at
#      97%), so the feasibility endgame conditions are preserved.
#
#   2. EPSILON-CONTRACTION alpha (menu §7.9, untried): instead of
#      floor -> bursts -> plateau -> spike, alpha follows ONE smooth law — an
#      ADMM-style constant moderate penalty (0.5*alpha0, fully decoupled from
#      lr) through the hot phase, then from 45% a single quartic-back-loaded
#      GEOMETRIC contraction of the enforced violation band, ending exactly at
#      the proven terminal anchor 5*alpha0*D/gamma_min. The quartic exponent
#      shapes the climb to shadow the proven trajectory (~8*alpha0 at 85%,
#      ~10^2*alpha0 at 92%) while being one continuous exponential, so the
#      tolerated-violation band shrinks monotonically to ~gamma_min only at
#      the end.
#
#   betas — beta1 runs the untried Sutskever/one-cycle INCREASING ramp,
#   anti-correlated with lr (0.05 at the hot apex, where momentum would carry
#   turbines through the boundary, rising to 0.28 as lr shrinks — momentum as
#   implicit ALM multiplier averaging constraint gradients), then the proven
#   gated cut to 0.02 for the terminal alpha climb. beta2 keeps the proven
#   0.2 -> 0.9 logistic, re-centered at 58% (where lr first drops below
#   ~1.2*D on the linear descent).
_F_PK = 0.30          # lr apex location; cosine warmup before, linear descent after
_LR_LO = 0.08         # warmup launch lr, in units of D
_PK = 2.0             # apex lr, in units of D — hotter than anything tried
_F_A = 0.45           # epsilon-contraction of alpha starts here
_P_A = 4.0            # quartic back-loading of the geometric contraction
_A_EXPL = 0.5         # constant decoupled exploration penalty, in alpha0 units
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven anchor)
_B1_PK = 0.05         # momentum at the hot apex (anti-correlated with lr)
_B1_LATE = 0.28       # momentum as lr -> small: implicit ALM multiplier
_B1_TERM = 0.02       # proven near-zero momentum during the terminal climb
_B1_CENTER = 0.90
_B1_WIDTH = 0.03
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # lr crosses ~1.2*D here on the linear descent
_B2_WIDTH = 0.05


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: one-cycle — cosine warmup to the apex, then straight to gmin ---
    # lr_up freezes at the apex once fu=1, so composing with the linear factor
    # gives cosine-up before _F_PK and an exact linear landing on gamma_min at
    # the last step, with no branches.
    fu = jnp.clip(frac / _F_PK, 0.0, 1.0)
    lr_up = (_LR_LO + (_PK - _LR_LO) * 0.5 * (1.0 - jnp.cos(jnp.pi * fu))) * Dj
    p = jnp.clip((frac - _F_PK) / (1.0 - _F_PK), 0.0, 1.0)   # descent progress
    lr = gmin + (lr_up - gmin) * (1.0 - p)

    # --- alpha: constant moderate penalty -> single geometric contraction ---
    # Enforced violation band ~ alpha0*D/alpha shrinks from ~2*D to gmin/5;
    # exact terminal value 5*alpha0*D/gmin, the proven feasibility anchor.
    u = jnp.clip((frac - _F_A) / (1.0 - _F_A), 0.0, 1.0) ** _P_A
    log_ratio = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_EXPL), 1.0))
    alpha = alpha0 * _A_EXPL * jnp.exp(u * log_ratio)

    # --- betas: rising momentum anti-correlated with lr, gated terminal cut ---
    b1_base = _B1_PK + (_B1_LATE - _B1_PK) * p               # low at apex, high late
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_base + (_B1_TERM - b1_base) * gate
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2