import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top+notch best (+0.0577%): the entire
# duty-cycle machinery (repair notches, alpha bursts, bounded plateau) is
# REMOVED and replaced by the two prior-art bets the search has never run
# together: a ONE-CYCLE lr thermal arc (§2/§6) and a pure epsilon-shrinking
# GEOMETRIC alpha contraction (§7.9), with the momentum-as-implicit-ALM
# beta1 anti-correlation (§2/§4) that no schedule so far has used.
#
#   lr    — one-cycle: smooth sin^2 warmup from 0.3*D to a 1.9*D superheat
#           peak at 15% (hotter than any sustained level tried; the arc's
#           built-in warmup replaces the 3% damp), then one proven-shape
#           LINEAR cool-down landing exactly on gamma_min at the last step.
#           No notches, no restarts — cumulatively hotter than the best's
#           1.5->1.1*D hold, with the same near-linear tail the menu backs.
#   alpha — DECOUPLED epsilon-contraction: flat 0.4*alpha0 exploration floor
#           (proven) until 55%, then a single quadratically back-loaded
#           GEOMETRIC climb straight to the proven 5*alpha0*D/gamma_min
#           terminal spike. No plateau, no bursts: the enforced violation
#           band contracts monotonically, passing the proven ~6*alpha0
#           plateau level near 78% on its way up, so the endgame trajectory
#           matches the 5/5-seed-feasible spike while starting the squeeze
#           earlier — safer, not riskier, than the burst machinery.
#   beta1 — anti-correlated with the lr arc (one-cycle prescription): ~0.55
#           at the cool ends, 0.05 at the superheat peak, rising through the
#           cool-down so momentum acts as an implicit ALM multiplier while
#           alpha is still moderate; then the proven logistic gate slams it
#           to 0.02 under the terminal spike (momentum must not fight the
#           final feasibility restoration).
#   beta2 — RAdam-style monotone ramp 0.2 -> 0.9 (proven high) centered at
#           mid-run, so adaptive-lr variance is tamed before the alpha climb
#           injects large constraint curvature.
_LR_START = 0.3       # arc start, in units of D — built-in warmup
_LR_PEAK = 1.9        # superheat peak, in units of D — above all tried holds
_F_PEAK = 0.15        # peak location; linear cool-down occupies the last 85%
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_F_CLIMB = 0.55       # geometric alpha contraction starts here
_POW = 2.0            # quadratic back-loading of the contraction
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B1_MAX = 0.65        # momentum when lr is cold (implicit ALM phase)
_B1_MIN = 0.05        # momentum at the superheat peak
_B1_LO = 0.02         # near-zero momentum under the terminal spike (proven)
_B1_CENTER = 0.88     # proven terminal momentum gate
_B1_WIDTH = 0.03
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # earlier than the best: cool-down starts at 15% here
_B2_WIDTH = 0.07


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: one-cycle arc — sin^2 rise to the peak, linear landing on gamma_min ---
    warm = jnp.clip(frac / _F_PEAK, 0.0, 1.0)
    up = _LR_START + (_LR_PEAK - _LR_START) * jnp.sin(0.5 * jnp.pi * warm) ** 2
    cool = jnp.clip((frac - _F_PEAK) / (1.0 - _F_PEAK), 0.0, 1.0)
    lr = gmin + (up * Dj - gmin) * (1.0 - cool)

    # --- alpha: floor -> single geometric epsilon-contraction to the spike ---
    s = jnp.clip((frac - _F_CLIMB) / (1.0 - _F_CLIMB), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_LO), 1.0))
    alpha = alpha0 * _A_LO * jnp.exp(s * log_term)   # ends at 5*alpha0*D/gmin

    # --- beta1: anti-correlated with the lr arc, then the proven terminal gate ---
    units_norm = (up / _LR_PEAK) * (1.0 - cool)      # ~ lr in units of the peak
    b1_base = _B1_MAX - (_B1_MAX - _B1_MIN) * units_norm
    g = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_base + (_B1_LO - b1_base) * g

    # --- beta2: RAdam-style monotone ramp, settled before the alpha climb ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2