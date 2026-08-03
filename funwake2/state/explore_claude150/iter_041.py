import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the two prior-art
# menu rows no lineage member has touched, fused into one schedule.
#
#   lr    — WSD / trapezoid (prior-art §6, "hold near c*D then (near-)linear
#           cool-down beats cosine"): 3% linear warmup, then a HOT DECAYING
#           HOLD at 1.60*D -> 1.10*D across [3%, 60%] — no cosine, no
#           restarts, no troughs. Time-integrated exploration energy is ~35%
#           above the SGDR parent (whose cycles average ~1.0*D), delivering
#           the "higher/longer peak" push structurally instead of by nudging
#           cosine constants. From 60% the proven straight linear tail lands
#           exactly on gamma_min at the last step, leaving a 40% polish phase.
#   alpha — eps-CONSTRAINED SHRINKING TOLERANCE (prior-art §7.9, untried):
#           the entire floor -> bursts -> logistic plateau -> cubic climb
#           stack is replaced by ONE smooth law. Alpha sits at the proven
#           0.4*alpha0 exploration floor through the hot hold, then from 50%
#           the enforced violation band contracts GEOMETRICALLY down to
#           gamma_min — alpha rises log-linearly (power-2.2 back-loading)
#           from the floor to the 5/5-seed-proven terminal 5*alpha0*D/gamma_min.
#           En route it passes the proven ~6*alpha0 ALM plateau scale near
#           78% and keeps contracting, so late repair happens at ever-smaller
#           lr (violators pinned without shredding interior AEP structure,
#           since the one-sided penalty gradient vanishes for feasible
#           turbines). Delayed ramp (§7.3), bounded mid-run scale (§7.2) and
#           terminal restoration spike (§7.5) all emerge from the single
#           contraction — a clean ablation against the hand-stacked parent.
#   betas — proven transitions only: beta2 0.2 -> 0.9 logistic aligned with
#           the hold end (adaptive scaling absorbs the growing constraint
#           curvature through the tail); beta1 0.1 -> 0.02 gated at 86% so
#           momentum never carries turbines back across the boundary once the
#           contraction turns near-strict.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.60        # hot hold ends here; linear tail to gamma_min at 100%
_HI0 = 1.60           # hold start, in units of D — sustained, not a cosine blip
_HI1 = 1.10           # hold end; the linear tail launches from here
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_DELAY = 0.50       # tolerance contraction starts mid-run (delayed ramp, §7.3)
_A_POW = 2.2          # back-loads the contraction so alpha stays moderate
                      # through early cool-down and spikes only late
_A_TERM = 5.0         # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the hold end
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum in the strict endgame
_B1_CENTER = 0.86
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> decaying WSD hold -> exact linear landing on gamma_min ---
    # h freezes at 1 past _F_HOLD, so the tail launches from _HI1 * D.
    h = jnp.clip(frac, 0.0, _F_HOLD) / _F_HOLD
    lr_hold = (_HI0 + (_HI1 - _HI0) * h) * Dj                 # sustained hot plateau
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # lr(1) = gamma_min exactly
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor, then one geometric eps-contraction to the terminal spike ---
    # log(alpha) climbs linearly in s from the floor to 5*alpha0*D/gamma_min,
    # i.e. the enforced violation band shrinks geometrically to gamma_min,
    # reaching it only at the final step (s = 1).
    u = jnp.clip((frac - _A_DELAY) / (1.0 - _A_DELAY), 0.0, 1.0)
    s = u ** _A_POW
    log_gain = jnp.log(jnp.maximum(_A_TERM * Dj / (gmin * _A_LO), 1.0))
    alpha = alpha0 * _A_LO * jnp.exp(s * log_gain)            # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2