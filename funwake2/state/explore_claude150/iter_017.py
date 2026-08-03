import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the two menu rows
# still untouched by the lineage, fused into one mechanism —
#   lr    : WSD / trapezoid (prior-art §6, §2) — NO restarts, NO cosine. A short
#           warmup, then a long TILTED HOLD that keeps the step size hot for 57%
#           of the run (far more *integrated* heat than any restart schedule:
#           the best's exploration averaged ~1.0*D; this holds 1.45*D -> 1.05*D),
#           then the proven straight linear tail landing exactly on gamma_min at
#           the last step. Tests the frozen hypothesis "hold near c*D, then
#           (near-)linear cool-down beats cosine/product decay", and the parent
#           guidance "higher/LONGER lr peak early" taken literally: the peak IS
#           the phase. The hold's tilt is chosen so the decay begins from
#           1.05*D — the exact coolest-peak value the proven tail starts from.
#   alpha : epsilon-CONSTRAINED SHRINKING TOLERANCE (§7.9) — no plateau, no
#           bursts, no logistic+spike stack. One continuous law: alpha follows
#           a single delayed GEOMETRIC contraction of the enforced violation
#           band, alpha = alpha0 * exp(interp) from the 0.5*alpha0 exploration
#           floor up to the terminal value. Because penalty gradients vanish
#           once a turbine is feasible, the monotone contraction never taxes
#           feasible AEP polish, yet it passes through ALM-plateau magnitudes
#           (~3*alpha0 by 70%, ~16*alpha0 by 78%) earlier than the old spike,
#           buying feasibility margin for the hotter hold. It lands EXACTLY on
#           the 5/5-seed-proven terminal 5*alpha0*D/gamma_min at the last step,
#           so the tolerance band contracts to gamma_min only at the end.
#   betas : the proven, validated transitions untouched — beta2 0.2 -> 0.9 as
#           the cool-down begins (absorbs the growing alpha*constraint
#           curvature), beta1 0.1 -> 0.02 gated at 88% so momentum never
#           carries turbines back across the boundary during the endgame.
# Clean ablation: sustained heat + continuous tolerance contraction vs cyclic
# heat + episodic repair, with the terminal restoration fully preserved.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_DECAY = 0.60       # hold ends here; straight linear tail to gamma_min at 100%
_LR_HI = 1.45         # hold entry lr, in units of D — sustained, not a blip
_LR_LO = 1.05         # hold exit lr = proven tail start value, in units of D
_A_FLOOR = 0.5        # exploration penalty floor, in alpha0 units
_F_CONTRACT = 0.55    # tolerance contraction starts just before the lr decay
_P_CONTRACT = 1.6     # back-loads the geometric climb (delayed ramp, §7.3/7.5)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven 5/5 scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum during the terminal contraction
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> long tilted hold -> straight linear tail ---
    # h freezes at 1 past _F_DECAY, so the tail launches from exactly _LR_LO*D.
    h = jnp.clip(frac, 0.0, _F_DECAY) / _F_DECAY
    lr_hold = (_LR_HI + (_LR_LO - _LR_HI) * h) * Dj
    p = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: single delayed geometric tolerance contraction ---
    # w = 0 through the hot hold (alpha pinned at the exploration floor), then
    # a back-loaded log-space climb; w = 1 at the last step lands alpha exactly
    # on the proven terminal 5*alpha0*D/gamma_min.
    s = jnp.clip((frac - _F_CONTRACT) / (1.0 - _F_CONTRACT), 0.0, 1.0)
    w = s ** _P_CONTRACT
    log_lo = jnp.log(_A_FLOOR)
    log_hi = jnp.log(jnp.maximum(_TERM_GAIN * Dj / gmin, 1.0))
    alpha = alpha0 * jnp.exp(log_lo + (log_hi - log_lo) * w)

    # --- betas: proven transitions, untouched ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2