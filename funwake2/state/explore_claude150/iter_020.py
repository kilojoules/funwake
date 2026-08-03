import jax.numpy as jnp

# STRUCTURALLY NEW vs the whole restart/burst lineage (best +0.0533%): the
# three menu rows still untried by ANY attempt, composed into one schedule —
# a ONE-CYCLE / WSD lr (hold near c*D, then LINEAR cool-down; prior-art §2/§6:
# "hold-then-linear beats cosine/product decay"), an EPSILON-CONSTRAINED
# CONTINUOUSLY-CONTRACTING PENALTY BAND (§7.9: schedule alpha so the enforced
# violation band shrinks geometrically and reaches gamma_min only at the very
# end — no bursts, no logistic plateau, one smooth monotone contraction), and
# ANTI-CORRELATED MOMENTUM (one-cycle §2: beta1 low while lr is hot, rising as
# lr anneals so late fine-scale descent is coherent).
#
#   lr    — proven 3% warmup, then a sustained HOT HOLD tilting 1.60*D ->
#           1.40*D over the first 40% (mean lr far above the ~1.0*D cycling
#           average of every restart parent: strictly more total exploration
#           displacement, per the "higher/longer peak" parent hint), then one
#           straight WSD linear tail over the remaining 60% — more annealing
#           steps than any parent — landing EXACTLY on gamma_min at the last
#           step (proven).
#   alpha — starts on a 0.3*alpha0 exploration floor through the hot hold
#           (basin hops trade violation for AEP freely), then from 30% climbs
#           GEOMETRICALLY (log-linear in a backloaded power of progress) all
#           the way to the proven 5-seed-feasible terminal 5*alpha0*D/gamma_min.
#           The tolerated violation band ~1/alpha therefore contracts smoothly
#           from D-scale to gamma_min in lockstep with the lr descent: debt is
#           repaid continuously along the anneal instead of in bursts or one
#           endgame spike, and from ~80% onward alpha is STRICTER than the
#           proven plateau line at comparable lr — feasibility-safe.
#   betas — beta2 keeps the proven 0.2 -> 0.9 transition, re-centered on the
#           new descent start (RAdam-flavored: high beta2 once past the noisy
#           hot phase). beta1 is anti-correlated with lr: 0.05 during the hot
#           hold (no overshoot amplification), ramping linearly with descent
#           progress to 0.25 mid-anneal (momentum as implicit ALM multiplier,
#           menu bet), then the proven terminal gate to 0.02 so the diverging
#           alpha never rides momentum.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.40        # hot hold ends here; linear decay to gamma_min at 100%
_HI0 = 1.60           # hold entry level, in units of D
_HI1 = 1.40           # hold exit level — the linear tail starts from here
_A_LO = 0.3           # exploration penalty floor, in alpha0 units
_F_A0 = 0.30          # contraction starts here (delayed ramp, §7.3/§7.5)
_P_A = 2.4            # backloading of the geometric alpha climb
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.50     # beta2 transition just after the descent begins
_B2_WIDTH = 0.06
_B1_HOT = 0.05        # low momentum while lr is hot (one-cycle)
_B1_MID = 0.25        # momentum rises as lr anneals (anti-correlated)
_B1_LO = 0.02         # near-zero momentum during the terminal alpha climb
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> single WSD linear tail ---
    # hold_frac freezes at 1 past _F_HOLD, so the tail starts from _HI1 * D.
    hold_frac = jnp.clip(frac / _F_HOLD, 0.0, 1.0)
    hi = _HI0 + (_HI1 - _HI0) * hold_frac                     # 1.60*D -> 1.40*D tilt
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (hi * Dj - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor -> one continuous geometric contraction to the terminal value ---
    # g runs 0 -> 1 (backloaded) from _F_A0 to the last step; alpha is log-linear
    # in g, so the enforced violation band shrinks geometrically from ~D-scale
    # onto gamma_min exactly at the end (epsilon-constrained tolerance, §7.9).
    g = jnp.clip((frac - _F_A0) / (1.0 - _F_A0), 0.0, 1.0) ** _P_A
    log_full = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_LO), 1.0))
    alpha = alpha0 * _A_LO * jnp.exp(g * log_full)            # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; anti-correlated beta1 with terminal gate ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_base = _B1_HOT + (_B1_MID - _B1_HOT) * p               # rises as lr falls
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_base + (_B1_LO - b1_base) * b1r

    return lr, alpha, beta1, beta2