import jax.numpy as jnp

# STRUCTURALLY NEW vs both the SGDR-decaying-peak parent (+0.0450%) and the
# anti-phased-burst best (+0.0533%): NO cycles, NO restarts, NO bursts, NO
# plateau+spike. Three menu bets the lineage has never tried, composed:
#
#   lr    — WSD / TRAPEZOID (prior-art §6, explicitly untried): 3% linear
#           warmup, then a CONSTANT hot hold at 1.25*D until 55%. Every
#           cyclic parent only touches its peak lr momentarily and spends
#           half of each cycle at the 0.65*D trough; the steady hold
#           delivers strictly more total hot-step budget than any restart
#           scheme tried, with no mid-run cooling that freezes the layout
#           into a premature basin. From 55% a 1-SQRT decay (the WSD
#           literature's winning tail shape) lands EXACTLY on gamma_min at
#           the last step: it sheds heat fast right after the hold, then
#           spends a long slow tail at tiny lr — precisely where the
#           strict-alpha endgame needs many small feasibility-polishing
#           steps.
#   alpha — ε-CONSTRAINED SHRINKING TOLERANCE (prior-art §7.9, untried):
#           instead of floor -> plateau -> terminal spike, ONE smooth
#           geometric contraction of the enforced violation band. alpha
#           climbs geometrically from a 0.5*alpha0 exploration floor to the
#           PROVEN 5/5-seed-feasible terminal 5*alpha0*D/gamma_min, with a
#           cubic delay so the hot hold stays nearly free (≈1.8*alpha0 when
#           the hold ends) and strictness concentrates where lr is already
#           tiny. Continuous tightening replaces burst/spike debt-repayment:
#           the band the optimizer may violate contracts every step and
#           reaches gamma_min only at the end, so there is never a sudden
#           penalty shock — and never a hot step taken under a slack
#           penalty it must later repay.
#   betas — ONE-CYCLE ANTI-CORRELATED MOMENTUM (menu bets §2/§4, untried):
#           beta1 rises 0.08 -> 0.30 as lr cools past the hold — momentum
#           acting as an implicit ALM multiplier, accumulating persistent
#           constraint gradients so the moderate mid-run alpha enforces like
#           a large one — then the PROVEN terminal gate drops it to 0.02 so
#           the diverging alpha never rides momentum. beta2 keeps the proven
#           0.2 -> 0.9 transition, aligned with the start of the cool-down.
_F_WARM = 0.03       # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.55       # hot hold ends here; 1-sqrt decay to gamma_min at 100%
_HI = 1.25           # hold lr, in units of D (≈ the cyclic parents' MEAN hot lr,
                     # held constantly instead of visited twice per cycle)
_A_START = 0.5       # exploration penalty floor, in alpha0 units
_TERM_GAIN = 5.0     # terminal alpha = 5*alpha0*D/gamma_min (proven 5/5 scale)
_A_POW = 3.0         # cubic delay of the geometric contraction (back-loaded)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58    # beta2 transition just after the cool-down starts
_B2_WIDTH = 0.05
_B1_EXPL = 0.08      # low momentum while lr is hot (anti-correlated)
_B1_POLISH = 0.30    # raised momentum once lr cools — implicit ALM multiplier
_B1_CENTER = 0.60
_B1_WIDTH = 0.05
_B1_LO = 0.02        # near-zero momentum under the terminal alpha (proven)
_B1_GATE = 0.88
_B1_GWIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> constant hot hold at _HI*D -> 1-sqrt decay to gamma_min ---
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    decay = 1.0 - jnp.sqrt(p)                # 1 during hold, 0 exactly at the end
    lr_env = gmin + (_HI * Dj - gmin) * decay
    warm = jnp.minimum(frac / _F_WARM, 1.0)  # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: single geometric contraction of the enforced violation band ---
    # alpha0*_A_START at the start -> _TERM_GAIN*alpha0*D/gamma_min at the last
    # step, back-loaded cubically so the hold phase stays nearly unconstrained.
    s = jnp.clip(frac, 0.0, 1.0) ** _A_POW
    log_span = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_START), 1.0))
    alpha = alpha0 * _A_START * jnp.exp(s * log_span)

    # --- betas: anti-correlated beta1 rise, proven terminal gate, proven beta2 ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    b1_mid = _B1_EXPL + (_B1_POLISH - _B1_EXPL) * b1r
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_GATE) / _B1_GWIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * gate

    return lr, alpha, beta1, beta2