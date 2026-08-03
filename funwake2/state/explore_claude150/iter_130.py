import jax.numpy as jnp

# STRUCTURALLY NEW vs both the parent (+0.0450%, cosine SGDR restarts) and the
# reigning best (+0.0577%, notched flat-top hold): ALL mid-run repair machinery
# is moved OFF the lr axis and ONTO the momentum axis. The lr waveform becomes
# a PURE ONE-CYCLE / WSD hold — no restarts, no notches, no troughs — so every
# exploration step runs at full heat (cumulatively hotter than either prior
# schedule, per the "push the peak higher/longer" directive). Feasibility is
# then enforced by the prior-art menu's one completely untried lever:
# **momentum as an implicit ALM multiplier** (one-cycle anti-correlation +
# Sutskever increasing ramp, §2/§4). Every attempt so far has kept beta1
# pinned near 0.1; here beta1 ramps 0.1 -> 0.8 exactly as lr cools, so the
# Adam first moment accumulates the plateau-weighted constraint gradient into
# a running multiplier estimate — a *moderate, bounded* alpha enforces the
# constraints without the notch/burst duty cycle that spent ~20% of the
# exploration budget below basin-hopping heat.
#
#   lr    — 3% linear warmup (proven) -> uninterrupted flat-top hold, envelope
#           easing 1.55*D -> 1.15*D (hold START above the hottest level tried,
#           sustained rather than momentary) -> exploration ends at the proven
#           62% -> proven straight linear tail landing exactly on gamma_min at
#           the last step.
#   alpha — the parent's PROVEN no-burst architecture, unchanged: 0.4*alpha0
#           exploration floor (frees the hot hold to trade violation for AEP),
#           one logistic ramp just after cool-down to the bounded 6*alpha0
#           ALM plateau, then the 5/5-seed-feasible cubic-delayed geometric
#           climb from 78% to the terminal 5*alpha0*D/gamma_min spike. The
#           parent proved this repays hot-phase debt WITHOUT lr notches.
#   beta1 — the new axis. 0.1 (native, proven for exploration) while lr is
#           hot; logistic ramp UP to 0.8 centered at 70% — high momentum only
#           once alpha sits on its plateau and lr has cooled, so the moment
#           vector integrates constraint gradients like an augmented-
#           Lagrangian multiplier while AEP still refines; then the proven
#           safety gate drops it to 0.02 through the terminal alpha spike so
#           the diverging penalty never rides accumulated momentum.
#   beta2 — proven transition only: 0.2 -> 0.9 logistic at the cool-down
#           start (adaptive-variance clamp absorbs the alpha-phase curvature).
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_HI0 = 1.55           # hold level at warmup end, in units of D — sustained
_HI1 = 1.15           # hold level at cool-down; the linear tail starts here
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.64      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_EXPLORE = 0.1     # native low momentum while lr is at full heat
_B1_POLISH = 0.8      # implicit-ALM momentum during the plateau polish window
_B1_TERM = 0.02       # near-zero momentum through the terminal alpha spike
_B1_UP_CENTER = 0.70  # momentum ramps up only after alpha reaches its plateau
_B1_UP_WIDTH = 0.04
_B1_DOWN_CENTER = 0.88  # proven safety gate for the terminal spike
_B1_DOWN_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> uninterrupted flat-top hold -> linear tail ---
    # fc freezes at 1 past _F_COOL, so the tail launches cleanly from _HI1*D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    lr_hold = (_HI0 + (_HI1 - _HI0) * fc) * Dj                # slowly easing hold
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor -> logistic ramp -> bounded plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    plateau = _A_LO + (_A_PLAT - _A_LO) * ramp                # in alpha0 units
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * plateau * jnp.exp(s * log_term)          # ends at 5*alpha0*D/gmin

    # --- beta2: proven adaptive-variance clamp at cool-down ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    # --- beta1 one-cycle: low while hot -> ALM-multiplier ramp -> spike gate ---
    up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_mid = _B1_EXPLORE + (_B1_POLISH - _B1_EXPLORE) * up
    down = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DOWN_CENTER) / _B1_DOWN_WIDTH))
    beta1 = b1_mid + (_B1_TERM - b1_mid) * down

    return lr, alpha, beta1, beta2