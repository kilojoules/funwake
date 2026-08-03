import jax.numpy as jnp

# REGIME SHIFT vs both the 3-restart best (+0.0533%) and the 8-cycle fast-CLR
# parent (+0.0467%): the exploration phase abandons OSCILLATION entirely and
# moves to the WSD / HOLD regime — prior-art §6's strongest untried lr bet
# ("hold near c*D, then (near-)linear cool-down beats cosine/product decay").
# Every lineage member so far is a cosine oscillator, so at any moment it is
# on average only halfway up its envelope; here lr sits ON a hot flat-top
# plateau for ~80% of the exploration phase, interrupted only by SIX brief
# REPAIR NOTCHES (sharpened, q=6) where lr dives to 0.50*D and an anti-phased
# alpha burst collects the accumulated violation debt at minimum step size.
# The mechanism is the dual of the parent: instead of "mostly cool with hot
# kicks" (cosine duty cycle ~50%), it is "mostly HOT with brief repairs" —
# far more total time above 1.3*D than any cosine schedule can deliver at the
# same peak, which is exactly the quantity the lineage's best runs correlate
# with AEP gains. The hold tilts 1.50*D -> 1.00*D so late exploration cools,
# and the notch cadence (~800 steps) sits between the parent's 600 and the
# best's 1650 — debt is never carried long. The min_spacing cap (0.9*ms) on
# the hold is kept as the generalization guard: a sustained hold must never
# exceed what a tight-spacing farm can tolerate per step.
# The PROVEN endgame is preserved verbatim: linear lr tail landing exactly on
# gamma_min, logistic alpha ramp to the bounded 6*alpha0 ALM plateau, the
# cubic-delayed geometric climb to the 5/5-seed-feasible terminal
# 5*alpha0*D/gamma_min, beta2 0.2 -> 0.9 at cool-down, and beta1 gated to
# 0.02 through the terminal spike, with the per-notch 0.05 momentum dip so
# momentum never drags turbines back out mid-repair.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.60        # exploration ends here; linear decay to gamma_min at 100%
_N_NOTCH = 6.0        # six brief repair notches inside the hold phase
_HI0 = 1.50           # hold level at the start of the plateau, in D units
_HI1 = 1.00           # hold level at cool-down start; proven tail starts here
_LO = 0.50            # notch-bottom lr, in D units — repairs at small steps
_Q = 6.0              # sharpens notches: lr stays ON the hold ~80% of the time
_MS_CAP = 0.9         # hold never exceeds 0.9 * min_spacing (proven guard)
_A_LO = 0.35          # alpha floor on the hold, in alpha0 units (free ascent)
_A_B0 = 2.0           # first repair-burst height, in alpha0 units
_A_B1 = 9.0           # last repair-burst height, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.64      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_NOTCH = 0.05      # reduced momentum inside each repair notch
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    ms = jnp.asarray(min_spacing) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot HOLD with 6 sharp repair notches -> tail ---
    # c = 1 at fc = 0, 1 and at every cycle boundary; notch = (1-c)^Q is ~0 on
    # the hold and rises to 1 only briefly at the N interior repair points, so
    # fc freezing at 1 past _F_COOL pins the cool-down start exactly on the
    # hold's end value, _HI1 * D — the proven linear-tail starting point.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    c = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_NOTCH * fc))
    notch = (1.0 - c) ** _Q                                   # ~0 on hold, 1 in notches
    hold = _HI0 + (_HI1 - _HI0) * fc                          # tilted flat-top envelope
    hold_lr = jnp.maximum(jnp.minimum(hold * Dj, _MS_CAP * ms), _LO * Dj)
    lr_expl = hold_lr - (hold_lr - _LO * Dj) * notch          # brief dives to 0.5*D
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_expl - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + anti-phased notch bursts -> plateau -> terminal climb ---
    # bursts share the notch shape, so every repair happens at minimum step
    # size; burst height grows 2 -> 9 alpha0 across the hold so the layout
    # enters the polish phase already near-feasible, and notch = 0 for
    # frac >= _F_COOL hands alpha cleanly to the proven logistic/plateau/climb.
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-notch beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * notch            # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2