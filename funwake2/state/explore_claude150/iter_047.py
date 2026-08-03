import jax.numpy as jnp

# STRUCTURAL BREAK from the SGDR+burst best (+0.0533%): the lineage is now
# saturated on CYCLIC schedules (restarts, anti-phased bursts) and small tweaks
# of them have flatlined. This attempt takes the three menu rows still untried
# by ANY parent, as one coherent design:
#
#   lr    — ONE-CYCLE / WSD TRAPEZOID (prior-art §6/§2, the table's top row:
#           "hold near c*D then (near-)linear cool-down beats cosine/product
#           decay"). 3% linear warmup, then a SUSTAINED hot hold that tilts
#           1.50*D -> 1.10*D over the first 55% — more total exploration heat
#           than the SGDR restarts deliver (their troughs waste half the phase
#           at 0.65*D), and hotter early, exactly as the parent guidance asks —
#           then the proven straight linear tail landing exactly on gamma_min
#           at the last step. No restarts: heat is spent continuously, and the
#           layout gets one long uninterrupted anneal instead of three short ones.
#   alpha — EPS-CONSTRAINED SHRINKING TOLERANCE (§7.9), fully decoupled from lr.
#           A flat 0.5*alpha0 exploration floor through the hold (violation is
#           freely traded for AEP while basins are still being chosen), then a
#           SINGLE geometric contraction law: alpha rises so the enforced
#           violation band ~ D/alpha shrinks smoothly from ~D-scale to
#           gamma_min, reaching the 5/5-seed-proven terminal 5*alpha0*D/gamma_min
#           exactly at the last step. The power-2.3 back-loading makes this one
#           law reproduce plateau-then-spike behavior (a few alpha0 mid-decay,
#           near-strict only in the last ~10%) without the hand-built
#           floor+logistic+spike stack — feasibility pressure is monotone, so
#           there is no late debt for bursts to repay.
#   betas — ONE-CYCLE MOMENTUM-AS-ALM (§2/§4, menu bet 4's untried half).
#           beta1 anti-correlates with lr: native 0.1 while hot (proven for
#           exploration), ramping UP to 0.5 through the cool-down so the
#           integrated constraint gradient acts as an implicit ALM multiplier —
#           letting the moderate mid-decay alpha enforce feasibility — then the
#           proven terminal gate drops it to 0.02 so momentum never fights the
#           final spike. beta2 keeps the proven 0.2 -> 0.9 transition, aligned
#           with the start of the cool-down.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.55        # hot hold ends here; linear decay to gamma_min at 100%
_HI0 = 1.50           # hold entry lr, in units of D — hot, but sustained not spiked
_HI1 = 1.10           # hold exit lr; the linear tail starts from here
_A_FLOOR = 0.5        # decoupled exploration penalty floor, in alpha0 units
_F_ACLIMB = 0.55      # tolerance contraction starts with the cool-down
_POW = 2.3            # back-loads the contraction: lenient mid-decay, strict late
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven feasible scale)
_B2_LO = 0.2          # native adaptive scaling while exploring (proven)
_B2_HI = 0.9          # tamed adaptive-lr variance in the feasibility phase (proven)
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_EXPLORE = 0.1     # native momentum during the hot hold (proven)
_B1_ALM = 0.5         # raised momentum in the cool-down: implicit ALM multiplier
_B1_UP_CENTER = 0.68  # momentum ramps up once lr has meaningfully cooled
_B1_UP_WIDTH = 0.05
_B1_TERM = 0.02       # near-zero momentum during the terminal alpha spike (proven)
_B1_DN_CENTER = 0.88
_B1_DN_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> linear cool-down to gamma_min ---
    # h freezes at 1 past _F_HOLD, so the tail starts exactly at _HI1 * D.
    h = jnp.clip(frac / _F_HOLD, 0.0, 1.0)
    lr_hold = (_HI0 + (_HI1 - _HI0) * h) * Dj                 # sustained, gently tilting
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: flat floor, then one geometric eps-contraction to the terminal scale ---
    # Implied enforced-violation band ~ TERM_GAIN*D/(alpha/alpha0) shrinks smoothly
    # to gamma_min exactly at the last step; monotone, so no mid-run debt cycles.
    u = jnp.clip((frac - _F_ACLIMB) / (1.0 - _F_ACLIMB), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_FLOOR), 1.0))
    alpha = alpha0 * _A_FLOOR * jnp.exp(u * log_term)         # ends at 5*alpha0*D/gmin

    # --- betas: one-cycle beta1 (up in the cool-down, gated off for the spike) ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_mid = _B1_EXPLORE + (_B1_ALM - _B1_EXPLORE) * b1_up
    b1_dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    beta1 = b1_mid + (_B1_TERM - b1_mid) * b1_dn

    return lr, alpha, beta1, beta2