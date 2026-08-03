import jax.numpy as jnp

# STRUCTURAL REGIME SHIFT vs both the anti-phased-burst best (+0.0533%) and
# the fast-CLR parent (+0.0467%): every schedule in the lineage is CYCLIC
# (restarts, kicks, bursts, pulses). This one has NO cycles at all — it is the
# two top-row prior-art bets that no attempt has embodied, composed:
#
#   lr    — WSD / ONE-CYCLE (prior-art §2/§6): short linear warmup, then a
#           long STABLE HOT PLATEAU at 1.45*D held from 4% to 55% of the run,
#           then the proven straight linear cool-down landing exactly on
#           gamma_min at the last step. The plateau gives far more total time
#           above 1.3*D than any cyclic parent (the cyclic schedules only
#           touch their hot peaks briefly), which is the "hotter/longer
#           exploration" the search state asks for — and §6 says the
#           hold-then-linear-decay shape beats cosine/product decay outright.
#           The gen-98 generalization guard is kept: the plateau is capped at
#           0.9*min_spacing so tight-spacing farms are never blown through.
#   alpha — ε-CONSTRAINED SHRINKING TOLERANCE (prior-art §7.9, the one menu
#           row never tried): instead of floor + pulses + logistic plateau +
#           terminal spike, alpha is a SINGLE cubic-back-loaded geometric
#           contraction of the enforced violation band, from a wide-band
#           0.5*alpha0 at the start to the proven 5/5-seed-feasible terminal
#           5*alpha0*D/gamma_min at the very last step. The band stays wide
#           (< ~2*alpha0) through the entire hot plateau — full exploration
#           freedom — then contracts continuously through the cool-down, so
#           violation debt is repaid smoothly as lr shrinks rather than in
#           discrete bursts; by ~90% alpha is already in the hundreds of
#           alpha0 and the endgame reproduces the proven terminal-spike
#           strictness without ever needing a discontinuous phase.
#   betas — beta2 keeps the proven 0.2 -> 0.9 logistic at the cool-down
#           start. beta1 is the one-cycle ANTI-CORRELATION with lr (§2, menu
#           bet 4): 0.05 on the hot plateau (momentum never compounds a
#           1.45*D step across a boundary), rising toward the native 0.1 as
#           lr decays into the polish phase, then the proven gate down to
#           0.02 through the terminal alpha contraction.
_F_WARM = 0.04        # linear lr warmup over the first 4%
_F_DECAY = 0.55       # stable hot plateau ends here; linear decay to gamma_min
_HI = 1.45            # plateau lr in units of D — long-hold exploration
_MS_CAP = 0.9         # plateau lr never exceeds 0.9*min_spacing (gen-98 guard)
_LO_FLOOR = 0.5       # cap can never push the plateau below 0.5*D
_A_LO = 0.5           # starting alpha in alpha0 units (widest tolerance band)
_POW = 3.0            # cubic back-loading of the band contraction (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.05        # low momentum while lr sits on the hot plateau
_B1_MID = 0.1         # native momentum as lr decays into the polish phase
_B1_LO = 0.02         # near-zero momentum through the terminal contraction
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    ms = jnp.asarray(min_spacing) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> stable hot plateau -> linear cool-down (WSD/one-cycle) ---
    hi_lr = jnp.maximum(jnp.minimum(_HI * Dj, _MS_CAP * ms), _LO_FLOOR * Dj)
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    p = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0)
    lr = (gmin + (hi_lr - gmin) * (1.0 - p)) * warm   # exact landing on gamma_min

    # --- alpha: one geometric tolerance-band contraction, cubic back-loaded ---
    # r stays tiny through the plateau (alpha < ~2*alpha0 -> free exploration),
    # then the band contracts continuously through the cool-down and lands on
    # the proven terminal strictness exactly at the last step.
    r = jnp.clip(frac, 0.0, 1.0) ** _POW
    log_end = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_LO), 1.0))
    alpha = alpha0 * _A_LO * jnp.exp(r * log_end)     # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 ramp + one-cycle beta1 anti-correlated with lr ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    lr_n = lr / jnp.maximum(hi_lr, 1e-30)             # ~1 on plateau, ~0 at end
    b1_exp = _B1_MID - (_B1_MID - _B1_HOT) * lr_n     # low momentum when lr is hot
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2