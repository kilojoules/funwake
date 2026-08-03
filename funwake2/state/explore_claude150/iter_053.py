import jax.numpy as jnp

# STRUCTURALLY NEW vs both the SGDR-restart lines: NO restarts, NO bursts.
# This is the ONE-CYCLE / WSD "tilted hot hold" (prior-art §2/§6 — the menu's
# top untried lr bet: "hold near c*D, then (near-)linear cool-down beats
# cosine/product decay") composed with the last genuinely untried search-state
# direction: an ADMM-STYLE MODERATE, DECOUPLED PENALTY held through the whole
# hot phase (§7.9 dynamic-penalty flavor), instead of a floor-plus-bursts or a
# 1/lr coupling. The proven feasibility endgame is preserved verbatim.
#
#   lr    — 3% linear warmup (proven), then a SUSTAINED TILTED HOLD from
#           1.60*D down to 1.15*D across the first 55%: the restart lines
#           spend most of the hot phase in 0.55-0.65*D troughs, so their
#           time-integrated exploration heat is only ~1.0*D; the hold
#           delivers ~1.4*D continuously — a strictly hotter, longer early
#           phase, exactly what the parent hint asks for — with no trough
#           downtime. From 55% the proven WSD straight linear tail lands
#           exactly on gamma_min at the last step.
#   alpha — ADMM-style bounded moderate penalty, fully decoupled from lr:
#           a gentle graduated climb 1.2 -> 2.4*alpha0 across the hold
#           (epsilon-constraint band contraction, §7.9). This matches the
#           TIME-AVERAGED penalty (~2*alpha0) that kept the burst line's
#           violation debt repayable, but applies it continuously, so the
#           layout never accumulates the deep violation spikes that bursts
#           must claw back — the hold's extra heat is licensed by constant
#           moderate containment. Then the proven endgame untouched: logistic
#           ramp (center 0.60) onto the bounded 6*alpha0 ALM plateau, and the
#           cubic-delayed geometric climb from 78% landing on the
#           5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#   betas — beta2 0.2 -> 0.9 logistic at the cool-down start (proven).
#           beta1 realizes menu bet: ANTI-CORRELATED WITH lr (one-cycle
#           momentum): 0.05 during the hot hold so big steps never ride
#           momentum across the boundary, rising to the proven 0.1 for the
#           cool-down polish, then gated to the proven 0.02 during the
#           terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.55        # hold ends here; straight linear decay to gamma_min at 100%
_HI0 = 1.60           # hold entry lr, in units of D — hot and SUSTAINED
_HI1 = 1.15           # hold exit lr; the linear tail starts from here
_A_LO = 1.2           # ADMM moderate penalty at hold entry, in alpha0 units
_A_HI = 2.4           # graduated to this by hold exit (violation band contracts)
_A_PLAT = 6.0         # bounded ALM plateau through the polish (proven)
_A_CENTER = 0.60      # logistic ramp onto the plateau, at the cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.05        # low momentum while lr is hot (one-cycle anti-correlation)
_B1_POLISH = 0.1      # proven native momentum for the cool-down polish
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_RISE_W = 0.05
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> straight linear tail onto gamma_min ---
    # fh runs 0 -> 1 across the hold and freezes at 1, so the tail starts
    # exactly from the hold-exit value _HI1 * D.
    fh = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    hold = (_HI0 + (_HI1 - _HI0) * fh) * Dj                   # sustained hot plateau
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (hold - gmin) * (1.0 - p)                 # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: graduated ADMM moderate penalty -> plateau -> terminal climb ---
    # Decoupled from lr everywhere: bounded moderate containment during the
    # hold, then the proven logistic ramp / plateau / geometric terminal spike.
    expl = _A_LO + (_A_HI - _A_LO) * fh                       # contracting violation band
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    base = expl + (_A_PLAT - expl) * ramp                     # in alpha0 units
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * base * jnp.exp(s * log_term)             # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; beta1 anti-correlated with lr ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_rise = 1.0 / (1.0 + jnp.exp(-(frac - _F_COOL) / _B1_RISE_W))
    b1_mid = _B1_HOT + (_B1_POLISH - _B1_HOT) * b1_rise       # 0.05 hot -> 0.1 polish
    b1g = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1g                  # gated for terminal spike

    return lr, alpha, beta1, beta2