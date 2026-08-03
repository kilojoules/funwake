import jax.numpy as jnp

# STRUCTURALLY NEW vs the whole cosine-restart lineage: NO cycles at all.
# Every attempt so far (decaying-peak SGDR, anti-phase bursts, growing bursts)
# oscillates lr and spends roughly half the exploration budget in troughs.
# This is the two prior-art menu bets the lineage has never touched:
#
#   lr    — ONE-CYCLE / WSD (warmup–stable–decay, §2/§6): 3% linear warmup to
#           a HOT SUSTAINED HOLD (a gently tilted trapezoid, 1.45*D -> 1.15*D
#           over the first 55%) — far more *integrated* hot-lr time than any
#           restart schedule dared, with zero trough dead-time — then the
#           proven straight linear cool-down landing EXACTLY on gamma_min at
#           the final step. The menu's hypothesis verbatim: "hold near c*D,
#           then (near-)linear cool-down beats cosine/product decay".
#   alpha — ADMM-STYLE MODERATE CONSTANT penalty through the hold (§7 / listed
#           as untried), given a gentle dynamic-penalty tilt 1.0 -> 2.2*alpha0
#           (α₀·(1+Ct)^p flavour, §7.9): with no lr troughs to host repair
#           bursts, a steady moderate alpha keeps the violation band bounded
#           *continuously* while the hot hold trades structure for AEP — an
#           ε-constrained band that contracts instead of a debt/repay cycle.
#           The proven endgame is preserved untouched: logistic ramp onto the
#           bounded 6*alpha0 ALM plateau at the cool-down, then the
#           cubic-delayed geometric terminal climb from 78% ending at
#           5*alpha0*D/gamma_min — the restoration that made the lineage
#           5/5-seed feasible.
#   betas — proven beta2 transition 0.2 -> 0.9 at the cool-down start, plus
#           menu bet 4 finally applied to beta1: ANTI-CORRELATED WITH lr
#           (one-cycle, §2/§4) — reduced momentum 0.05 through the hot hold
#           (big steps never compound), native 0.1 through the polish, gated
#           to 0.02 during the terminal alpha spike as proven.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.55        # hold ends here; linear decay to gamma_min at 100%
_HI0 = 1.45           # hold entry lr, in units of D — hot and SUSTAINED
_HI1 = 1.15           # hold exit lr — tilted trapezoid, cool-down starts here
_A0 = 1.0             # ADMM-style moderate penalty at hold entry, alpha0 units
_A1 = 2.2             # gentle dynamic-penalty growth across the hold (§7.9)
_A_PLAT = 6.0         # bounded ALM plateau through the polish (proven)
_A_CENTER = 0.62      # logistic ramp onto the plateau (proven constants)
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.05        # reduced momentum during the hot hold (one-cycle)
_B1_HI = 0.1          # native momentum through the polish
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> linear tail onto gamma_min ---
    # fc freezes at 1 past _F_COOL, so the cool-down starts exactly at _HI1*D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    hold = (_HI0 + (_HI1 - _HI0) * fc) * Dj                   # sustained hot trapezoid
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (hold - gmin) * (1.0 - p)                 # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: moderate growing constant -> logistic plateau -> terminal climb ---
    # No bursts: with lr held hot continuously, feasibility pressure is applied
    # continuously and moderately, then the proven bounded endgame collects.
    a_hold = _A0 + (_A1 - _A0) * fc                           # contracting violation band
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    units = a_hold + (_A_PLAT - a_hold) * ramp                # in alpha0 units
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * units * jnp.exp(s * log_term)            # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; beta1 anti-correlated with lr ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_base = _B1_HOT + (_B1_HI - _B1_HOT) * b2r              # rises as lr cools
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_base + (_B1_LO - b1_base) * b1r                # gated in the spike

    return lr, alpha, beta1, beta2