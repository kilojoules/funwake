import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst/SGDR best (+0.0533%): the two menu directions
# no lineage member has tried, fused into one coherent regime — a WSD /
# trapezoid lr (prior-art §6: warmup -> long HOT STABLE phase -> straight
# decay to gamma_min; hypothesized to beat cosine/product decay) driving an
# ADMM-STYLE CONSTANT MODERATE PENALTY (explicitly listed untried). The whole
# cosine-restart machinery is removed: no cycles, no anti-phased bursts.
#
#   Mechanism bet: the best's restarts spend most of the exploration budget
#   either accelerating out of a trough or braking into one — only the brief
#   peaks actually hop basins. A sustained tilted plateau at 1.40->1.20 * D
#   delivers MORE cumulative hot-phase displacement than three decaying peaks
#   (parent's cyclic mean lr ~ 1.0*D over the same window), while a constant
#   1.5*alpha0 companion penalty (ADMM-style: fixed, moderate, never coupled
#   to lr) keeps violation debt bounded the entire time instead of letting it
#   balloon at each peak and clawing it back at each trough. The layout that
#   enters the cool-down is therefore both more explored AND less indebted.
#
#   lr    — 4% linear warmup -> stable phase tilted 1.40*D -> 1.20*D until
#           60% (one-cycle/WSD trapezoid, §2/§6) -> the proven straight
#           linear tail landing exactly on gamma_min at the last step. The
#           40%-long anneal replaces the settling work the restart troughs
#           used to do, but does it once, monotonically, with no re-heating
#           to undo it.
#   alpha — ADMM-style CONSTANT 1.5*alpha0 through warmup + stable phase
#           (bounded debt, well-conditioned, decoupled from lr), then the
#           PROVEN endgame kept verbatim for feasibility: logistic ramp to
#           the bounded 6*alpha0 ALM plateau at cool-down, and the cubic-
#           delayed geometric climb from 78% landing on the 5/5-seed-feasible
#           terminal 5*alpha0*D/gamma_min spike.
#   betas — proven transitions only, no per-cycle gymnastics: native 0.1/0.2
#           while hot; beta2 logistic 0.2 -> 0.9 aligned with the cool-down
#           (absorbs the alpha-ramp curvature, §4); beta1 gated 0.1 -> 0.02
#           inside the terminal spike so momentum cannot carry turbines back
#           across the boundary while the spike pulls them in.
_F_WARM = 0.04        # linear lr warmup over the first 4%
_F_COOL = 0.60        # stable phase ends here; linear decay to gamma_min at 100%
_HI0 = 1.40           # stable-phase lr at its start, in units of D
_HI1 = 1.20           # stable-phase lr at cool-down start (gentle tilt)
_A_STAB = 1.5         # ADMM-style constant penalty during exploration, alpha0 units
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
_B1_HI = 0.1          # native momentum while exploring and annealing
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot plateau -> straight linear tail (WSD) ---
    # fs freezes at 1 past _F_COOL, so the tail starts exactly from _HI1 * D.
    fs = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    lr_stab = (_HI0 + (_HI1 - _HI0) * fs) * Dj                # gently tilted plateau
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_stab - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> bounded plateau -> proven terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_STAB + (_A_PLAT - _A_STAB) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions only ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2