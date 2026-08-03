import jax.numpy as jnp

# MOMENTUM REGIME SHIFT vs the flat-top-hold best (+0.0577%): every schedule
# in this lineage has explored with the near-momentum-free NATIVE moments
# (beta1~0.1, beta2=0.2 — sign-SGD-like, diffusive dithering) and spent all
# of its search budget reshaping the lr waveform. The one axis the prior-art
# menu flags that the lineage has never touched (§2 one-cycle, §4 momentum
# ramps, and the survey's own "standard Adam vs TopFarm betas" ablation) is
# the SEARCH DYNAMICS themselves. This schedule keeps the proven duty-cycle-
# inverted lr waveform (the mechanism behind the current best) and changes
# WHAT KIND of explorer runs on top of it:
#   * on the hot hold the moments become HEAVY-BALL / standard-Adam-like
#     (beta1=0.88, beta2=0.97): gradients are direction-smoothed over ~10
#     steps, so turbines make coherent BALLISTIC transits across the polygon
#     — large-scale rearrangement between crowded and sparse regions — rather
#     than local diffusive wiggles. In Adam the step MAGNITUDE stays ~lr, so
#     per-step boundary overshoot is no worse than the parent's; only the
#     trajectory coherence changes.
#   * beta1 is ANTI-CORRELATED with lr (one-cycle, §2): inside each repair
#     notch it crashes to 0.03 and beta2 to the native 0.2, so momentum dies
#     within a step or two and every notch is the same surgical, native-style
#     repair the lineage has already proven — ballistic debt can never be
#     carried THROUGH a repair window.
#   * a logistic HANDOFF centered just before cool-down returns the moments
#     to the proven polish values (beta1 0.1, beta2 0.9) so the entire tail,
#     plateau and terminal spike run with exactly the 5/5-seed-feasible
#     dynamics.
# AEP push licensed by the new regime: the hold opens slightly hotter
# (1.6*D vs 1.5*D), retaining the parent's min_spacing cap so tight-spacing
# farms are protected. The bursts strengthen slightly (4 -> 9 alpha0) to
# repay the extra ballistic debt per notch. The PROVEN feasibility machinery
# is otherwise verbatim: 3 sin^12 notches to 0.35*D, exploration ending at
# 62%, linear lr tail landing exactly on gamma_min, logistic alpha ramp to
# the bounded 6*alpha0 ALM plateau, cubic-delayed geometric climb to the
# terminal 5*alpha0*D/gamma_min spike, and the beta1 gate to 0.02 through it.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three repair notches inside the exploration phase (proven)
_Q = 6.0              # notch = sin^(2Q); ~80% of each cycle at full hold lr
_HI0 = 1.6            # initial hold level, in D units — slightly hotter than best
_HI1 = 1.1            # final hold level; the linear tail starts from here (proven)
_LO = 0.35            # notch-bottom lr — deep, surgical repair windows (proven)
_MS_CAP = 0.9         # hold lr never exceeds 0.9*min_spacing (parent's guard)
_A_LO = 0.4           # exploration penalty floor at full heat, in alpha0 units
_A_B0 = 4.0           # first restoration burst height (strengthened from 3)
_A_B1 = 9.0           # last restoration burst height (strengthened from 8)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B1_BALL = 0.88       # NEW: heavy-ball momentum on the hot hold
_B1_NOTCH = 0.03      # momentum killed inside each repair notch
_B1_POLISH = 0.1      # proven polish momentum after the handoff
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88     # terminal beta1 gate (proven)
_B1_WIDTH = 0.03
_B2_BALL = 0.97       # NEW: standard-Adam-like smoothing on the hot hold
_B2_NOTCH = 0.2       # native fast adaptation inside repair notches
_B2_HI = 0.9          # proven polish beta2
_HAND_CENTER = 0.60   # ballistic -> proven-polish moment handoff (pre-cooldown)
_HAND_WIDTH = 0.025


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    ms = jnp.asarray(min_spacing) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat-top hold with three deep narrow notches -> tail ---
    # notch = sin(pi*N*fc)^(2Q): 0 at fc=0 and fc=1 (tail launches from the
    # clean hold level _HI1*D), ~0 for ~80% of each cycle, 1 briefly at each
    # cycle midpoint. fc freezes at 1 past _F_COOL.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    notch = jnp.sin(jnp.pi * _N_CYC * fc) ** (2.0 * _Q)
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    hi_lr = jnp.maximum(jnp.minimum(hi * Dj, _MS_CAP * ms), _LO * Dj)  # spacing-aware cap
    lr_hold = _LO * Dj + (hi_lr - _LO * Dj) * (1.0 - notch)
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + notch-synchronized growing bursts -> plateau -> climb ---
    # Bursts fire exactly inside the lr notches (repair when steps are small
    # AND momentum is dead) and vanish for frac >= _F_COOL; the proven bounded
    # endgame then takes over.
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: ballistic on the hold, native in notches, proven after handoff ---
    # beta1 anti-correlated with lr (one-cycle): heavy-ball at full heat,
    # crashed inside every notch. The logistic handoff returns both moments to
    # the proven polish values before the alpha plateau engages, and the
    # proven terminal gate then takes beta1 to 0.02 through the spike.
    hand = 1.0 / (1.0 + jnp.exp(-(frac - _HAND_CENTER) / _HAND_WIDTH))
    b1_exp = _B1_BALL - (_B1_BALL - _B1_NOTCH) * notch        # ballistic, notch-killed
    b1_mid = b1_exp + (_B1_POLISH - b1_exp) * hand
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1r
    b2_exp = _B2_BALL - (_B2_BALL - _B2_NOTCH) * notch        # smooth hold, native notch
    beta2 = b2_exp + (_B2_HI - b2_exp) * hand

    return lr, alpha, beta1, beta2